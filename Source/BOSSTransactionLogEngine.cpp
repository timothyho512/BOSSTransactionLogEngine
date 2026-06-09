#include <BOSS.hpp>
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <Utilities.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <string>
#include <cstdint>

#ifndef BOSS_WAL_INSTRUMENTATION
#define BOSS_WAL_INSTRUMENTATION 1
#endif

using std::string_literals::operator""s;
using boss::utilities::operator""_;
using boss::ComplexExpression;
using boss::Span;
using boss::Symbol;

using boss::Expression;

using boss::expressions::generic::get;
using boss::expressions::generic::get_if;

// ============================================================
// WAL Index Structure
// ============================================================
//
// The WAL is indexed by (tableName, rowID).
// Each row gets its own RowBucket.
//
// RowBucket uses per-column entry lists with sequence numbers:
//   - columnEntries: maps each column name to its own vector of WALEntry
//   - deleteEntry:   stores the latest Delete for this row (if any)
//   - nextSeq:       monotonically increasing counter for global arrival order
//
// A WALEntry holds the raw expression and a sequence number.
// The sequence number encodes the global arrival order across ALL columns
// in the bucket — so comparing seq values tells you which write came first
// regardless of which column was written.
//
// This design enables:
//   1. Same-column dependency folding — unchanged from before
//   2. Cross-column dependency resolution — when resolving column X at seq=s,
//      find the latest entry for dependency column Y with seq < s
//   3. Selective column flush — each column's vector is independent
//
// Capture:  O(c) per write
// Flush:    O(k_col) per column, O(k_col_dep) for each cross-column dependency
 
using RowID = std::variant<int32_t, int64_t>;
 
// normalise RowID to int64_t for use as map key
static int64_t toInt64(RowID const& id) {
  return std::visit([](auto const& v) { return static_cast<int64_t>(v); }, id);
}
 
using WALKey = std::pair<std::string, int64_t>; // (tableName, rowID)
 
struct WALKeyHash {
  size_t operator()(WALKey const& k) const {
    size_t h1 = std::hash<std::string>{}(k.first);
    size_t h2 = std::hash<int64_t>{}(k.second);
    // combine the two hashes — shift h2 to avoid trivial cancellation
    return h1 ^ (h2 * 2654435761ULL);
  }
};

// One WAL entry: the raw expression and its global arrival sequence number
struct WALEntry {
  Expression expr;
  int seq;
  bool isBlindWrite = false; // true if this entry is a blind write (no Symbol dependencies)
};

// SelectTarget — result of parsing a Select or Project(Select(...)) expression
// key:     (tableName, rowID) — which row to flush
// columns: which columns to flush — empty means flush all columns (bare Select case)
struct SelectTarget {
  WALKey key;
  std::vector<std::string> columns;
};

 
struct RowBucket {
  // per-column entry lists — each column has its own independent vector
  // key = column name, value = entries in arrival order (seq ascending)
  std::unordered_map<std::string, std::vector<WALEntry>> columnEntries;
 
  // latest Delete for this row — stored separately since Delete is not a column write
  // seq inside deleteEntry tells us which column entries it supersedes (those with seq < deleteEntry.seq)
  std::optional<WALEntry> deleteEntry;
 
  // monotonically increasing counter — assigned to each new entry across all columns
  int nextSeq = 0;
};
 
// the WAL index: (tableName, rowID) → bucket
static std::unordered_map<WALKey, RowBucket, WALKeyHash> walIndex;
 
// insertion order of keys — so flush emits entries in the order rows were first touched
static std::vector<WALKey> walOrder;
 
// total number of entries across all buckets — for threshold check
static size_t walTotalEntries = 0;
 
// Threshold for WAL flush
static size_t WAL_THRESHOLD = 200'000'000;
// V2A: reserve capacity for per-column WAL entry vectors.
// The profiling run used chain length 100, so 128 gives enough space
// for a typical hot row-column chain while keeping memory overhead modest.
// reserve() changes vector capacity only; it does not create WAL entries.
static constexpr size_t WAL_COLUMN_ENTRY_RESERVE = 128;

// ============================================================
// WAL instrumentation counters
// ============================================================
//
// These counters are for benchmark analysis only.
// They do not change WAL semantics. They only count:
//   - how many logical WAL entries were created
//   - how many entries were flushed
//   - how many physical expressions survived optimisation
//   - what triggered the flush
//
// Important:
//   Incrementing counters is cheap.
//   Printing inside the hot path is NOT cheap, so we only print when
//   PrintWALStats(...) is explicitly called from the benchmark.

#if BOSS_WAL_INSTRUMENTATION

enum class FlushReason {
  ChainThreshold,
  ReadTriggered,
  Manual,
  Other
};

struct WALInstrumentationStats {
  uint64_t walEntriesCreated = 0;

  uint64_t flushCalls = 0;
  uint64_t walEntriesFlushed = 0;

  uint64_t physicalUpdatesApplied = 0;
  uint64_t physicalDeletesApplied = 0;

  uint64_t readTriggeredFlushes = 0;
  uint64_t chainTriggeredFlushes = 0;
  uint64_t manualFlushes = 0;
  uint64_t otherFlushes = 0;

  uint64_t totalEntriesPerFlush = 0;
  uint64_t maxEntriesInSingleFlush = 0;
};

static WALInstrumentationStats walStats;

static void resetWALStats() {
  walStats = WALInstrumentationStats{};
}

static void recordFlushStart(FlushReason reason, size_t entriesBeforeFlush) {
  walStats.flushCalls++;
  walStats.walEntriesFlushed += entriesBeforeFlush;
  walStats.totalEntriesPerFlush += entriesBeforeFlush;

  walStats.maxEntriesInSingleFlush =
    std::max<uint64_t>(walStats.maxEntriesInSingleFlush, entriesBeforeFlush);

  switch(reason) {
    case FlushReason::ChainThreshold:
      walStats.chainTriggeredFlushes++;
      break;
    case FlushReason::ReadTriggered:
      walStats.readTriggeredFlushes++;
      break;
    case FlushReason::Manual:
      walStats.manualFlushes++;
      break;
    default:
      walStats.otherFlushes++;
      break;
  }
}

static void recordPhysicalExpressions(std::vector<Expression> const& expressions) {
  for(auto const& expression : expressions) {
    auto const* complex = get_if<ComplexExpression>(&expression);
    if(!complex) continue;

    if(complex->getHead() == "Update"_) {
      walStats.physicalUpdatesApplied++;
    } else if(complex->getHead() == "Delete"_) {
      walStats.physicalDeletesApplied++;
    }
  }
}

static void printWALStats(std::string const& label = "") {
  std::cout << "\n--- WAL Instrumentation";
  if(!label.empty()) {
    std::cout << ": " << label;
  }
  std::cout << " ---\n";

  std::cout << "walEntriesCreated          = " << walStats.walEntriesCreated << "\n";
  std::cout << "walEntriesFlushed          = " << walStats.walEntriesFlushed << "\n";
  std::cout << "physicalUpdatesApplied     = " << walStats.physicalUpdatesApplied << "\n";
  std::cout << "physicalDeletesApplied     = " << walStats.physicalDeletesApplied << "\n";

  std::cout << "flushCalls                 = " << walStats.flushCalls << "\n";
  std::cout << "readTriggeredFlushes       = " << walStats.readTriggeredFlushes << "\n";
  std::cout << "chainTriggeredFlushes      = " << walStats.chainTriggeredFlushes << "\n";
  std::cout << "manualFlushes              = " << walStats.manualFlushes << "\n";
  std::cout << "otherFlushes               = " << walStats.otherFlushes << "\n";

  if(walStats.flushCalls > 0) {
    std::cout << "avgEntriesPerFlush         = "
              << static_cast<double>(walStats.totalEntriesPerFlush) /
                   static_cast<double>(walStats.flushCalls)
              << "\n";

    std::cout << "maxEntriesInSingleFlush    = "
              << walStats.maxEntriesInSingleFlush << "\n";
  }

  if(walStats.physicalUpdatesApplied > 0) {
    std::cout << "walToPhysicalUpdateRatio   = "
              << static_cast<double>(walStats.walEntriesFlushed) /
                   static_cast<double>(walStats.physicalUpdatesApplied)
              << "x\n";
  }

  std::cout << "pendingWALEntriesNow       = " << walTotalEntries << "\n";
  std::cout << "------------------------------------------\n";
}

#else

enum class FlushReason {
  ChainThreshold,
  ReadTriggered,
  Manual,
  Other
};

static void resetWALStats() {}

static void recordFlushStart(FlushReason, size_t) {}

static void recordPhysicalExpressions(std::vector<Expression> const&) {}

static void printWALStats(std::string const& = "") {
  std::cout << "\nWAL instrumentation disabled at compile time.\n";
}

#endif

// WAL - just a list of expression for now
// static std:: vector<Expression> writeAheadLog;


// Helper: extract table name from a WAL entry
// static std::string getTableName(ComplexExpression const& entry) {
//   if(entry.getArguments().size() == 0) return "";
//   auto arg = entry.getArguments()[0];
//   auto const* table = get_if<Symbol>(&arg);
//   return table ? table->getName() : "";
// }

// Helper: extract row IDs from a WAL entry (second argument is "id"_(List(...)))
// static std::vector<RowID> getRowIDs(ComplexExpression const& entry) {
//   std::vector<RowID> ids;
//   if(entry.getArguments().size() < 2) return ids;
  
//   // arg 1 is id(List(...))
//   auto idArg = entry.getArguments()[1];
//   auto const* idExpr = get_if<ComplexExpression>(&idArg);
//   if(!idExpr) return ids;

//   // inside id(...) is List(...)
//   auto listArg = idExpr->getArguments()[0];
//   auto const* listExpr = get_if<ComplexExpression>(&listArg);
//   if(!listExpr) return ids;

//   // Case 1: plain integers in dynamic arguments
//   for(size_t i = 0; i < listExpr->getArguments().size(); i++) {
//     auto val = listExpr->getArguments()[i];
//     if(auto const* id32 = get_if<int32_t>(&val)) {
//       ids.push_back(*id32);
//     } else if(auto const* id64 = get_if<int64_t>(&val)) {
//       ids.push_back(*id64);
//     }
//   }

//   // Case 2: Spans in span arguments
//   // expression arrives from Velox - type unknown at compile time
//   for (auto const& spanArg : listExpr->getSpanArguments()) {
//     std::visit([&ids](auto const& typedSpan) {
//       using T = std::decay_t<decltype(*typedSpan.begin())>;
//       if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
//         for(auto it = typedSpan.begin(); it != typedSpan.end(); ++it) {
//           ids.push_back(*it);
//         }
//       }
//       // non-numeric types (float, string etc) are silently skipped
//       // because row IDs will always be integers
//     }, spanArg);
//   }
//   return ids;
// }

// Initial design decision is for the prototype
// we only allow one row, or one ID per expression
// to prevent optimisation complexity
// need to fix the code below
// static bool sameTableAndRow(ComplexExpression const& a, ComplexExpression const& b) {
//   if(getTableName(a) != getTableName(b)) return false;
//   auto idsA = getRowIDs(a);
//   auto idsB = getRowIDs(b);
//   if(idsA.empty() || idsB.empty()) return false;

//   // cast both to int64_t as this does not change the value
//   auto toInt64 = [](RowID const& id) {
//     return std::visit([](auto const& v) { return static_cast<int64_t>(v); }, id); 
//   };
//   return toInt64(idsA[0]) == toInt64(idsB[0]);
// }

// Helper: iterate over all row IDs in a List expression
// handles both plain integers (our WAL format) and Spans (from Velox)
// calls callback(idValue) for each ID found, preserving original type
template<typename Callback>
static void visitRowIDs(ComplexExpression const& idListExpr, Callback callback) {
  // case 1: plain integers
  for(size_t i = 0; i < idListExpr.getArguments().size(); i++) {
    auto idVal = idListExpr.getArguments()[i];
    if(auto const* id32 = get_if<int32_t>(&idVal)) {
      callback(*id32);
    } else if(auto const* id64 = get_if<int64_t>(&idVal)) {
      callback(*id64);
    }
  }

  // case 2: Spans from Velox
  for (auto const& spanArg : idListExpr.getSpanArguments()) {
    std::visit([&](auto const& typedSpan) {
      using T = std::decay_t<decltype(*typedSpan.begin())>;
      if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
        for (auto it = typedSpan.begin(); it != typedSpan.end(); ++it) {
          callback(*it);
        }
      }
    }, spanArg);
  }
}

// Helper function for the optimsation rule

static bool containsSymbol(Expression const& expr);

template <typename T>
static bool containsSymbolValue(T const& value);

template <typename WrappedArgument>
static bool containsSymbolArgument(WrappedArgument const& wrappedArg);

template <typename T>
static bool containsSymbolValue(T const& value) {
  using Decayed = std::decay_t<T>;

  if constexpr(std::is_same_v<Decayed, Symbol>) {
    return true;
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    auto const& args = value.getArguments();

    for(auto const& arg : args) {
      if(containsSymbolArgument(arg)) return true;
    }

    return false;
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    return containsSymbol(value);
  } else {
    return false;
  }
}

template <typename WrappedArgument>
static bool containsSymbolArgument(WrappedArgument const& wrappedArg) {
  return std::visit(
    [](auto const& unwrapped) -> bool {
      using Decayed = std::decay_t<decltype(unwrapped)>;

      if constexpr(boss::utilities::isInstanceOfTemplate<
                     Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
        return containsSymbolValue(unwrapped.get());
      } else {
        return containsSymbolValue(unwrapped);
      }
    },
    wrappedArg.getArgument()
  );
}

// Check if a value expression contains any Symbol references
// A Symbol in a value expression means it reads a column — making it a dependent write
// e.g. Plus(price, 1) contains Symbol "price" → dependent write
// e.g. 100.0 contains no Symbols → blind write
static bool containsSymbol(Expression const& expr) {
  return std::visit(
    [](auto const& value) -> bool {
      return containsSymbolValue(value);
    },
    expr
  );
}

// Check if a column assignment in Set(...) is a blind write
// e.g. price(100.0) → blind write, price(Plus(price, 1)) → dependent write
static bool isBlindWrite(ComplexExpression const& colAssign) {
  auto const& args = colAssign.getArguments();

  if(args.empty()) return true;

  return !containsSymbolArgument(args[0]);
}

// Helper to extract a double from any numeric Expression
// returns nullopt if not a numeric concrete value
static std::optional<double> toDouble(Expression const& expr) {
  if(auto const* v = get_if<int32_t>(&expr))  return static_cast<double>(*v);
  if(auto const* v = get_if<int64_t>(&expr))  return static_cast<double>(*v);
  if(auto const* v = get_if<float>(&expr))    return static_cast<double>(*v);
  if(auto const* v = get_if<double>(&expr))   return *v;
  return std::nullopt;
}

// Read a numeric value from either a raw atom, an Expression, or a BOSS wrapper.
// This avoids cloneArgument() when tryConstantFold only needs to inspect constants.
template <typename T>
static std::optional<double> toDoubleValue(T const& value) {
  using Decayed = std::decay_t<T>;

  if constexpr(boss::utilities::isInstanceOfTemplate<
                 Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
    return toDoubleValue(value.get());
  } else if constexpr(std::is_same_v<Decayed, int32_t>) {
    return static_cast<double>(value);
  } else if constexpr(std::is_same_v<Decayed, int64_t>) {
    return static_cast<double>(value);
  } else if constexpr(std::is_same_v<Decayed, float>) {
    return static_cast<double>(value);
  } else if constexpr(std::is_same_v<Decayed, double>) {
    return value;
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    return toDouble(value);
  } else {
    return std::nullopt;
  }
}

// Read a ComplexExpression pointer from either a raw ComplexExpression,
// an Expression, or a BOSS wrapper.
// This avoids cloning an argument just to check whether it is a ComplexExpression.
template <typename T>
static ComplexExpression const* asComplexExpressionPtr(T const& value) {
  using Decayed = std::decay_t<T>;

  if constexpr(boss::utilities::isInstanceOfTemplate<
                 Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
    return asComplexExpressionPtr(value.get());
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    return &value;
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    return get_if<ComplexExpression>(&value);
  } else {
    return nullptr;
  }
}

// Create an owned Expression from an existing argument only when rebuilding
// the simplified expression. This keeps ownership cloning only at output time.
template <typename T>
static Expression cloneExpressionValue(T const& value) {
  using Decayed = std::decay_t<T>;

  if constexpr(boss::utilities::isInstanceOfTemplate<
                 Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
    return cloneExpressionValue(value.get());
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    return value.clone();
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    return value.clone();
  } else {
    return value;
  }
}

template <typename WrappedArgument>
static Expression cloneWrappedArgument(WrappedArgument const& wrappedArg) {
  return std::visit(
    [](auto const& unwrapped) -> Expression {
      return cloneExpressionValue(unwrapped);
    },
    wrappedArg.getArgument()
  );
}

// Visit an argument by reference without cloning.
// Use this when we only need to inspect an argument, not store it.
template <typename WrappedArgument, typename Visitor>
static decltype(auto) visitArgumentByReference(WrappedArgument const& wrappedArg,
                                               Visitor&& visitor) {
  return std::visit(
    [&](auto const& unwrapped) -> decltype(auto) {
      using Decayed = std::decay_t<decltype(unwrapped)>;

      if constexpr(boss::utilities::isInstanceOfTemplate<
                     Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
        return visitor(unwrapped.get());
      } else {
        return visitor(unwrapped);
      }
    },
    wrappedArg.getArgument()
  );
}

// Attempt constant folding on a folded expression
// handles same-operator chains where both constants are concrete numbers
// e.g. Plus(Plus(price, 1.0), 1.0)   → Plus(price, 2.0)
// e.g. Times(Times(price, 2.0), 3.0) → Times(price, 6.0)
// e.g. Minus(Minus(price, 1.0), 2.0) → Minus(price, 3.0)
// e.g. Divide(Divide(price, 2.0), 2.0) → Divide(price, 4.0)
// returns nullopt if pattern doesn't match — expression stays as-is
static std::optional<Expression> tryConstantFold(Expression const& expr) {
  auto const* outer = get_if<ComplexExpression>(&expr);
  if(!outer) return std::nullopt;

  auto const& outerArgs = outer->getArguments();
  if(outerArgs.size() < 2) return std::nullopt;

  auto const& outerHead = outer->getHead();

  // only handle our four arithmetic operators
  if(outerHead != "Plus"_  && outerHead != "Minus"_ &&
     outerHead != "Times"_ && outerHead != "Divide"_) return std::nullopt;

  // outer's first arg must be a ComplexExpression with the SAME operator.
  // V2E-A: inspect by wrapper/reference instead of cloneArgument(0).
  auto const* inner = std::visit(
    [](auto const& unwrapped) -> ComplexExpression const* {
      return asComplexExpressionPtr(unwrapped);
    },
    outerArgs[0].getArgument()
  );

  if(!inner) return std::nullopt;
  if(inner->getHead() != outerHead) return std::nullopt;

  auto const& innerArgs = inner->getArguments();
  if(innerArgs.size() < 2) return std::nullopt;

  // inner's second arg must be a concrete number (c1).
  // V2E-A: inspect by wrapper/reference instead of cloneArgument(1).
  auto c1 = std::visit(
    [](auto const& unwrapped) -> std::optional<double> {
      return toDoubleValue(unwrapped);
    },
    innerArgs[1].getArgument()
  );

  if(!c1) return std::nullopt;

  // outer's second arg must be a concrete number (c2).
  // V2E-A: inspect by wrapper/reference instead of cloneArgument(1).
  auto c2 = std::visit(
    [](auto const& unwrapped) -> std::optional<double> {
      return toDoubleValue(unwrapped);
    },
    outerArgs[1].getArgument()
  );

  if(!c2) return std::nullopt;

  // compute the folded constant based on operator
  double folded;
  if(outerHead == "Plus"_)        folded = *c1 + *c2;
  else if(outerHead == "Minus"_)  folded = *c1 + *c2; // Minus(Minus(x,a),b) = Minus(x, a+b)
  else if(outerHead == "Times"_)  folded = *c1 * *c2;
  else if(outerHead == "Divide"_) folded = *c1 * *c2; // Divide(Divide(x,a),b) = Divide(x, a*b)
  else return std::nullopt;

  // Rebuild: outerHead(inner->arg0, folded)
  // We still need an owned copy of inner arg0 because it becomes part of
  // the new simplified expression.
  boss::ExpressionArguments newArgs;
  newArgs.push_back(cloneWrappedArgument(innerArgs[0]));
  newArgs.push_back(folded);

  return ComplexExpression(outerHead, {}, std::move(newArgs), {});
}

// Fold two dependent writes on the same column into one composed expression
// e.g. price(Plus(price, 1)) followed by price(Plus(price, 1))
// becomes price(Plus(Plus(price, 1), 1))
// the earlier write's value gets substituted as the "current value" for the later write
static Expression foldColumnWrites(ComplexExpression const& earlier,
                                   ComplexExpression const& later) {
  // earlier = price(Plus(price, 1))
  // later   = price(Plus(price, 1))
  // result  = price(Plus(Plus(price, 1), 1))
  // we substitute the earlier's value expression wherever
  // "price" Symbol appears in the later's value expression

  auto colName = later.getHead(); // e.g. "price"

  // We still need an owned copy of the earlier value because it may be inserted
  // into the new folded expression wherever the column Symbol is found.
  auto earlierValue = earlier.cloneArgument(0);

  std::function<Expression(Expression const&)> substituteExpr;

  auto substituteValue = [&](auto const& value) -> Expression {
    using Decayed = std::decay_t<decltype(value)>;

    if constexpr(std::is_same_v<Decayed, Symbol>) {
      if(value == colName) return earlierValue.clone();
      return value;
    } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
      boss::ExpressionArguments newArgs;
      auto const& args = value.getArguments();

      for(auto const& arg : args) {
        newArgs.push_back(std::visit(
          [&](auto const& unwrapped) -> Expression {
            using Inner = std::decay_t<decltype(unwrapped)>;

            if constexpr(boss::utilities::isInstanceOfTemplate<
                           Inner, boss::expressions::generic::MovableReferenceWrapper>::value) {
              return substituteExpr(unwrapped.get());
            } else {
              return substituteExpr(unwrapped);
            }
          },
          arg.getArgument()
        ));
      }

      return ComplexExpression(value.getHead(), {}, std::move(newArgs), {});
    } else if constexpr(std::is_same_v<Decayed, Expression>) {
      return substituteExpr(value);
    } else {
      return value;
    }
  };

  substituteExpr = [&](Expression const& expr) -> Expression {
    return std::visit(
      [&](auto const& value) -> Expression {
        return substituteValue(value);
      },
      expr
    );
  };

  auto const& laterArgs = later.getArguments();
  if(laterArgs.empty()) {
    boss::ExpressionArguments colArgs;
    colArgs.push_back(earlierValue.clone());
    return ComplexExpression(colName, {}, std::move(colArgs), {});
  }

  auto foldedValue = std::visit(
    [&](auto const& unwrapped) -> Expression {
      using Inner = std::decay_t<decltype(unwrapped)>;

      if constexpr(boss::utilities::isInstanceOfTemplate<
                     Inner, boss::expressions::generic::MovableReferenceWrapper>::value) {
        return substituteExpr(unwrapped.get());
      } else {
        return substituteExpr(unwrapped);
      }
    },
    laterArgs[0].getArgument()
  );

  // attempt constant folding — simplifies same-operator chains
  // e.g. Plus(Plus(price, 1), 1) → Plus(price, 2)
  if(auto simplified = tryConstantFold(foldedValue)) {
    foldedValue = std::move(*simplified);
  }

  boss::ExpressionArguments colArgs;
  colArgs.push_back(std::move(foldedValue));
  return ComplexExpression(colName, {}, std::move(colArgs), {});
}

// ============================================================
// walIndexPush — capture one raw entry into the right bucket
// ============================================================
//
// For Update: extract each column from Set(...), push a WALEntry
//             into that column's vector with the current seq number
// For Delete: store as deleteEntry — not in any column vector
// O(c) per write — one push per column in Set(...)

static void walIndexPush(WALKey const& key, Expression walEntry) {
  // create bucket if this is the first entry for this row
  if(walIndex.find(key) == walIndex.end()) {
    walIndex[key] = RowBucket{};
    walOrder.push_back(key);
  }

  RowBucket& bucket = walIndex[key];

  if(auto const* entry = get_if<ComplexExpression>(&walEntry)) {
    if(entry->getHead() == "Update"_ && entry->getArguments().size() >= 3) {
      // grab table name and id expression — shared across all columns in this Update
      // we need these to reconstruct a per-column Update expression for each column
      auto const& tableArg = entry->getArguments()[0]; // e.g. Symbol "Customer"
      auto const& idArg    = entry->getArguments()[1]; // e.g. id(List(5))

      auto const& setArg = entry->getArguments()[2];
      if(auto const* setExpr = get_if<ComplexExpression>(&setArg)) {
        // iterate columns in Set(...) in order — order matters for cross-column dependencies
        // e.g. Set(price(100), total(Times(price, 2))) — price must come before total
        // we assign a fresh seq to each column so they sort correctly at flush time
        for(size_t ci = 0; ci < setExpr->getArguments().size(); ci++) {
          auto const& colArg = setExpr->getArguments()[ci];
          if(auto const* colExpr = get_if<ComplexExpression>(&colArg)) {
            std::string colName = colExpr->getHead().getName();

            // assign a unique seq for this column — increments per column not per Update
            // this ensures columns from the same Update sort in their original order
            int seq = bucket.nextSeq++;

            // build a new Update containing only this column in its Set
            // e.g. Update(Customer, id(List(5)), Set(price(100)))
            // storing only the relevant column avoids redundant data per entry
            boss::ExpressionArguments singleSetArgs;
            singleSetArgs.push_back(colExpr->clone()); // just this column assignment
            auto singleSet = ComplexExpression("Set"_, {}, std::move(singleSetArgs), {});

            boss::ExpressionArguments singleUpdateArgs;
            singleUpdateArgs.push_back(cloneWrappedArgument(tableArg)); // table name
            singleUpdateArgs.push_back(cloneWrappedArgument(idArg));    // row id
            singleUpdateArgs.push_back(std::move(singleSet));
            auto singleUpdate = ComplexExpression("Update"_, {}, std::move(singleUpdateArgs), {});

            auto& colEntries = bucket.columnEntries[colName];

            if(colEntries.empty()) {
              colEntries.reserve(WAL_COLUMN_ENTRY_RESERVE);
            }

            bool blind = isBlindWrite(*colExpr);
            colEntries.push_back({std::move(singleUpdate), seq, blind});
            walTotalEntries++; // once per column, not once per Update
            #if BOSS_WAL_INSTRUMENTATION
            walStats.walEntriesCreated++;
            #endif
          }
        }
      }
    } else if(entry->getHead() == "Delete"_) {
      // Delete gets its own seq — after all column seqs if called after an Update
      // but Delete is captured independently so seq reflects actual arrival order
      int seq = bucket.nextSeq++;
      if(!bucket.deleteEntry.has_value() || seq > bucket.deleteEntry->seq) {
        bucket.deleteEntry = {walEntry.clone(), seq, false};
      }
      walTotalEntries++;
      #if BOSS_WAL_INSTRUMENTATION
      walStats.walEntriesCreated++;
      #endif
    }
  }
}

// ============================================================
// resolveColumnEntries — resolve one column's entry list into a final expression
// ============================================================
//
// Takes the column's entry list (in arrival order, seq ascending) and the
// bucket (for cross-column dependency lookup), plus the cutoff seq — only
// entries with seq <= cutoffSeq are considered.
//
// Returns the resolved column assignment expression e.g. price(Plus(price, 100))
// or nullopt if no surviving entries.
//
// Same-column dependency: fold backwards through this column's entries
// Cross-column dependency: when a Symbol Y appears in the expression for column X,
//   and Y != X, look up Y's entries in the bucket and resolve Y up to cutoffSeq
//
// The in-memory engine processes Set(...) columns sequentially and updates
// row state as it goes — so emitting columns in seq order gives correct results
// without needing to substitute concrete values here.

static std::optional<Expression> resolveColumnEntries(
  std::string const& colName,
  std::vector<WALEntry> const& entries,
  RowBucket const& bucket,
  int cutoffSeq)
{
  std::optional<Expression> resolvedValue;
 
  // walk entry list backwards — latest entry first
  for(int idx = static_cast<int>(entries.size()) - 1; idx >= 0; idx--) {
    WALEntry const& walEntry = entries[idx];
 
    // only consider entries at or before cutoffSeq
    if(walEntry.seq > cutoffSeq) continue;
 
    // discard entries at or before the delete seq
    if(bucket.deleteEntry.has_value() && walEntry.seq <= bucket.deleteEntry->seq) break;
 
    // extract the column assignment for colName from this entry's Set(...)
    auto const* entryExpr = get_if<ComplexExpression>(&walEntry.expr);
    if(!entryExpr || entryExpr->getArguments().size() < 3) continue;
 
    auto setArg = entryExpr->getArguments()[2];
    auto const* setExpr = get_if<ComplexExpression>(&setArg);
    if(!setExpr) continue;
 
    // Set contains exactly one column after Change 1 — read it directly
    auto colArg = setExpr->getArguments()[0];
    auto const* colAssign = get_if<ComplexExpression>(&colArg);
    if(!colAssign) continue;
 
    if(!resolvedValue.has_value()) {
      // first (latest) entry for this column
      if(walEntry.isBlindWrite) {
        resolvedValue = colAssign->clone();
        break; // Rule 1a: blind write, stop immediately
      } else {
        resolvedValue = colAssign->clone(); // Rule 1b: dependent, keep walking
      }
    } else {
      auto const* resolvedExpr = get_if<ComplexExpression>(&*resolvedValue);
      if(!resolvedExpr) break;
      if(walEntry.isBlindWrite) {
        resolvedValue = foldColumnWrites(*colAssign, *resolvedExpr);
        break; // blind base found, stop
      } else {
        resolvedValue = foldColumnWrites(*colAssign, *resolvedExpr);
      }
    }
  }
 
  return resolvedValue;
}

// ============================================================
// collectSymbolNames — find all Symbol names in an expression
// ============================================================
//
// Used to detect cross-column dependencies: if column X's expression
// contains Symbol "price", and "price" is a column in the same bucket,
// then X depends on price.
 
template <typename T>
static void collectSymbolNamesValue(T const& value, std::vector<std::string>& names);

template <typename WrappedArgument>
static void collectSymbolNamesArgument(WrappedArgument const& wrappedArg,
                                       std::vector<std::string>& names) {
  visitArgumentByReference(
    wrappedArg,
    [&](auto const& unwrapped) {
      collectSymbolNamesValue(unwrapped, names);
    }
  );
}

static void collectSymbolNames(Expression const& expr, std::vector<std::string>& names) {
  std::visit(
    [&](auto const& value) {
      collectSymbolNamesValue(value, names);
    },
    expr
  );
}

template <typename T>
static void collectSymbolNamesValue(T const& value, std::vector<std::string>& names) {
  using Decayed = std::decay_t<T>;

  if constexpr(boss::utilities::isInstanceOfTemplate<
                 Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
    collectSymbolNamesValue(value.get(), names);
  } else if constexpr(std::is_same_v<Decayed, Symbol>) {
    names.push_back(value.getName());
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    auto const& args = value.getArguments();

    for(auto const& arg : args) {
      collectSymbolNamesArgument(arg, names);
    }
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    collectSymbolNames(value, names);
  } else {
    // concrete literals do not contain Symbol dependencies
  }
}

// ============================================================
// optimiseBucketImpl — compact bucket in place
// ============================================================
//
// Resolves each column's entry list into a single final expression.
// Handles:
//   Rule 1a: blind write — last-write-wins, discard earlier entries
//   Rule 1b: same-column dependent write — fold chain backwards
//   Rule 1c: cross-column dependency — when column X at seq s depends on
//            Symbol Y (different column), resolve Y's entries up to seq s-1
//            and emit Y before X in the merged Set(...)
//   Rule 2:  Delete — discard all entries before deleteEntry.seq
//   Rule 3:  merge columns — all surviving columns in one Update (implicit)
//
// After this call the bucket's columnEntries are replaced with at most
// one entry per column (the resolved result), and deleteEntry is preserved.
 
static void optimiseBucketImpl(RowBucket& bucket) {
  // deleteSeq: entries at or before this seq are discarded
  int deleteSeq = bucket.deleteEntry.has_value() ? bucket.deleteEntry->seq : -1;
 
  // find the maximum seq across all surviving entries — used as cutoff for resolution
  // "surviving" means seq > deleteSeq
  int maxSeq = deleteSeq; // start at deleteSeq, will be updated as we find surviving entries
  for(auto const& [colName, entries] : bucket.columnEntries) {
    for(auto const& e : entries) {
      if(e.seq > deleteSeq && e.seq > maxSeq) maxSeq = e.seq;
    }
  }
 
  // resolve each column using all its entries up to maxSeq
  // result: map from column name to resolved expression
  // we also track the seq of the latest surviving entry per column
  // so we can emit columns in correct seq order for cross-column dependencies
  struct ResolvedCol {
    Expression expr;   // the resolved column assignment e.g. price(Plus(price,100))
    int latestSeq;     // seq of the latest entry that contributed to this result
  };
  std::vector<ResolvedCol> resolved;
 
  // save table name and id for rebuilding the Update expression
  std::optional<Expression> savedTableName;
  std::optional<Expression> savedIdExpr;
 
  for(auto const& [colName, entries] : bucket.columnEntries) {
    // find the latest surviving entry seq for this column
    int colLatestSeq = -1;
    for(auto const& e : entries) {
      if(e.seq > deleteSeq && e.seq > colLatestSeq) colLatestSeq = e.seq;
    }
    if(colLatestSeq == -1) continue; // all entries before delete, skip
 
    // resolve this column's entries
    auto result = resolveColumnEntries(colName, entries, bucket, maxSeq);
    if(!result.has_value()) continue;
 
    // grab table name and id from any surviving entry
    if(!savedTableName.has_value()) {
      for(auto const& e : entries) {
        if(e.seq > deleteSeq) {
          auto const* entryExpr = get_if<ComplexExpression>(&e.expr);
          if(entryExpr && entryExpr->getArguments().size() >= 2) {
            savedTableName = entryExpr->cloneArgument(0);
            savedIdExpr    = entryExpr->cloneArgument(1);
            break;
          }
        }
      }
    }
 
    resolved.push_back({std::move(*result), colLatestSeq});
  }
 
  // sort resolved columns by latestSeq ascending
  // this ensures columns are emitted in the order they were last written
  // which is critical for cross-column dependencies:
  // if price was written before total, price must appear before total in Set(...)
  // so the in-memory engine evaluates price first, then total reads the updated price
  std::sort(resolved.begin(), resolved.end(),
    [](ResolvedCol const& a, ResolvedCol const& b) {
      return a.latestSeq < b.latestSeq;
    });
 
  // clear the old column entries
  bucket.columnEntries.clear();
 
  // rebuild: one merged Update with all surviving columns in seq order
  if(!resolved.empty() && savedTableName.has_value()) {
    boss::ExpressionArguments resolvedCols;
    for(auto& rc : resolved) {
      resolvedCols.push_back(std::move(rc.expr));
    }
 
    boss::ExpressionArguments updateArgs;
    updateArgs.push_back(std::move(*savedTableName));
    updateArgs.push_back(std::move(*savedIdExpr));
    updateArgs.push_back(ComplexExpression("Set"_, {}, std::move(resolvedCols), {}));
    auto mergedUpdate = ComplexExpression("Update"_, {}, std::move(updateArgs), {});
 
    // re-register the merged Update's columns at seq 0 in the new columnEntries
    // (all at the same seq since they are now a single compacted entry)
    auto const& setArg = mergedUpdate.getArguments()[2];
    if(auto const* setExpr = get_if<ComplexExpression>(&setArg)) {
      for(size_t ci = 0; ci < setExpr->getArguments().size(); ci++) {
        auto const& colArg = setExpr->getArguments()[ci];
        if(auto const* colExpr = get_if<ComplexExpression>(&colArg)) {
          auto& colEntries = bucket.columnEntries[colExpr->getHead().getName()];

          if(colEntries.empty()) {
            colEntries.reserve(WAL_COLUMN_ENTRY_RESERVE);
          }

          bool blind = isBlindWrite(*colExpr);
          colEntries.push_back({mergedUpdate.clone(), 0, blind});
        }
      }
    }
    bucket.nextSeq = 1; // reset seq counter after compaction
  }
  // deleteEntry is preserved unchanged — caller decides what to emit
}

// ============================================================
// optimiseBucket — apply rules, compact bucket in place
// ============================================================
//
// Resolves the bucket's chain into a compacted set of expressions.
// Rebuilds chain and columnChain from the resolved results.
// The bucket stays alive after this call — WAL remains populated.
//
// Rules applied:
//   Rule 1a (blind write):     last-write-wins — earlier writes to same column discarded
//   Rule 1b (dependent write): fold chain backwards — substitute earlier value into later
//   Rule 2 (Delete):           all entries before the latest Delete are discarded
//   Rule 3 (merge columns):    all surviving columns merged into one Update — implicit

// static void optimiseBucketImpl(RowBucket& bucket) {
//   // find the latest Delete — walk backwards
//   int deletePos = -1;
//   for(int i = static_cast<int>(bucket.chain.size()) - 1; i >= 0; i--) {
//     if(auto const* entry = get_if<ComplexExpression>(&bucket.chain[i])) {
//       if(entry->getHead() == "Delete"_) {
//         deletePos = i;
//         break;
//       }
//     }
//   }
 
//   // save the Delete expression NOW before we clear the chain
//   std::optional<Expression> savedDelete;
//   if(deletePos >= 0)
//     savedDelete = bucket.chain[deletePos].clone();
 
//   // resolve each column
//   boss::ExpressionArguments resolvedCols;
//   ComplexExpression const* representativeEntry = nullptr;
 
//   for(auto const& [colName, positions] : bucket.columnChain) {
//     std::optional<Expression> resolvedValue;
 
//     for(int idx = static_cast<int>(positions.size()) - 1; idx >= 0; idx--) {
//       int pos = positions[idx];
//       if(pos <= deletePos) break; // Rule 2: before Delete, discard
 
//       auto const* entry = get_if<ComplexExpression>(&bucket.chain[pos]);
//       if(!entry || entry->getArguments().size() < 3) continue;
 
//       auto setArg = entry->getArguments()[2];
//       auto const* setExpr = get_if<ComplexExpression>(&setArg);
//       if(!setExpr) continue;
 
//       ComplexExpression const* colAssign = nullptr;
//       for(size_t ci = 0; ci < setExpr->getArguments().size(); ci++) {
//         auto colArg = setExpr->getArguments()[ci];
//         if(auto const* colExpr = get_if<ComplexExpression>(&colArg)) {
//           if(colExpr->getHead().getName() == colName) { colAssign = colExpr; break; }
//         }
//       }
//       if(!colAssign) continue;
 
//       if(representativeEntry == nullptr) representativeEntry = entry;
 
//       if(!resolvedValue.has_value()) {
//         if(isBlindWrite(*colAssign)) {
//           resolvedValue = colAssign->clone();
//           break; // Rule 1a: blind write wins, stop
//         } else {
//           resolvedValue = colAssign->clone(); // Rule 1b: dependent, keep walking
//         }
//       } else {
//         auto const* resolvedExpr = get_if<ComplexExpression>(&*resolvedValue);
//         if(!resolvedExpr) break;
//         if(isBlindWrite(*colAssign)) {
//           resolvedValue = foldColumnWrites(*colAssign, *resolvedExpr);
//           break; // blind base found, stop
//         } else {
//           resolvedValue = foldColumnWrites(*colAssign, *resolvedExpr); // keep walking
//         }
//       }
//     }
 
//     if(resolvedValue.has_value())
//       resolvedCols.push_back(std::move(*resolvedValue));
//   }

//   // save these BEFORE clearing the chain
//   std::optional<Expression> savedTableName;
//   std::optional<Expression> savedIdExpr;
//   if(representativeEntry != nullptr) {
//     savedTableName = representativeEntry->cloneArgument(0);
//     savedIdExpr    = representativeEntry->cloneArgument(1);
//   }
 
//   // now clear the bucket — old chain and columnChain are no longer needed
//   bucket.chain.clear();
//   bucket.columnChain.clear();

//   // re-add the saved Delete at the end of the compacted chain
//   if(savedDelete.has_value())
//     bucket.chain.push_back(std::move(*savedDelete));
 
//   // rebuild: one merged Update if any columns survived
//   if(!resolvedCols.empty() && savedTableName.has_value()) {
//     boss::ExpressionArguments updateArgs;
//     updateArgs.push_back(std::move(*savedTableName)); // table name
//     updateArgs.push_back(std::move(*savedIdExpr)); // id(List(...))
//     updateArgs.push_back(ComplexExpression("Set"_, {}, std::move(resolvedCols), {}));
//     auto mergedUpdate = ComplexExpression("Update"_, {}, std::move(updateArgs), {});
 
//     // re-register columns in columnChain at position 0
//     auto setArg = mergedUpdate.getArguments()[2];
//     if(auto const* setExpr = get_if<ComplexExpression>(&setArg)) {
//       for(size_t ci = 0; ci < setExpr->getArguments().size(); ci++) {
//         auto colArg = setExpr->getArguments()[ci];
//         if(auto const* colExpr = get_if<ComplexExpression>(&colArg))
//           bucket.columnChain[colExpr->getHead().getName()].push_back(0);
//       }
//     }
//     bucket.chain.push_back(std::move(mergedUpdate));
//   }
// }

// ============================================================
// flushBucket — optimise, collect, clear
// ============================================================
 
static std::vector<Expression> flushBucket(
  WALKey const& key,
  std::vector<std::string> const& columns = {},
  FlushReason reason = FlushReason::Other)
{
  auto it = walIndex.find(key);
  if(it == walIndex.end()) return {};

  RowBucket& bucket = it->second;

  // count entries before any changes for walTotalEntries accounting
  ssize_t countBefore = 0;

  auto flushingAllColumns = columns.empty();

  if(flushingAllColumns) {
    for(auto const& [col, entries] : bucket.columnEntries) {
      countBefore += entries.size();
    }
  } else {
    for(auto const& col : columns) {
      auto it = bucket.columnEntries.find(col);
      if(it != bucket.columnEntries.end()) {
        countBefore += it->second.size();
      }
    }
  }

  // A delete invalidates the whole row, so count it whenever it is present.
  // In practice, if there is a deleteEntry, the flush logic should treat the row
  // as deleted rather than just flushing one column.
  if(bucket.deleteEntry.has_value()) {
    countBefore++;
  }

  recordFlushStart(reason, countBefore);

  // compact the bucket in place
  optimiseBucketImpl(bucket);

  std::vector<Expression> result;

  if(columns.empty()) {
    // ── whole-row flush ───────────────────────────────────────────────────
    // emit Delete first if one exists
    if(bucket.deleteEntry.has_value()) {
      result.push_back(bucket.deleteEntry->expr.clone());
    }

    // emit the merged Update — after optimiseBucketImpl there is at most
    // one entry per column, all sharing the same compacted Update expression
    // grab from any column since they all point to the same merged Update
    bool updateEmitted = false;
    for(auto const& [col, entries] : bucket.columnEntries) {
      if(!entries.empty() && !updateEmitted) {
        result.push_back(entries[0].expr.clone());
        updateEmitted = true;
        break;
      }
    }

    // whole-row flush — remove the bucket entirely
    walTotalEntries -= countBefore;
    walIndex.erase(it);
    walOrder.erase(std::remove(walOrder.begin(), walOrder.end(), key), walOrder.end());

  } else {
    // ── column-selective flush ────────────────────────────────────────────
    // only emit and remove the requested columns
    // the remaining columns stay in the bucket for future reads

    // collect the resolved expressions for the requested columns only
    // and build a single merged Update containing just those columns
    boss::ExpressionArguments selectedCols;
    std::optional<Expression> savedTableName;
    std::optional<Expression> savedIdExpr;
    size_t removedEntries = 0;

    for(auto const& colName : columns) {
      auto colIt = bucket.columnEntries.find(colName);
      if(colIt == bucket.columnEntries.end()) continue;
      if(colIt->second.empty()) continue;

      // after optimiseBucketImpl there is exactly one entry per column
      // grab the column assignment from Set's first argument
      auto const* entryExpr = get_if<ComplexExpression>(&colIt->second[0].expr);
      if(!entryExpr || entryExpr->getArguments().size() < 3) continue;

      // save table name and id from the first column we find
      if(!savedTableName.has_value()) {
        savedTableName = entryExpr->cloneArgument(0);
        savedIdExpr    = entryExpr->cloneArgument(1);
      }

      // grab the column assignment from Set's first argument
      auto const& setArg = entryExpr->getArguments()[2];
      auto const* setExpr = get_if<ComplexExpression>(&setArg);
      if(!setExpr || setExpr->getArguments().size() < 1) continue;

      selectedCols.push_back(setExpr->cloneArgument(0));

      // remove this column from the bucket — it is being flushed now
      removedEntries += colIt->second.size();
      bucket.columnEntries.erase(colIt);
    }

    // emit a merged Update with only the selected columns
    if(!selectedCols.empty() && savedTableName.has_value()) {
      boss::ExpressionArguments updateArgs;
      updateArgs.push_back(std::move(*savedTableName));
      updateArgs.push_back(std::move(*savedIdExpr));
      updateArgs.push_back(ComplexExpression("Set"_, {}, std::move(selectedCols), {}));
      result.push_back(ComplexExpression("Update"_, {}, std::move(updateArgs), {}));
    }

    // update walTotalEntries — only subtract what was actually removed
    walTotalEntries -= removedEntries;

    // if all columns have been flushed, remove the bucket entirely
    // otherwise leave it alive for the remaining columns
    if(bucket.columnEntries.empty() && !bucket.deleteEntry.has_value()) {
      walIndex.erase(it);
      walOrder.erase(std::remove(walOrder.begin(), walOrder.end(), key), walOrder.end());
    }
  }

  recordPhysicalExpressions(result);
  return result;
}
// ============================================================
// extractSelectKeys
// ============================================================
 
// extractSelectKeys — get (tableName, rowID) pairs from a Select expression
//
// Handles two formats:
//
// Format 1: raw Select arriving directly from application (our benchmark pipeline)
//   Select(Customer, Where(Equal(id, 5)))
//   → tableName = "Customer" from arg 0 (Symbol)
//   → rowID = 5 from Equal(id, 5)
//
// Format 2: pre-resolved Select with explicit Table (full pipeline with Velox)
//   Select(Table(Customer, id(List(5, 6))), Where(...))
//   → tableName = "Customer" from Table arg 0 (Symbol)
//   → rowIDs from id(List(...))
//
// Returns empty vector if neither format matches — caller falls back to full flush.

static std::vector<SelectTarget> extractSelectKeys(ComplexExpression const& expr) {
  std::vector<SelectTarget> targets;

  // ── Format 3: Project(Select(Customer, Where(Equal(id, 5))), As(price_, price_)) ──
  //
  // Outermost head is Project — column-selective read
  // arg 0 is the inner Select expression
  // arg 1 is As(...) containing pairs of (outputName, inputName)
  // we extract the row ID from the inner Select and the column names from As(...)
  if(expr.getHead() == "Project"_) {
    if(expr.getArguments().size() < 2) return targets;

    // arg 0 must be a Select expression
    auto innerArg = expr.getArguments()[0];
    auto const* innerSelect = get_if<ComplexExpression>(&innerArg);
    if(!innerSelect || innerSelect->getHead() != "Select"_) return targets;

    // arg 1 must be As(...)
    auto asArg = expr.getArguments()[1];
    auto const* asExpr = get_if<ComplexExpression>(&asArg);
    if(!asExpr || asExpr->getHead() != "As"_) return targets;

    // recursively call extractSelectKeys on the inner Select to get the row keys
    // this reuses the Format 1 / Format 2 logic already written below
    auto innerTargets = extractSelectKeys(*innerSelect);
    if(innerTargets.empty()) return targets;

    // extract column names from As(...)
    // As contains pairs: (outputName, inputExpression)
    // for a simple passthrough As(price_, price_) both are the same Symbol
    // we want the input name (even-indexed args: 0, 2, 4...)
    // actually As stores them as alternating pairs so arg 0 = output, arg 1 = input
    // e.g. As(FirstName_, FirstName_, LastName_, LastName_)
    // we take every other argument starting at index 1 — the input expressions
    std::vector<std::string> columns;
    for(size_t i = 1; i < asExpr->getArguments().size(); i += 2) {
      auto colArg = asExpr->getArguments()[i];
      if(auto const* colSym = get_if<Symbol>(&colArg)) {
        columns.push_back(colSym->getName());
      }
    }

    // combine each row key with the column list
    for(auto& t : innerTargets) {
      targets.push_back({t.key, columns});
    }
    return targets;
  }

  // ── Format 1: Select(Customer, Where(Equal(id, 5))) ──────────────────────
  //
  // arg 0 is a plain Symbol — the table name
  // arg 1 is Where(...) containing the condition
  if(expr.getArguments().size() < 1) return targets;
  auto arg0 = expr.getArguments()[0];
  if(auto const* tableSymbol = get_if<boss::Symbol>(&arg0)) {
    std::string tableName = tableSymbol->getName();

    if(expr.getArguments().size() < 2) return targets;
    auto arg1 = expr.getArguments()[1];
    auto const* whereExpr = get_if<ComplexExpression>(&arg1);
    if(!whereExpr || whereExpr->getHead() != "Where"_) return targets;

    if(whereExpr->getArguments().size() < 1) return targets;
    auto condArg = whereExpr->getArguments()[0];
    auto const* condExpr = get_if<ComplexExpression>(&condArg);
    if(!condExpr || condExpr->getHead() != "Equal"_) return targets;

    if(condExpr->getArguments().size() < 2) return targets;
    auto colArg = condExpr->getArguments()[0];
    auto valArg = condExpr->getArguments()[1];

    auto const* colSymbol = get_if<boss::Symbol>(&colArg);
    if(!colSymbol || colSymbol->getName() != "id") return targets;

    if(auto const* id32 = get_if<int32_t>(&valArg)) {
      // empty columns = flush whole row
      targets.push_back({{tableName, static_cast<int64_t>(*id32)}, {}});
      return targets;
    }
    if(auto const* id64 = get_if<int64_t>(&valArg)) {
      targets.push_back({{tableName, *id64}, {}});
      return targets;
    }
    return targets;
  }

  // ── Format 2: Select(Table(Customer, id(List(5, 6))), Where(...)) ─────────
  //
  // arg 0 is a ComplexExpression with head Table
  auto const* tableExpr = get_if<ComplexExpression>(&arg0);
  if(!tableExpr) return targets;

  if(tableExpr->getArguments().size() < 1) return targets;
  auto nameArg = tableExpr->getArguments()[0];
  auto const* nameSymbol = get_if<boss::Symbol>(&nameArg);
  if(!nameSymbol) return targets;
  std::string tableName = nameSymbol->getName();

  for(size_t i = 1; i < tableExpr->getArguments().size(); i++) {
    auto colArg = tableExpr->getArguments()[i];
    auto const* colExpr = get_if<ComplexExpression>(&colArg);
    if(!colExpr || colExpr->getArguments().size() < 1) continue;

    auto listArg = colExpr->getArguments()[0];
    auto const* listExpr = get_if<ComplexExpression>(&listArg);
    if(!listExpr) continue;

    visitRowIDs(*listExpr, [&](auto idValue) {
      // empty columns = flush whole row
      targets.push_back({{tableName, static_cast<int64_t>(idValue)}, {}});
    });

    if(!targets.empty()) break;
  }

  return targets;
}

static Expression flushAllBuckets(FlushReason reason = FlushReason::Manual) {
  // commented out for better testing output
  // std::cout << "WAL: flushing all " << walTotalEntries << " entries" << std::endl;
  boss::ExpressionArguments entries;
  auto keysCopy = walOrder; // copy because flushBucket modifies walOrder
  for(auto const& key : keysCopy) {
    auto flushed = flushBucket(key, {}, reason);
    for(auto& e : flushed) entries.push_back(std::move(e));
  }
  // commented out for better testing output
  // std::cout << "WAL: emitting " << entries.size() << " entries" << std::endl;
  return ComplexExpression("ApplyWAL"_, {}, std::move(entries), {});
}
 
// // flushWAL - flush all buckets in insertion order, return ApplyWAL(...)
// static Expression flushWAL() {
//   std::cout << "WAL: flushing " << walTotalEntries << " entries across "
//             << walOrder.size() << " rows" << std::endl;
 
//   boss::ExpressionArguments entries;
 
//   for(auto const& key : walOrder) {
//     auto it = walIndex.find(key);
//     if(it == walIndex.end()) continue;
//     auto flushed = flushBucket(it->second);
//     for(auto& e : flushed) {
//       entries.push_back(std::move(e));
//     }
//   }
 
//   walIndex.clear();
//   walOrder.clear();
//   walTotalEntries = 0;
 
//   std::cout << "WAL: " << entries.size() << " entries after optimisation" << std::endl;
//   return ComplexExpression("ApplyWAL"_, {}, std::move(entries), {});
// }

// // OptimiseWAL - apply three optimisation rules:
// // Rule 1: later Update on same row and same column
// // if later write is blind → earlier is redundant (last-write-wins)
// // if later write is dependent → fold the two writes together into entry j
// //   by substituting entry i's value into entry j's expression
// // 2. Delete eliminates pending Updates on same row
// // 3. Merge Updates on same row with different columns
// static void optimiseWAL() {
//   std::vector<Expression> optimised;

//   for (size_t i = 0; i < writeAheadLog.size(); i++) {
//     auto const& entryI = get<ComplexExpression>(writeAheadLog[i]);
//     bool redundant = false;

//     // check if a later entry makes this one redundant
//     for (size_t j = i + 1; j < writeAheadLog.size(); j++) {
//       auto const& entryJ = get<ComplexExpression>(writeAheadLog[j]);
//       if (!sameTableAndRow(entryI, entryJ)) continue;

//       // Rule 2: Delete eliminates any pending Update on same row
//       if (entryJ.getHead() == "Delete"_) {
//         redundant = true;
//         break;
//       }

//       // Rule 1: later Update on same row and same column
//       // if later write is blind → earlier is redundant (last-write-wins)
//       // if later write is dependent → fold the two writes together into entry j
//       //   by substituting entry i's value into entry j's expression
//       if(entryI.getHead() == "Update"_ && entryJ.getHead() == "Update"_) {
//         auto setArgI = entryI.getArguments()[2];
//         auto const* setExprI = get_if<ComplexExpression>(&setArgI);
//         if(!setExprI) continue;
//         auto setArgJ = entryJ.getArguments()[2];
//         auto const* setExprJ = get_if<ComplexExpression>(&setArgJ);
//         if(!setExprJ) continue;

//         for(size_t ci = 0; ci < setExprI->getArguments().size(); ci++) {
//           auto colIArg = setExprI->getArguments()[ci];
//           auto const* colI = get_if<ComplexExpression>(&colIArg);
//           if(!colI) continue;
//           for(size_t cj = 0; cj < setExprJ->getArguments().size(); cj++) {
//             auto colJArg = setExprJ->getArguments()[cj];
//             auto const* colJ = get_if<ComplexExpression>(&colJArg);
//             if(!colJ) continue;
//             if(colI->getHead() != colJ->getHead()) continue;

//             // same column — check if later write is blind or dependent
//             if(isBlindWrite(*colJ)) {
//               // blind write — earlier is simply redundant
//               redundant = true;
//             } else {
//               // dependent write — fold: substitute earlier value into later expression
//               // modify entry j in the WAL to contain the folded expression
//               auto& entryJMutable = get<ComplexExpression>(writeAheadLog[j]);
//               auto [jHead, jStatics, jArgs, jSpans] = std::move(entryJMutable).decompose();
//               auto setArgJMut = std::move(jArgs[2]);
//               auto [setHead, setStatics, setArgs, setSpans] = 
//                 std::move(get<ComplexExpression>(setArgJMut)).decompose();

//               // replace the matching column in entry j's Set with the folded value
//               for(size_t k = 0; k < setArgs.size(); k++) {
//                 auto const* setCol = get_if<ComplexExpression>(&setArgs[k]);
//                 if(!setCol || setCol->getHead() != colI->getHead()) continue;
//                 setArgs[k] = foldColumnWrites(*colI, *setCol);
//                 break;
//               }

//               jArgs[2] = ComplexExpression(setHead, {}, std::move(setArgs), {});
//               writeAheadLog[j] = ComplexExpression(jHead, {}, std::move(jArgs), {});
//               redundant = true; // earlier entry is now absorbed into j
//             }
//             break;
//           }
//           if(redundant) break;
//         }
//       }
//       if(redundant) break;
//     }
//     if(!redundant) {
//       optimised.push_back(entryI.clone());
//     }
//   }

//   // Rule 3: merge Updates on same row with different columns
//   std::vector<Expression> merged;
//   std::vector<bool> mergedFlag(optimised.size(), false);

//   for (size_t i = 0; i < optimised.size(); i++) {
//     if(mergedFlag[i]) continue;
//     auto const& entryI = get<ComplexExpression>(optimised[i]);

//     if(entryI.getHead() != "Update"_) {
//       merged.push_back(entryI.clone());
//       continue;
//     }


//     // clone each argument to avoid ArgumentWrapper temporary issue
//     auto setArgI = entryI.getArguments()[2];
//     auto const* setExprI = get_if<ComplexExpression>(&setArgI);
//     boss::ExpressionArguments mergedSetArgs;
//     for(size_t ci = 0; ci < setExprI->getArguments().size(); ci++) {
//       mergedSetArgs.push_back(setExprI->cloneArgument(ci));
//     }
  
//     // mergedArgs.push_back(entryI.getArguments()[0]); // table
//     // mergedArgs.push_back(entryI.getArguments()[1]); // ID
//     // for(size_t ci = 2; ci < entryI.getArguments().size(); ci++) {
//     //   mergedArgs.push_back(entryI.getArguments()[ci]);
//     // }

//     for(size_t j = i + 1; j < optimised.size(); j++) {
//       if(mergedFlag[j]) continue;
//       auto const& entryJ = get<ComplexExpression>(optimised[j]);
//       if(entryJ.getHead() != "Update"_) continue;
//       if(!sameTableAndRow(entryI, entryJ)) continue;

//       // merge columns from entryJ
//       auto setArgJ = entryJ.getArguments()[2];
//       auto const* setExprJ = get_if<ComplexExpression>(&setArgJ);
//       for(size_t cj = 0; cj < setExprJ->getArguments().size(); cj++) {
//         mergedSetArgs.push_back(setExprJ->cloneArgument(cj));
//       }
//       mergedFlag[j] = true;
//     }
//     // build merged WAL entry
//     boss::ExpressionArguments mergedArgs;
//     mergedArgs.push_back(entryI.cloneArgument(0)); // table name
//     mergedArgs.push_back(entryI.cloneArgument(1)); // id
//     mergedArgs.push_back(ComplexExpression("Set"_, {}, std::move(mergedSetArgs), {}));
//     merged.push_back(ComplexExpression("Update"_, {}, std::move(mergedArgs), {}));
//   }
//   writeAheadLog = std::move(merged);
// }

// // flushWAL - optimise and return all WAL entries as a List for Arrow storage to apply
// // called when WAL hit threshold
// static Expression flushWAL() {
//   std::cout << "WAL: flushing" << writeAheadLog.size() << " entries" << std::endl;
//   optimiseWAL();
//   std::cout << "WAL: " << writeAheadLog.size() << " entries after optimisation" << std::endl;

//   boss::ExpressionArguments entries;
//   for (auto& entry : writeAheadLog) {
//     entries.push_back(entry.clone());
//   }
//   writeAheadLog.clear();
//   return ComplexExpression("ApplyWAL"_, {}, std::move(entries), {});
// }

static Expression evaluate(Expression &&e) {
  return std::visit(
    [](auto &&expr) -> Expression {
      if constexpr(std::is_same_v<std::decay_t<decltype(expr)>, ComplexExpression>) {
        auto head = expr.getHead();

        // Update - defer to WAL
        if (head == "Update"_) {
          // arg 0 = table name symbol
          // arg 1 = "Table"_(columns...)
          // arg 2 = "Set"_(column assignments)
          // Extract table name - arg 0
          auto tableArg = expr.getArguments()[0];
          auto const* tableSymbol = get_if<Symbol>(&tableArg);
          if(!tableSymbol) return std::move(expr);

          // Extract "Table"_ expression
          auto tableExprArg = expr.getArguments()[1];
          auto const* tableExpr = get_if<ComplexExpression>(&tableExprArg);
          if(!tableExpr) return std::move(expr);

          // find the "id"_ column inside Table
          ComplexExpression const* idListExpr = nullptr;
          auto firstColArg = tableExpr->getArguments()[0];
          auto const* firstColExpr = get_if<ComplexExpression>(&firstColArg);
          if(!firstColExpr) return std::move(expr);

          // use its actual column name for the WAL entry
          auto idColName = firstColExpr->getHead();
          auto listArg = firstColExpr->getArguments()[0];
          idListExpr = get_if<ComplexExpression>(&listArg);
          if (!idListExpr) return std::move(expr);

          // Extract the "Set"_ expression
          auto setArg = expr.getArguments()[2];
          auto const* setExpr = get_if<ComplexExpression>(&setArg);
          if(!setExpr) return std::move(expr);

          visitRowIDs(*idListExpr, [&](auto idValue) {
            // boss::ExpressionArguments walArgs;
            // walArgs.push_back(*tableSymbol);
            // boss::ExpressionArguments idListArgs;
            // idListArgs.push_back(idValue);
            // auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});
            // boss::ExpressionArguments idColArgs;
            // idColArgs.push_back(std::move(idList));
            // walArgs.push_back(ComplexExpression(idColName, {}, std::move(idColArgs), {}));
            // walArgs.push_back(setExpr->clone());
            // writeAheadLog.push_back(
            //   ComplexExpression("Update"_, {}, std::move(walArgs), {})
            // );

            WALKey key{tableSymbol->getName(), static_cast<int64_t>(idValue)};
            boss::ExpressionArguments walArgs;
            walArgs.push_back(*tableSymbol);
            boss::ExpressionArguments idListArgs;
            idListArgs.push_back(idValue);
            auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});
            boss::ExpressionArguments idColArgs;
            idColArgs.push_back(std::move(idList));
            walArgs.push_back(ComplexExpression(idColName, {}, std::move(idColArgs), {}));
            walArgs.push_back(setExpr->clone());
            walIndexPush(key, ComplexExpression("Update"_, {}, std::move(walArgs), {}));
          });

          // commented out for better testing output
          // std::cout << "WAL: captured Update" << std::endl;

          // check if WAL has hit the threshold - if so flush to Arrow storage
          if(walTotalEntries >= WAL_THRESHOLD) {
            return flushAllBuckets(FlushReason::ChainThreshold);
          }

          return "Update_Logged"_();
        }

        // Delete - defer to WAL
        if (head == "Delete"_) {
          // arg 0 = table name symbol
          // arg 1 = "Table"_(columns...)

          auto tableArg = expr.getArguments()[0];
          auto const* tableSymbol = get_if<Symbol>(&tableArg);
          if(!tableSymbol) return std::move(expr);

          auto tableExprArg = expr.getArguments()[1];
          auto const* tableExpr = get_if<ComplexExpression>(&tableExprArg);
          if(!tableExpr) return std::move(expr);

          // find "id"_ column inside "Table"_
          // which is the first column
          ComplexExpression const* idListExpr = nullptr;
          auto firstColArg = tableExpr->getArguments()[0];
          auto const* firstColExpr = get_if<ComplexExpression>(&firstColArg);
          if(!firstColExpr) return std::move(expr);

          // use its actual head name for the WAL entry
          auto idColName = firstColExpr->getHead();
          auto listArg = firstColExpr->getArguments()[0];
          idListExpr = get_if<ComplexExpression>(&listArg);
          if(!idListExpr) return std::move(expr);

          visitRowIDs(*idListExpr, [&](auto idValue) {
            // boss::ExpressionArguments walArgs;
            // walArgs.push_back(*tableSymbol);
            // boss::ExpressionArguments idListArgs;
            // idListArgs.push_back(idValue);  // preserves original type
            // auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});
            // boss::ExpressionArguments idColArgs;
            // idColArgs.push_back(std::move(idList));
            // walArgs.push_back(ComplexExpression(idColName, {}, std::move(idColArgs), {}));
            // writeAheadLog.push_back(ComplexExpression("Delete"_, {}, std::move(walArgs), {}));
            WALKey key{tableSymbol->getName(), static_cast<int64_t>(idValue)};
            boss::ExpressionArguments walArgs;
            walArgs.push_back(*tableSymbol);
            boss::ExpressionArguments idListArgs;
            idListArgs.push_back(idValue);
            auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});
            boss::ExpressionArguments idColArgs;
            idColArgs.push_back(std::move(idList));
            walArgs.push_back(ComplexExpression(idColName, {}, std::move(idColArgs), {}));
            walIndexPush(key, ComplexExpression("Delete"_, {}, std::move(walArgs), {}));
          });

          // commented out for better testing output
          // std::cout << "WAL: captured Delete" << std::endl;

          // check if WAL has hit the threshold - if so flush to Arrow storage
          if(walTotalEntries >= WAL_THRESHOLD) {
            return flushAllBuckets(FlushReason::ChainThreshold);
          }
          return "Delete_Logged"_();
        }

        // GetWAL — return raw chain contents flattened in walOrder sequence
        if (head == "GetWAL"_) {
          boss::ExpressionArguments entries;
          for(auto const& key : walOrder) {
            auto it = walIndex.find(key);
            if(it == walIndex.end()) continue;
            auto const& bucket = it->second;
 
            // collect all entries across columns with their seq numbers
            std::vector<std::pair<int, Expression>> allEntries;
            for(auto const& [col, colEntries] : bucket.columnEntries) {
              for(auto const& e : colEntries)
                allEntries.push_back({e.seq, e.expr.clone()});
            }
            if(bucket.deleteEntry.has_value())
              allEntries.push_back({bucket.deleteEntry->seq, bucket.deleteEntry->expr.clone()});
 
            // sort by seq to return in arrival order
            std::sort(allEntries.begin(), allEntries.end(),
              [](auto const& a, auto const& b) { return a.first < b.first; });
 
            // deduplicate — multiple columns in same Update share same seq
            // we only want to emit each Update expression once
            int lastSeq = -1;
            for(auto& [seq, expr] : allEntries) {
              if(seq != lastSeq) {
                entries.push_back(std::move(expr));
                lastSeq = seq;
              }
            }
          }
          // commented out for better testing output
          // std::cout << "WAL: returning " << entries.size() << " raw entries" << std::endl;
          return ComplexExpression("List"_, {}, std::move(entries), {});
        }

        // ResetWALStats — reset instrumentation counters only.
        // Does not clear the WAL itself.
        if(head == "ResetWALStats"_) {
          resetWALStats();
          return "WALStats_Reset"_();
        }

        // PrintWALStats(label?) — print instrumentation counters.
        // Optional label can be a string or Symbol.
        if(head == "PrintWALStats"_) {
          std::string label;

          if(!expr.getArguments().empty()) {
            auto labelArg = expr.getArguments()[0];

            if(auto const* s = get_if<std::string>(&labelArg)) {
              label = *s;
            } else if(auto const* sym = get_if<Symbol>(&labelArg)) {
              label = sym->getName();
            }
          }

          printWALStats(label);
          return "WALStats_Printed"_();
        }
        
        // ClearWAL - empty the log
        if (head == "ClearWAL"_) {
          walIndex.clear();
          walOrder.clear();
          walTotalEntries = 0;
          return "WAL_Cleared"_();
        }
        // OptimiseWAL — compact every bucket in place, WAL stays alive
        if(head == "OptimiseWAL"_) {
          // commented out for better testing output
          // std::cout << "WAL: optimising " << walTotalEntries << " entries" << std::endl;
          for(auto const& key : walOrder) {
            auto it = walIndex.find(key);
            if(it == walIndex.end()) continue;
            RowBucket& bucket = it->second;

            // count entries before optimise
            size_t countBefore = 0;
            for(auto const& [col, entries] : bucket.columnEntries) countBefore += entries.size();
            if(bucket.deleteEntry.has_value()) countBefore++;

            optimiseBucketImpl(bucket);

            // count entries after optimise
            size_t countAfter = 0;
            for(auto const& [col, entries] : bucket.columnEntries) countAfter += entries.size();
            if(bucket.deleteEntry.has_value()) countAfter++;

            walTotalEntries -= countBefore;
            walTotalEntries += countAfter;
          }
          // commented out for better testing output
          // std::cout << "WAL: " << walTotalEntries << " entries after optimisation" << std::endl;
          return "WAL_Optimised"_();
        }

        // Project(Select(...), As(...)) - column-selective flush before read
        // handles the case where only specific columns are being read
        // we flush only those columns for the affected row, leaving others buffered
        if(head == "Project"_) {
          auto targets = extractSelectKeys(expr);
          boss::ExpressionArguments applyArgs;

          if(!targets.empty()) {
            // selective flush — only requested columns for affected rows
            for(auto const& target : targets) {
              auto flushed = flushBucket(target.key, target.columns, FlushReason::ReadTriggered);
              for(auto& e : flushed) applyArgs.push_back(std::move(e));
            }
          } else {
            // could not parse — fall back to full flush for correctness
            // std::cout << "WAL: Project fallback to full flush" << std::endl;
            auto keysCopy = walOrder;
            for(auto const& key : keysCopy) {
              auto flushed = flushBucket(key, {}, FlushReason::ReadTriggered);
              for(auto& e : flushed) applyArgs.push_back(std::move(e));
            }
          }

          // append the original Project expression as the last argument
          // the downstream engine will evaluate the Project normally after WAL is applied
          applyArgs.push_back(std::move(expr));
          return ComplexExpression("ApplyWAL"_, {}, std::move(applyArgs), {});
        }

        // Select - triggers WAL flush before read
        // bare Select flushes the whole row — no column filter
        if(head == "Select"_) {
          auto targets = extractSelectKeys(expr);
          boss::ExpressionArguments applyArgs;

          if(!targets.empty()) {
            // selective flush — only affected rows, but all columns
            for(auto const& target : targets) {
              auto flushed = flushBucket(target.key, {}, FlushReason::ReadTriggered); // no column filter — whole row
              for(auto& e : flushed) applyArgs.push_back(std::move(e));
            }
          } else {
            // could not parse row IDs — fall back to full flush for correctness
            // commented out for better testing output
            // std::cout << "WAL: Select fallback to full flush" << std::endl;
            auto keysCopy = walOrder;
            for(auto const& key : keysCopy) {
              auto flushed = flushBucket(key, {}, FlushReason::ReadTriggered);
              for(auto& e : flushed) applyArgs.push_back(std::move(e));
            }
          }

          // append the original Select as the last argument
          applyArgs.push_back(std::move(expr));
          return ComplexExpression("ApplyWAL"_, {}, std::move(applyArgs), {});
        }
        
        
        // for flushing the WAL
        // for flushing the WAL
        if(head == "FlushWAL"_) {
          // FlushWAL() — flush everything
          if(expr.getArguments().size() == 0) {
            return flushAllBuckets(FlushReason::Manual);
          }

          // FlushWAL(tableName, rowId) — flush only one specific row
          if(expr.getArguments().size() == 2) {
            auto tableArg = expr.getArguments()[0];
            auto const* tableSymbol = get_if<Symbol>(&tableArg);
            if(!tableSymbol) return flushAllBuckets(FlushReason::Manual); // fallback

            auto rowArg = expr.getArguments()[1];
            int64_t rowId = 0;
            if(auto const* id32 = get_if<int32_t>(&rowArg)) rowId = *id32;
            else if(auto const* id64 = get_if<int64_t>(&rowArg)) rowId = *id64;
            else return flushAllBuckets(FlushReason::Manual); // fallback

            WALKey key{tableSymbol->getName(), rowId};
            auto flushed = flushBucket(key, {}, FlushReason::Manual);
            boss::ExpressionArguments entries;
            for(auto& e : flushed) entries.push_back(std::move(e));
            return ComplexExpression("ApplyWAL"_, {}, std::move(entries), {});
          }

          return flushAllBuckets(FlushReason::Manual); // fallback for any other form
        }

        // for changing the WAL threshold at runtime
        if(head == "SetWALThreshold"_) {
          WAL_THRESHOLD = get<int32_t>(expr.getArguments()[0]);
          return "WALThreshold_Set"_();
        }
      }
      return std::move(expr);
    }, std::move(e));
};

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
};
