#include <BOSS.hpp>
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <Utilities.hpp>
#include "ankerl/unordered_dense.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <string>
#include <cstdint>
#include <functional>
#include <limits>

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
 
using WALKey = std::pair<int32_t, int64_t>; // (tableId, rowID)

struct WALKeyHash {
  size_t operator()(WALKey const& k) const {
    size_t h1 = std::hash<int32_t>{}(k.first);
    size_t h2 = std::hash<int64_t>{}(k.second);
    return h1 ^ (h2 * 2654435761ULL);
  }
};

// Intern tables — convert low-cardinality string keys (table names, column names)
// to small integers so WAL index and column entry map lookups hash integers, not strings.
static std::unordered_map<std::string, int32_t> tableNameIntern;
static std::unordered_map<std::string, int32_t> columnNameIntern;
// Reverse lookup: tableId -> Symbol (populated alongside tableNameIntern)
static std::vector<Symbol> tableIdToSymbol;
// Reverse lookup: colId → Symbol (populated alongside columnNameIntern)
static std::vector<Symbol> columnIdToSymbol;
static int32_t nextTableId = 0;
static int32_t nextColumnId = 0;

static int32_t internTableName(std::string const& name) {
  auto [it, inserted] = tableNameIntern.try_emplace(name, nextTableId);
  if(inserted) {
    tableIdToSymbol.emplace_back(name);
    ++nextTableId;
  }
  return it->second;
}

static int32_t internColumnName(std::string const& name) {
  auto [it, inserted] = columnNameIntern.try_emplace(name, nextColumnId);
  if(inserted) {
    columnIdToSymbol.emplace_back(name);
    ++nextColumnId;
  }
  return it->second;
}

struct CellKey {
  int32_t tableId = -1;
  int64_t rowId = 0;
  int32_t colId = -1;
};

static bool operator==(CellKey const& lhs, CellKey const& rhs) {
  return lhs.tableId == rhs.tableId &&
         lhs.rowId == rhs.rowId &&
         lhs.colId == rhs.colId;
}

struct CellKeyHash {
  size_t operator()(CellKey const& k) const {
    size_t h1 = std::hash<int32_t>{}(k.tableId);
    size_t h2 = std::hash<int64_t>{}(k.rowId);
    size_t h3 = std::hash<int32_t>{}(k.colId);
    return h1 ^ (h2 * 2654435761ULL) ^ (h3 * 1099511628211ULL);
  }
};

// One WAL entry: the raw expression and its global arrival sequence number
struct WALEntry {
  Expression valueExpr;
  int seq;
  bool isBlindWrite = false;
  std::vector<int32_t> referencedCols;
  std::vector<CellKey> referencedCells;
};

struct DeleteEntry {
  int seq;
};

using OperationId = uint64_t;

struct WALAssignment {
  int32_t colId;
  Expression valueExpr;
  bool isBlindWrite = false;
  std::vector<int32_t> referencedCols;
  std::vector<CellKey> referencedCells;
};

struct WALOperation {
  OperationId opId;
  int32_t tableId;
  int32_t idColumnId = -1;
  std::vector<RowID> rowIds;
  std::vector<WALAssignment> assignments;
  int seqBase;
  uint32_t liveRefCount = 0;
};

struct WALEntryRef {
  OperationId opId;
  uint32_t assignmentIndex;
  int seq;
};

using WALColumnEntry = std::variant<WALEntry, WALEntryRef>;

struct EntryHandle {
  int32_t colId;
  int seq;
};

struct ReverseDepList {
  int32_t referencedColId = -1;
  std::vector<EntryHandle> readers;
};

struct CellEntryHandle {
  CellKey cell;
  int seq;
};

// SelectTarget — result of parsing a Select or Project(Select(...)) expression
// key:     (tableName, rowID) — which row to flush
// columns: which columns to flush — empty means flush all columns (bare Select case)
struct SelectTarget {
  WALKey key;
  std::vector<std::string> columns;
};

 
// Fix 3: Inline small array replacing unordered_map<int32_t, vector<WALEntry>>.
// Most OLTP rows touch ≤8 columns; storing them inline avoids two heap pointer
// hops per lookup and keeps the column map entirely on the stack / in-object.
struct ColEntry {
  int32_t colId = -1;
  std::vector<WALColumnEntry> entries;
};

struct InlineColVec {
  static constexpr int N = 8;
  ColEntry data[N];
  int size = 0;

  ColEntry* find(int32_t id) {
    for(int i = 0; i < size; ++i)
      if(data[i].colId == id) return &data[i];
    return nullptr;
  }
  ColEntry const* find(int32_t id) const {
    for(int i = 0; i < size; ++i)
      if(data[i].colId == id) return &data[i];
    return nullptr;
  }

  // Returns existing entry or inserts a new slot.
  // Returns {nullptr, false} if the array is full (should not happen for ≤N columns).
  std::pair<ColEntry*, bool> try_emplace(int32_t id) {
    if(auto* p = find(id)) return {p, false};
    if(size < N) {
      data[size].colId = id;
      data[size].entries.clear();
      return {&data[size++], true};
    }
    return {nullptr, false}; // caller must handle null
  }

  ColEntry* begin() { return data; }
  ColEntry* end()   { return data + size; }
  ColEntry const* begin() const { return data; }
  ColEntry const* end()   const { return data + size; }
  bool empty() const { return size == 0; }
  void erase(int32_t id) {
    for(int i = 0; i < size; ++i) {
      if(data[i].colId == id) {
        data[i].entries.clear();
        for(int j = i + 1; j < size; ++j) {
          data[j - 1] = std::move(data[j]);
        }
        --size;
        data[size].colId = -1;
        data[size].entries.clear();
        return;
      }
    }
  }
  void clear() {
    for(int i = 0; i < size; ++i) data[i].entries.clear();
    size = 0;
  }
};

struct RowBucket {
  // per-column entry lists — inline fixed-size array (Fix 3)
  InlineColVec columnEntries;

  // latest Delete for this row
  std::optional<DeleteEntry> deleteEntry;

  // Canonical row identity. Keep semantic scalars here and reconstruct BOSS
  // output expressions at the flush boundary instead of cloning stored syntax.
  int32_t tableId = -1;
  RowID rowId = int64_t{0};
  int32_t idColumnId = -1;

  // Same-row cross-column dependency metadata.
  // referenced column id -> pending entries in this row that read that column.
  std::vector<ReverseDepList> reverseDeps;
};
 
// the WAL index: (tableName, rowID) → bucket
// Fix 2: ankerl::unordered_dense stores entries in a flat contiguous array,
// eliminating one heap pointer hop per lookup compared to std::unordered_map.
static ankerl::unordered_dense::map<WALKey, RowBucket, WALKeyHash> walIndex;
static ankerl::unordered_dense::map<OperationId, WALOperation> walOperations;
static OperationId nextOperationId = 0;
static int globalNextSeq = 0;
static ankerl::unordered_dense::map<CellKey, std::vector<CellEntryHandle>, CellKeyHash> reverseDepsByCell;
static size_t pendingCrossRowRefEntries = 0;

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

  uint64_t dependencyEntriesConsumed = 0;
  uint64_t dependencyPhysicalUpdatesEmitted = 0;

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

static size_t countPhysicalUpdates(std::vector<Expression> const& expressions) {
  size_t updates = 0;
  for(auto const& expression : expressions) {
    auto const* complex = get_if<ComplexExpression>(&expression);
    if(complex && complex->getHead() == "Update"_) {
      updates++;
    }
  }
  return updates;
}

static void recordDependencyMaterialisation(
  size_t entriesConsumed,
  std::vector<Expression> const& physicalExpressions) {
  walStats.dependencyEntriesConsumed += entriesConsumed;
  walStats.dependencyPhysicalUpdatesEmitted += countPhysicalUpdates(physicalExpressions);
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
  std::cout << "dependencyEntriesConsumed  = "
            << walStats.dependencyEntriesConsumed << "\n";
  std::cout << "dependencyPhysicalUpdates  = "
            << walStats.dependencyPhysicalUpdatesEmitted << "\n";

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

  if(walStats.dependencyPhysicalUpdatesEmitted > 0) {
    std::cout << "dependencyEntryToUpdateRatio = "
              << static_cast<double>(walStats.dependencyEntriesConsumed) /
                   static_cast<double>(walStats.dependencyPhysicalUpdatesEmitted)
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

static void recordDependencyMaterialisation(size_t, std::vector<Expression> const&) {}

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
  auto const& args = idListExpr.getArguments();
  for(size_t i = 0; i < args.size(); i++) {
    auto const& idVal = args[i];
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

static bool isBlindValueWrite(Expression const& valueExpr) {
  return !containsSymbol(valueExpr);
}

// Check if a column assignment in Set(...) is a blind write
// e.g. price(100.0) → blind write, price(Plus(price, 1)) → dependent write
static bool isBlindWrite(ComplexExpression const& colAssign) {
  auto const& args = colAssign.getArguments();

  if(args.empty()) return true;

  return !containsSymbolArgument(args[0]);
}

static std::vector<int32_t> collectReferencedColumnIds(Expression const& expr);
static std::vector<CellKey> collectReferencedCellKeys(Expression const& expr);

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
// This avoids materialising argument copies when constant folding only needs to inspect constants.
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

// Attempt constant folding on a folded expression.
// handles same-operator chains where both constants are concrete numbers
// e.g. Plus(Plus(price, 1.0), 1.0)   → Plus(price, 2.0)
// e.g. Times(Times(price, 2.0), 3.0) → Times(price, 6.0)
// e.g. Minus(Minus(price, 1.0), 2.0) → Minus(price, 3.0)
// e.g. Divide(Divide(price, 2.0), 2.0) → Divide(price, 4.0)
// If the pattern does not match, the original expression is returned unchanged.
static Expression simplifyConstantFold(Expression expr) {
  auto const* outer = get_if<ComplexExpression>(&expr);
  if(!outer) return expr;

  auto const& outerArgs = outer->getArguments();
  if(outerArgs.size() < 2) return expr;

  auto const& outerHead = outer->getHead();

  // only handle our four arithmetic operators
  if(outerHead != "Plus"_  && outerHead != "Minus"_ &&
     outerHead != "Times"_ && outerHead != "Divide"_) return expr;

  // outer's first arg must be a ComplexExpression with the SAME operator.
  // V2E-A: inspect by wrapper/reference instead of materialising argument 0.
  auto const* inner = std::visit(
    [](auto const& unwrapped) -> ComplexExpression const* {
      return asComplexExpressionPtr(unwrapped);
    },
    outerArgs[0].getArgument()
  );

  if(!inner) return expr;
  if(inner->getHead() != outerHead) return expr;

  auto const& innerArgs = inner->getArguments();
  if(innerArgs.size() < 2) return expr;

  // inner's second arg must be a concrete number (c1).
  // V2E-A: inspect by wrapper/reference instead of materialising argument 1.
  auto c1 = std::visit(
    [](auto const& unwrapped) -> std::optional<double> {
      return toDoubleValue(unwrapped);
    },
    innerArgs[1].getArgument()
  );

  if(!c1) return expr;

  // outer's second arg must be a concrete number (c2).
  // V2E-A: inspect by wrapper/reference instead of materialising argument 1.
  auto c2 = std::visit(
    [](auto const& unwrapped) -> std::optional<double> {
      return toDoubleValue(unwrapped);
    },
    outerArgs[1].getArgument()
  );

  if(!c2) return expr;

  // compute the folded constant based on operator
  double folded;
  if(outerHead == "Plus"_)        folded = *c1 + *c2;
  else if(outerHead == "Minus"_)  folded = *c1 + *c2; // Minus(Minus(x,a),b) = Minus(x, a+b)
  else if(outerHead == "Times"_)  folded = *c1 * *c2;
  else if(outerHead == "Divide"_) folded = *c1 * *c2; // Divide(Divide(x,a),b) = Divide(x, a*b)
  else return expr;

  auto* ownedOuter = get_if<ComplexExpression>(&expr);
  auto [outerMoveHead, outerStatics, outerDynamics, outerSpans] =
    std::move(*ownedOuter).decompose();
  auto* ownedInner = get_if<ComplexExpression>(&outerDynamics[0]);
  if(!ownedInner) {
    return expr;
  }
  auto [innerMoveHead, innerStatics, innerDynamics, innerSpans] =
    std::move(*ownedInner).decompose();
  if(innerDynamics.empty()) {
    return expr;
  }

  // Rebuild by moving the operator head and inner->arg0 into the simplified expression.
  boss::ExpressionArguments newArgs;
  newArgs.push_back(std::move(innerDynamics[0]));
  newArgs.push_back(folded);

  return ComplexExpression(std::move(outerMoveHead), {}, std::move(newArgs), {});
}

enum class SymbolUse {
  Zero,
  One,
  Many
};

template <typename WrappedArgument>
static SymbolUse classifySymbolUseArgument(WrappedArgument const& wrappedArg,
                                           Symbol const& target);

static SymbolUse classifySymbolUse(Expression const& expr, Symbol const& target);

static SymbolUse combineSymbolUse(SymbolUse accumulated, SymbolUse next) {
  if(accumulated == SymbolUse::Many || next == SymbolUse::Many) return SymbolUse::Many;
  if(accumulated == SymbolUse::One && next == SymbolUse::One) return SymbolUse::Many;
  if(accumulated == SymbolUse::One || next == SymbolUse::One) return SymbolUse::One;
  return SymbolUse::Zero;
}

template <typename T>
static SymbolUse classifySymbolUseValue(T const& value, Symbol const& target) {
  using Decayed = std::decay_t<T>;

  if constexpr(boss::utilities::isInstanceOfTemplate<
                 Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
    return classifySymbolUseValue(value.get(), target);
  } else if constexpr(std::is_same_v<Decayed, Symbol>) {
    return value == target ? SymbolUse::One : SymbolUse::Zero;
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    SymbolUse use = SymbolUse::Zero;
    auto const& args = value.getArguments();

    for(auto const& arg : args) {
      use = combineSymbolUse(use, classifySymbolUseArgument(arg, target));
      if(use == SymbolUse::Many) return SymbolUse::Many;
    }

    return use;
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    return classifySymbolUse(value, target);
  } else {
    return SymbolUse::Zero;
  }
}

template <typename WrappedArgument>
static SymbolUse classifySymbolUseArgument(WrappedArgument const& wrappedArg,
                                           Symbol const& target) {
  return visitArgumentByReference(
    wrappedArg,
    [&](auto const& unwrapped) -> SymbolUse {
      return classifySymbolUseValue(unwrapped, target);
    }
  );
}

static SymbolUse classifySymbolUse(Expression const& expr, Symbol const& target) {
  return std::visit(
    [&](auto const& value) -> SymbolUse {
      return classifySymbolUseValue(value, target);
    },
    expr
  );
}

static Expression buildTableNameExpression(RowBucket const& bucket) {
  return tableIdToSymbol[bucket.tableId];
}

static Expression buildRowIdExpression(RowBucket const& bucket) {
  boss::ExpressionArguments idListArgs;
  std::visit([&](auto id) { idListArgs.push_back(id); }, bucket.rowId);
  auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});

  boss::ExpressionArguments idColArgs;
  idColArgs.push_back(std::move(idList));

  if(bucket.idColumnId >= 0) {
    return ComplexExpression(columnIdToSymbol[bucket.idColumnId], {}, std::move(idColArgs), {});
  }
  return ComplexExpression("id"_, {}, std::move(idColArgs), {});
}

static Expression buildDeleteExpression(RowBucket const& bucket) {
  boss::ExpressionArguments deleteArgs;
  deleteArgs.push_back(buildTableNameExpression(bucket));
  deleteArgs.push_back(buildRowIdExpression(bucket));
  return ComplexExpression("Delete"_, {}, std::move(deleteArgs), {});
}

static Expression buildDeleteDebugSummary(RowBucket const& bucket, int seq) {
  boss::ExpressionArguments args;
  args.push_back(tableIdToSymbol[bucket.tableId]);
  std::visit([&](auto rowId) { args.push_back(rowId); }, bucket.rowId);
  args.push_back(static_cast<int64_t>(seq));
  return ComplexExpression("WALDelete"_, {}, std::move(args), {});
}

static Expression buildRowIdExpression(int32_t idColumnId, std::vector<RowID> const& rowIds) {
  boss::ExpressionArguments idListArgs;
  idListArgs.reserve(rowIds.size());
  for(auto const& rowId : rowIds) {
    std::visit([&](auto id) { idListArgs.push_back(id); }, rowId);
  }
  auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});

  boss::ExpressionArguments idColArgs;
  idColArgs.push_back(std::move(idList));

  if(idColumnId >= 0) {
    return ComplexExpression(columnIdToSymbol[idColumnId], {}, std::move(idColArgs), {});
  }
  return ComplexExpression("id"_, {}, std::move(idColArgs), {});
}

template <typename T>
static void captureBucketRowIdentity(RowBucket& bucket, T const& idExpression) {
  auto const* idExpr = asComplexExpressionPtr(idExpression);
  if(!idExpr) return;

  bucket.idColumnId = internColumnName(idExpr->getHead().getName());

  auto const& idArgs = idExpr->getArguments();
  if(idArgs.empty()) return;

  auto const& listArg = idArgs[0];
  auto const* listExpr = get_if<ComplexExpression>(&listArg);
  if(!listExpr) return;

  auto const& listArgs = listExpr->getArguments();
  if(listArgs.empty()) return;

  auto const& firstId = listArgs[0];
  if(auto const* id32 = get_if<int32_t>(&firstId)) {
    bucket.rowId = *id32;
  } else if(auto const* id64 = get_if<int64_t>(&firstId)) {
    bucket.rowId = *id64;
  }
}

static bool captureAssignmentsFromSet(Expression setExpression, WALOperation& operation) {
  auto* setExpr = get_if<ComplexExpression>(&setExpression);
  if(!setExpr || setExpr->getHead() != "Set"_) {
    return false;
  }

  auto [setHead, setStatics, setDynamics, setSpans] = std::move(*setExpr).decompose();
  operation.assignments.reserve(setDynamics.size());

  for(auto& setDynamic : setDynamics) {
    Expression colExpression = std::move(setDynamic);
    auto* colExpr = get_if<ComplexExpression>(&colExpression);
    if(!colExpr) {
      continue;
    }

    auto [colHead, colStatics, colDynamics, colSpans] = std::move(*colExpr).decompose();
    if(colDynamics.empty()) {
      continue;
    }

    Expression valueExpr = std::move(colDynamics[0]);
    bool blind = isBlindValueWrite(valueExpr);
    auto referencedCols = collectReferencedColumnIds(valueExpr);
    auto referencedCells = collectReferencedCellKeys(valueExpr);
    operation.assignments.push_back({
      internColumnName(colHead.getName()),
      std::move(valueExpr),
      blind,
      std::move(referencedCols),
      std::move(referencedCells)
    });
  }

  return !operation.assignments.empty();
}

static WALEntry consumeColumnEntryRef(WALEntryRef const& ref) {
  auto opIt = walOperations.find(ref.opId);
  if(opIt == walOperations.end() || ref.assignmentIndex >= opIt->second.assignments.size()) {
    return {int32_t{0}, ref.seq, true};
  }

  WALAssignment& assignment = opIt->second.assignments[ref.assignmentIndex];
  WALEntry entry{
    std::move(assignment.valueExpr),
    ref.seq,
    assignment.isBlindWrite,
    std::move(assignment.referencedCols),
    std::move(assignment.referencedCells)
  };
  walOperations.erase(opIt);
  return entry;
}

static void discardColumnEntryRef(WALEntryRef const& ref) {
  walOperations.erase(ref.opId);
}

static std::optional<Expression> buildUpdateFromOperation(WALOperation& operation) {
  if(operation.tableId < 0 || operation.assignments.empty() || operation.rowIds.empty()) {
    return std::nullopt;
  }

  boss::ExpressionArguments setArgs;
  setArgs.reserve(operation.assignments.size());
  for(auto& assignment : operation.assignments) {
    boss::ExpressionArguments colArgs;
    colArgs.push_back(std::move(assignment.valueExpr));
    setArgs.push_back(
      ComplexExpression(columnIdToSymbol[assignment.colId], {}, std::move(colArgs), {}));
  }

  boss::ExpressionArguments updateArgs;
  updateArgs.push_back(tableIdToSymbol[operation.tableId]);
  updateArgs.push_back(buildRowIdExpression(operation.idColumnId, operation.rowIds));
  updateArgs.push_back(ComplexExpression("Set"_, {}, std::move(setArgs), {}));

  return ComplexExpression("Update"_, {}, std::move(updateArgs), {});
}

static Expression buildOperationDebugSummary(WALOperation const& operation) {
  boss::ExpressionArguments args;
  args.push_back(static_cast<int64_t>(operation.opId));
  args.push_back(tableIdToSymbol[operation.tableId]);
  args.push_back(static_cast<int64_t>(operation.rowIds.size()));
  args.push_back(static_cast<int64_t>(operation.assignments.size()));
  return ComplexExpression("WALOperation"_, {}, std::move(args), {});
}

static Expression buildRefDebugSummary(RowBucket const& bucket, int32_t colId, WALEntryRef const& ref) {
  boss::ExpressionArguments args;
  args.push_back(static_cast<int64_t>(ref.opId));
  args.push_back(tableIdToSymbol[bucket.tableId]);
  std::visit([&](auto rowId) { args.push_back(rowId); }, bucket.rowId);
  args.push_back(columnIdToSymbol[colId]);
  args.push_back(static_cast<int64_t>(ref.seq));

  auto opIt = walOperations.find(ref.opId);
  bool isBlind = false;
  if(opIt != walOperations.end() && ref.assignmentIndex < opIt->second.assignments.size()) {
    isBlind = opIt->second.assignments[ref.assignmentIndex].isBlindWrite;
  }
  args.push_back(isBlind ? "blind"s : "dependent"s);

  return ComplexExpression("WALRef"_, {}, std::move(args), {});
}

static Expression buildLocalDebugSummary(RowBucket const& bucket, int32_t colId, WALEntry const& entry) {
  boss::ExpressionArguments args;
  args.push_back(static_cast<int64_t>(-1));
  args.push_back(tableIdToSymbol[bucket.tableId]);
  std::visit([&](auto rowId) { args.push_back(rowId); }, bucket.rowId);
  args.push_back(columnIdToSymbol[colId]);
  args.push_back(static_cast<int64_t>(entry.seq));
  args.push_back(entry.isBlindWrite ? "blind"s : "dependent"s);

  return ComplexExpression("WALRef"_, {}, std::move(args), {});
}

static bool isSharedOperationRef(WALEntryRef const& ref) {
  auto opIt = walOperations.find(ref.opId);
  return opIt != walOperations.end() && opIt->second.rowIds.size() > 1;
}

static int getColumnEntrySeq(WALColumnEntry const& entry) {
  return std::visit([](auto const& e) { return e.seq; }, entry);
}

static void addUniqueColumnId(std::vector<int32_t>& ids, int32_t colId) {
  if(std::find(ids.begin(), ids.end(), colId) == ids.end()) {
    ids.push_back(colId);
  }
}

static void addUniqueCellKey(std::vector<CellKey>& cells, CellKey cell) {
  if(std::find(cells.begin(), cells.end(), cell) == cells.end()) {
    cells.push_back(cell);
  }
}

static std::optional<CellKey> parseLevel1CellReference(ComplexExpression const& expr) {
  if(expr.getHead() != "Cell"_) {
    return std::nullopt;
  }

  auto const& args = expr.getArguments();
  if(args.size() != 4) {
    return std::nullopt;
  }

  std::optional<Symbol> tableSym;
  std::optional<Symbol> valueColSym;
  std::optional<int64_t> rowId;

  visitArgumentByReference(args[0], [&](auto const& unwrapped) {
    using Decayed = std::decay_t<decltype(unwrapped)>;
    if constexpr(std::is_same_v<Decayed, Symbol>) {
      tableSym = unwrapped;
    } else if constexpr(std::is_same_v<Decayed, Expression>) {
      if(auto const* symbol = get_if<Symbol>(&unwrapped)) {
        tableSym = *symbol;
      }
    }
  });
  visitArgumentByReference(args[2], [&](auto const& unwrapped) {
    using Decayed = std::decay_t<decltype(unwrapped)>;
    if constexpr(std::is_integral_v<Decayed> && !std::is_same_v<Decayed, bool>) {
      rowId = static_cast<int64_t>(unwrapped);
    } else if constexpr(std::is_same_v<Decayed, Expression>) {
      if(auto const* id32 = get_if<int32_t>(&unwrapped)) {
        rowId = static_cast<int64_t>(*id32);
      } else if(auto const* id64 = get_if<int64_t>(&unwrapped)) {
        rowId = *id64;
      }
    }
  });
  visitArgumentByReference(args[3], [&](auto const& unwrapped) {
    using Decayed = std::decay_t<decltype(unwrapped)>;
    if constexpr(std::is_same_v<Decayed, Symbol>) {
      valueColSym = unwrapped;
    } else if constexpr(std::is_same_v<Decayed, Expression>) {
      if(auto const* symbol = get_if<Symbol>(&unwrapped)) {
        valueColSym = *symbol;
      }
    }
  });

  if(!tableSym.has_value() || !rowId.has_value() || !valueColSym.has_value()) {
    return std::nullopt;
  }

  return CellKey{
    internTableName(tableSym->getName()),
    *rowId,
    internColumnName(valueColSym->getName())
  };
}

template <typename T>
static void collectReferencedColumnIdsValue(T const& value, std::vector<int32_t>& out);

template <typename WrappedArgument>
static void collectReferencedColumnIdsArgument(WrappedArgument const& wrappedArg,
                                               std::vector<int32_t>& out) {
  visitArgumentByReference(
    wrappedArg,
    [&](auto const& unwrapped) {
      collectReferencedColumnIdsValue(unwrapped, out);
    }
  );
}

static std::vector<int32_t> collectReferencedColumnIds(Expression const& expr);

template <typename T>
static void collectReferencedColumnIdsValue(T const& value, std::vector<int32_t>& out) {
  using Decayed = std::decay_t<T>;

  if constexpr(boss::utilities::isInstanceOfTemplate<
                 Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
    collectReferencedColumnIdsValue(value.get(), out);
  } else if constexpr(std::is_same_v<Decayed, Symbol>) {
    addUniqueColumnId(out, internColumnName(value.getName()));
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    if(value.getHead() == "Cell"_) {
      return;
    }
    for(auto const& arg : value.getArguments()) {
      collectReferencedColumnIdsArgument(arg, out);
    }
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    std::visit(
      [&](auto const& nested) {
        collectReferencedColumnIdsValue(nested, out);
      },
      value
    );
  }
}

static std::vector<int32_t> collectReferencedColumnIds(Expression const& expr) {
  std::vector<int32_t> referenced;
  referenced.reserve(4);
  std::visit(
    [&](auto const& value) {
      collectReferencedColumnIdsValue(value, referenced);
    },
    expr
  );
  return referenced;
}

template <typename T>
static void collectReferencedCellKeysValue(T const& value, std::vector<CellKey>& out);

template <typename WrappedArgument>
static void collectReferencedCellKeysArgument(WrappedArgument const& wrappedArg,
                                              std::vector<CellKey>& out) {
  visitArgumentByReference(
    wrappedArg,
    [&](auto const& unwrapped) {
      collectReferencedCellKeysValue(unwrapped, out);
    }
  );
}

static std::vector<CellKey> collectReferencedCellKeys(Expression const& expr);

template <typename T>
static void collectReferencedCellKeysValue(T const& value, std::vector<CellKey>& out) {
  using Decayed = std::decay_t<T>;

  if constexpr(boss::utilities::isInstanceOfTemplate<
                 Decayed, boss::expressions::generic::MovableReferenceWrapper>::value) {
    collectReferencedCellKeysValue(value.get(), out);
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    if(auto cell = parseLevel1CellReference(value)) {
      addUniqueCellKey(out, *cell);
      return;
    }
    for(auto const& arg : value.getArguments()) {
      collectReferencedCellKeysArgument(arg, out);
    }
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    std::visit(
      [&](auto const& nested) {
        collectReferencedCellKeysValue(nested, out);
      },
      value
    );
  }
}

static std::vector<CellKey> collectReferencedCellKeys(Expression const& expr) {
  std::vector<CellKey> referenced;
  referenced.reserve(2);
  std::visit(
    [&](auto const& value) {
      collectReferencedCellKeysValue(value, referenced);
    },
    expr
  );
  return referenced;
}

static void registerReverseDependency(RowBucket& bucket,
                                      int32_t referencedColId,
                                      EntryHandle reader) {
  auto it = std::find_if(
    bucket.reverseDeps.begin(),
    bucket.reverseDeps.end(),
    [&](ReverseDepList const& dep) {
      return dep.referencedColId == referencedColId;
    });

  if(it == bucket.reverseDeps.end()) {
    bucket.reverseDeps.push_back(ReverseDepList{referencedColId, {}});
    it = std::prev(bucket.reverseDeps.end());
  }

  auto& readers = it->readers;
  if(readers.empty() || readers.back().seq < reader.seq) {
    readers.push_back(reader);
    return;
  }

  auto insertPos = readers.begin();
  for(; insertPos != readers.end(); ++insertPos) {
    if(insertPos->colId == reader.colId && insertPos->seq == reader.seq) {
      return;
    }
    if(insertPos->seq > reader.seq) {
      break;
    }
  }

  readers.insert(insertPos, reader);
}

static void registerReverseDependencies(RowBucket& bucket,
                                        int32_t readerColId,
                                        int seq,
                                        std::vector<int32_t> const& referencedCols) {
  for(int32_t referencedColId : referencedCols) {
    if(referencedColId == readerColId) {
      continue;
    }
    registerReverseDependency(bucket, referencedColId, EntryHandle{readerColId, seq});
  }
}

static void unregisterReverseDependency(RowBucket& bucket,
                                        int32_t referencedColId,
                                        EntryHandle reader) {
  auto it = std::find_if(
    bucket.reverseDeps.begin(),
    bucket.reverseDeps.end(),
    [&](ReverseDepList const& dep) {
      return dep.referencedColId == referencedColId;
    });
  if(it == bucket.reverseDeps.end()) {
    return;
  }

  auto& readers = it->readers;
  readers.erase(
    std::remove_if(readers.begin(), readers.end(),
      [&](EntryHandle const& existing) {
        return existing.colId == reader.colId && existing.seq == reader.seq;
      }),
    readers.end());

  if(readers.empty()) {
    bucket.reverseDeps.erase(it);
  }
}

static void unregisterReverseDependencies(RowBucket& bucket,
                                          int32_t readerColId,
                                          int seq,
                                          std::vector<int32_t> const& referencedCols) {
  for(int32_t referencedColId : referencedCols) {
    unregisterReverseDependency(bucket, referencedColId, EntryHandle{readerColId, seq});
  }
}

static void registerCellReverseDependency(CellKey referencedCell,
                                          CellEntryHandle reader) {
  auto& readers = reverseDepsByCell[referencedCell];
  if(readers.empty() || readers.back().seq < reader.seq) {
    readers.push_back(reader);
    return;
  }

  auto insertPos = readers.begin();
  for(; insertPos != readers.end(); ++insertPos) {
    if(insertPos->cell == reader.cell && insertPos->seq == reader.seq) {
      return;
    }
    if(insertPos->seq > reader.seq) {
      break;
    }
  }

  readers.insert(insertPos, reader);
}

static void registerCellReverseDependencies(CellKey readerCell,
                                            int seq,
                                            std::vector<CellKey> const& referencedCells) {
  for(auto const& referencedCell : referencedCells) {
    registerCellReverseDependency(referencedCell, CellEntryHandle{readerCell, seq});
  }
  if(!referencedCells.empty()) {
    pendingCrossRowRefEntries++;
  }
}

static void unregisterCellReverseDependency(CellKey referencedCell,
                                            CellEntryHandle reader) {
  auto it = reverseDepsByCell.find(referencedCell);
  if(it == reverseDepsByCell.end()) {
    return;
  }

  auto& readers = it->second;
  readers.erase(
    std::remove_if(readers.begin(), readers.end(),
      [&](CellEntryHandle const& existing) {
        return existing.cell == reader.cell && existing.seq == reader.seq;
      }),
    readers.end());

  if(readers.empty()) {
    reverseDepsByCell.erase(it);
  }
}

static void unregisterCellReverseDependencies(CellKey readerCell,
                                              int seq,
                                              std::vector<CellKey> const& referencedCells) {
  for(auto const& referencedCell : referencedCells) {
    unregisterCellReverseDependency(referencedCell, CellEntryHandle{readerCell, seq});
  }
  if(!referencedCells.empty() && pendingCrossRowRefEntries > 0) {
    pendingCrossRowRefEntries--;
  }
}

static CellKey cellKeyForBucketColumn(RowBucket const& bucket, int32_t colId) {
  return CellKey{bucket.tableId, toInt64(bucket.rowId), colId};
}

static void registerCellReverseDependencies(RowBucket const& bucket,
                                            int32_t readerColId,
                                            int seq,
                                            std::vector<CellKey> const& referencedCells) {
  registerCellReverseDependencies(
    cellKeyForBucketColumn(bucket, readerColId), seq, referencedCells);
}

static void unregisterCellReverseDependencies(RowBucket const& bucket,
                                              int32_t readerColId,
                                              int seq,
                                              std::vector<CellKey> const& referencedCells) {
  unregisterCellReverseDependencies(
    cellKeyForBucketColumn(bucket, readerColId), seq, referencedCells);
}

static void registerAllCellReverseDependencies(RowBucket const& bucket) {
  for(auto const& col : bucket.columnEntries) {
    for(auto const& entry : col.entries) {
      if(auto const* local = std::get_if<WALEntry>(&entry)) {
        registerCellReverseDependencies(bucket, col.colId, local->seq, local->referencedCells);
      } else if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
        auto opIt = walOperations.find(ref->opId);
        if(opIt == walOperations.end() ||
           ref->assignmentIndex >= opIt->second.assignments.size()) {
          continue;
        }
        auto const& assignment = opIt->second.assignments[ref->assignmentIndex];
        registerCellReverseDependencies(bucket, col.colId, ref->seq, assignment.referencedCells);
      }
    }
  }
}

static void unregisterAllCellReverseDependencies(RowBucket const& bucket) {
  for(auto const& col : bucket.columnEntries) {
    for(auto const& entry : col.entries) {
      if(auto const* local = std::get_if<WALEntry>(&entry)) {
        unregisterCellReverseDependencies(bucket, col.colId, local->seq, local->referencedCells);
      } else if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
        auto opIt = walOperations.find(ref->opId);
        if(opIt == walOperations.end() ||
           ref->assignmentIndex >= opIt->second.assignments.size()) {
          continue;
        }
        auto const& assignment = opIt->second.assignments[ref->assignmentIndex];
        unregisterCellReverseDependencies(bucket, col.colId, ref->seq, assignment.referencedCells);
      }
    }
  }
}

static void rebuildReverseDependencies(RowBucket& bucket) {
  bucket.reverseDeps.clear();

  for(auto const& col : bucket.columnEntries) {
    for(auto const& entry : col.entries) {
      if(auto const* local = std::get_if<WALEntry>(&entry)) {
        registerReverseDependencies(bucket, col.colId, local->seq, local->referencedCols);
      } else if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
        auto opIt = walOperations.find(ref->opId);
        if(opIt == walOperations.end() ||
           ref->assignmentIndex >= opIt->second.assignments.size()) {
          continue;
        }
        auto const& assignment = opIt->second.assignments[ref->assignmentIndex];
        registerReverseDependencies(bucket, col.colId, ref->seq, assignment.referencedCols);
      }
    }
  }
}

static void rebuildCellReverseDependencies() {
  reverseDepsByCell.clear();
  pendingCrossRowRefEntries = 0;

  for(auto const& [key, bucket] : walIndex) {
    for(auto const& col : bucket.columnEntries) {
      CellKey readerCell{bucket.tableId, toInt64(bucket.rowId), col.colId};
      for(auto const& entry : col.entries) {
        if(auto const* local = std::get_if<WALEntry>(&entry)) {
          registerCellReverseDependencies(readerCell, local->seq, local->referencedCells);
        } else if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
          auto opIt = walOperations.find(ref->opId);
          if(opIt == walOperations.end() ||
             ref->assignmentIndex >= opIt->second.assignments.size()) {
            continue;
          }
          auto const& assignment = opIt->second.assignments[ref->assignmentIndex];
          registerCellReverseDependencies(readerCell, ref->seq, assignment.referencedCells);
        }
      }
    }
  }
}

static void maybeRebuildCellReverseDependencies() {
  if(reverseDepsByCell.empty() && pendingCrossRowRefEntries == 0) {
    return;
  }
  rebuildCellReverseDependencies();
}

static WALEntryRef const* getColumnEntryRef(WALColumnEntry const& entry) {
  return std::get_if<WALEntryRef>(&entry);
}

static bool isSharedOperationEntry(WALColumnEntry const& entry) {
  auto const* ref = getColumnEntryRef(entry);
  return ref && isSharedOperationRef(*ref);
}

static bool isBlindSharedAssignment(OperationId opId, int32_t colId) {
  auto opIt = walOperations.find(opId);
  if(opIt == walOperations.end() || opIt->second.rowIds.size() <= 1) {
    return false;
  }

  for(auto const& assignment : opIt->second.assignments) {
    if(assignment.colId == colId) {
      return assignment.isBlindWrite;
    }
  }
  return false;
}

static bool bucketHasPendingWork(RowBucket const& bucket) {
  if(bucket.deleteEntry.has_value()) {
    return true;
  }
  for(auto const& col : bucket.columnEntries) {
    if(!col.entries.empty()) {
      return true;
    }
  }
  return false;
}

// ============================================================
// substituteAndFold — recursive substitution without std::function overhead
// ============================================================
//
// Replaces all occurrences of colName with replacement inside expr,
// applying constant folding after rebuilding each ComplexExpression node (V2M).
//
// V2O: foldColumnValues previously used a std::function<Expression(Expression
// const&)> for mutual recursion between two lambdas (substituteExpr and
// substituteValue). std::function uses type erasure, so every recursive call
// went through virtual dispatch, plus each argument passed to it was
// implicitly converted into a temporary Expression (a clone for
// ComplexExpression). Replacing it with two ordinary mutually recursive
// functions removes both the dispatch overhead and the implicit clones —
// visitArgumentByReference already gives us the concrete unwrapped type, so
// substituteAndFoldValue can dispatch on it directly via if constexpr.

static Expression substituteAndFold(Expression expr,
                                    Symbol const& colName,
                                    Expression& replacement,
                                    bool& moved);

template <typename T>
static Expression substituteAndFoldValue(T&& value,
                                          Symbol const& colName,
                                          Expression& replacement,
                                          bool& moved)
{
  using Decayed = std::decay_t<T>;

  if constexpr(std::is_same_v<Decayed, Symbol>) {
    if(value == colName) {
      if(!moved) {
        moved = true;
        return std::move(replacement);
      }
    }
    return value;
  } else if constexpr(std::is_same_v<Decayed, ComplexExpression>) {
    auto [head, statics, dynamics, spans] = std::move(value).decompose();
    boss::ExpressionArguments newArgs;
    newArgs.reserve(dynamics.size());
    for(auto& arg : dynamics) {
      newArgs.push_back(substituteAndFold(std::move(arg), colName, replacement, moved));
    }
    Expression rebuilt = ComplexExpression(head, {}, std::move(newArgs), {});

    // V2M: apply constant folding immediately during recursive substitution.
    return simplifyConstantFold(std::move(rebuilt));
  } else if constexpr(std::is_same_v<Decayed, Expression>) {
    return substituteAndFold(std::move(value), colName, replacement, moved);
  } else {
    return value;
  }
}

static Expression substituteAndFold(Expression expr,
                                    Symbol const& colName,
                                    Expression& replacement,
                                    bool& moved)
{
  return std::visit([&](auto& value) -> Expression {
    return substituteAndFoldValue(std::move(value), colName, replacement, moved);
  }, expr);
}

static std::optional<Expression> foldColumnValues(Symbol const& colName,
                                                  Expression earlierValueExpr,
                                                  Expression laterValueExpr) {
  // earlierValueExpr = Plus(price, 1)
  // laterValueExpr   = Plus(price, 1)
  //
  // result:
  // substitute price in laterValueExpr with earlierValueExpr
  // => Plus(Plus(price, 1), 1)

  SymbolUse replacementUse = classifySymbolUse(laterValueExpr, colName);
  if(replacementUse == SymbolUse::Zero) {
    return std::move(laterValueExpr);
  }

  if(replacementUse == SymbolUse::Many) {
    return std::nullopt;
  }
  bool earlierValueMoved = false;

  Expression foldedValue = substituteAndFold(
    std::move(laterValueExpr), colName, earlierValueExpr, earlierValueMoved);

  // top-level constant fold (inner nodes already folded by substituteAndFoldValue)
  return simplifyConstantFold(std::move(foldedValue));
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
  // Create bucket if this is the first entry for this row.
  // V2J: use try_emplace to avoid find(key) followed by walIndex[key],
  // which performs repeated hash-table lookup work.
  auto [bucketIt, inserted] = walIndex.try_emplace(key);

  RowBucket& bucket = bucketIt->second;
  if(inserted) {
    bucket.tableId = key.first;
    bucket.rowId = key.second;
  }

  if(auto* entry = get_if<ComplexExpression>(&walEntry)) {
    if(entry->getHead() == "Update"_ && entry->getArguments().size() >= 3) {
      // V2Q: decompose the incoming Update expression so we can move its
      // sub-expressions directly into WALEntries instead of cloning them.
      auto [updateHead, updateStatics, updateDynamics, updateSpans] =
        std::move(*entry).decompose();

      if(bucket.idColumnId < 0) {
        captureBucketRowIdentity(bucket, updateDynamics[1]);
      }

      Expression setExpression = std::move(updateDynamics[2]);
      if(auto* setExpr = get_if<ComplexExpression>(&setExpression)) {
        // iterate columns in Set(...) in order — order matters for cross-column dependencies
        // e.g. Set(price(100), total(Times(price, 2))) — price must come before total
        // we assign a fresh seq to each column so they sort correctly at flush time
        auto [setHead, setStatics, setDynamics, setSpans] = std::move(*setExpr).decompose();

        for(size_t ci = 0; ci < setDynamics.size(); ci++) {
          Expression colExpression = std::move(setDynamics[ci]);
          if(auto* colPtr = get_if<ComplexExpression>(&colExpression)) {
            auto [colHead, colStatics, colDynamics, colSpans] = std::move(*colPtr).decompose();

            if(colDynamics.empty()) {
              continue;
            }

            std::string colName = colHead.getName();
            int32_t colId = internColumnName(colName);
            Expression valueExpr = std::move(colDynamics[0]);

            bool blind = isBlindValueWrite(valueExpr);
            auto referencedCols = collectReferencedColumnIds(valueExpr);
            auto referencedCells = collectReferencedCellKeys(valueExpr);

            int seq = globalNextSeq++;

            auto [colPtr2, colInserted] = bucket.columnEntries.try_emplace(colId);
            if(!colPtr2) continue; // too many columns (>N) — should not happen for OLTP
            auto& colEntries = colPtr2->entries;

            if(colInserted) {
              colEntries.reserve(WAL_COLUMN_ENTRY_RESERVE);
            }

            colEntries.push_back(WALEntry{
              std::move(valueExpr),
              seq,
              blind,
              referencedCols,
              referencedCells
            });
            registerReverseDependencies(bucket, colId, seq, referencedCols);
            registerCellReverseDependencies(
              bucket,
              colId,
              seq,
              referencedCells);
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
      if(bucket.idColumnId < 0 && entry->getArguments().size() >= 2) {
        auto const& deleteArgs = entry->getArguments();
        auto const& idArg = deleteArgs[1];
        visitArgumentByReference(idArg, [&](auto const& unwrapped) {
          captureBucketRowIdentity(bucket, unwrapped);
        });
      }
      int seq = globalNextSeq++;
      if(!bucket.deleteEntry.has_value() || seq > bucket.deleteEntry->seq) {
        bucket.deleteEntry = DeleteEntry{seq};
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

// Result of folding one column's entry list.
// foldedValue: the combined expression for the entries that were safely folded
//              (newest entry down to the fold break point, or all entries).
// pendingEntries: entries that could NOT be safely folded in (oldest first).
//   These arise when an expression has the column symbol more than once —
//   folding such an entry would substitute the accumulated expression into
//   multiple positions, causing exponential tree growth.
//   Callers must emit these as separate ordered Updates before the merged one.
struct ColumnFoldResult {
  std::optional<Expression> foldedValue;
  std::vector<WALEntry> pendingEntries;
};

static ColumnFoldResult resolveColumnEntries(
  std::vector<WALEntry> entries,
  RowBucket const& bucket,
  int cutoffSeq,
  int32_t colId,
  std::function<void(int32_t, WALEntry const&)> const* beforeApplyEntry = nullptr)
{
  Symbol const& colSym = columnIdToSymbol[colId];
  std::optional<Expression> resolvedValue;

  for(int idx = static_cast<int>(entries.size()) - 1; idx >= 0; idx--) {
    WALEntry& walEntry = entries[idx];

    if(walEntry.seq > cutoffSeq) continue;

    if(bucket.deleteEntry.has_value() && walEntry.seq <= bucket.deleteEntry->seq) {
      break;
    }

    if(beforeApplyEntry) {
      (*beforeApplyEntry)(colId, walEntry);
    }

    if(!resolvedValue.has_value()) {
      if(walEntry.isBlindWrite) {
        resolvedValue = std::move(walEntry.valueExpr);
        break;
      } else {
        resolvedValue = std::move(walEntry.valueExpr);
      }
    } else {
      // Before folding this (older) entry into the accumulated (newer) value,
      // check if the fold would cause exponential expression growth.
      //
      // foldColumnValues substitutes colSym in resolvedValue with walEntry.valueExpr.
      // If resolvedValue has K occurrences and walEntry has M occurrences, the result
      // has K*M occurrences. Folding is safe only when both K ≤ 1 and M ≤ 1.
      //
      // Blind writes have 0 occurrences of colSym, but the accumulated newer
      // expression can still be a multi-use boundary such as Plus(price, price).
      SymbolUse k = classifySymbolUse(*resolvedValue, colSym);
      SymbolUse m = walEntry.isBlindWrite ? SymbolUse::Zero
                                          : classifySymbolUse(walEntry.valueExpr, colSym);
      if(k == SymbolUse::Many || m == SymbolUse::Many) {
        // Collect entries 0..idx (oldest first) as pending — they were not moved.
        std::vector<WALEntry> pending;
        for(int j = 0; j <= idx; j++) {
          WALEntry& e = entries[j];
          if(e.seq > cutoffSeq) continue;
          if(bucket.deleteEntry.has_value() && e.seq <= bucket.deleteEntry->seq) continue;
          pending.push_back(std::move(e));
        }
        return {std::move(resolvedValue), std::move(pending)};
      }

      if(walEntry.isBlindWrite) {
        auto folded = foldColumnValues(
          colSym,
          std::move(walEntry.valueExpr),
          std::move(*resolvedValue)
        );
        if(!folded.has_value()) return {std::move(resolvedValue), {}};
        resolvedValue = std::move(*folded);
        break;
      } else {
        auto folded = foldColumnValues(
          colSym,
          std::move(walEntry.valueExpr),
          std::move(*resolvedValue)
        );
        if(!folded.has_value()) return {std::move(resolvedValue), {}};
        resolvedValue = std::move(*folded);
      }
    }
  }

  return {std::move(resolvedValue), {}};
}

static ColumnFoldResult resolveColumnEntries(
  std::vector<WALColumnEntry>& columnEntries,
  RowBucket const& bucket,
  int cutoffSeq,
  int32_t colId,
  std::function<void(int32_t, WALEntry const&)> const* beforeApplyEntry = nullptr)
{
  std::vector<WALEntry> entries;
  entries.reserve(columnEntries.size());
  for(auto& entry : columnEntries) {
    if(auto* local = std::get_if<WALEntry>(&entry)) {
      entries.push_back(std::move(*local));
    } else if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
      entries.push_back(consumeColumnEntryRef(*ref));
    }
  }

  return resolveColumnEntries(std::move(entries), bucket, cutoffSeq, colId, beforeApplyEntry);
}

static std::optional<Expression> buildSingleColumnUpdate(RowBucket const& bucket,
                                                         int32_t colId,
                                                         Expression valueExpr) {
  if(bucket.tableId < 0) {
    return std::nullopt;
  }

  boss::ExpressionArguments colArgs;
  colArgs.push_back(std::move(valueExpr));

  boss::ExpressionArguments setArgs;
  setArgs.push_back(
    ComplexExpression(columnIdToSymbol[colId], {}, std::move(colArgs), {}));

  boss::ExpressionArguments updateArgs;
  updateArgs.push_back(buildTableNameExpression(bucket));
  updateArgs.push_back(buildRowIdExpression(bucket));
  updateArgs.push_back(ComplexExpression("Set"_, {}, std::move(setArgs), {}));

  return ComplexExpression("Update"_, {}, std::move(updateArgs), {});
}

static std::vector<Expression> materialiseLocalEntries(RowBucket const& bucket,
                                                       int32_t colId,
                                                       std::vector<WALEntry> entries);

static std::vector<Expression> materialiseLocalRefSegment(RowBucket const& bucket,
                                                          int32_t colId,
                                                          std::vector<WALEntry> entries) {
  std::vector<Expression> result;
  if(entries.empty()) {
    return result;
  }

  auto foldResult = resolveColumnEntries(
    std::move(entries), bucket, std::numeric_limits<int>::max(), colId);

  if(!foldResult.pendingEntries.empty()) {
    auto pendingUpdates = materialiseLocalRefSegment(
      bucket, colId, std::move(foldResult.pendingEntries));
    for(auto& update : pendingUpdates) {
      result.push_back(std::move(update));
    }
  }

  if(foldResult.foldedValue.has_value()) {
    if(auto update = buildSingleColumnUpdate(bucket, colId, std::move(*foldResult.foldedValue))) {
      result.push_back(std::move(*update));
    }
  }

  return result;
}

static std::vector<Expression> materialiseLocalEntries(RowBucket const& bucket,
                                                       int32_t colId,
                                                       std::vector<WALEntry> entries) {
  return materialiseLocalRefSegment(bucket, colId, std::move(entries));
}

static std::vector<Expression> materialiseLocalRefSegment(RowBucket const& bucket,
                                                          int32_t colId,
                                                          std::vector<WALColumnEntry> columnEntries) {
  std::vector<WALEntry> entries;
  entries.reserve(columnEntries.size());
  for(auto& entry : columnEntries) {
    if(auto* local = std::get_if<WALEntry>(&entry)) {
      unregisterCellReverseDependencies(bucket, colId, local->seq, local->referencedCells);
      entries.push_back(std::move(*local));
    } else if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
      auto opIt = walOperations.find(ref->opId);
      if(opIt != walOperations.end() &&
         ref->assignmentIndex < opIt->second.assignments.size()) {
        auto const& assignment = opIt->second.assignments[ref->assignmentIndex];
        unregisterCellReverseDependencies(bucket, colId, ref->seq, assignment.referencedCells);
      }
      entries.push_back(consumeColumnEntryRef(*ref));
    }
  }

  return materialiseLocalRefSegment(bucket, colId, std::move(entries));
}

static void discardLocalRefSegment(RowBucket const& bucket,
                                   int32_t colId,
                                   std::vector<WALColumnEntry> const& entries) {
  for(auto const& entry : entries) {
    if(auto const* local = std::get_if<WALEntry>(&entry)) {
      unregisterCellReverseDependencies(bucket, colId, local->seq, local->referencedCells);
    } else if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
      auto opIt = walOperations.find(ref->opId);
      if(opIt != walOperations.end() &&
         ref->assignmentIndex < opIt->second.assignments.size()) {
        auto const& assignment = opIt->second.assignments[ref->assignmentIndex];
        unregisterCellReverseDependencies(bucket, colId, ref->seq, assignment.referencedCells);
      }
      discardColumnEntryRef(*ref);
    }
  }
}

enum class LocalSegmentMode {
  Emit,
  Discard
};

static std::vector<Expression> materialiseColumnChainUntil(
  WALKey const& key,
  int32_t colId,
  std::optional<OperationId> stopBeforeOp,
  LocalSegmentMode localMode);

static void removeOperationRefsFromTouchedColumns(OperationId opId,
                                                  std::vector<RowID> const& rowIds,
                                                  std::vector<int32_t> const& colIds,
                                                  int32_t tableId) {
  for(auto const& rowId : rowIds) {
    WALKey key{tableId, toInt64(rowId)};
    auto bucketIt = walIndex.find(key);
    if(bucketIt == walIndex.end()) {
      continue;
    }

    for(int32_t colId : colIds) {
      auto* col = bucketIt->second.columnEntries.find(colId);
      if(!col) {
        continue;
      }

      auto& entries = col->entries;
      entries.erase(
        std::remove_if(entries.begin(), entries.end(),
          [&](WALColumnEntry const& entry) {
            auto const* ref = std::get_if<WALEntryRef>(&entry);
            if(!ref || ref->opId != opId) {
              return false;
            }
            auto opIt = walOperations.find(ref->opId);
            if(opIt != walOperations.end() &&
               ref->assignmentIndex < opIt->second.assignments.size()) {
              auto const& assignment = opIt->second.assignments[ref->assignmentIndex];
              unregisterCellReverseDependencies(
                bucketIt->second, colId, ref->seq, assignment.referencedCells);
            }
            return true;
          }),
        entries.end());
    }
  }
}

static std::vector<Expression> materialiseSharedBoundary(OperationId opId) {
  std::vector<Expression> result;
  auto opIt = walOperations.find(opId);
  if(opIt == walOperations.end()) {
    return result;
  }

  int32_t tableId = opIt->second.tableId;
  std::vector<RowID> rowIds = opIt->second.rowIds;
  std::vector<int32_t> colIds;
  std::vector<bool> assignmentBlindness;
  colIds.reserve(opIt->second.assignments.size());
  assignmentBlindness.reserve(opIt->second.assignments.size());
  for(auto const& assignment : opIt->second.assignments) {
    colIds.push_back(assignment.colId);
    assignmentBlindness.push_back(assignment.isBlindWrite);
  }
  size_t liveRefCount = opIt->second.liveRefCount;

  for(auto const& rowId : rowIds) {
    WALKey key{tableId, toInt64(rowId)};
    for(size_t ai = 0; ai < colIds.size(); ++ai) {
      auto prefixMode = assignmentBlindness[ai]
        ? LocalSegmentMode::Discard
        : LocalSegmentMode::Emit;
      auto prefix = materialiseColumnChainUntil(key, colIds[ai], opId, prefixMode);
      for(auto& update : prefix) {
        result.push_back(std::move(update));
      }
    }
  }

  opIt = walOperations.find(opId);
  if(opIt == walOperations.end()) {
    return result;
  }

  if(auto update = buildUpdateFromOperation(opIt->second)) {
    result.push_back(std::move(*update));
  }

  if(walTotalEntries >= liveRefCount) {
    walTotalEntries -= liveRefCount;
  } else {
    walTotalEntries = 0;
  }

  removeOperationRefsFromTouchedColumns(opId, rowIds, colIds, tableId);
  walOperations.erase(opId);

  return result;
}

static std::vector<Expression> materialiseColumnChainUntil(
  WALKey const& key,
  int32_t colId,
  std::optional<OperationId> stopBeforeOp,
  LocalSegmentMode localMode)
{
  std::vector<Expression> result;

  while(true) {
    auto bucketIt = walIndex.find(key);
    if(bucketIt == walIndex.end()) {
      return result;
    }

    RowBucket& bucket = bucketIt->second;
    auto* col = bucket.columnEntries.find(colId);
    if(!col || col->entries.empty()) {
      return result;
    }

    auto& entries = col->entries;
    size_t boundaryIndex = entries.size();
    std::optional<OperationId> boundaryOp;

    for(size_t i = 0; i < entries.size(); ++i) {
      auto const* ref = std::get_if<WALEntryRef>(&entries[i]);
      if(stopBeforeOp.has_value() && ref && ref->opId == *stopBeforeOp) {
        boundaryIndex = i;
        boundaryOp = std::nullopt;
        break;
      }
      if(ref && isSharedOperationRef(*ref)) {
        boundaryIndex = i;
        boundaryOp = ref->opId;
        break;
      }
    }

    std::vector<WALColumnEntry> segment(
      std::make_move_iterator(entries.begin()),
      std::make_move_iterator(entries.begin() + boundaryIndex));
    size_t segmentSize = segment.size();
    if(!segment.empty()) {
      auto effectiveLocalMode = localMode;
      if(boundaryOp.has_value() && isBlindSharedAssignment(*boundaryOp, colId)) {
        effectiveLocalMode = LocalSegmentMode::Discard;
      }

      if(effectiveLocalMode == LocalSegmentMode::Emit) {
        auto localUpdates = materialiseLocalRefSegment(bucket, colId, std::move(segment));
        for(auto& update : localUpdates) {
          result.push_back(std::move(update));
        }
      } else {
        discardLocalRefSegment(bucket, colId, segment);
      }

      if(walTotalEntries >= segmentSize) {
        walTotalEntries -= segmentSize;
      } else {
        walTotalEntries = 0;
      }
      entries.erase(entries.begin(), entries.begin() + boundaryIndex);
    }

    if(stopBeforeOp.has_value()) {
      auto const* frontRef = entries.empty()
        ? nullptr
        : std::get_if<WALEntryRef>(&entries.front());
      if(frontRef && frontRef->opId == *stopBeforeOp) {
        return result;
      }
      continue;
    }

    if(boundaryOp.has_value()) {
      auto sharedUpdates = materialiseSharedBoundary(*boundaryOp);
      for(auto& update : sharedUpdates) {
        result.push_back(std::move(update));
      }
      continue;
    }

    return result;
  }
}

static bool selectedColumnsContainSharedRefs(RowBucket const& bucket,
                                             std::vector<std::string> const& columns) {
  for(auto const& col : bucket.columnEntries) {
    if(!columns.empty()) {
      Symbol const& colSym = columnIdToSymbol[col.colId];
      if(std::find(columns.begin(), columns.end(), colSym.getName()) == columns.end()) {
        continue;
      }
    }

    for(auto const& entry : col.entries) {
      if(isSharedOperationEntry(entry)) {
        return true;
      }
    }
  }
  return false;
}

static bool bucketContainsSharedRefs(RowBucket const& bucket) {
  return selectedColumnsContainSharedRefs(bucket, {});
}

static WALEntry const* findLocalEntry(RowBucket const& bucket, int32_t colId, int seq) {
  auto const* col = bucket.columnEntries.find(colId);
  if(!col) {
    return nullptr;
  }

  for(auto const& entry : col->entries) {
    auto const* local = std::get_if<WALEntry>(&entry);
    if(local && local->seq == seq) {
      return local;
    }
  }
  return nullptr;
}

static WALEntry* findLocalEntry(RowBucket& bucket, int32_t colId, int seq) {
  auto* col = bucket.columnEntries.find(colId);
  if(!col) {
    return nullptr;
  }

  for(auto& entry : col->entries) {
    auto* local = std::get_if<WALEntry>(&entry);
    if(local && local->seq == seq) {
      return local;
    }
  }
  return nullptr;
}

static std::optional<int> latestLocalSeq(RowBucket const& bucket, int32_t colId) {
  auto const* col = bucket.columnEntries.find(colId);
  if(!col) {
    return std::nullopt;
  }

  std::optional<int> latest;
  for(auto const& entry : col->entries) {
    if(auto const* local = std::get_if<WALEntry>(&entry)) {
      if(!latest.has_value() || local->seq > *latest) {
        latest = local->seq;
      }
    }
  }
  return latest;
}

static std::optional<int> latestLocalSeqBefore(RowBucket const& bucket,
                                               int32_t colId,
                                               int beforeSeq) {
  auto const* col = bucket.columnEntries.find(colId);
  if(!col) {
    return std::nullopt;
  }

  std::optional<int> latest;
  for(auto const& entry : col->entries) {
    if(auto const* local = std::get_if<WALEntry>(&entry)) {
      if(local->seq < beforeSeq && (!latest.has_value() || local->seq > *latest)) {
        latest = local->seq;
      }
    }
  }
  return latest;
}

static bool overwritesOwnColumn(WALEntry const& entry, int32_t colId) {
  Symbol const& colSym = columnIdToSymbol[colId];
  return classifySymbolUse(entry.valueExpr, colSym) == SymbolUse::Zero;
}

static std::vector<EntryHandle> collectRelevantEntriesForColumnCutoff(
  RowBucket const& bucket,
  int32_t colId,
  int cutoffSeq)
{
  std::vector<EntryHandle> relevant;
  auto const* col = bucket.columnEntries.find(colId);
  if(!col) {
    return relevant;
  }

  for(auto it = col->entries.rbegin(); it != col->entries.rend(); ++it) {
    auto const* local = std::get_if<WALEntry>(&*it);
    if(!local) {
      continue;
    }

    if(local->seq > cutoffSeq) {
      continue;
    }

    if(bucket.deleteEntry.has_value() && local->seq <= bucket.deleteEntry->seq) {
      break;
    }

    relevant.push_back(EntryHandle{colId, local->seq});

    if(overwritesOwnColumn(*local, colId)) {
      break;
    }
  }

  std::reverse(relevant.begin(), relevant.end());
  return relevant;
}

static bool isRelevantReaderBefore(RowBucket const& bucket,
                                   EntryHandle reader,
                                   int beforeSeq) {
  auto const* col = bucket.columnEntries.find(reader.colId);
  if(!col) {
    return false;
  }

  int cutoffSeq = beforeSeq - 1;
  for(auto it = col->entries.rbegin(); it != col->entries.rend(); ++it) {
    auto const* local = std::get_if<WALEntry>(&*it);
    if(!local) {
      continue;
    }

    if(local->seq > cutoffSeq) {
      continue;
    }

    if(bucket.deleteEntry.has_value() && local->seq <= bucket.deleteEntry->seq) {
      return false;
    }

    if(local->seq == reader.seq) {
      return true;
    }

    if(overwritesOwnColumn(*local, reader.colId)) {
      return false;
    }
  }

  return false;
}

static bool entryHasCrossColumnDependency(WALEntry const& entry, int32_t colId);

static bool relevantColumnPrefixHasCrossColumnDependency(RowBucket const& bucket,
                                                         int32_t colId,
                                                         int cutoffSeq) {
  auto const* col = bucket.columnEntries.find(colId);
  if(!col) {
    return false;
  }

  for(auto it = col->entries.rbegin(); it != col->entries.rend(); ++it) {
    auto const* local = std::get_if<WALEntry>(&*it);
    if(!local) {
      continue;
    }

    if(local->seq > cutoffSeq) {
      continue;
    }

    if(bucket.deleteEntry.has_value() && local->seq <= bucket.deleteEntry->seq) {
      break;
    }

    if(entryHasCrossColumnDependency(*local, colId)) {
      return true;
    }

    if(overwritesOwnColumn(*local, colId)) {
      break;
    }
  }

  return false;
}

static std::vector<EntryHandle> readersOfColumnBefore(RowBucket const& bucket,
                                                       int32_t referencedColId,
                                                       int beforeSeq) {
  std::vector<EntryHandle> readers;
  auto it = std::find_if(
    bucket.reverseDeps.begin(),
    bucket.reverseDeps.end(),
    [&](ReverseDepList const& dep) {
      return dep.referencedColId == referencedColId;
    });
  if(it == bucket.reverseDeps.end()) {
    return readers;
  }

  for(auto const& reader : it->readers) {
    if(reader.seq >= beforeSeq) {
      break;
    }
    if(reader.colId == referencedColId) {
      continue;
    }
    if(findLocalEntry(bucket, reader.colId, reader.seq) &&
       isRelevantReaderBefore(bucket, reader, beforeSeq)) {
      readers.push_back(reader);
    }
  }
  return readers;
}

static bool entryHasCrossColumnDependency(WALEntry const& entry, int32_t colId) {
  return std::any_of(
    entry.referencedCols.begin(),
    entry.referencedCols.end(),
    [&](int32_t referencedColId) {
      return referencedColId != colId;
    });
}

static bool selectionNeedsDependencyPath(RowBucket const& bucket,
                                         std::vector<std::string> const& columns) {
  if(columns.empty() || bucketContainsSharedRefs(bucket)) {
    return false;
  }

  for(auto const& name : columns) {
    auto nameIt = columnNameIntern.find(name);
    if(nameIt == columnNameIntern.end()) {
      continue;
    }

    int32_t colId = nameIt->second;
    auto latestSeq = latestLocalSeq(bucket, colId);
    if(!latestSeq.has_value()) {
      continue;
    }

    if(relevantColumnPrefixHasCrossColumnDependency(bucket, colId, *latestSeq)) {
      return true;
    }

    if(!readersOfColumnBefore(bucket, colId, *latestSeq).empty()) {
      return true;
    }
  }

  return false;
}

struct DependencyFlushContext {
  WALKey key;
  RowBucket& bucket;
  std::vector<Expression> emitted;
  std::vector<int> readyCutoffByCol;
  std::vector<std::pair<int32_t, int>> processing;
  size_t consumedEntries = 0;

  DependencyFlushContext(WALKey key, RowBucket& bucket)
      : key(key), bucket(bucket), readyCutoffByCol(nextColumnId, -1) {}

  void ensureStateSize() {
    if(readyCutoffByCol.size() < static_cast<size_t>(nextColumnId)) {
      readyCutoffByCol.resize(nextColumnId, -1);
    }
  }

  bool isProcessing(int32_t colId, int cutoffSeq) const {
    return std::find(
      processing.begin(),
      processing.end(),
      std::pair<int32_t, int>{colId, cutoffSeq}) != processing.end();
  }

  void beforeApplyEntry(int32_t currentColId, WALEntry const& entry) {
    for(int32_t depColId : entry.referencedCols) {
      if(depColId == currentColId) {
        continue;
      }
      ensureColumnReady(depColId, entry.seq - 1);
    }
  }

  std::vector<WALEntry> consumeRelevantSegment(int32_t colId, int cutoffSeq) {
    std::vector<WALEntry> segment;
    auto* col = bucket.columnEntries.find(colId);
    if(!col) {
      return segment;
    }

    auto& entries = col->entries;
    std::vector<size_t> selectedIndexes;

    for(size_t index = entries.size(); index > 0; --index) {
      size_t i = index - 1;
      auto const* local = std::get_if<WALEntry>(&entries[i]);
      if(!local) {
        continue;
      }

      if(local->seq > cutoffSeq) {
        continue;
      }

      if(bucket.deleteEntry.has_value() && local->seq <= bucket.deleteEntry->seq) {
        break;
      }

      selectedIndexes.push_back(i);

      if(overwritesOwnColumn(*local, colId)) {
        break;
      }
    }

    if(selectedIndexes.empty()) {
      return segment;
    }

    segment.reserve(selectedIndexes.size());

    size_t firstSelected = selectedIndexes.back();
    size_t lastSelectedExclusive = selectedIndexes.front() + 1;
    bool selectedRangeIsContiguous =
      lastSelectedExclusive - firstSelected == selectedIndexes.size();

    if(selectedRangeIsContiguous) {
      for(size_t i = firstSelected; i < lastSelectedExclusive; ++i) {
        auto* local = std::get_if<WALEntry>(&entries[i]);
        if(local) {
          unregisterReverseDependencies(bucket, colId, local->seq, local->referencedCols);
          unregisterCellReverseDependencies(bucket, colId, local->seq, local->referencedCells);
          segment.push_back(std::move(*local));
        }
      }
      entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(firstSelected),
                    entries.begin() + static_cast<std::ptrdiff_t>(lastSelectedExclusive));
    } else {
      for(auto it = selectedIndexes.rbegin(); it != selectedIndexes.rend(); ++it) {
        auto* local = std::get_if<WALEntry>(&entries[*it]);
        if(local) {
          unregisterReverseDependencies(bucket, colId, local->seq, local->referencedCols);
          unregisterCellReverseDependencies(bucket, colId, local->seq, local->referencedCells);
          segment.push_back(std::move(*local));
        }
      }

      for(size_t index : selectedIndexes) {
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(index));
      }
    }

    if(entries.empty()) {
      bucket.columnEntries.erase(colId);
    }

    consumedEntries += segment.size();
    return segment;
  }

  void materialiseSegmentEntries(int32_t colId, std::vector<WALEntry> entries) {
    if(entries.empty()) {
      return;
    }

    std::function<void(int32_t, WALEntry const&)> beforeApply =
      [&](int32_t currentColId, WALEntry const& entry) {
        beforeApplyEntry(currentColId, entry);
      };

    auto foldResult = resolveColumnEntries(
      std::move(entries),
      bucket,
      std::numeric_limits<int>::max(),
      colId,
      &beforeApply);

    if(!foldResult.pendingEntries.empty()) {
      materialiseSegmentEntries(colId, std::move(foldResult.pendingEntries));
    }

    if(foldResult.foldedValue.has_value()) {
      if(auto update = buildSingleColumnUpdate(
           bucket, colId, std::move(*foldResult.foldedValue))) {
        emitted.push_back(std::move(*update));
      }
    }
  }

  void ensureColumnReady(int32_t colId, int cutoffSeq) {
    if(cutoffSeq < 0) {
      return;
    }

    ensureStateSize();
    if(colId < 0 || colId >= static_cast<int32_t>(readyCutoffByCol.size())) {
      return;
    }

    if(readyCutoffByCol[colId] >= cutoffSeq) {
      return;
    }

    if(isProcessing(colId, cutoffSeq)) {
      return;
    }

    processing.push_back({colId, cutoffSeq});

    auto readers = readersOfColumnBefore(bucket, colId, cutoffSeq + 1);
    for(auto const& reader : readers) {
      ensureColumnReady(colId, reader.seq - 1);
      ensureColumnReady(reader.colId, reader.seq);
    }

    if(readyCutoffByCol[colId] < cutoffSeq) {
      auto segment = consumeRelevantSegment(colId, cutoffSeq);
      materialiseSegmentEntries(colId, std::move(segment));
      readyCutoffByCol[colId] = std::max(readyCutoffByCol[colId], cutoffSeq);
    }

    processing.pop_back();
  }
};

static std::vector<Expression> materialiseSameRowDependencySegments(
  WALKey const& key,
  RowBucket& bucket,
  std::vector<std::string> const& requestedColumns,
  size_t& materialisedEntries) {
  DependencyFlushContext context(key, bucket);

  for(auto const& name : requestedColumns) {
    auto nameIt = columnNameIntern.find(name);
    if(nameIt == columnNameIntern.end()) {
      continue;
    }

    auto latestSeq = latestLocalSeq(bucket, nameIt->second);
    if(latestSeq.has_value()) {
      context.ensureColumnReady(nameIt->second, *latestSeq);
    }
  }

  materialisedEntries = context.consumedEntries;
  return std::move(context.emitted);
}

static WALKey rowKeyForCell(CellKey cell);
static RowBucket* findBucketMutable(CellKey cell);
static std::optional<int> latestLocalSeq(CellKey cell);
static std::vector<CellEntryHandle> readersOfCellBefore(CellKey referencedCell,
                                                        int beforeSeq);

struct CellDependencyFlushContext {
  std::vector<Expression> emitted;
  std::vector<std::pair<CellKey, int>> readyCutoffs;
  std::vector<std::pair<CellKey, int>> processing;
  std::vector<WALKey> touchedKeys;
  size_t consumedEntries = 0;

  bool isProcessing(CellKey cell, int cutoffSeq) const {
    return std::find(
      processing.begin(),
      processing.end(),
      std::pair<CellKey, int>{cell, cutoffSeq}) != processing.end();
  }

  bool isReady(CellKey cell, int cutoffSeq) const {
    auto it = std::find_if(
      readyCutoffs.begin(),
      readyCutoffs.end(),
      [&](std::pair<CellKey, int> const& ready) {
        return ready.first == cell;
      });
    return it != readyCutoffs.end() && it->second >= cutoffSeq;
  }

  void markReady(CellKey cell, int cutoffSeq) {
    auto it = std::find_if(
      readyCutoffs.begin(),
      readyCutoffs.end(),
      [&](std::pair<CellKey, int> const& ready) {
        return ready.first == cell;
      });
    if(it == readyCutoffs.end()) {
      readyCutoffs.push_back({cell, cutoffSeq});
    } else {
      it->second = std::max(it->second, cutoffSeq);
    }
  }

  void markTouched(CellKey cell) {
    WALKey key = rowKeyForCell(cell);
    if(std::find(touchedKeys.begin(), touchedKeys.end(), key) == touchedKeys.end()) {
      touchedKeys.push_back(key);
    }
  }

  void beforeApplyEntry(CellKey currentCell, WALEntry const& entry) {
    for(int32_t depColId : entry.referencedCols) {
      if(depColId == currentCell.colId) {
        continue;
      }
      ensureCellReady(CellKey{currentCell.tableId, currentCell.rowId, depColId},
                      entry.seq - 1);
    }

    for(auto const& depCell : entry.referencedCells) {
      if(depCell == currentCell) {
        continue;
      }
      ensureCellReady(depCell, entry.seq - 1);
    }
  }

  std::vector<WALEntry> consumeRelevantCellSegment(CellKey cell, int cutoffSeq) {
    std::vector<WALEntry> segment;
    auto* bucket = findBucketMutable(cell);
    if(!bucket) {
      return segment;
    }

    auto* col = bucket->columnEntries.find(cell.colId);
    if(!col) {
      return segment;
    }

    auto& entries = col->entries;
    std::vector<size_t> selectedIndexes;

    for(size_t index = entries.size(); index > 0; --index) {
      size_t i = index - 1;
      auto const* local = std::get_if<WALEntry>(&entries[i]);
      if(!local) {
        continue;
      }

      if(local->seq > cutoffSeq) {
        continue;
      }

      if(bucket->deleteEntry.has_value() && local->seq <= bucket->deleteEntry->seq) {
        break;
      }

      selectedIndexes.push_back(i);

      if(overwritesOwnColumn(*local, cell.colId)) {
        break;
      }
    }

    if(selectedIndexes.empty()) {
      return segment;
    }

    segment.reserve(selectedIndexes.size());

    size_t firstSelected = selectedIndexes.back();
    size_t lastSelectedExclusive = selectedIndexes.front() + 1;
    bool selectedRangeIsContiguous =
      lastSelectedExclusive - firstSelected == selectedIndexes.size();

    if(selectedRangeIsContiguous) {
      for(size_t i = firstSelected; i < lastSelectedExclusive; ++i) {
        auto* local = std::get_if<WALEntry>(&entries[i]);
        if(local) {
          unregisterReverseDependencies(*bucket, cell.colId, local->seq, local->referencedCols);
          unregisterCellReverseDependencies(cell, local->seq, local->referencedCells);
          segment.push_back(std::move(*local));
        }
      }
      entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(firstSelected),
                    entries.begin() + static_cast<std::ptrdiff_t>(lastSelectedExclusive));
    } else {
      for(auto it = selectedIndexes.rbegin(); it != selectedIndexes.rend(); ++it) {
        auto* local = std::get_if<WALEntry>(&entries[*it]);
        if(local) {
          unregisterReverseDependencies(*bucket, cell.colId, local->seq, local->referencedCols);
          unregisterCellReverseDependencies(cell, local->seq, local->referencedCells);
          segment.push_back(std::move(*local));
        }
      }

      for(size_t index : selectedIndexes) {
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(index));
      }
    }

    if(entries.empty()) {
      bucket->columnEntries.erase(cell.colId);
    }

    consumedEntries += segment.size();
    markTouched(cell);
    return segment;
  }

  void materialiseCellSegmentEntries(CellKey cell, std::vector<WALEntry> entries) {
    if(entries.empty()) {
      return;
    }

    auto* bucket = findBucketMutable(cell);
    if(!bucket) {
      return;
    }

    std::function<void(int32_t, WALEntry const&)> beforeApply =
      [&](int32_t currentColId, WALEntry const& entry) {
        beforeApplyEntry(CellKey{cell.tableId, cell.rowId, currentColId}, entry);
      };

    auto foldResult = resolveColumnEntries(
      std::move(entries),
      *bucket,
      std::numeric_limits<int>::max(),
      cell.colId,
      &beforeApply);

    if(!foldResult.pendingEntries.empty()) {
      materialiseCellSegmentEntries(cell, std::move(foldResult.pendingEntries));
    }

    if(foldResult.foldedValue.has_value()) {
      auto* updateBucket = findBucketMutable(cell);
      if(updateBucket) {
        if(auto update = buildSingleColumnUpdate(
             *updateBucket, cell.colId, std::move(*foldResult.foldedValue))) {
          emitted.push_back(std::move(*update));
        }
      }
    }
  }

  void ensureCellReady(CellKey cell, int cutoffSeq) {
    if(cutoffSeq < 0) {
      return;
    }

    if(isReady(cell, cutoffSeq)) {
      return;
    }

    if(isProcessing(cell, cutoffSeq)) {
      return;
    }

    processing.push_back({cell, cutoffSeq});

    auto readers = readersOfCellBefore(cell, cutoffSeq + 1);
    for(auto const& reader : readers) {
      ensureCellReady(cell, reader.seq - 1);
      ensureCellReady(reader.cell, reader.seq);
    }

    if(!isReady(cell, cutoffSeq)) {
      auto segment = consumeRelevantCellSegment(cell, cutoffSeq);
      materialiseCellSegmentEntries(cell, std::move(segment));
      markReady(cell, cutoffSeq);
    }

    processing.pop_back();
  }

  void cleanupTouchedBuckets() {
    for(auto const& key : touchedKeys) {
      auto bucketIt = walIndex.find(key);
      if(bucketIt != walIndex.end() && !bucketHasPendingWork(bucketIt->second)) {
        walIndex.erase(bucketIt);
      }
    }
  }
};

static std::vector<Expression> materialiseCrossRowDependencySegments(
  RowBucket const& bucket,
  std::vector<std::string> const& requestedColumns,
  size_t& materialisedEntries) {
  CellDependencyFlushContext context;

  for(auto const& name : requestedColumns) {
    auto nameIt = columnNameIntern.find(name);
    if(nameIt == columnNameIntern.end()) {
      continue;
    }

    CellKey cell{bucket.tableId, toInt64(bucket.rowId), nameIt->second};
    auto latestSeq = latestLocalSeq(cell);
    if(latestSeq.has_value()) {
      context.ensureCellReady(cell, *latestSeq);
    }
  }

  context.cleanupTouchedBuckets();
  materialisedEntries = context.consumedEntries;
  return std::move(context.emitted);
}

static WALKey rowKeyForCell(CellKey cell) {
  return WALKey{cell.tableId, cell.rowId};
}

static RowBucket const* findBucket(CellKey cell) {
  auto it = walIndex.find(rowKeyForCell(cell));
  if(it == walIndex.end()) {
    return nullptr;
  }
  return &it->second;
}

static RowBucket* findBucketMutable(CellKey cell) {
  auto it = walIndex.find(rowKeyForCell(cell));
  if(it == walIndex.end()) {
    return nullptr;
  }
  return &it->second;
}

static WALEntry const* findLocalEntry(CellKey cell, int seq) {
  auto const* bucket = findBucket(cell);
  return bucket ? findLocalEntry(*bucket, cell.colId, seq) : nullptr;
}

static std::optional<int> latestLocalSeq(CellKey cell) {
  auto const* bucket = findBucket(cell);
  return bucket ? latestLocalSeq(*bucket, cell.colId) : std::nullopt;
}

static bool isRelevantCellReaderBefore(CellEntryHandle reader, int beforeSeq) {
  auto const* bucket = findBucket(reader.cell);
  if(!bucket) {
    return false;
  }

  auto const* col = bucket->columnEntries.find(reader.cell.colId);
  if(!col) {
    return false;
  }

  int cutoffSeq = beforeSeq - 1;
  for(auto it = col->entries.rbegin(); it != col->entries.rend(); ++it) {
    auto const* local = std::get_if<WALEntry>(&*it);
    if(!local) {
      continue;
    }

    if(local->seq > cutoffSeq) {
      continue;
    }

    if(bucket->deleteEntry.has_value() && local->seq <= bucket->deleteEntry->seq) {
      return false;
    }

    if(local->seq == reader.seq) {
      return true;
    }

    if(overwritesOwnColumn(*local, reader.cell.colId)) {
      return false;
    }
  }

  return false;
}

static std::optional<int> latestBlindOverwriteSeqBefore(CellKey cell, int beforeSeq) {
  auto const* bucket = findBucket(cell);
  if(!bucket) {
    return std::nullopt;
  }

  auto const* col = bucket->columnEntries.find(cell.colId);
  if(!col) {
    return std::nullopt;
  }

  int cutoffSeq = beforeSeq - 1;
  for(auto it = col->entries.rbegin(); it != col->entries.rend(); ++it) {
    auto const* local = std::get_if<WALEntry>(&*it);
    if(!local) {
      continue;
    }
    if(local->seq > cutoffSeq) {
      continue;
    }
    if(bucket->deleteEntry.has_value() && local->seq <= bucket->deleteEntry->seq) {
      break;
    }
    if(overwritesOwnColumn(*local, cell.colId)) {
      return local->seq;
    }
  }

  return std::nullopt;
}

static std::vector<CellEntryHandle> readersOfCellBefore(CellKey referencedCell,
                                                        int beforeSeq) {
  std::vector<CellEntryHandle> readers;
  auto it = reverseDepsByCell.find(referencedCell);
  if(it == reverseDepsByCell.end()) {
    return readers;
  }

  std::vector<std::pair<CellKey, std::optional<int>>> latestBlindByReaderCell;

  for(auto const& reader : it->second) {
    if(reader.seq >= beforeSeq) {
      break;
    }
    if(reader.cell == referencedCell) {
      continue;
    }

    auto blindIt = std::find_if(
      latestBlindByReaderCell.begin(),
      latestBlindByReaderCell.end(),
      [&](std::pair<CellKey, std::optional<int>> const& cached) {
        return cached.first == reader.cell;
      });
    if(blindIt == latestBlindByReaderCell.end()) {
      latestBlindByReaderCell.push_back(
        {reader.cell, latestBlindOverwriteSeqBefore(reader.cell, beforeSeq)});
      blindIt = std::prev(latestBlindByReaderCell.end());
    }

    if(blindIt->second.has_value() && *blindIt->second > reader.seq) {
      continue;
    }

    if(findLocalEntry(reader.cell, reader.seq) &&
       isRelevantCellReaderBefore(reader, beforeSeq)) {
      readers.push_back(reader);
    }
  }
  return readers;
}

static bool relevantCellPrefixHasCrossRowDependency(CellKey cell, int cutoffSeq) {
  auto const bucketIt = walIndex.find(rowKeyForCell(cell));
  if(bucketIt == walIndex.end()) {
    return false;
  }

  auto const& bucket = bucketIt->second;
  auto const* col = bucket.columnEntries.find(cell.colId);
  if(!col) {
    return false;
  }

  for(auto it = col->entries.rbegin(); it != col->entries.rend(); ++it) {
    auto const* local = std::get_if<WALEntry>(&*it);
    if(!local) {
      continue;
    }
    if(local->seq > cutoffSeq) {
      continue;
    }
    if(bucket.deleteEntry.has_value() && local->seq <= bucket.deleteEntry->seq) {
      break;
    }

    if(!local->referencedCells.empty()) {
      return true;
    }
    if(overwritesOwnColumn(*local, cell.colId)) {
      break;
    }
  }

  return false;
}

static bool selectionNeedsCrossRowDependencyPath(RowBucket const& bucket,
                                                 std::vector<std::string> const& columns) {
  if(columns.empty() || bucket.deleteEntry.has_value() || bucketContainsSharedRefs(bucket)) {
    return false;
  }

  for(auto const& name : columns) {
    auto nameIt = columnNameIntern.find(name);
    if(nameIt == columnNameIntern.end()) {
      continue;
    }

    CellKey cell{bucket.tableId, toInt64(bucket.rowId), nameIt->second};
    auto latestSeq = latestLocalSeq(cell);
    if(!latestSeq.has_value()) {
      continue;
    }

    if(relevantCellPrefixHasCrossRowDependency(cell, *latestSeq) ||
       !readersOfCellBefore(cell, *latestSeq).empty()) {
      return true;
    }
  }

  return false;
}

// ============================================================
// ResolvedCol / resolveBucketColumns — read-only column resolution
// ============================================================
//
// Resolves each column's entry list into a single final expression, sorted
// by latestSeq ascending (the order columns were last written — needed so
// cross-column dependencies are emitted correctly: if price was written
// before total, price must appear before total in Set(...)).
//
// V2P: this is the resolution half of what optimiseBucketImpl used to do,
// pulled out so callers that are about to consume the result directly
// (flushBucket) don't have to round-trip it through bucket.columnEntries
// first. Does NOT mutate the bucket — see optimiseBucketImpl below for the
// "resolve and persist back into columnEntries" version.
struct ResolvedCol {
  int32_t colId;
  Symbol columnName;
  Expression valueExpr;
  int latestSeq;
  bool isBlind;
  std::vector<WALEntry> pendingEntries; // entries that couldn't be safely folded, oldest first
};

struct BucketResolution {
  std::vector<ResolvedCol> columns; // sorted by latestSeq ascending
  int maxSeq;
};

static BucketResolution resolveBucketColumns(RowBucket& bucket) {
  // deleteSeq: entries at or before this seq are discarded
  int deleteSeq = bucket.deleteEntry.has_value() ? bucket.deleteEntry->seq : -1;

  // Single pass: compute global maxSeq and per-column colLatestSeq together,
  // avoiding a separate first pass just for maxSeq.
  int maxSeq = deleteSeq;
  int colLatestSeqs[InlineColVec::N];
  std::fill(colLatestSeqs, colLatestSeqs + InlineColVec::N, -1);

  for(int ci = 0; ci < bucket.columnEntries.size; ++ci) {
    for(auto const& e : bucket.columnEntries.data[ci].entries) {
      int seq = getColumnEntrySeq(e);
      if(seq > deleteSeq) {
        if(seq > maxSeq) maxSeq = seq;
        if(seq > colLatestSeqs[ci]) colLatestSeqs[ci] = seq;
      }
    }
  }

  std::vector<ResolvedCol> resolved;

  for(int ci = 0; ci < bucket.columnEntries.size; ++ci) {
    if(colLatestSeqs[ci] == -1) {
      for(auto const& entry : bucket.columnEntries.data[ci].entries) {
        if(auto const* ref = std::get_if<WALEntryRef>(&entry)) {
          discardColumnEntryRef(*ref);
        }
      }
      continue; // all entries before delete, skip
    }

    int32_t colId = bucket.columnEntries.data[ci].colId;
    auto& entries = bucket.columnEntries.data[ci].entries;

    auto foldResult = resolveColumnEntries(entries, bucket, maxSeq, colId);
    if(!foldResult.foldedValue.has_value()) continue;

    Symbol const& colSym = columnIdToSymbol[colId];
    bool blind = isBlindValueWrite(*foldResult.foldedValue);

    resolved.push_back({
      colId,
      colSym,
      std::move(*foldResult.foldedValue),
      colLatestSeqs[ci],
      blind,
      std::move(foldResult.pendingEntries)
    });
  }

  // sort resolved columns by latestSeq ascending
  std::sort(resolved.begin(), resolved.end(),
    [](ResolvedCol const& a, ResolvedCol const& b) {
      return a.latestSeq < b.latestSeq;
    });

  return { std::move(resolved), maxSeq };
}

// ============================================================
// optimiseBucketImpl — compact bucket in place
// ============================================================
//
// Resolves each column's entry list into a single final expression and
// persists that compacted state back into bucket.columnEntries. Handles:
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
//
// Used by the standalone OptimiseWAL command, and by flushBucket's
// column-selective path for the columns that must survive the flush.
// flushBucket's whole-row path does NOT call this — see resolveBucketColumns
// above and the V2P comment in flushBucket for why.
static void optimiseBucketImpl(RowBucket& bucket) {
  auto resolution = resolveBucketColumns(bucket);

  // clear the old column entries
  unregisterAllCellReverseDependencies(bucket);
  bucket.columnEntries.clear();

  // V2L: rebuild compacted bucket as one structured WAL entry per column.
  // The resolved result is already the folded RHS value expression.
  // Do not rebuild price(valueExpr) here.
  for(auto& rc : resolution.columns) {
    auto [colPtr, colInserted] = bucket.columnEntries.try_emplace(rc.colId);
    if(!colPtr) continue; // too many columns — should not happen for OLTP
    auto& colEntries = colPtr->entries;

    if(colInserted) {
      colEntries.reserve(1 + rc.pendingEntries.size());
    }

    // Pending entries (older, could not be safely folded) go back first.
    for(auto& pe : rc.pendingEntries) {
      colEntries.push_back(WALEntry{
        std::move(pe.valueExpr),
        pe.seq,
        pe.isBlindWrite,
        std::move(pe.referencedCols),
        std::move(pe.referencedCells)
      });
    }

    // Keep latestSeq so flushBucket can emit columns in dependency-safe order.
    auto referencedCols = collectReferencedColumnIds(rc.valueExpr);
    auto referencedCells = collectReferencedCellKeys(rc.valueExpr);
    colEntries.push_back(WALEntry{
      std::move(rc.valueExpr),
      rc.latestSeq,
      rc.isBlind,
      std::move(referencedCols),
      std::move(referencedCells)
    });
  }

  rebuildReverseDependencies(bucket);
  registerAllCellReverseDependencies(bucket);
  // deleteEntry is preserved unchanged — caller decides what to emit
}

// ============================================================
// buildUpdateFromResolvedColumns — assemble Update(...) directly from
// already-resolved columns, without touching bucket.columnEntries
// ============================================================
//
// V2P: takes a vector of ResolvedCol (from resolveBucketColumns) instead of
// reading bucket.columnEntries. The caller decides which resolved columns
// to pass in — typically the ones being flushed right now, which would
// otherwise be written into columnEntries only to be read back out once
// here and then discarded/erased.
//
// selected is already sorted by latestSeq ascending (resolveBucketColumns
// guarantees this, and partitioning it preserves relative order), so no
// re-sort is needed here.
static std::optional<Expression> buildUpdateFromResolvedColumns(
  RowBucket const& bucket,
  std::vector<ResolvedCol>& selected)
{
  if(bucket.tableId < 0) {
    return std::nullopt;
  }

  if(selected.empty()) {
    return std::nullopt;
  }

  boss::ExpressionArguments selectedCols;
  selectedCols.reserve(selected.size());

  for(auto& rc : selected) {
    boss::ExpressionArguments colArgs;
    colArgs.push_back(std::move(rc.valueExpr));
    selectedCols.push_back(ComplexExpression(rc.columnName, {}, std::move(colArgs), {}));
  }

  boss::ExpressionArguments updateArgs;
  updateArgs.push_back(buildTableNameExpression(bucket));
  updateArgs.push_back(buildRowIdExpression(bucket));
  updateArgs.push_back(ComplexExpression("Set"_, {}, std::move(selectedCols), {}));

  return ComplexExpression("Update"_, {}, std::move(updateArgs), {});
}

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
  std::vector<Expression> result;

  // If a row-level Delete is pending, a column-selective flush is not safe.
  // A Delete affects the whole row, so force this to become a whole-row flush.
  std::vector<std::string> effectiveColumns = columns;
  if(bucket.deleteEntry.has_value()) {
    effectiveColumns.clear();
  }

  bool flushingAllColumns = effectiveColumns.empty();

  // Count the whole bucket before resolution.
  // This is needed for walTotalEntries accounting because resolveBucketColumns()
  // resolves the entire bucket, even during a selective flush.
  size_t countBeforeAll = 0;
  for(auto const& col : bucket.columnEntries) {
    countBeforeAll += col.entries.size();
  }
  if(bucket.deleteEntry.has_value()) {
    countBeforeAll++;
  }

  // Count only the entries logically being flushed.
  // This is used for instrumentation only.
  size_t countBeforeFlushed = 0;
  if(flushingAllColumns) {
    countBeforeFlushed = countBeforeAll;
  } else {
    for(auto const& colName : effectiveColumns) {
      auto nameIt = columnNameIntern.find(colName);
      if(nameIt == columnNameIntern.end()) continue;
      auto* colPtr = bucket.columnEntries.find(nameIt->second);
      if(colPtr) {
        countBeforeFlushed += colPtr->entries.size();
      }
    }
  }

  recordFlushStart(reason, countBeforeFlushed);

  if(selectedColumnsContainSharedRefs(bucket, effectiveColumns)) {
    std::vector<int32_t> colIdsToFlush;
    if(flushingAllColumns) {
      colIdsToFlush.reserve(bucket.columnEntries.size);
      for(auto const& col : bucket.columnEntries) {
        colIdsToFlush.push_back(col.colId);
      }
    } else {
      for(auto const& colName : effectiveColumns) {
        auto nameIt = columnNameIntern.find(colName);
        if(nameIt == columnNameIntern.end()) continue;
        if(bucket.columnEntries.find(nameIt->second)) {
          colIdsToFlush.push_back(nameIt->second);
        }
      }
    }

    for(int32_t colId : colIdsToFlush) {
      auto columnUpdates = materialiseColumnChainUntil(key, colId, std::nullopt, LocalSegmentMode::Emit);
      for(auto& update : columnUpdates) {
        result.push_back(std::move(update));
      }
    }

    if(bucket.deleteEntry.has_value() && flushingAllColumns) {
      result.push_back(buildDeleteExpression(bucket));
      if(walTotalEntries > 0) {
        walTotalEntries--;
      }
    }

    if(flushingAllColumns || !bucketHasPendingWork(bucket)) {
      walIndex.erase(it);
    }

    recordPhysicalExpressions(result);
    return result;
  }

  if(!flushingAllColumns && selectionNeedsCrossRowDependencyPath(bucket, effectiveColumns)) {
    size_t materialisedEntries = 0;
    auto dependencyUpdates = materialiseCrossRowDependencySegments(
      bucket, effectiveColumns, materialisedEntries);
    recordDependencyMaterialisation(materialisedEntries, dependencyUpdates);
    for(auto& update : dependencyUpdates) {
      result.push_back(std::move(update));
    }

    if(walTotalEntries >= materialisedEntries) {
      walTotalEntries -= materialisedEntries;
    } else {
      walTotalEntries = 0;
    }

    recordPhysicalExpressions(result);
    return result;
  }

  if(!flushingAllColumns && selectionNeedsDependencyPath(bucket, effectiveColumns)) {
    size_t materialisedEntries = 0;
    auto dependencyUpdates = materialiseSameRowDependencySegments(
      key, bucket, effectiveColumns, materialisedEntries);
    recordDependencyMaterialisation(materialisedEntries, dependencyUpdates);
    for(auto& update : dependencyUpdates) {
      result.push_back(std::move(update));
    }

    if(walTotalEntries >= materialisedEntries) {
      walTotalEntries -= materialisedEntries;
    } else {
      walTotalEntries = 0;
    }

    if(!bucketHasPendingWork(bucket)) {
      walIndex.erase(it);
    }

    recordPhysicalExpressions(result);
    return result;
  }

  // V2P: resolve all columns once, without writing the result back into
  // bucket.columnEntries yet. Previously optimiseBucketImpl() wrote every
  // resolved column into the map unconditionally, even though:
  //   - in a whole-row flush, the entire bucket (map included) is destroyed
  //     a few lines later — so the rewrite was read once then thrown away
  //   - in a column-selective flush, the columns being flushed right now
  //     are written into the map only to be read back out and erased again
  // Only columns that need to remain in the bucket afterward are persisted
  // below; columns being flushed now go straight from resolution to output.
  auto resolution = resolveBucketColumns(bucket);

  if(flushingAllColumns) {
    // ── whole-row flush ───────────────────────────────────────────────────
    // emit Delete first if one exists
    if(bucket.deleteEntry.has_value()) {
      result.push_back(buildDeleteExpression(bucket));
    }

    // Emit pending entries (unfoldable due to multi-occurrence expressions) as
    // separate ordered Updates BEFORE the merged one. The in-memory engine will
    // apply them in sequence, then apply the merged folded state last.
    for(auto& rc : resolution.columns) {
      if(!rc.pendingEntries.empty()) {
        auto pendingUpdates = materialiseLocalEntries(
          bucket, rc.colId, std::move(rc.pendingEntries));
        for(auto& update : pendingUpdates) {
          result.push_back(std::move(update));
        }
      }
    }

    // V2P: build the final physical Update directly from the resolved
    // columns — every column is being flushed, so none need to be written
    // into bucket.columnEntries before the bucket is erased below.
    if(auto update = buildUpdateFromResolvedColumns(bucket, resolution.columns)) {
      result.push_back(std::move(*update));
    }

    // whole-row flush — the bucket (and its columnEntries array) is about to
    // be destroyed entirely, so there is nothing left to persist.
    walTotalEntries -= countBeforeAll;
    unregisterAllCellReverseDependencies(bucket);
    walIndex.erase(it);

  } else {
    // ── column-selective flush ────────────────────────────────────────────
    // only emit the requested columns; the remaining columns stay in the
    // bucket for future reads, so only those need to be persisted back.

    // Partition the resolved columns into "being flushed now" (selected)
    // and "must survive in the bucket" (surviving). Both vectors preserve
    // resolveBucketColumns()'s latestSeq-ascending order.
    std::vector<ResolvedCol> selected;
    std::vector<ResolvedCol> surviving;

    for(auto& rc : resolution.columns) {
      bool isSelected = std::find(effectiveColumns.begin(), effectiveColumns.end(),
                                   rc.columnName.getName()) != effectiveColumns.end();
      if(isSelected) {
        selected.push_back(std::move(rc));
      } else {
        surviving.push_back(std::move(rc));
      }
    }

    // Emit pending entries for selected columns before the merged Update.
    for(auto& rc : selected) {
      if(!rc.pendingEntries.empty()) {
        auto pendingUpdates = materialiseLocalEntries(
          bucket, rc.colId, std::move(rc.pendingEntries));
        for(auto& update : pendingUpdates) {
          result.push_back(std::move(update));
        }
      }
    }

    // V2P: build the final physical Update directly from the selected
    // columns, without ever writing them into bucket.columnEntries.
    if(auto update = buildUpdateFromResolvedColumns(bucket, selected)) {
      result.push_back(std::move(*update));
    }

    // Persist only the surviving columns. The old (pre-resolution) entries
    // for every column — including the ones just flushed — are discarded
    // here; the flushed ones don't need to be written back at all.
    unregisterAllCellReverseDependencies(bucket);
    bucket.columnEntries.clear();
    for(auto& rc : surviving) {
      auto [colPtr, colInserted] = bucket.columnEntries.try_emplace(rc.colId);
      if(!colPtr) continue; // too many columns — should not happen for OLTP
      auto& colEntries = colPtr->entries;

      if(colInserted) {
        colEntries.reserve(1 + rc.pendingEntries.size());
      }

      // Pending entries (older, could not be safely folded) go back first.
      for(auto& pe : rc.pendingEntries) {
        colEntries.push_back(WALEntry{
          std::move(pe.valueExpr),
          pe.seq,
          pe.isBlindWrite,
          std::move(pe.referencedCols),
          std::move(pe.referencedCells)
        });
      }

      auto referencedCols = collectReferencedColumnIds(rc.valueExpr);
      auto referencedCells = collectReferencedCellKeys(rc.valueExpr);
      colEntries.push_back(WALEntry{
        std::move(rc.valueExpr),
        rc.latestSeq,
        rc.isBlind,
        std::move(referencedCols),
        std::move(referencedCells)
      });
    }
    rebuildReverseDependencies(bucket);
    registerAllCellReverseDependencies(bucket);

    // deleteEntry is guaranteed absent here (forced to flushingAllColumns
    // above otherwise). Count pending entries for surviving columns too since
    // they are written back into the bucket.
    size_t countAfterAll = 0;
    for(auto const& rc : surviving) {
      countAfterAll += 1 + rc.pendingEntries.size();
    }

    walTotalEntries -= countBeforeAll;
    walTotalEntries += countAfterAll;

    // if all columns have been flushed, remove the bucket entirely
    // otherwise leave it alive for the remaining columns
    if(bucket.columnEntries.empty() && !bucket.deleteEntry.has_value()) {
      walIndex.erase(it);
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
    auto const& args = expr.getArguments();
    if(args.size() < 2) return targets;

    // arg 0 must be a Select expression
    auto const& innerArg = args[0];
    auto const* innerSelect = get_if<ComplexExpression>(&innerArg);
    if(!innerSelect || innerSelect->getHead() != "Select"_) return targets;

    // arg 1 must be As(...)
    auto const& asArg = args[1];
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
    auto const& asArgs = asExpr->getArguments();
    for(size_t i = 1; i < asArgs.size(); i += 2) {
      auto const& colArg = asArgs[i];
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
  auto const& args = expr.getArguments();
  if(args.size() < 1) return targets;
  auto const& arg0 = args[0];
  if(auto const* tableSymbol = get_if<boss::Symbol>(&arg0)) {
    std::string tableName = tableSymbol->getName();
    auto tableIt = tableNameIntern.find(tableName);
    if(tableIt == tableNameIntern.end()) return targets; // table not in WAL

    if(args.size() < 2) return targets;
    auto const& arg1 = args[1];
    auto const* whereExpr = get_if<ComplexExpression>(&arg1);
    if(!whereExpr || whereExpr->getHead() != "Where"_) return targets;

    auto const& whereArgs = whereExpr->getArguments();
    if(whereArgs.size() < 1) return targets;
    auto const& condArg = whereArgs[0];
    auto const* condExpr = get_if<ComplexExpression>(&condArg);
    if(!condExpr || condExpr->getHead() != "Equal"_) return targets;

    auto const& condArgs = condExpr->getArguments();
    if(condArgs.size() < 2) return targets;
    auto const& colArg = condArgs[0];
    auto const& valArg = condArgs[1];

    auto const* colSymbol = get_if<boss::Symbol>(&colArg);
    if(!colSymbol || colSymbol->getName() != "id") return targets;

    int32_t tableId = tableIt->second;
    if(auto const* id32 = get_if<int32_t>(&valArg)) {
      targets.push_back({{tableId, static_cast<int64_t>(*id32)}, {}});
      return targets;
    }
    if(auto const* id64 = get_if<int64_t>(&valArg)) {
      targets.push_back({{tableId, *id64}, {}});
      return targets;
    }
    return targets;
  }

  // ── Format 2: Select(Table(Customer, id(List(5, 6))), Where(...)) ─────────
  //
  // arg 0 is a ComplexExpression with head Table
  auto const* tableExpr = get_if<ComplexExpression>(&arg0);
  if(!tableExpr) return targets;

  auto const& tableArgs = tableExpr->getArguments();
  if(tableArgs.size() < 1) return targets;
  auto const& nameArg = tableArgs[0];
  auto const* nameSymbol = get_if<boss::Symbol>(&nameArg);
  if(!nameSymbol) return targets;
  std::string tableName = nameSymbol->getName();
  auto tableIt2 = tableNameIntern.find(tableName);
  if(tableIt2 == tableNameIntern.end()) return targets; // table not in WAL
  int32_t tableId2 = tableIt2->second;

  for(size_t i = 1; i < tableArgs.size(); i++) {
    auto const& colArg = tableArgs[i];
    auto const* colExpr = get_if<ComplexExpression>(&colArg);
    if(!colExpr) continue;

    auto const& colArgs = colExpr->getArguments();
    if(colArgs.size() < 1) continue;
    auto const& listArg = colArgs[0];
    auto const* listExpr = get_if<ComplexExpression>(&listArg);
    if(!listExpr) continue;

    visitRowIDs(*listExpr, [&](auto idValue) {
      targets.push_back({{tableId2, static_cast<int64_t>(idValue)}, {}});
    });

    if(!targets.empty()) break;
  }

  return targets;
}

static Expression flushAllBuckets(FlushReason reason = FlushReason::Manual) {
  // commented out for better testing output
  // std::cout << "WAL: flushing all " << walTotalEntries << " entries" << std::endl;
  boss::ExpressionArguments entries;
  // Copy keys because flushBucket modifies walIndex (erases buckets)
  std::vector<WALKey> keys;
  keys.reserve(walIndex.size());
  for(auto const& [k, _] : walIndex) keys.push_back(k);
  for(auto const& key : keys) {
    auto flushed = flushBucket(key, {}, reason);
    for(auto& e : flushed) entries.push_back(std::move(e));
  }
  // commented out for better testing output
  // std::cout << "WAL: emitting " << entries.size() << " entries" << std::endl;
  return ComplexExpression("ApplyWAL"_, {}, std::move(entries), {});
}

static bool captureMultiRowUpdateOperation(
  int32_t tableId,
  int32_t idColumnId,
  std::vector<RowID> rowIds,
  Expression setExpression,
  boss::ExpressionArguments& prefixFlushes)
{
  if(rowIds.size() <= 1) {
    return false;
  }

  WALOperation operation{
    nextOperationId++,
    tableId,
    idColumnId,
    std::move(rowIds),
    {},
    0,
    0
  };

  if(!captureAssignmentsFromSet(std::move(setExpression), operation)) {
    return false;
  }

  OperationId opId = operation.opId;
  size_t assignmentCount = operation.assignments.size();
  operation.liveRefCount = static_cast<uint32_t>(operation.rowIds.size() * assignmentCount);
  std::vector<int> assignmentSeqs;
  assignmentSeqs.reserve(assignmentCount);
  for(size_t assignmentIndex = 0; assignmentIndex < assignmentCount; ++assignmentIndex) {
    assignmentSeqs.push_back(globalNextSeq++);
  }
  operation.seqBase = assignmentSeqs.empty() ? 0 : assignmentSeqs.front();

  for(auto const& rowId : operation.rowIds) {
    WALKey key{tableId, toInt64(rowId)};
    auto [bucketIt, inserted] = walIndex.try_emplace(key);
    RowBucket& bucket = bucketIt->second;
    if(inserted) {
      bucket.tableId = key.first;
      bucket.rowId = rowId;
    }
    bucket.idColumnId = idColumnId;

    for(uint32_t assignmentIndex = 0;
        assignmentIndex < static_cast<uint32_t>(operation.assignments.size());
        ++assignmentIndex) {
      int seq = assignmentSeqs[assignmentIndex];
      int32_t colId = operation.assignments[assignmentIndex].colId;
      auto [colPtr, colInserted] = bucket.columnEntries.try_emplace(colId);
      if(!colPtr) continue;
      if(colInserted) {
        colPtr->entries.reserve(WAL_COLUMN_ENTRY_RESERVE);
      }
      colPtr->entries.push_back(WALEntryRef{opId, assignmentIndex, seq});
      registerReverseDependencies(
        bucket,
        colId,
        seq,
        operation.assignments[assignmentIndex].referencedCols);
      registerCellReverseDependencies(
        bucket,
        colId,
        seq,
        operation.assignments[assignmentIndex].referencedCells);
    }
  }

  walTotalEntries += operation.rowIds.size() * assignmentCount;
  #if BOSS_WAL_INSTRUMENTATION
  walStats.walEntriesCreated += operation.rowIds.size() * assignmentCount;
  #endif

  walOperations.emplace(opId, std::move(operation));
  return true;
}

static Expression evaluate(Expression &&e) {
  return std::visit(
    [](auto &&expr) -> Expression {
      if constexpr(std::is_same_v<std::decay_t<decltype(expr)>, ComplexExpression>) {
        auto head = expr.getHead();
        auto const& args = expr.getArguments();

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

          auto const& setArgRef = args[2];
          bool setArgIsValid = visitArgumentByReference(setArgRef, [](auto const& unwrapped) {
            auto const* setExpr = asComplexExpressionPtr(unwrapped);
            return setExpr && setExpr->getHead() == "Set"_;
          });
          if(!setArgIsValid) return std::move(expr);

          std::vector<RowID> rowIds;
          visitRowIDs(*idListExpr, [&](auto idValue) { rowIds.push_back(idValue); });
          if(rowIds.empty()) return "Update_Logged"_();

          int32_t tableId = internTableName(tableSymbol->getName());
          int32_t idColumnId = internColumnName(idColName.getName());
          Symbol tableNameSymbol = *tableSymbol;
          Symbol idColNameSymbol = idColName;

          auto [updateHead, updateStatics, updateDynamics, updateSpans] =
            std::move(expr).decompose();
          Expression setArg = std::move(updateDynamics[2]);

          if(rowIds.size() > 1) {
            boss::ExpressionArguments prefixFlushes;
            if(!captureMultiRowUpdateOperation(
                 tableId, idColumnId, std::move(rowIds), std::move(setArg), prefixFlushes)) {
              return "Update_Ignored"_();
            }

            if(walTotalEntries >= WAL_THRESHOLD && prefixFlushes.empty()) {
              return flushAllBuckets(FlushReason::ChainThreshold);
            }

            if(!prefixFlushes.empty()) {
              return ComplexExpression("ApplyWAL"_, {}, std::move(prefixFlushes), {});
            }

            return "Update_Logged"_();
          }

          WALKey key{tableId, toInt64(rowIds[0])};
          {
            boss::ExpressionArguments walArgs;
            walArgs.push_back(tableNameSymbol);
            boss::ExpressionArguments idListArgs;
            std::visit([&](auto id) { idListArgs.push_back(id); }, rowIds[0]);
            auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});
            boss::ExpressionArguments idColArgs;
            idColArgs.push_back(std::move(idList));
            walArgs.push_back(ComplexExpression(idColNameSymbol, {}, std::move(idColArgs), {}));
            walArgs.push_back(std::move(setArg));
            walIndexPush(key, ComplexExpression("Update"_, {}, std::move(walArgs), {}));
          }

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

          auto const& tableArg = args[0];
          auto const* tableSymbol = get_if<Symbol>(&tableArg);
          if(!tableSymbol) return std::move(expr);

          auto const& tableExprArg = args[1];
          auto const* tableExpr = get_if<ComplexExpression>(&tableExprArg);
          if(!tableExpr) return std::move(expr);

          // find "id"_ column inside "Table"_
          // which is the first column
          ComplexExpression const* idListExpr = nullptr;
          auto const& tableExprArgs = tableExpr->getArguments();
          auto const& firstColArg = tableExprArgs[0];
          auto const* firstColExpr = get_if<ComplexExpression>(&firstColArg);
          if(!firstColExpr) return std::move(expr);

          // use its actual head name for the WAL entry
          auto idColName = firstColExpr->getHead();
          auto const& firstColArgs = firstColExpr->getArguments();
          auto const& listArg = firstColArgs[0];
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
            WALKey key{internTableName(tableSymbol->getName()), static_cast<int64_t>(idValue)};
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

        // GetWAL — return non-consuming metadata summaries flattened in seq order
        if (head == "GetWAL"_) {
          boss::ExpressionArguments entries;
          std::vector<OperationId> seenOperations;
          for(auto const& [key, bucket] : walIndex) {

            // collect all entries across columns with their seq numbers
            std::vector<std::pair<int, Expression>> allEntries;
            for(auto const& col : bucket.columnEntries) {
              for(auto const& e : col.entries) {
                if(auto const* local = std::get_if<WALEntry>(&e)) {
                  allEntries.push_back({
                    local->seq,
                    buildLocalDebugSummary(bucket, col.colId, *local)
                  });
                  continue;
                }

                auto const* ref = std::get_if<WALEntryRef>(&e);
                if(!ref) {
                  continue;
                }

                if(isSharedOperationRef(*ref)) {
                  if(std::find(seenOperations.begin(), seenOperations.end(), ref->opId)
                       != seenOperations.end()) {
                    continue;
                  }
                  auto opIt = walOperations.find(ref->opId);
                  if(opIt == walOperations.end()) {
                    continue;
                  }
                  seenOperations.push_back(ref->opId);
                  allEntries.push_back({ref->seq, buildOperationDebugSummary(opIt->second)});
                  continue;
                }
                allEntries.push_back({ref->seq, buildRefDebugSummary(bucket, col.colId, *ref)});
              }
            }
            if(bucket.deleteEntry.has_value())
              allEntries.push_back({
                bucket.deleteEntry->seq,
                buildDeleteDebugSummary(bucket, bucket.deleteEntry->seq)
              });

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

          if(!args.empty()) {
            auto const& labelArg = args[0];

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
          walOperations.clear();
          reverseDepsByCell.clear();
          nextOperationId = 0;
          globalNextSeq = 0;
          pendingCrossRowRefEntries = 0;
          walTotalEntries = 0;
          tableNameIntern.clear();
          tableIdToSymbol.clear();
          columnNameIntern.clear();
          columnIdToSymbol.clear();
          nextTableId = 0;
          nextColumnId = 0;
          return "WAL_Cleared"_();
        }
        // OptimiseWAL — compact every bucket in place, WAL stays alive
        if(head == "OptimiseWAL"_) {
          // commented out for better testing output
          // std::cout << "WAL: optimising " << walTotalEntries << " entries" << std::endl;
          for(auto& [key, bucket] : walIndex) {
            if(selectedColumnsContainSharedRefs(bucket, {})) {
              continue;
            }

            // count entries before optimise
            size_t countBefore = 0;
            for(auto const& col : bucket.columnEntries) countBefore += col.entries.size();
            if(bucket.deleteEntry.has_value()) countBefore++;

            optimiseBucketImpl(bucket);

            // count entries after optimise
            size_t countAfter = 0;
            for(auto const& col : bucket.columnEntries) countAfter += col.entries.size();
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
            std::vector<WALKey> keysCopy;
            keysCopy.reserve(walIndex.size());
            for(auto const& [k, _] : walIndex) keysCopy.push_back(k);
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
            std::vector<WALKey> keysCopy;
            keysCopy.reserve(walIndex.size());
            for(auto const& [k, _] : walIndex) keysCopy.push_back(k);
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
          if(args.size() == 0) {
            return flushAllBuckets(FlushReason::Manual);
          }

          // FlushWAL(tableName, rowId) — flush only one specific row
          if(args.size() == 2) {
            auto const& tableArg = args[0];
            auto const* tableSymbol = get_if<Symbol>(&tableArg);
            if(!tableSymbol) return flushAllBuckets(FlushReason::Manual); // fallback

            auto const& rowArg = args[1];
            int64_t rowId = 0;
            if(auto const* id32 = get_if<int32_t>(&rowArg)) rowId = *id32;
            else if(auto const* id64 = get_if<int64_t>(&rowArg)) rowId = *id64;
            else return flushAllBuckets(FlushReason::Manual); // fallback

            WALKey key{internTableName(tableSymbol->getName()), rowId};
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
