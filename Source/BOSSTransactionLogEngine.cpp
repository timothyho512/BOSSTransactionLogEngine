#include <BOSS.hpp>
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <Utilities.hpp>
#include <iostream>
#include <vector>

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
// Instead of a flat vector, the WAL is indexed by (tableName, rowID).
// Each row gets its own RowBucket containing:
//   - chain:       raw WAL entries in arrival order (never evaluated at capture time)
//   - columnChain: maps each column name to the list of positions in chain
//                  that write to that column
//
// columnChain is pure bookkeeping — just integers, no expression evaluation.
// At flush time, for each column we walk its position list backwards,
// applying the three optimisation rules only over that column's entries.
// Delete entries sit in the chain like any other entry — the backwards
// walk stops naturally when it hits one.
//
// This eliminates the O(k^2 * c^2) flush cost of the old flat WAL.
// Capture is O(c) per write. Flush is O(k_col) per column per row.
 
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
 
struct RowBucket {
  std::vector<Expression> chain;                                  // raw entries, arrival order
  std::unordered_map<std::string, std::vector<int>> columnChain; // column → positions in chain
  // Note: Delete entries are stored in chain like any other entry.
  // They are NOT tracked in columnChain — the backwards walk in
  // flushBucket() stops when it hits a Delete naturally.
};
 
// the WAL index: (tableName, rowID) → bucket
static std::unordered_map<WALKey, RowBucket, WALKeyHash> walIndex;
 
// insertion order of keys — so flush emits entries in the order rows were first touched
static std::vector<WALKey> walOrder;
 
// total number of entries across all buckets — for threshold check
static size_t walTotalEntries = 0;
 
// Threshold for WAL flush
static const size_t WAL_THRESHOLD = 10;

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

// Check if a value expression contains any Symbol references
// A Symbol in a value expression means it reads a column — making it a dependent write
// e.g. Plus(price, 1) contains Symbol "price" → dependent write
// e.g. 100.0 contains no Symbols → blind write
static bool containsSymbol(Expression const& expr) {
  if(get_if<Symbol>(&expr)) return true;
  if(auto const* complex = get_if<ComplexExpression>(&expr)) {
    for(size_t i = 0; i < complex->getArguments().size(); i++) {
      if(containsSymbol(complex->cloneArgument(i))) return true;
    }
  }
  return false;
}

// Check if a column assignment in Set(...) is a blind write
// e.g. price(100.0) → blind write, price(Plus(price, 1)) → dependent write
static bool isBlindWrite(ComplexExpression const& colAssign) {
  if(colAssign.getArguments().empty()) return true;
  return !containsSymbol(colAssign.cloneArgument(0));
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
  if(outer->getArguments().size() < 2) return std::nullopt;

  auto const& outerHead = outer->getHead();

  // only handle our four arithmetic operators
  if(outerHead != "Plus"_  && outerHead != "Minus"_ &&
     outerHead != "Times"_ && outerHead != "Divide"_) return std::nullopt;

  // outer's first arg must be a ComplexExpression with the SAME operator
  auto outerArg0 = outer->cloneArgument(0);
  auto const* inner = get_if<ComplexExpression>(&outerArg0);
  if(!inner) return std::nullopt;
  if(inner->getHead() != outerHead) return std::nullopt;
  if(inner->getArguments().size() < 2) return std::nullopt;

  // inner's second arg must be a concrete number (c1)
  auto c1Expr = inner->cloneArgument(1);
  auto c1 = toDouble(c1Expr);
  if(!c1) return std::nullopt;

  // outer's second arg must be a concrete number (c2)
  auto c2Expr = outer->cloneArgument(1);
  auto c2 = toDouble(c2Expr);
  if(!c2) return std::nullopt;

  // compute the folded constant based on operator
  double folded;
  if(outerHead == "Plus"_)        folded = *c1 + *c2;
  else if(outerHead == "Minus"_)  folded = *c1 + *c2; // Minus(Minus(x,a),b) = Minus(x, a+b)
  else if(outerHead == "Times"_)  folded = *c1 * *c2;
  else if(outerHead == "Divide"_) folded = *c1 * *c2; // Divide(Divide(x,a),b) = Divide(x, a*b)
  else return std::nullopt;

  // rebuild: outerHead(inner->arg0, folded)
  boss::ExpressionArguments newArgs;
  newArgs.push_back(inner->cloneArgument(0)); // the Symbol or deeper expression
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
  auto laterValue = later.cloneArgument(0); // e.g. Plus(price, 1)
  auto earlierValue = earlier.cloneArgument(0); // e.g. Plus(price, 1)

  // substitute colName Symbol in laterValue with earlierValue
  std::function<Expression(Expression)> substitute = [&](Expression expr) -> Expression {
    if(auto const* sym = get_if<Symbol>(&expr)) {
      if(*sym == colName) return earlierValue.clone();
      return expr;
    }
    if(auto const* complex = get_if<ComplexExpression>(&expr)) {
      boss::ExpressionArguments newArgs;
      for(size_t i = 0; i < complex->getArguments().size(); i++) {
        newArgs.push_back(substitute(complex->cloneArgument(i)));
      }
      return ComplexExpression(complex->getHead(), {}, std::move(newArgs), {});
    }
    return expr;
  };

  auto foldedValue = substitute(std::move(laterValue));

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
// WAL Index Operations
// ============================================================
 
// Push one raw WAL entry into the right bucket and update columnChain.
// For Update: register each column in Set(...) in columnChain.
// For Delete: push to chain only — no columnChain entry.
//             The backwards walk in flushBucket stops when it hits a Delete.
static void walIndexPush(WALKey const& key, Expression walEntry) {
  // create bucket if first time seeing this row
  if(walIndex.find(key) == walIndex.end()) {
    walIndex[key] = RowBucket{};
    walOrder.push_back(key);
  }
 
  RowBucket& bucket = walIndex[key];
  int pos = static_cast<int>(bucket.chain.size());
 
  // for Update: register each column in columnChain
  if(auto const* entry = get_if<ComplexExpression>(&walEntry)) {
    if(entry->getHead() == "Update"_ && entry->getArguments().size() >= 3) {
      auto setArg = entry->getArguments()[2];
      if(auto const* setExpr = get_if<ComplexExpression>(&setArg)) {
        for(size_t ci = 0; ci < setExpr->getArguments().size(); ci++) {
          auto colArg = setExpr->getArguments()[ci];
          if(auto const* colExpr = get_if<ComplexExpression>(&colArg)) {
            std::string colName = colExpr->getHead().getName();
            bucket.columnChain[colName].push_back(pos);
          }
        }
      }
    }
    // Delete: just pushed to chain, no columnChain entry needed
  }
 
  bucket.chain.push_back(std::move(walEntry));
  walTotalEntries++;
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

static void optimiseBucketImpl(RowBucket& bucket) {
  // find the latest Delete — walk backwards
  int deletePos = -1;
  for(int i = static_cast<int>(bucket.chain.size()) - 1; i >= 0; i--) {
    if(auto const* entry = get_if<ComplexExpression>(&bucket.chain[i])) {
      if(entry->getHead() == "Delete"_) {
        deletePos = i;
        break;
      }
    }
  }
 
  // save the Delete expression NOW before we clear the chain
  std::optional<Expression> savedDelete;
  if(deletePos >= 0)
    savedDelete = bucket.chain[deletePos].clone();
 
  // resolve each column
  boss::ExpressionArguments resolvedCols;
  ComplexExpression const* representativeEntry = nullptr;
 
  for(auto const& [colName, positions] : bucket.columnChain) {
    std::optional<Expression> resolvedValue;
 
    for(int idx = static_cast<int>(positions.size()) - 1; idx >= 0; idx--) {
      int pos = positions[idx];
      if(pos <= deletePos) break; // Rule 2: before Delete, discard
 
      auto const* entry = get_if<ComplexExpression>(&bucket.chain[pos]);
      if(!entry || entry->getArguments().size() < 3) continue;
 
      auto setArg = entry->getArguments()[2];
      auto const* setExpr = get_if<ComplexExpression>(&setArg);
      if(!setExpr) continue;
 
      ComplexExpression const* colAssign = nullptr;
      for(size_t ci = 0; ci < setExpr->getArguments().size(); ci++) {
        auto colArg = setExpr->getArguments()[ci];
        if(auto const* colExpr = get_if<ComplexExpression>(&colArg)) {
          if(colExpr->getHead().getName() == colName) { colAssign = colExpr; break; }
        }
      }
      if(!colAssign) continue;
 
      if(representativeEntry == nullptr) representativeEntry = entry;
 
      if(!resolvedValue.has_value()) {
        if(isBlindWrite(*colAssign)) {
          resolvedValue = colAssign->clone();
          break; // Rule 1a: blind write wins, stop
        } else {
          resolvedValue = colAssign->clone(); // Rule 1b: dependent, keep walking
        }
      } else {
        auto const* resolvedExpr = get_if<ComplexExpression>(&*resolvedValue);
        if(!resolvedExpr) break;
        if(isBlindWrite(*colAssign)) {
          resolvedValue = foldColumnWrites(*colAssign, *resolvedExpr);
          break; // blind base found, stop
        } else {
          resolvedValue = foldColumnWrites(*colAssign, *resolvedExpr); // keep walking
        }
      }
    }
 
    if(resolvedValue.has_value())
      resolvedCols.push_back(std::move(*resolvedValue));
  }

  // save these BEFORE clearing the chain
  std::optional<Expression> savedTableName;
  std::optional<Expression> savedIdExpr;
  if(representativeEntry != nullptr) {
    savedTableName = representativeEntry->cloneArgument(0);
    savedIdExpr    = representativeEntry->cloneArgument(1);
  }
 
  // now clear the bucket — old chain and columnChain are no longer needed
  bucket.chain.clear();
  bucket.columnChain.clear();

  // re-add the saved Delete at the end of the compacted chain
  if(savedDelete.has_value())
    bucket.chain.push_back(std::move(*savedDelete));
 
  // rebuild: one merged Update if any columns survived
  if(!resolvedCols.empty() && savedTableName.has_value()) {
    boss::ExpressionArguments updateArgs;
    updateArgs.push_back(std::move(*savedTableName)); // table name
    updateArgs.push_back(std::move(*savedIdExpr)); // id(List(...))
    updateArgs.push_back(ComplexExpression("Set"_, {}, std::move(resolvedCols), {}));
    auto mergedUpdate = ComplexExpression("Update"_, {}, std::move(updateArgs), {});
 
    // re-register columns in columnChain at position 0
    auto setArg = mergedUpdate.getArguments()[2];
    if(auto const* setExpr = get_if<ComplexExpression>(&setArg)) {
      for(size_t ci = 0; ci < setExpr->getArguments().size(); ci++) {
        auto colArg = setExpr->getArguments()[ci];
        if(auto const* colExpr = get_if<ComplexExpression>(&colArg))
          bucket.columnChain[colExpr->getHead().getName()].push_back(0);
      }
    }
    bucket.chain.push_back(std::move(mergedUpdate));
  }
}

// ============================================================
// flushBucket — optimise bucket, collect entries, clear bucket
// ============================================================
//
// Calls optimiseBucketImpl to compact the bucket in place.
// Then collects all entries from the compacted chain.
// Then clears the bucket entirely (chain + columnChain).
// Also removes the key from walOrder and decrements walTotalEntries.
//
// Returns the collected expressions — ready to put into ApplyWAL.
 
static std::vector<Expression> flushBucket(WALKey const& key) {
  auto it = walIndex.find(key);
  if(it == walIndex.end()) return {}; // no bucket for this key
 
  RowBucket& bucket = it->second;
 
  // optimise first — compacts chain and columnChain in place
  optimiseBucketImpl(bucket);
 
  // collect the compacted entries
  std::vector<Expression> result;
  for(auto& entry : bucket.chain)
    result.push_back(std::move(entry));
 
  // subtract this bucket's entry count from the global total
  walTotalEntries -= result.size();
 
  // clear the bucket and remove from index
  walIndex.erase(it);
 
  // remove key from walOrder
  walOrder.erase(
    std::remove(walOrder.begin(), walOrder.end(), key),
    walOrder.end()
  );
 
  return result;
}

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

static std::vector<WALKey> extractSelectKeys(ComplexExpression const& selectExpr) {
  std::vector<WALKey> keys;

  if(selectExpr.getArguments().size() < 1) return keys;

  // ── Format 1: Select(Customer, Where(Equal(id, 5))) ──────────────────────
  //
  // arg 0 is a plain Symbol — the table name
  // arg 1 is Where(...) containing the condition
  auto arg0 = selectExpr.getArguments()[0];
  if(auto const* tableSymbol = get_if<boss::Symbol>(&arg0)) {
    std::string tableName = tableSymbol->getName();

    // look for Where(...) in arg 1
    if(selectExpr.getArguments().size() < 2) return keys;
    auto arg1 = selectExpr.getArguments()[1];
    auto const* whereExpr = get_if<ComplexExpression>(&arg1);
    if(!whereExpr || whereExpr->getHead() != "Where"_) return keys;

    // inside Where(...) is the condition — e.g. Equal(id, 5)
    if(whereExpr->getArguments().size() < 1) return keys;
    auto condArg = whereExpr->getArguments()[0];
    auto const* condExpr = get_if<ComplexExpression>(&condArg);
    if(!condExpr || condExpr->getHead() != "Equal"_) return keys;

    // Equal(id, 5) — arg 0 is the column Symbol, arg 1 is the value
    if(condExpr->getArguments().size() < 2) return keys;
    auto colArg = condExpr->getArguments()[0];
    auto valArg = condExpr->getArguments()[1];

    // only handle equality on the id column — anything else falls back to full flush
    auto const* colSymbol = get_if<boss::Symbol>(&colArg);
    if(!colSymbol || colSymbol->getName() != "id") return keys;

    // extract the row ID — handle both int32 and int64
    if(auto const* id32 = get_if<int32_t>(&valArg)) {
      keys.push_back({tableName, static_cast<int64_t>(*id32)});
      return keys;
    }
    if(auto const* id64 = get_if<int64_t>(&valArg)) {
      keys.push_back({tableName, *id64});
      return keys;
    }

    // value was not an integer — fall back to full flush
    return keys;
  }

  // ── Format 2: Select(Table(Customer, id(List(5, 6))), Where(...)) ─────────
  //
  // arg 0 is a ComplexExpression with head Table
  auto const* tableExpr = get_if<ComplexExpression>(&arg0);
  if(!tableExpr) return keys;

  // arg 0 of Table is the table name Symbol
  if(tableExpr->getArguments().size() < 1) return keys;
  auto nameArg = tableExpr->getArguments()[0];
  auto const* nameSymbol = get_if<boss::Symbol>(&nameArg);
  if(!nameSymbol) return keys;
  std::string tableName = nameSymbol->getName();

  // remaining args of Table are columns — find the id column (contains a List)
  for(size_t i = 1; i < tableExpr->getArguments().size(); i++) {
    auto colArg = tableExpr->getArguments()[i];
    auto const* colExpr = get_if<ComplexExpression>(&colArg);
    if(!colExpr || colExpr->getArguments().size() < 1) continue;

    auto listArg = colExpr->getArguments()[0];
    auto const* listExpr = get_if<ComplexExpression>(&listArg);
    if(!listExpr) continue;

    // collect each row ID from the List
    visitRowIDs(*listExpr, [&](auto idValue) {
      keys.push_back({tableName, static_cast<int64_t>(idValue)});
    });

    if(!keys.empty()) break; // found the id column
  }

  return keys;
}

static Expression flushAllBuckets() {
  std::cout << "WAL: flushing all " << walTotalEntries << " entries" << std::endl;
  boss::ExpressionArguments entries;
  auto keysCopy = walOrder; // copy because flushBucket modifies walOrder
  for(auto const& key : keysCopy) {
    auto flushed = flushBucket(key);
    for(auto& e : flushed) entries.push_back(std::move(e));
  }
  std::cout << "WAL: emitting " << entries.size() << " entries" << std::endl;
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

          std::cout << "WAL: captured Update" << std::endl;

          // check if WAL has hit the threshold - if so flush to Arrow storage
          if(walTotalEntries >= WAL_THRESHOLD) {
            return flushAllBuckets();
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

          std::cout << "WAL: captured Delete" << std::endl;

          // check if WAL has hit the threshold - if so flush to Arrow storage
          if(walTotalEntries >= WAL_THRESHOLD) {
            return flushAllBuckets();
          }
          return "Delete_Logged"_();
        }

        // GetWAL — return raw chain contents flattened in walOrder sequence
        if (head == "GetWAL"_) {
          boss::ExpressionArguments entries;
          for(auto const& key : walOrder) {
            auto it = walIndex.find(key);
            if(it == walIndex.end()) continue;
            for(auto const& entry : it->second.chain)
              entries.push_back(entry.clone());
          }
          std::cout << "WAL: returning " << entries.size() << " raw entries" << std::endl;
          return ComplexExpression("List"_, {}, std::move(entries), {});
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
          std::cout << "WAL: optimising " << walTotalEntries << " entries" << std::endl;
          for(auto const& key : walOrder) {
            auto it = walIndex.find(key);
            if(it == walIndex.end()) continue;
            // subtract old count, optimise, add new count
            walTotalEntries -= it->second.chain.size();
            optimiseBucketImpl(it->second);
            walTotalEntries += it->second.chain.size();
          }
          std::cout << "WAL: " << walTotalEntries << " entries after optimisation" << std::endl;
          return "WAL_Optimised"_();
        }

        // Select - triggers WAL flush before read
        if(head == "Select"_) {
          auto keys = extractSelectKeys(expr);
          boss::ExpressionArguments applyArgs;
 
          if(!keys.empty()) {
            // selective flush — only affected rows
            for(auto const& key : keys) {
              auto flushed = flushBucket(key);
              for(auto& e : flushed) applyArgs.push_back(std::move(e));
            }
          } else {
            // could not parse row IDs — fall back to full flush for correctness
            std::cout << "WAL: Select fallback to full flush" << std::endl;
            auto keysCopy = walOrder;
            for(auto const& key : keysCopy) {
              auto flushed = flushBucket(key);
              for(auto& e : flushed) applyArgs.push_back(std::move(e));
            }
          }
 
          // append the original Select as the last argument
          applyArgs.push_back(std::move(expr));
          return ComplexExpression("ApplyWAL"_, {}, std::move(applyArgs), {});
        }
        
        // for flushing the WAL
        if(head == "FlushWAL"_) {
          return flushAllBuckets();
        }
      }
      return std::move(expr);
    }, std::move(e));
};

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
};
