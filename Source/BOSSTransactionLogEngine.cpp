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

// WAL - just a list of expression for now
static std:: vector<Expression> writeAheadLog;

// Threshold for WAL flush - when WAL reaches this size, flush to Arrow storage
static const size_t WAL_THRESHOLD = 10;

// Helper: extract table name from a WAL entry
static std::string getTableName(ComplexExpression const& entry) {
  if(entry.getArguments().size() == 0) return "";
  auto arg = entry.getArguments()[0];
  auto const* table = get_if<Symbol>(&arg);
  return table ? table->getName() : "";
}

// Helper: extract row IDs from a WAL entry (second argument is "id"_(List(...)))
using RowID = std::variant<int32_t, int64_t>;
static std::vector<RowID> getRowIDs(ComplexExpression const& entry) {
  std::vector<RowID> ids;
  if(entry.getArguments().size() < 2) return ids;
  
  // arg 1 is id(List(...))
  auto idArg = entry.getArguments()[1];
  auto const* idExpr = get_if<ComplexExpression>(&idArg);
  if(!idExpr) return ids;

  // inside id(...) is List(...)
  auto listArg = idExpr->getArguments()[0];
  auto const* listExpr = get_if<ComplexExpression>(&listArg);
  if(!listExpr) return ids;

  // Case 1: plain integers in dynamic arguments
  for(size_t i = 0; i < listExpr->getArguments().size(); i++) {
    auto val = listExpr->getArguments()[i];
    if(auto const* id32 = get_if<int32_t>(&val)) {
      ids.push_back(*id32);
    } else if(auto const* id64 = get_if<int64_t>(&val)) {
      ids.push_back(*id64);
    }
  }

  // Case 2: Spans in span arguments
  // expression arrives from Velox - type unknown at compile time
  for (auto const& spanArg : listExpr->getSpanArguments()) {
    std::visit([&ids](auto const& typedSpan) {
      using T = std::decay_t<decltype(*typedSpan.begin())>;
      if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
        for(auto it = typedSpan.begin(); it != typedSpan.end(); ++it) {
          ids.push_back(*it);
        }
      }
      // non-numeric types (float, string etc) are silently skipped
      // because row IDs will always be integers
    }, spanArg);
  }
  return ids;
}

// Initial design decision is for the prototype
// we only allow one row, or one ID per expression
// to prevent optimisation complexity
// need to fix the code below
static bool sameTableAndRow(ComplexExpression const& a, ComplexExpression const& b) {
  if(getTableName(a) != getTableName(b)) return false;
  auto idsA = getRowIDs(a);
  auto idsB = getRowIDs(b);
  if(idsA.empty() || idsB.empty()) return false;

  // cast both to int64_t as this does not change the value
  auto toInt64 = [](RowID const& id) {
    return std::visit([](auto const& v) { return static_cast<int64_t>(v); }, id); 
  };
  return toInt64(idsA[0]) == toInt64(idsB[0]);
}

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

// OptimiseWAL - apply three optimisation rules:
// Rule 1: later Update on same row and same column
// if later write is blind → earlier is redundant (last-write-wins)
// if later write is dependent → fold the two writes together into entry j
//   by substituting entry i's value into entry j's expression
// 2. Delete eliminates pending Updates on same row
// 3. Merge Updates on same row with different columns
static void optimiseWAL() {
  std::vector<Expression> optimised;

  for (size_t i = 0; i < writeAheadLog.size(); i++) {
    auto const& entryI = get<ComplexExpression>(writeAheadLog[i]);
    bool redundant = false;

    // check if a later entry makes this one redundant
    for (size_t j = i + 1; j < writeAheadLog.size(); j++) {
      auto const& entryJ = get<ComplexExpression>(writeAheadLog[j]);
      if (!sameTableAndRow(entryI, entryJ)) continue;

      // Rule 2: Delete eliminates any pending Update on same row
      if (entryJ.getHead() == "Delete"_) {
        redundant = true;
        break;
      }

      // Rule 1: later Update on same row and same column
      // if later write is blind → earlier is redundant (last-write-wins)
      // if later write is dependent → fold the two writes together into entry j
      //   by substituting entry i's value into entry j's expression
      if(entryI.getHead() == "Update"_ && entryJ.getHead() == "Update"_) {
        auto setArgI = entryI.getArguments()[2];
        auto const* setExprI = get_if<ComplexExpression>(&setArgI);
        if(!setExprI) continue;
        auto setArgJ = entryJ.getArguments()[2];
        auto const* setExprJ = get_if<ComplexExpression>(&setArgJ);
        if(!setExprJ) continue;

        for(size_t ci = 0; ci < setExprI->getArguments().size(); ci++) {
          auto colIArg = setExprI->getArguments()[ci];
          auto const* colI = get_if<ComplexExpression>(&colIArg);
          if(!colI) continue;
          for(size_t cj = 0; cj < setExprJ->getArguments().size(); cj++) {
            auto colJArg = setExprJ->getArguments()[cj];
            auto const* colJ = get_if<ComplexExpression>(&colJArg);
            if(!colJ) continue;
            if(colI->getHead() != colJ->getHead()) continue;

            // same column — check if later write is blind or dependent
            if(isBlindWrite(*colJ)) {
              // blind write — earlier is simply redundant
              redundant = true;
            } else {
              // dependent write — fold: substitute earlier value into later expression
              // modify entry j in the WAL to contain the folded expression
              auto& entryJMutable = get<ComplexExpression>(writeAheadLog[j]);
              auto [jHead, jStatics, jArgs, jSpans] = std::move(entryJMutable).decompose();
              auto setArgJMut = std::move(jArgs[2]);
              auto [setHead, setStatics, setArgs, setSpans] = 
                std::move(get<ComplexExpression>(setArgJMut)).decompose();

              // replace the matching column in entry j's Set with the folded value
              for(size_t k = 0; k < setArgs.size(); k++) {
                auto const* setCol = get_if<ComplexExpression>(&setArgs[k]);
                if(!setCol || setCol->getHead() != colI->getHead()) continue;
                setArgs[k] = foldColumnWrites(*colI, *setCol);
                break;
              }

              jArgs[2] = ComplexExpression(setHead, {}, std::move(setArgs), {});
              writeAheadLog[j] = ComplexExpression(jHead, {}, std::move(jArgs), {});
              redundant = true; // earlier entry is now absorbed into j
            }
            break;
          }
          if(redundant) break;
        }
      }
      if(redundant) break;
    }
    if(!redundant) {
      optimised.push_back(entryI.clone());
    }
  }

  // Rule 3: merge Updates on same row with different columns
  std::vector<Expression> merged;
  std::vector<bool> mergedFlag(optimised.size(), false);

  for (size_t i = 0; i < optimised.size(); i++) {
    if(mergedFlag[i]) continue;
    auto const& entryI = get<ComplexExpression>(optimised[i]);

    if(entryI.getHead() != "Update"_) {
      merged.push_back(entryI.clone());
      continue;
    }


    // clone each argument to avoid ArgumentWrapper temporary issue
    auto setArgI = entryI.getArguments()[2];
    auto const* setExprI = get_if<ComplexExpression>(&setArgI);
    boss::ExpressionArguments mergedSetArgs;
    for(size_t ci = 0; ci < setExprI->getArguments().size(); ci++) {
      mergedSetArgs.push_back(setExprI->cloneArgument(ci));
    }
  
    // mergedArgs.push_back(entryI.getArguments()[0]); // table
    // mergedArgs.push_back(entryI.getArguments()[1]); // ID
    // for(size_t ci = 2; ci < entryI.getArguments().size(); ci++) {
    //   mergedArgs.push_back(entryI.getArguments()[ci]);
    // }

    for(size_t j = i + 1; j < optimised.size(); j++) {
      if(mergedFlag[j]) continue;
      auto const& entryJ = get<ComplexExpression>(optimised[j]);
      if(entryJ.getHead() != "Update"_) continue;
      if(!sameTableAndRow(entryI, entryJ)) continue;

      // merge columns from entryJ
      auto setArgJ = entryJ.getArguments()[2];
      auto const* setExprJ = get_if<ComplexExpression>(&setArgJ);
      for(size_t cj = 0; cj < setExprJ->getArguments().size(); cj++) {
        mergedSetArgs.push_back(setExprJ->cloneArgument(cj));
      }
      mergedFlag[j] = true;
    }
    // build merged WAL entry
    boss::ExpressionArguments mergedArgs;
    mergedArgs.push_back(entryI.cloneArgument(0)); // table name
    mergedArgs.push_back(entryI.cloneArgument(1)); // id
    mergedArgs.push_back(ComplexExpression("Set"_, {}, std::move(mergedSetArgs), {}));
    merged.push_back(ComplexExpression("Update"_, {}, std::move(mergedArgs), {}));
  }
  writeAheadLog = std::move(merged);
}

// flushWAL - optimise and return all WAL entries as a List for Arrow storage to apply
// called when WAL hit threshold
static Expression flushWAL() {
  std::cout << "WAL: flushing" << writeAheadLog.size() << " entries" << std::endl;
  optimiseWAL();
  std::cout << "WAL: " << writeAheadLog.size() << " entries after optimisation" << std::endl;

  boss::ExpressionArguments entries;
  for (auto& entry : writeAheadLog) {
    entries.push_back(entry.clone());
  }
  writeAheadLog.clear();
  return ComplexExpression("ApplyWAL"_, {}, std::move(entries), {});
}

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
            boss::ExpressionArguments walArgs;
            walArgs.push_back(*tableSymbol);
            boss::ExpressionArguments idListArgs;
            idListArgs.push_back(idValue);
            auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});
            boss::ExpressionArguments idColArgs;
            idColArgs.push_back(std::move(idList));
            walArgs.push_back(ComplexExpression(idColName, {}, std::move(idColArgs), {}));
            walArgs.push_back(setExpr->clone());
            writeAheadLog.push_back(
              ComplexExpression("Update"_, {}, std::move(walArgs), {})
            );
          });

          std::cout << "WAL: captured Update" << std::endl;

          // check if WAL has hit the threshold - if so flush to Arrow storage
          if(writeAheadLog.size() >= WAL_THRESHOLD) {
            return flushWAL();
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
            boss::ExpressionArguments walArgs;
            walArgs.push_back(*tableSymbol);
            boss::ExpressionArguments idListArgs;
            idListArgs.push_back(idValue);  // preserves original type
            auto idList = ComplexExpression("List"_, {}, std::move(idListArgs), {});
            boss::ExpressionArguments idColArgs;
            idColArgs.push_back(std::move(idList));
            walArgs.push_back(ComplexExpression(idColName, {}, std::move(idColArgs), {}));
            writeAheadLog.push_back(ComplexExpression("Delete"_, {}, std::move(walArgs), {}));
          });

          std::cout << "WAL: captured Delete" << std::endl;

          // check if WAL has hit the threshold - if so flush to Arrow storage
          if(writeAheadLog.size() >= WAL_THRESHOLD) {
            return flushWAL();
          }

          return "Delete_Logged"_();
        }

        // GetWAL - return current WAL contents
        if (head == "GetWAL"_) {
          std::cout << "WAL: returning " << writeAheadLog.size() << " entries" << std::endl;
          boss::ExpressionArguments entries;
          for (auto& entry : writeAheadLog) {
            entries.push_back(entry.clone());
          }
          return ComplexExpression("List"_, {}, std::move(entries), {});
        }
        // ClearWAL - empty the log
        if (head == "ClearWAL"_) {
          writeAheadLog.clear();
          return "WAL_Cleared"_();
        }
        // OptimiseWAL - apply optimisation rules
        if(head == "OptimiseWAL"_) {
          std::cout << "WAL: optimising " << writeAheadLog.size() << " entries" << std::endl;
          optimiseWAL();
          std::cout << "WAL: " << writeAheadLog.size() << " entries after optimisation" << std::endl;
          return "WAL_Optimised"_();
        }

        // Select - triggers WAL flush before read
        if(head == "Select"_) {
          optimiseWAL();

          // build ApplyWAL(Update(...), Delete(...), ..., Select(...))
          boss::ExpressionArguments applyArgs;

          // add all flushed WAL entries
          for(auto& entry : writeAheadLog) {
            applyArgs.push_back(entry.clone());
          }
          writeAheadLog.clear();

          // add the original Select as the last argument
          applyArgs.push_back(std::move(expr));

          return ComplexExpression("ApplyWAL"_, {}, std::move(applyArgs), {});
        }
        
        // for flushing the WAL
        if(head == "FlushWAL"_) {
            return flushWAL();
        }
      }
      return std::move(expr);
    }, std::move(e));
};

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
};
