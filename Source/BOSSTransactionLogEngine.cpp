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

// Helper: extract table name from a WAL entry
static std::string getTableName(ComplexExpression const& entry) {
  if(entry.getArguments().size() == 0) return "";
  auto arg = entry.getArguments()[0];
  auto const* table = get_if<Symbol>(&arg);
  return table ? table->getName() : "";
}

// Helper: extract row IDs from a WAL entry (second argument is "id"_(List(...)))
static std::vector<int32_t> getRowIDs(ComplexExpression const& entry) {
  std::vector<int32_t> ids;
  if(entry.getArguments().size() < 2) return ids;
  auto idArg = entry.getArguments()[1];
  auto const* idExpr = get_if<ComplexExpression>(&idArg);
  if(!idExpr) return ids;
  // might need to change later as this is hardcoded, (primary key column might no always be ids)
  if(idExpr->getHead().getName() != "id") return ids;

  auto listArg = idExpr->getArguments()[0];
  auto const* listExpr = get_if<ComplexExpression>(&listArg);
  if(!listExpr) return ids;
  for(size_t i = 0; i < listExpr->getArguments().size(); i++) {
    auto val = listExpr->getArguments()[i];
    auto const* id = get_if<int32_t>(&val);
    if(id) ids.push_back(*id);
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
  return idsA[0] == idsB[0];
}

// OptimiseWAL - apply three optimisation rules:
// 1. Last write wins on same row and column
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

      // Rule 1: later Update on same row and same column wins
      if (entryI.getHead() == "Update"_ && entryJ.getHead() == "Update"_) {
        // get "Set"_ from entryI - arg index 2
        auto setArgI = entryI.getArguments()[2];
        auto const* setExprI = get_if<ComplexExpression>(&setArgI);
        if(!setExprI) continue;

        // get "Set"_ from entryJ - arg index 2  
        auto setArgJ = entryJ.getArguments()[2];
        auto const* setExprJ = get_if<ComplexExpression>(&setArgJ);
        if(!setExprJ) continue;

        // check if they share any column
        for (size_t ci = 0; ci < setExprI->getArguments().size(); ci++) {
          auto colIArg = setExprI->getArguments()[ci];
          auto const* colI = get_if<ComplexExpression>(&colIArg);
          if(!colI) continue;
          for (size_t cj = 0; cj < setExprJ->getArguments().size(); cj++) {
            auto colJArg = setExprJ->getArguments()[cj];
            auto const* colJ = get_if<ComplexExpression>(&colJArg);
            if(!colJ) continue;
            if(colI->getHead() == colJ->getHead()) {
              redundant = true;
              break;
            }
          }
          if (redundant) break;
        }
      }
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

static Expression evaluate(Expression &&e) {
  return std::visit(
    [](auto &&expr) -> Expression {
      if constexpr(std::is_same_v<std::decay_t<decltype(expr)>, ComplexExpression>) {
        auto head = expr.getHead();

        // InsertInto - blind write, execute eagerly, pass through
        // if (head == "InsertInto"_) {
        //   std::cout << "WAL: InsertInto passing through eagerly" << std::endl;
        //   // writeAheadLog.push_back(expr.clone());
        //   // return "InsertInto_Logged"_();
        //   return std::move(expr);
        // }

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
          for (size_t i = 0; i < tableExpr->getArguments().size(); i++) {
            auto colArg = tableExpr->getArguments()[i];
            auto const* colExpr = get_if<ComplexExpression>(&colArg);
            if(!colExpr) continue;
            if (colExpr->getHead().getName() == "id") {
              //get the "List"_ inside "id"_
              auto listArg = colExpr->getArguments()[0];
              idListExpr = get_if<ComplexExpression>(&listArg);
              break;
            }
          }
          if(!idListExpr) return std::move(expr);

          // Extract the "Set"_ expression
          auto setArg = expr.getArguments()[2];
          auto const* setExpr = get_if<ComplexExpression>(&setArg);
          if(!setExpr) return std::move(expr);

          // Loop over each id and push one WAL entry per row
          for (size_t i = 0; i < idListExpr->getArguments().size(); i++) {
            auto idVal = idListExpr->getArguments()[i];
            auto const* id = get_if<int32_t>(&idVal);
            if(!id) continue;

            // build: "Update"_("Customer"_, "id"_("List"_(5)), "Set"_(...))
            boss::ExpressionArguments walArgs;
            walArgs.push_back(*tableSymbol);
            // TODO: hardcoded "id" column name - should use first column of Table instead
            walArgs.push_back("id"_("List"_(*id)));
            walArgs.push_back(setExpr->clone());
            writeAheadLog.push_back(
              ComplexExpression("Update"_, {}, std::move(walArgs), {})
            );
          }

          std::cout << "WAL: captured Update, " 
                    << idListExpr->getArguments().size() 
                    << " row(s)" << std::endl;
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
          ComplexExpression const* idListExpr = nullptr;
          for (size_t i = 0; i < tableExpr->getArguments().size(); i++) {
            auto colArg = tableExpr->getArguments()[i];
            auto const* colExpr = get_if<ComplexExpression>(&colArg);
            if(!colExpr) continue;
            if(colExpr->getHead().getName() == "id") {
              auto listArg = colExpr->getArguments()[0];
              idListExpr = get_if<ComplexExpression>(&listArg);
              break;
            }
          }
          if(!idListExpr) return std::move(expr);

          // loop over each id and push one WAL entry per row
          for(size_t i = 0; i < idListExpr->getArguments().size(); i++) {
            auto idVal = idListExpr->getArguments()[i];
            auto const* id = get_if<int32_t>(&idVal);
            if(!id) continue;

            // build: "Delete"_("Customer"_, "id"_("List"_(5)))
            boss::ExpressionArguments walArgs;
            walArgs.push_back(*tableSymbol);
            walArgs.push_back("id"_("List"_(*id)));
            writeAheadLog.push_back(
              ComplexExpression("Delete"_, {}, std::move(walArgs), {})
            );
          }
          std::cout << "WAL: captured Delete, "
                    << idListExpr->getArguments().size()
                    << " row(s)" << std::endl;
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
        // if (head == "DetectConflicts"_) {
        //   boss::ExpressionArguments conflicts;
        //   for (size_t i = 0; i < writeAheadLog.size(); i++) {
        //     for (size_t j = i + 1; j < writeAheadLog.size(); j++) {
        //       auto const& entryI = get<ComplexExpression>(writeAheadLog[i]);
        //       auto const& entryJ = get<ComplexExpression>(writeAheadLog[j]);
        //       if (entryI.getArguments().size() == 0 || entryJ.getArguments().size() == 0) continue;
        //       auto argI = entryI.getArguments()[0];
        //       auto argJ = entryJ.getArguments()[0];
        //       auto const* tableI = get_if<Symbol>(&argI);
        //       auto const* tableJ = get_if<Symbol>(&argJ);
        //       if (tableI && tableJ && tableI->getName() == tableJ->getName()) {
        //         conflicts.push_back(
        //           "Conflict"_(entryI.getHead(), entryJ.getHead(), *tableI)
        //         );
        //       }
        //     }
        //   }
        //   return ComplexExpression("List"_, {}, std::move(conflicts), {});
        // }
        // OptimiseWAL - apply optimisation rules
        if(head == "OptimiseWAL"_) {
          std::cout << "WAL: optimising " << writeAheadLog.size() << " entries" << std::endl;
          optimiseWAL();
          std::cout << "WAL: " << writeAheadLog.size() << " entries after optimisation" << std::endl;
          return "WAL_Optimised"_();
        }
      }
      return std::move(expr);
    }, std::move(e));
};

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
};
