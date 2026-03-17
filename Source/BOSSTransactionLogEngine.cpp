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

// Helper: extract row IDs from a WAL entry (second argument is ID(List(...)))
static std::vector<int32_t> getRowIDs(ComplexExpression const& entry) {
  std::vector<int32_t> ids;
  if(entry.getArguments().size() < 2) return ids;
  auto idArg = entry.getArguments()[1];
  auto const* idExpr = get_if<ComplexExpression>(&idArg);
  if(!idExpr) return ids;
  auto listArg = idExpr->getArguments()[0];   // List(5, 9)
  auto const* listExpr = get_if<ComplexExpression>(&listArg);  // List(...)
  if(!listExpr) return ids;
  for(size_t i = 0; i < listExpr->getArguments().size(); i++) {
    auto val = listExpr->getArguments()[i];
    auto const* id = get_if<int32_t>(&val);
    if(id) ids.push_back(*id);
  }
  return ids;
}

// Note: if entries have multiple row IDs, any overlap causes the entire
// earlier entry to be treated as redundant. Row-level splitting of
// partially overlapping multi-row updates is left as future work.
// Helper: check if two entries share any row IDs on the same table
static bool sameTableAndRow(ComplexExpression const& a, ComplexExpression const& b) {
  if(getTableName(a) != getTableName(b)) return false;
  auto idsA = getRowIDs(a);
  auto idsB = getRowIDs(b);
  for(auto idA : idsA) {
    for(auto idB : idsB) {
      if(idA == idB) return true;
    }
  }
  return false;
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
        // check if they share any column
        for (size_t ci = 2; ci < entryI.getArguments().size(); ci++) {
          auto colI = entryI.getArguments()[ci];
          auto const* colIExpr = get_if<ComplexExpression>(&colI);
          if(!colIExpr) continue;
          for (size_t cj = 2; cj < entryJ.getArguments().size(); cj++) {
            auto colJ = entryJ.getArguments()[cj];
            auto const* colJExpr = get_if<ComplexExpression>(&colJ);
            if(!colJExpr) continue;
            if(colIExpr->getHead() == colJExpr->getHead()) {
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

    // collect all columns from this and subsequent Updates on same row
    boss::ExpressionArguments mergedArgs;
    // clone each argument to avoid ArgumentWrapper temporary issue
    for (size_t ci = 0; ci < entryI.getArguments().size(); ci++) {
      mergedArgs.push_back(entryI.cloneArgument(ci));
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
      for(size_t cj = 2; cj < entryJ.getArguments().size(); cj++) {
        mergedArgs.push_back(entryJ.cloneArgument(cj));
      }
      mergedFlag[j] = true;
    }

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
          std::cout << "WAL: captured Update" << std::endl;
          writeAheadLog.push_back(expr.clone());
          return "Update_Logged"_();
        }

        // Delete - defer to WAL
        if (head == "Delete"_) {
          std::cout << "WAL: captured Delete" << std::endl;
          writeAheadLog.push_back(expr.clone());
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
