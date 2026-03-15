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

static Expression evaluate(Expression &&e) {
  return std::visit(
    [](auto &&expr) -> Expression {
      if constexpr(std::is_same_v<std::decay_t<decltype(expr)>, ComplexExpression>) {
        auto head = expr.getHead();
        if (head == "InsertInto"_) {
          std::cout << "WAL: captured InsertInto" << std::endl;
          writeAheadLog.push_back(expr.clone());
          return "InsertInto_Logged"_();
        }
        if (head == "Update"_) {
          std::cout << "WAL: captured Update" << std::endl;
          writeAheadLog.push_back(expr.clone());
          return "Update_Logged"_();
        }
        if (head == "Delete"_) {
          std::cout << "WAL: captured Delete" << std::endl;
          writeAheadLog.push_back(expr.clone());
          return "Delete_Logged"_();
        }
        if (head == "GetWAL"_) {
          std::cout << "WAL: returning " << writeAheadLog.size() << " entries" << std::endl;
          boss::ExpressionArguments entries;
          for (auto& entry : writeAheadLog) {
            entries.push_back(entry.clone());
          }
          return ComplexExpression("List"_, {}, std::move(entries), {});
        }
        if (head == "ClearWAL"_) {
          writeAheadLog.clear();
          return "WAL_Cleared"_();
        }
        if (head == "DetectConflicts"_) {
          boss::ExpressionArguments conflicts;
          for (size_t i = 0; i < writeAheadLog.size(); i++) {
            for (size_t j = i + 1; j < writeAheadLog.size(); j++) {
              auto const& entryI = get<ComplexExpression>(writeAheadLog[i]);
              auto const& entryJ = get<ComplexExpression>(writeAheadLog[j]);
              if (entryI.getArguments().size() == 0 || entryJ.getArguments().size() == 0) continue;
              auto argI = entryI.getArguments()[0];
              auto argJ = entryJ.getArguments()[0];
              auto const* tableI = get_if<Symbol>(&argI);
              auto const* tableJ = get_if<Symbol>(&argJ);
              if (tableI && tableJ && tableI->getName() == tableJ->getName()) {
                conflicts.push_back(
                  "Conflict"_(entryI.getHead(), entryJ.getHead(), *tableI)
                );
              }
            }
          }
          return ComplexExpression("List"_, {}, std::move(conflicts), {});
        }
      }
      return std::move(expr);
    }, std::move(e));
};

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
};
