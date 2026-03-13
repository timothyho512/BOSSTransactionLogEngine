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
      }
      return std::move(expr);
    }, std::move(e));
};

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
};
