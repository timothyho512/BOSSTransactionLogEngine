#include "BOSS.hpp"
#include "Expression.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using boss::Expression;
using boss::ExpressionArguments;
using boss::Symbol;
using boss::expressions::ComplexExpression;
using boss::expressions::generic::get_if;

namespace {

std::string trim(std::string value) {
  auto notSpace = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
  value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
  return value;
}

bool bossExpressionNeedsMoreInput(std::string const& text) {
  int depth = 0;
  char quote = 0;
  bool escaped = false;
  for(char const current : text) {
    if(escaped) {
      escaped = false;
      continue;
    }
    if(quote != 0) {
      if(current == '\\') escaped = true;
      else if(current == quote) quote = 0;
      continue;
    }
    if(current == '\'' || current == '"') quote = current;
    else if(current == '(') ++depth;
    else if(current == ')') --depth;
  }
  return quote != 0 || depth > 0;
}

std::string upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

bool iequals(std::string const& left, std::string const& right) {
  return upper(left) == upper(right);
}

std::vector<std::string> splitTopLevel(std::string const& text, char delimiter) {
  std::vector<std::string> parts;
  size_t start = 0;
  int depth = 0;
  char quote = 0;
  for(size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if(quote != 0) {
      if(c == quote && (i == 0 || text[i - 1] != '\\')) quote = 0;
      continue;
    }
    if(c == '\'' || c == '"') quote = c;
    else if(c == '(') ++depth;
    else if(c == ')') --depth;
    else if(c == delimiter && depth == 0) {
      parts.push_back(trim(text.substr(start, i - start)));
      start = i + 1;
    }
  }
  parts.push_back(trim(text.substr(start)));
  return parts;
}

ComplexExpression complex(std::string const& head, ExpressionArguments arguments = {}) {
  return ComplexExpression(Symbol(head), {}, std::move(arguments), {});
}

template <typename... Args>
ComplexExpression make(std::string const& head, Args&&... args) {
  ExpressionArguments arguments;
  (arguments.emplace_back(std::forward<Args>(args)), ...);
  return complex(head, std::move(arguments));
}

Expression parseValue(std::string token) {
  token = trim(token);
  if(token.size() >= 2 && ((token.front() == '\'' && token.back() == '\'') ||
                           (token.front() == '"' && token.back() == '"'))) {
    return token.substr(1, token.size() - 2);
  }
  if(iequals(token, "TRUE")) return true;
  if(iequals(token, "FALSE")) return false;

  int64_t integer = 0;
  auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), integer);
  if(error == std::errc() && end == token.data() + token.size()) {
    if(integer >= INT32_MIN && integer <= INT32_MAX) return static_cast<int32_t>(integer);
    return integer;
  }

  char* doubleEnd = nullptr;
  double floating = std::strtod(token.c_str(), &doubleEnd);
  if(doubleEnd == token.c_str() + token.size() && doubleEnd != token.c_str()) return floating;

  if(std::regex_match(token, std::regex(R"([A-Za-z_][A-Za-z0-9_]*)"))) {
    return Symbol(token);
  }
  throw std::runtime_error("cannot parse value or column reference: " + token);
}

// Parse the textual S-expression form used by BOSS frontends, for example:
//   (Update Stock (Table (id (List 1)))
//                 (Set (value (Plus value 1.0))))
//
// The parser is deliberately generic: it builds an expression tree without
// maintaining an operator allow-list. Whether an expression can be evaluated
// is decided by the selected engine pipeline, not by this interface.
class BossExpressionParser {
  std::string_view input;
  size_t position = 0;

  void skipWhitespace() {
    while(position < input.size() &&
          std::isspace(static_cast<unsigned char>(input[position]))) {
      ++position;
    }
  }

  std::string parseQuotedString() {
    char const quote = input[position++];
    std::string value;
    while(position < input.size()) {
      char const current = input[position++];
      if(current == quote) return value;
      if(current != '\\') {
        value.push_back(current);
        continue;
      }
      if(position >= input.size()) break;
      char const escaped = input[position++];
      if(escaped == 'n') value.push_back('\n');
      else if(escaped == 't') value.push_back('\t');
      else value.push_back(escaped);
    }
    throw std::runtime_error("unterminated quoted string in BOSS expression");
  }

  std::string parseToken() {
    skipWhitespace();
    size_t const start = position;
    while(position < input.size() &&
          !std::isspace(static_cast<unsigned char>(input[position])) &&
          input[position] != '(' && input[position] != ')') {
      ++position;
    }
    if(start == position) {
      throw std::runtime_error("expected an atom in BOSS expression");
    }
    return std::string(input.substr(start, position - start));
  }

  Expression parseAtom() {
    skipWhitespace();
    if(position >= input.size()) throw std::runtime_error("unexpected end of BOSS expression");
    if(input[position] == '\'' || input[position] == '"') return parseQuotedString();

    auto token = parseToken();
    if(iequals(token, "TRUE")) return true;
    if(iequals(token, "FALSE")) return false;

    int64_t integer = 0;
    auto [integerEnd, integerError] =
      std::from_chars(token.data(), token.data() + token.size(), integer);
    if(integerError == std::errc() && integerEnd == token.data() + token.size()) {
      if(integer >= INT32_MIN && integer <= INT32_MAX) return static_cast<int32_t>(integer);
      return integer;
    }

    char* floatingEnd = nullptr;
    double floating = std::strtod(token.c_str(), &floatingEnd);
    if(floatingEnd == token.c_str() + token.size() && floatingEnd != token.c_str()) {
      return floating;
    }
    return Symbol(std::move(token));
  }

  Expression parseOne() {
    skipWhitespace();
    if(position >= input.size()) throw std::runtime_error("unexpected end of BOSS expression");
    if(input[position] != '(') return parseAtom();

    ++position;
    skipWhitespace();
    if(position >= input.size() || input[position] == ')') {
      throw std::runtime_error("a BOSS expression requires an operator head");
    }
    if(input[position] == '\'' || input[position] == '"' || input[position] == '(') {
      throw std::runtime_error("a BOSS expression head must be a symbol");
    }
    auto head = parseToken();
    ExpressionArguments arguments;
    while(true) {
      skipWhitespace();
      if(position >= input.size()) throw std::runtime_error("missing ')' in BOSS expression");
      if(input[position] == ')') {
        ++position;
        break;
      }
      arguments.emplace_back(parseOne());
    }
    return complex(head, std::move(arguments));
  }

public:
  explicit BossExpressionParser(std::string_view text) : input(text) {}

  Expression parse() {
    auto expression = parseOne();
    skipWhitespace();
    if(position != input.size()) {
      throw std::runtime_error("unexpected text after the BOSS expression");
    }
    return expression;
  }
};

std::string expressionHead(Expression const& expression) {
  auto const* complexExpression = get_if<ComplexExpression>(&expression);
  return complexExpression == nullptr ? std::string{} : complexExpression->getHead().getName();
}

bool isWriteExpression(Expression const& expression) {
  auto const head = expressionHead(expression);
  return head == "CreateTable" || head == "InsertInto" || head == "Update" ||
         head == "Delete" || head == "CreateIndex" || head == "DropIndex" ||
         head == "DropTable" || head == "PopulateTable" ||
         head == "SetRecordPayloadBytes";
}

Expression parseScalarExpression(std::string text) {
  text = trim(text);
  std::smatch match;
  static std::regex const cellPattern(
    R"(^CELL\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(-?[0-9]+)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$)",
    std::regex::icase);
  if(std::regex_match(text, match, cellPattern)) {
    auto row = std::stoll(match[2].str());
    Expression rowExpression = row >= INT32_MIN && row <= INT32_MAX
      ? Expression(static_cast<int32_t>(row)) : Expression(static_cast<int64_t>(row));
    return make("Cell", Symbol(match[1].str()), Symbol("id"),
                std::move(rowExpression), Symbol(match[3].str()));
  }

  for(std::string const operators : {"+-", "*/"}) {
    int depth = 0;
    char quote = 0;
    for(size_t i = text.size(); i-- > 0;) {
      char c = text[i];
      if(quote != 0) {
        if(c == quote && (i == 0 || text[i - 1] != '\\')) quote = 0;
        continue;
      }
      if(c == '\'' || c == '"') quote = c;
      else if(c == ')') ++depth;
      else if(c == '(') --depth;
      else if(depth == 0 && operators.find(c) != std::string::npos && i > 0) {
        auto left = trim(text.substr(0, i));
        auto right = trim(text.substr(i + 1));
        if(left.empty() || right.empty()) continue;
        std::string head = c == '+' ? "Plus" : c == '-' ? "Minus" : c == '*' ? "Times" : "Divide";
        return make(head, parseScalarExpression(left), parseScalarExpression(right));
      }
    }
  }
  return parseValue(text);
}

std::vector<int64_t> parseIds(std::string text) {
  text = trim(text);
  if(text.front() == '(' && text.back() == ')') text = text.substr(1, text.size() - 2);
  std::vector<int64_t> ids;
  for(auto const& part : splitTopLevel(text, ',')) {
    try {
      ids.push_back(std::stoll(trim(part)));
    } catch(...) {
      throw std::runtime_error("row id must be an integer: " + part);
    }
  }
  if(ids.empty()) throw std::runtime_error("at least one row id is required");
  return ids;
}

ComplexExpression idTable(std::vector<int64_t> const& ids) {
  ExpressionArguments values;
  for(auto id : ids) {
    if(id >= INT32_MIN && id <= INT32_MAX) values.emplace_back(static_cast<int32_t>(id));
    else values.emplace_back(id);
  }
  return make("Table", make("id", complex("List", std::move(values))));
}

Expression parsePredicate(std::string text) {
  std::vector<std::string> terms;
  std::regex const andPattern(R"(\s+AND\s+)", std::regex::icase);
  std::sregex_token_iterator cursor(text.begin(), text.end(), andPattern, -1), end;
  for(; cursor != end; ++cursor) terms.push_back(trim(cursor->str()));

  ExpressionArguments predicates;
  std::regex const equalityPattern(
    R"(^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$)", std::regex::icase);
  for(auto const& term : terms) {
    std::smatch match;
    if(!std::regex_match(term, match, equalityPattern)) {
      throw std::runtime_error("SELECT supports equality predicates joined by AND");
    }
    predicates.emplace_back(make("Equal", Symbol(match[1].str()), parseValue(match[2].str())));
  }
  if(predicates.size() == 1) return std::move(predicates.front());
  return complex("And", std::move(predicates));
}

struct ParsedCommand {
  Expression expression;
  bool write = false;
  std::string label;
};

ParsedCommand parseBoss(std::string const& statement) {
  auto expression = BossExpressionParser(statement).parse();
  auto head = expressionHead(expression);
  bool const write = isWriteExpression(expression);
  return {std::move(expression), write, head.empty() ? "BOSS atom" : head};
}

ParsedCommand parseSql(std::string statement) {
  statement = trim(statement);
  if(!statement.empty() && statement.back() == ';') statement.pop_back();
  std::smatch match;

  static std::regex const createTable(
    R"(^CREATE\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)$)", std::regex::icase);
  if(std::regex_match(statement, match, createTable)) {
    return {make("CreateTable", Symbol(match[1].str())), true, "create table"};
  }

  static std::regex const insert(
    R"(^INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.+)\)\s+VALUES\s*\((.+)\)$)",
    std::regex::icase);
  if(std::regex_match(statement, match, insert)) {
    auto columns = splitTopLevel(match[2].str(), ',');
    auto values = splitTopLevel(match[3].str(), ',');
    if(columns.size() != values.size()) throw std::runtime_error("column/value count mismatch");
    if(columns.empty() || !iequals(columns.front(), "id")) {
      throw std::runtime_error("the first INSERT column must be id");
    }
    ExpressionArguments arguments;
    arguments.emplace_back(Symbol(match[1].str()));
    for(size_t i = 0; i < columns.size(); ++i) {
      auto column = trim(columns[i]);
      if(!std::regex_match(column, std::regex(R"([A-Za-z_][A-Za-z0-9_]*)"))) {
        throw std::runtime_error("invalid column name: " + column);
      }
      arguments.emplace_back(make(column, parseValue(values[i])));
    }
    return {complex("InsertInto", std::move(arguments)), true, "insert"};
  }

  static std::regex const update(
    R"(^UPDATE\s+([A-Za-z_][A-Za-z0-9_]*)\s+SET\s+(.+)\s+WHERE\s+id\s*(?:=|IN)\s*(.+)$)",
    std::regex::icase);
  if(std::regex_match(statement, match, update)) {
    ExpressionArguments assignments;
    for(auto const& assignment : splitTopLevel(match[2].str(), ',')) {
      auto equals = assignment.find('=');
      if(equals == std::string::npos) throw std::runtime_error("invalid SET assignment: " + assignment);
      auto column = trim(assignment.substr(0, equals));
      if(!std::regex_match(column, std::regex(R"([A-Za-z_][A-Za-z0-9_]*)"))) {
        throw std::runtime_error("invalid SET column: " + column);
      }
      assignments.emplace_back(make(column, parseScalarExpression(assignment.substr(equals + 1))));
    }
    return {make("Update", Symbol(match[1].str()), idTable(parseIds(match[3].str())),
                 complex("Set", std::move(assignments))), true, "update"};
  }

  static std::regex const remove(
    R"(^DELETE\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+WHERE\s+id\s*=\s*(-?[0-9]+)$)",
    std::regex::icase);
  if(std::regex_match(statement, match, remove)) {
    return {make("Delete", Symbol(match[1].str()), idTable(parseIds(match[2].str()))),
            true, "delete"};
  }

  static std::regex const select(
    R"(^SELECT\s+(.+)\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+WHERE\s+(.+)$)",
    std::regex::icase);
  if(std::regex_match(statement, match, select)) {
    Expression query = make("Select", Symbol(match[2].str()),
                            make("Where", parsePredicate(match[3].str())));
    auto projection = trim(match[1].str());
    if(projection != "*") {
      ExpressionArguments mappings;
      for(auto const& column : splitTopLevel(projection, ',')) {
        if(!std::regex_match(column, std::regex(R"([A-Za-z_][A-Za-z0-9_]*)"))) {
          throw std::runtime_error("invalid projected column: " + column);
        }
        mappings.emplace_back(Symbol(column));
        mappings.emplace_back(Symbol(column));
      }
      query = make("Project", std::move(query), complex("As", std::move(mappings)));
    }
    return {std::move(query), false, "select"};
  }

  static std::regex const createIndex(
    R"(^CREATE\s+INDEX\s+([A-Za-z_][A-Za-z0-9_]*)\s+ON\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.+)\)$)",
    std::regex::icase);
  if(std::regex_match(statement, match, createIndex)) {
    ExpressionArguments columns;
    for(auto const& column : splitTopLevel(match[3].str(), ',')) columns.emplace_back(Symbol(column));
    return {make("CreateIndex", match[1].str(), Symbol(match[2].str()),
                 complex("By", std::move(columns))), true, "create index"};
  }

  static std::regex const dropIndex(
    R"(^DROP\s+INDEX\s+([A-Za-z_][A-Za-z0-9_]*)$)", std::regex::icase);
  if(std::regex_match(statement, match, dropIndex)) {
    return {make("DropIndex", match[1].str()), true, "drop index"};
  }

  throw std::runtime_error("unrecognised command; type HELP for the supported syntax");
}

void printExpression(Expression const& expression, std::ostream& output, int indent = 0) {
  std::visit([&](auto const& value) {
    using T = std::decay_t<decltype(value)>;
    if constexpr(std::is_same_v<T, std::string>) output << std::quoted(value);
    else if constexpr(std::is_same_v<T, Symbol>) output << value.getName();
    else if constexpr(std::is_same_v<T, ComplexExpression>) {
      output << '(' << value.getHead().getName();
      auto const& arguments = value.getArguments();
      bool tableLike = value.getHead().getName() == "Table" || value.getHead().getName() == "List";
      for(size_t i = 0; i < arguments.size(); ++i) {
        if(tableLike) output << '\n' << std::string(static_cast<size_t>(indent + 2), ' ');
        else output << ' ';
        auto argument = value.cloneArgument(i);
        printExpression(argument, output, indent + 2);
      }
      if(tableLike && !arguments.empty()) output << '\n' << std::string(static_cast<size_t>(indent), ' ');
      output << ')';
    } else if constexpr(std::is_same_v<T, bool>) output << (value ? "true" : "false");
    else output << value;
  }, expression);
}

class EnginePipeline {
  using Evaluate = BOSSExpression* (*)(BOSSExpression*);
  struct Library {
    void* handle = nullptr;
    Evaluate evaluate = nullptr;
  };
  std::vector<Library> libraries;

public:
  explicit EnginePipeline(std::vector<std::string> const& paths) {
    for(auto const& path : paths) {
      void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NODELETE);
      if(handle == nullptr) throw std::runtime_error("cannot load " + path + ": " + dlerror());
      auto evaluate = reinterpret_cast<Evaluate>(dlsym(handle, "evaluate"));
      if(evaluate == nullptr) throw std::runtime_error(path + " does not export evaluate");
      libraries.push_back({handle, evaluate});
    }
  }

  ~EnginePipeline() {
    for(auto it = libraries.rbegin(); it != libraries.rend(); ++it) dlclose(it->handle);
  }

  Expression evaluate(Expression expression) {
    auto* wrapper = new BOSSExpression{std::move(expression)};
    for(auto const& library : libraries) {
      auto* previous = wrapper;
      wrapper = library.evaluate(wrapper);
      freeBOSSExpression(previous);
      if(wrapper == nullptr) throw std::runtime_error("engine returned a null expression");
    }
    Expression result = std::move(wrapper->delegate);
    freeBOSSExpression(wrapper);
    return result;
  }
};

void printHelp() {
  std::cout << R"HELP(
BOSS expressions (may span several lines):
  (CreateTable Accounts)
  (InsertInto Accounts (id 1) (owner "Ada") (balance 100.0))
  (Update Accounts
    (Table (id (List 1)))
    (Set (balance (Plus balance 25.0))))
  (Select Accounts (Where (Equal id 1)))

The parser accepts any well-formed BOSS expression. The selected transaction
and in-memory engines determine whether an operator can actually be executed.

Transaction/demo commands:
  BEGIN                 buffer subsequent writes
  COMMIT                submit the buffered logical transaction in order
  ABORT                 discard the buffer
  MODE LAZY|EAGER       defer writes, or flush after each accepted group
  WAL                   inspect pending deferred operations
  FLUSH [table id]      materialise all operations, or one row
  FLUSH CHAIN table id  row materialisation attributed to a chain bound
  CLEAR WAL             discard pending operations (debug/reset command)
  THRESHOLD n           set the automatic global WAL-entry threshold
  INSERT DRAIN n        set pending-insert drain threshold (0 disables it)
  INDEX STATS name      inspect an in-memory index
  STATS / RESET STATS   print or reset instrumentation
  DEMO ALL              run a guided end-to-end feature demonstration
  HELP, EXIT

Scope: this is a deterministic single-threaded prototype. BEGIN/COMMIT groups
write expressions and submits them in order. It does not claim atomic rollback,
concurrent isolation, durability, or recovery.
)HELP";
}

class DemoSession {
  EnginePipeline pipeline;
  bool eager = false;
  bool transactionOpen = false;
  std::vector<ParsedCommand> transaction;

  Expression execute(Expression expression, bool write, bool quiet = false) {
    auto result = pipeline.evaluate(std::move(expression));
    if(eager && write) result = pipeline.evaluate(make("FlushWAL"));
    if(!quiet) {
      std::cout << "=> ";
      printExpression(result, std::cout);
      std::cout << '\n';
    }
    return result;
  }

  void runGuidedDemo() {
    std::cout << "\n--- lazy insert, update, selective read, and WAL ---\n";
    run("MODE LAZY");
    run("(CreateTable products)");
    run("(InsertInto products (id 10) (name \"Keyboard\") (price 50.0) (stock 20))");
    run("(InsertInto products (id 20) (name \"Mouse\") (price 25.0) (stock 40))");
    run("(Update products (Table (id (List 10))) "
        "(Set (price (Plus price 5)) (stock (Minus stock 2))))");
    run("WAL");
    run("(Project (Select products (Where (Equal id 10))) "
        "(As id id name name price price stock stock))");

    std::cout << "\n--- buffered multi-operation transaction and ABORT ---\n";
    run("BEGIN");
    run("(Update products (Table (id (List 10))) (Set (stock (Minus stock 3))))");
    run("(Update products (Table (id (List 20))) (Set (stock (Plus stock 3))))");
    run("COMMIT");
    run("BEGIN");
    run("(Delete products (Table (id (List 20))))");
    run("ABORT");
    run("(Select products (Where (Equal id 20)))");

    std::cout << "\n--- same-row and cross-row/cross-column dependencies ---\n";
    run("(CreateTable orders)");
    run("(InsertInto orders (id 7) (quantity 2) (total 0.0))");
    run("(Update orders (Table (id (List 7))) "
        "(Set (total (Times quantity (Cell products id 10 price)))))");
    run("(Update products (Table (id (List 10))) "
        "(Set (price (Plus price 10)) (stock (Plus stock price))))");
    run("(Select orders (Where (Equal id 7)))");

    std::cout << "\n--- multi-row update, index lifecycle, delete, and eager mode ---\n";
    run("(Update products (Table (id (List 10 20))) (Set (stock (Plus stock 1))))");
    run("FLUSH");
    run("(CreateIndex \"product_name_idx\" products (By name id))");
    run("(Project (Select products (Where (Equal name \"Keyboard\"))) (As id id name name))");
    run("INDEX STATS product_name_idx");
    run("(DropIndex \"product_name_idx\")");
    run("THRESHOLD 1000");
    run("INSERT DRAIN 0");
    run("MODE EAGER");
    run("(Delete products (Table (id (List 20))))");
    run("(Select products (Where (Equal id 20)))");
    run("STATS");
    std::cout << "--- demonstration complete ---\n\n";
  }

public:
  explicit DemoSession(std::vector<std::string> paths) : pipeline(paths) {
    pipeline.evaluate(make("SetWALThreshold", int32_t{1000}));
  }

  bool run(std::string line) {
    line = trim(line);
    if(line.empty()) return true;
    if(!line.empty() && line.back() == ';') line.pop_back();
    auto command = upper(trim(line));

    if(command == "EXIT" || command == "QUIT") return false;
    if(command == "HELP") { printHelp(); return true; }
    if(command == "DEMO ALL") { runGuidedDemo(); return true; }
    if(command == "BEGIN") {
      if(transactionOpen) throw std::runtime_error("a transaction is already open");
      transactionOpen = true;
      transaction.clear();
      std::cout << "transaction opened; writes will be buffered\n";
      return true;
    }
    if(command == "ABORT") {
      if(!transactionOpen) throw std::runtime_error("ABORT requires an open transaction");
      auto count = transaction.size();
      transaction.clear();
      transactionOpen = false;
      std::cout << "transaction aborted; discarded " << count << " statement(s)\n";
      return true;
    }
    if(command == "COMMIT") {
      if(!transactionOpen) throw std::runtime_error("COMMIT requires an open transaction");
      auto buffered = std::move(transaction);
      transaction.clear();
      transactionOpen = false;
      auto oldEager = eager;
      eager = false;
      Expression result = Symbol("EmptyTransaction");
      try {
        for(auto& item : buffered) result = execute(std::move(item.expression), item.write, true);
      } catch(...) {
        eager = oldEager;
        throw;
      }
      eager = oldEager;
      if(eager && !buffered.empty()) result = pipeline.evaluate(make("FlushWAL"));
      std::cout << "COMMIT submitted " << buffered.size() << " buffered write expression(s) => ";
      printExpression(result, std::cout);
      std::cout << '\n';
      return true;
    }
    if(command == "MODE LAZY" || command == "MODE EAGER") {
      if(transactionOpen) throw std::runtime_error("change mode before BEGIN or after COMMIT/ABORT");
      eager = command == "MODE EAGER";
      std::cout << "mode: " << (eager ? "eager" : "lazy") << '\n';
      return true;
    }
    if(command == "WAL") { execute(make("GetWAL"), false); return true; }
    if(command == "FLUSH") { execute(make("FlushWAL"), false); return true; }
    {
      std::smatch match;
      std::regex const rowFlush(
        R"(^FLUSH\s+([A-Za-z_][A-Za-z0-9_]*)\s+(-?[0-9]+)$)", std::regex::icase);
      std::regex const chainFlush(
        R"(^FLUSH\s+CHAIN\s+([A-Za-z_][A-Za-z0-9_]*)\s+(-?[0-9]+)$)", std::regex::icase);
      if(std::regex_match(line, match, chainFlush) || std::regex_match(line, match, rowFlush)) {
        auto id = std::stoll(match[2].str());
        Expression row = id >= INT32_MIN && id <= INT32_MAX
          ? Expression(static_cast<int32_t>(id)) : Expression(static_cast<int64_t>(id));
        execute(make(command.starts_with("FLUSH CHAIN") ? "FlushWALChain" : "FlushWAL",
                     Symbol(match[1].str()), std::move(row)), false);
        return true;
      }
    }
    if(command == "CLEAR WAL") { execute(make("ClearWAL"), false); return true; }
    {
      std::smatch match;
      std::regex const threshold(R"(^THRESHOLD\s+([0-9]+)$)", std::regex::icase);
      std::regex const insertDrain(R"(^INSERT\s+DRAIN\s+([0-9]+)$)", std::regex::icase);
      if(std::regex_match(line, match, threshold) || std::regex_match(line, match, insertDrain)) {
        auto value = std::stoll(match[1].str());
        if(value > INT32_MAX) throw std::runtime_error("threshold is too large");
        execute(make(command.starts_with("THRESHOLD") ? "SetWALThreshold"
                                                       : "SetWALInsertDrainThreshold",
                     static_cast<int32_t>(value)), false);
        return true;
      }
    }
    if(command == "STATS") {
      execute(make("PrintWALStats", std::string("interactive-demo")), false);
      execute(make("PrintInMemoryStats", std::string("interactive-demo")), false);
      return true;
    }
    if(command == "RESET STATS") {
      execute(make("ResetWALStats"), false);
      execute(make("ResetInMemoryStats"), false);
      return true;
    }
    if(command.starts_with("INDEX STATS ")) {
      execute(make("GetIndexStats", trim(line.substr(12))), false);
      return true;
    }

    auto parsed = !line.empty() && line.front() == '('
      ? parseBoss(line)
      : parseSql(line); // retained as a compatibility input path
    if(transactionOpen) {
      if(!parsed.write) throw std::runtime_error("reads are not buffered; COMMIT or ABORT first");
      transaction.push_back(std::move(parsed));
      std::cout << "buffered " << transaction.back().label << " (#" << transaction.size() << ")\n";
    } else {
      execute(std::move(parsed.expression), parsed.write);
    }
    return true;
  }
};

std::string argumentValue(int argc, char** argv, std::string const& option) {
  for(int i = 1; i + 1 < argc; ++i) if(argv[i] == option) return argv[i + 1];
  return {};
}

} // namespace

int main(int argc, char** argv) try {
  auto transactionPath = argumentValue(argc, argv, "--transaction-engine");
  auto storagePath = argumentValue(argc, argv, "--storage-engine");
  if(transactionPath.empty()) transactionPath = "build/libBOSSTransactionLogEngine.so";
  if(storagePath.empty()) storagePath = "../BOSSInMemoryWriteEngine/build/libBOSSInMemoryWriteEngine.so";

  if(!std::filesystem::exists(transactionPath) || !std::filesystem::exists(storagePath)) {
    std::cerr << "Engine library not found. Build both engines or pass:\n"
              << "  --transaction-engine PATH --storage-engine PATH\n";
    return 2;
  }

  DemoSession session({std::filesystem::absolute(transactionPath).string(),
                       std::filesystem::absolute(storagePath).string()});
  std::cout << "BOSS lazy transaction demo (" << "single-threaded prototype" << ")\n"
            << "Type BOSS expressions directly; HELP shows examples and controls.\n";
  std::string line;
  std::string statement;
  while(std::cout << (statement.empty() ? "boss> " : "...> ") && std::getline(std::cin, line)) {
    if(!statement.empty()) statement.push_back('\n');
    statement += line;
    auto const candidate = trim(statement);
    if(!candidate.empty() && candidate.front() == '(' &&
       bossExpressionNeedsMoreInput(candidate)) {
      continue;
    }
    try {
      bool const keepRunning = session.run(statement);
      statement.clear();
      if(!keepRunning) break;
    } catch(std::exception const& error) {
      std::cerr << "error: " << error.what() << '\n';
      statement.clear();
    }
  }
  if(!trim(statement).empty()) std::cerr << "error: incomplete BOSS expression\n";
  return 0;
} catch(std::exception const& error) {
  std::cerr << "fatal: " << error.what() << '\n';
  return 1;
}
