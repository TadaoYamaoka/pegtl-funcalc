#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <tao/pegtl.hpp>

namespace pegtl = tao::pegtl;

// 抽象構文木（AST）のノード定義
class ASTNode {
public:
    virtual double eval(std::map<std::string, double>& vars) const = 0;
    virtual ~ASTNode() = default;
};

// 数値ノード
class NumberNode : public ASTNode {
    double value_;
public:
    NumberNode(double value) : value_(value) {}
    double eval(std::map<std::string, double>& vars) const override { return value_; }
};

// 変数参照ノード
class VariableNode : public ASTNode {
    std::string name_;
public:
    VariableNode(const std::string& name) : name_(name) {}
    double eval(std::map<std::string, double>& vars) const override {
        if (vars.find(name_) == vars.end())
            throw std::runtime_error("未定義の変数: " + name_);
        return vars[name_];
    }
};

// 二項演算子ノード
class BinaryOpNode : public ASTNode {
    char op_;
    std::unique_ptr<ASTNode> left_, right_;
public:
    BinaryOpNode(char op, std::unique_ptr<ASTNode> left, std::unique_ptr<ASTNode> right)
        : op_(op), left_(std::move(left)), right_(std::move(right)) {}

    double eval(std::map<std::string, double>& vars) const override {
        double lval = left_->eval(vars);
        double rval = right_->eval(vars);

        switch (op_) {
            case '+': return lval + rval;
            case '-': return lval - rval;
            case '*': return lval * rval;
            case '/': 
                if (rval == 0) throw std::runtime_error("ゼロ除算エラー");
                return lval / rval;
            default: throw std::runtime_error("無効な演算子");
        }
    }
};

// 関数呼び出しノード
class FunctionNode : public ASTNode {
    std::string name_;
    std::unique_ptr<ASTNode> arg_;
public:
    FunctionNode(const std::string& name, std::unique_ptr<ASTNode> arg)
        : name_(name), arg_(std::move(arg)) {}

    double eval(std::map<std::string, double>& vars) const override {
        double arg_val = arg_->eval(vars);

        if (name_ == "sin") return std::sin(arg_val);
        if (name_ == "cos") return std::cos(arg_val);
        if (name_ == "tan") return std::tan(arg_val);
        if (name_ == "sqrt") return std::sqrt(arg_val);
        if (name_ == "log") return std::log(arg_val);
        if (name_ == "exp") return std::exp(arg_val);
        if (name_ == "abs") return std::abs(arg_val);

        throw std::runtime_error("未定義の関数: " + name_);
    }
};

// 代入ノード
class AssignmentNode : public ASTNode {
    std::string name_;
    std::unique_ptr<ASTNode> expr_;
public:
    AssignmentNode(const std::string& name, std::unique_ptr<ASTNode> expr)
        : name_(name), expr_(std::move(expr)) {}

    double eval(std::map<std::string, double>& vars) const override {
        double val = expr_->eval(vars);
        vars[name_] = val;
        return val;
    }
};

// パーサーの状態
struct ParserState {
    std::vector<std::unique_ptr<ASTNode>> values;  // 値のスタック
    bool in_function_call = false;                 // 関数呼び出し処理中フラグ
    std::string current_function_name;             // 現在処理中の関数名
    char temp_op;                                  // 二項演算子一時保管用

    // 評価結果を取得
    std::unique_ptr<ASTNode> get_result() {
        if (values.size() != 1) {
            throw std::runtime_error("構文エラー: 式が正しく構築されていません");
        }
        return std::move(values.back());
    }
};

// 文法定義
namespace calculator {
    using namespace pegtl;

    // 前方宣言
    struct expr;

    // 基本ルール
    struct ws : one<' ', '\t'> {};
    struct spaces : star<ws> {};

    struct integer : plus<digit> {};
    struct decimal : seq<opt<one<'+', '-'>>, integer, opt<seq<one<'.'>, star<digit>>>> {};
    struct number : decimal {};

    // 識別子（基本ルール）
    struct identifier : seq<ranges<'a', 'z', 'A', 'Z'>, star<ranges<'a', 'z', 'A', 'Z', '0', '9'>>> {};

    // 変数参照用（右辺）
    struct var : identifier {};

    // 代入左辺用（区別のため）
    struct assign_id : identifier {};

    // 関数呼び出し
    struct func_name : identifier {};
    struct func_args : if_must<one<'('>, spaces, expr, spaces, one<')'>> {};
    struct function_call : seq<func_name, spaces, func_args> {};

    // 括弧式
    struct parenthesized : if_must<one<'('>, spaces, expr, spaces, one<')'>> {};

    // 一次式
    struct primary : sor<number, function_call, var, parenthesized> {};

    // 乗除算
    struct mul_operator : one<'*', '/'> {};
    struct mul_op : seq<spaces, mul_operator, spaces, primary> {};
    struct multiplication : seq<primary, star<mul_op>> {};

    // 加減算
    struct add_operator : one<'+', '-'> {};
    struct add_op : seq<spaces, add_operator, spaces, multiplication> {};
    struct addition : seq<multiplication, star<add_op>> {};

    // 代入
    struct assignment : seq<assign_id, spaces, one<'='>, spaces, expr> {};

    // 式
    struct expr : sor<assignment, addition> {};

    // 文法全体
    struct grammar : must<spaces, expr, spaces, eof> {};

    // アクション定義
    template<typename Rule>
    struct action : nothing<Rule> {};

    // 数値：NumberNode を生成
    template<>
    struct action<number> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.values.push_back(std::make_unique<NumberNode>(std::stod(in.string())));
        }
    };

    // 変数参照：VariableNode を生成
    template<>
    struct action<var> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.values.push_back(std::make_unique<VariableNode>(in.string()));
        }
    };

    // 代入左辺用 assign_id：何も行わない
    template<>
    struct action<assign_id> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            // assignment のアクションで入力全体から左辺を抽出するため、ここでは何もしない
        }
    };

    // 関数呼び出し：関数名の取得
    template<>
    struct action<func_name> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 関数名を記録
            state.current_function_name = in.string();
            state.in_function_call = true;
        }
    };

    // 関数呼び出し引数：終了処理
    template<>
    struct action<func_args> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 関数呼び出しの処理完了
            state.in_function_call = false;
        }
    };

    // 関数呼び出し全体：FunctionNode を生成
    template<>
    struct action<function_call> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (state.values.empty()) {
                throw std::runtime_error("関数呼び出しの構文が不正です");
            }
    
            // 引数は既にスタックの最上部にある
            auto arg = std::move(state.values.back());
            state.values.pop_back();
            // 関数ノードを作成してスタックに追加
            state.values.push_back(std::make_unique<FunctionNode>(state.current_function_name, std::move(arg)));
        }
    };

    // 加減算：演算子取得
    template<>
    struct action<add_operator> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.temp_op = in.string()[0];
        }
    };

    // 加減算：左結合で AST を構築
    template<>
    struct action<add_op> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (state.values.size() < 2)
                throw std::runtime_error("加算演算子の構文エラー");
            auto right = std::move(state.values.back());
            state.values.pop_back();
            auto left = std::move(state.values.back());
            state.values.pop_back();
            state.values.push_back(std::make_unique<BinaryOpNode>(state.temp_op, std::move(left), std::move(right)));
        }
    };

    // 乗除算：演算子取得
    template<>
    struct action<mul_operator> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.temp_op = in.string()[0];
        }
    };

    // 乗除算：左結合で AST を構築
    template<>
    struct action<mul_op> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (state.values.size() < 2)
                throw std::runtime_error("乗算演算子の構文エラー");
            auto right = std::move(state.values.back());
            state.values.pop_back();
            auto left = std::move(state.values.back());
            state.values.pop_back();
            state.values.push_back(std::make_unique<BinaryOpNode>(state.temp_op, std::move(left), std::move(right)));
        }
    };

    // 代入：入力全体から左辺を抽出して AssignmentNode を生成
    template<>
    struct action<assignment> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            if (state.values.empty())
                throw std::runtime_error("代入式の右辺がありません");
            auto right = std::move(state.values.back());
            state.values.pop_back();
            std::string input = in.string();
            size_t eq_pos = input.find('=');
            if (eq_pos == std::string::npos)
                throw std::runtime_error("代入式の構文エラー");
            std::string var_name = input.substr(0, eq_pos);
            var_name.erase(0, var_name.find_first_not_of(" \t"));
            var_name.erase(var_name.find_last_not_of(" \t") + 1);
            state.values.push_back(std::make_unique<AssignmentNode>(var_name, std::move(right)));
        }
    };
}

int main() {
    std::map<std::string, double> variables;

    std::cout << "C++20関数電卓 (exitで終了)" << std::endl;

    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line == "exit") break;
        if (line.empty()) continue;

        try {
            ParserState state;
            pegtl::memory_input input(line, "input");

            if (pegtl::parse<calculator::grammar, calculator::action>(input, state)) {
                auto result = state.get_result();
                if (result) {
                    double value = result->eval(variables);
                    std::cout << "= " << value << std::endl;
                } else {
                    std::cerr << "構文解析エラー：結果が得られませんでした" << std::endl;
                }
            }
        } catch (const pegtl::parse_error& e) {
            std::cerr << "構文エラー: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "エラー: " << e.what() << std::endl;
        }
    }

    return 0;
}
