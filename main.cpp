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
    std::vector<char> operators;                   // 演算子のスタック
    bool in_function_call = false;                 // 関数呼び出し処理中フラグ
    std::string current_function_name;             // 現在処理中の関数名

    // 演算子をプッシュ（優先順位を考慮）
    void push_operator(char op) {
        while (!operators.empty() && precedence(operators.back()) >= precedence(op) && operators.back() != '(') {
            apply_operator();
        }
        operators.push_back(op);
    }

    // 括弧内の式を終了
    void close_parenthesis() {
        while (!operators.empty() && operators.back() != '(') {
            apply_operator();
        }

        if (!operators.empty() && operators.back() == '(') {
            operators.pop_back(); // 開き括弧を削除
        }
    }

    // 最上位の演算子を適用
    void apply_operator() {
        if (operators.empty() || values.size() < 2) return;

        char op = operators.back();
        operators.pop_back();

        auto right = std::move(values.back());
        values.pop_back();

        auto left = std::move(values.back());
        values.pop_back();

        values.push_back(std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right)));
    }

    // 残りの演算子をすべて適用
    void apply_remaining_operators() {
        while (!operators.empty()) {
            if (operators.back() == '(') {
                operators.pop_back(); // 未対応の開き括弧は無視
                continue;
            }
            apply_operator();
        }
    }

    // 演算子の優先順位
    int precedence(char op) const {
        switch (op) {
            case '+':
            case '-': return 1;
            case '*':
            case '/': return 2;
            default: return 0;
        }
    }

    // 評価結果を取得
    std::unique_ptr<ASTNode> get_result() {
        apply_remaining_operators();

        if (values.empty()) {
            return nullptr;
        }
        if (values.size() > 1) {
            throw std::runtime_error("構文エラー: 複数の式が残っています");
        }

        return std::move(values[0]);
    }
};

// 文法定義
namespace calculator {
    using namespace pegtl;

    // 基本ルール
    struct ws : one<' ', '\t'> {};
    struct spaces : star<ws> {};

    struct plus_minus : one<'+', '-'> {};
    struct mult_div : one<'*', '/'> {};

    struct integer : plus<digit> {};
    struct decimal : seq<opt<one<'+', '-'>>, integer, opt<seq<one<'.'>, star<digit>>>> {};
    struct number : decimal {};

    struct identifier : seq<ranges<'a', 'z', 'A', 'Z'>, star<ranges<'a', 'z', 'A', 'Z', '0', '9'>>> {};

    // 前方宣言
    struct expr;

    // 関数呼び出し - 関数名を明示的に捕捉
    struct func_name : identifier {};
    struct func_args : if_must<one<'('>, spaces, expr, spaces, one<')'>> {};
    struct function_call : seq<func_name, spaces, func_args> {};

    // 括弧式
    struct parenthesized : if_must<one<'('>, spaces, expr, spaces, one<')'>> {};

    // 一次式
    struct primary : sor<number, function_call, identifier, parenthesized> {};

    // 乗除算
    struct multiplication : list<primary, seq<spaces, mult_div, spaces>> {};

    // 加減算
    struct addition : list<multiplication, seq<spaces, plus_minus, spaces>> {};

    // 代入
    struct assignment : seq<identifier, spaces, one<'='>, spaces, expr> {};

    // 式
    struct expr : sor<assignment, addition> {};

    // 文法全体
    struct grammar : must<spaces, expr, spaces, eof> {};

    // アクション定義
    template<typename Rule>
    struct action : nothing<Rule> {};

    template<>
    struct action<number> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.values.push_back(std::make_unique<NumberNode>(std::stod(in.string())));
        }
    };

    template<>
    struct action<identifier> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 関数呼び出し内部では何もしない
            if (state.in_function_call) return;

            // 変数として処理
            state.values.push_back(std::make_unique<VariableNode>(in.string()));
        }
    };

    template<>
    struct action<func_name> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 関数名を記録
            state.current_function_name = in.string();
            state.in_function_call = true;
        }
    };

    template<>
    struct action<func_args> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 関数呼び出しの処理完了
            state.in_function_call = false;
        }
    };

    template<>
    struct action<function_call> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            if (state.values.size() < 2) {
                throw std::runtime_error("関数呼び出しの構文が不正です");
            }
    
            // 引数は既にスタックの最上部にある
            auto arg = std::move(state.values.back());
            state.values.pop_back();
    
            // 関数名のVariableNodeをスタックから取り除く（ここを追加）
            state.values.pop_back();
    
            // 関数ノードを作成してスタックに追加
            state.values.push_back(std::make_unique<FunctionNode>(
                state.current_function_name, std::move(arg)));
        }
    };
    
    template<>
    struct action<one<'('>> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 関数呼び出しの一部でない場合のみ演算子として処理
            if (!state.in_function_call) {
                state.operators.push_back('(');
            }
        }
    };

    template<>
    struct action<one<')'>> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 関数呼び出しの一部でない場合のみ括弧閉じとして処理
            if (!state.in_function_call) {
                state.close_parenthesis();
            }
        }
    };

    template<>
    struct action<plus_minus> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.push_operator(in.string()[0]);
        }
    };

    template<>
    struct action<mult_div> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.push_operator(in.string()[0]);
        }
    };

    template<>
    struct action<assignment> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            // 代入式の右辺
            state.apply_remaining_operators();
            if (state.values.empty()) {
                throw std::runtime_error("代入式の右辺がありません");
            }
            auto right = std::move(state.values.back());
            state.values.pop_back();
    
            // 左辺の変数ノードをスタックから取り除く（ここを追加）
            if (state.values.empty()) {
                throw std::runtime_error("代入式の左辺がありません");
            }
            state.values.pop_back();  // 左辺のVariableNodeを削除
    
            // 左辺の変数名を抽出（変更なし）
            std::string input = in.string();
            size_t eq_pos = input.find('=');
            if (eq_pos == std::string::npos) return;
    
            std::string var_name = input.substr(0, eq_pos);
            // 先頭と末尾の空白を削除
            var_name.erase(0, var_name.find_first_not_of(" \t"));
            var_name.erase(var_name.find_last_not_of(" \t") + 1);
    
            // 代入ノードを作成
            state.values.push_back(std::make_unique<AssignmentNode>(var_name, std::move(right)));
        }
    };
    
    template<>
    struct action<expr> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.apply_remaining_operators();
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
