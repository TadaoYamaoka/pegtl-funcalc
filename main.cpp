#include <iostream>
#include <map>
#include <cmath>
#include <numbers>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <tao/pegtl.hpp>
#include <boost/rational.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace pegtl = tao::pegtl;
namespace mp = boost::multiprecision;

// 有理数型の定義
using Rational = boost::rational<mp::cpp_int>;
// 高精度浮動小数点型（超越関数計算用）
using cpp_dec_float = mp::cpp_dec_float_100;
// 変換精度（10^100の精度）
const mp::cpp_int precision = mp::pow(mp::cpp_int(10), 100);

// 文字列から有理数への変換ヘルパー関数
Rational stringToRational(const std::string& str) {
    // 科学的表記法を扱う場合は一時的に高精度浮動小数点数へ変換
    if (str.find('e') != std::string::npos || str.find('E') != std::string::npos) {
        cpp_dec_float d(str);
        // 十分な精度で有理数に近似
        return Rational(mp::cpp_int(d * cpp_dec_float(precision)), precision);
    }

    // 小数点を含む場合は分子と分母を計算
    size_t dot_pos = str.find('.');
    if (dot_pos != std::string::npos) {
        std::string int_part = str.substr(0, dot_pos);
        std::string frac_part = str.substr(dot_pos + 1);

        // 符号の処理
        bool negative = false;
        if (!int_part.empty() && int_part[0] == '-') {
            negative = true;
            int_part = int_part.substr(1);
        }

        // 分子 = 整数部 * 10^(小数部の桁数) + 小数部
        mp::cpp_int numerator = 0;
        if (!int_part.empty()) {
            numerator = mp::cpp_int(int_part);
        }

        mp::cpp_int multiplier = 1;
        for (size_t i = 0; i < frac_part.length(); ++i) {
            multiplier *= 10;
        }

        numerator = numerator * multiplier;
        if (!frac_part.empty()) {
            numerator += mp::cpp_int(frac_part);
        }

        // 分母 = 10^(小数部の桁数)
        mp::cpp_int denominator = multiplier;

        // 符号を適用
        if (negative) {
            numerator = -numerator;
        }

        return Rational(numerator, denominator);
    } else {
        // 整数の場合
        return Rational(mp::cpp_int(str));
    }
}

// ユーザー定義関数を保持するためのグローバル定義
struct UserFunction {
    std::vector<std::string> parameters;
    std::unique_ptr<class ASTNode> body;
};

std::map<std::string, UserFunction> userFunctions;

// 抽象構文木（AST）のノード定義
class ASTNode {
public:
    virtual Rational eval(std::map<std::string, Rational>& vars) = 0;
    virtual ~ASTNode() = default;
};

// 数値ノード
class NumberNode : public ASTNode {
    Rational value_;
public:
    NumberNode(const std::string& value) : value_(stringToRational(value)) {}
    Rational eval(std::map<std::string, Rational>& /*vars*/) override { return value_; }
};

// 変数参照ノード
class VariableNode : public ASTNode {
    std::string name_;
public:
    VariableNode(const std::string& name) : name_(name) {}
    Rational eval(std::map<std::string, Rational>& vars) override {
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

    Rational eval(std::map<std::string, Rational>& vars) override {
        Rational lval = left_->eval(vars);
        Rational rval = right_->eval(vars);

        switch (op_) {
            case '+': return lval + rval;
            case '-': return lval - rval;
            case '*': return lval * rval;
            case '/':
                if (rval.numerator() == 0)
                    throw std::runtime_error("ゼロ除算エラー");
                return lval / rval;
            case '^': {
                // べき乗は整数の場合のみ正確に計算
                if (rval.denominator() == 1) {
                    mp::cpp_int exp = rval.numerator();
                    bool neg_exp = exp < 0;
                    exp = mp::abs(exp);

                    // 底が0の場合の特殊処理
                    if (lval.numerator() == 0) {
                        if (exp == 0) {
                            // 0^0 を1として定義（数学的な慣習に従う）
                            return Rational(1);
                        } else if (neg_exp) {
                            throw std::domain_error("0の負のべき乗は定義されていません");
                        } else {
                            // 0^n (n>0) = 0
                            return Rational(0);
                        }
                    }

                    // 整数のべき乗を計算
                    if (exp <= std::numeric_limits<unsigned int>::max()) {
                        // boost::multiprecisionの特化版を利用
                        unsigned int uint_exp = static_cast<unsigned int>(exp.convert_to<unsigned int>());
                        if (neg_exp) {
                            return Rational(1) / Rational(mp::pow(lval.numerator(), uint_exp),
                                                         mp::pow(lval.denominator(), uint_exp));
                        } else {
                            return Rational(mp::pow(lval.numerator(), uint_exp),
                                           mp::pow(lval.denominator(), uint_exp));
                        }
                    } else {
                        // 大きな指数の場合はループ計算
                        Rational result(1);
                        for (mp::cpp_int i = 0; i < exp; ++i) {
                            result *= lval;
                        }

                        if (neg_exp) {
                            return Rational(1) / result;
                        }
                        return result;
                    }
                } else {
                    // 底が負の場合、指数が有理数だとドメインエラー
                    if (lval < 0) {
                        throw std::domain_error("負の底の有理数べき乗は実数の範囲で定義されていません");
                    }

                    // 底が0で指数が0より小さい場合はドメインエラー
                    if (lval.numerator() == 0 && rval < 0) {
                        throw std::domain_error("0の負のべき乗は定義されていません");
                    }

                    // 有理数のべき乗は高精度浮動小数点を使って近似
                    cpp_dec_float lval_f = cpp_dec_float(lval.numerator()) / cpp_dec_float(lval.denominator());
                    cpp_dec_float rval_f = cpp_dec_float(rval.numerator()) / cpp_dec_float(rval.denominator());
                    cpp_dec_float result_f = mp::pow(lval_f, rval_f);

                    // 結果を有理数に変換
                    cpp_dec_float scaled = result_f * cpp_dec_float(precision);
                    return Rational(mp::cpp_int(scaled), precision);
                }
            }
            case '%':
                if (rval.numerator() == 0)
                    throw std::runtime_error("ゼロでの剰余演算エラー");
                // 浮動小数点数に変換してから計算
                {
                    cpp_dec_float lval_f = cpp_dec_float(lval.numerator()) / cpp_dec_float(lval.denominator());
                    cpp_dec_float rval_f = cpp_dec_float(rval.numerator()) / cpp_dec_float(rval.denominator());
                    cpp_dec_float result_f = mp::fmod(lval_f, rval_f);

                    // 結果を有理数に変換
                    return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
                }
            default:
                throw std::runtime_error("無効な演算子");
        }
    }
};

// 単項演算子ノード
class UnaryOpNode : public ASTNode {
    char op_;
    std::unique_ptr<ASTNode> operand_;
public:
    UnaryOpNode(char op, std::unique_ptr<ASTNode> operand)
        : op_(op), operand_(std::move(operand)) {}
    Rational eval(std::map<std::string, Rational>& vars) override {
        Rational val = operand_->eval(vars);
        if (op_ == '-') return -val;
        if (op_ == '+') return val;
        throw std::runtime_error("無効な単項演算子");
    }
};

// 組み込み単一引数関数ノード（加えてユーザー定義関数をチェック）
class FunctionNode : public ASTNode {
    std::string name_;
    std::unique_ptr<ASTNode> arg_;
public:
    FunctionNode(const std::string& name, std::unique_ptr<ASTNode> arg)
        : name_(name), arg_(std::move(arg)) {}

    Rational eval(std::map<std::string, Rational>& vars) override {
        Rational arg_val = arg_->eval(vars);

        // 数学関数
        if (name_ == "sqrt") {
            // 非負のチェック
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            if (arg_f < 0)
                throw std::domain_error("平方根の引数は非負の値である必要があります");
            cpp_dec_float result_f = mp::sqrt(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "log") {
            // 正のチェック
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            if (arg_f <= 0)
                throw std::domain_error("対数関数の引数は正の値である必要があります");
            cpp_dec_float result_f = mp::log(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "exp") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::exp(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "abs") {
            // absは有理数でそのまま計算可能
            return Rational(mp::abs(arg_val.numerator()), mp::abs(arg_val.denominator()));
        }
        else if (name_ == "log2") {
            // 正のチェック
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            if (arg_f <= 0)
                throw std::domain_error("log2関数の引数は正の値である必要があります");
            cpp_dec_float result_f = mp::log(arg_f) / mp::log(cpp_dec_float(2));
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "log10") {
            // 正のチェック
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            if (arg_f <= 0)
                throw std::domain_error("log10関数の引数は正の値である必要があります");
            cpp_dec_float result_f = mp::log10(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "cbrt") {
            // cbrtは任意の値で定義されています
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::pow(arg_f, cpp_dec_float(1) / cpp_dec_float(3));
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "exp2") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::pow(cpp_dec_float(2), arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "expm1") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::exp(arg_f) - cpp_dec_float(1);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "log1p") {
            // x > -1のチェック
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            if (arg_f <= -1)
                throw std::domain_error("log1p関数の引数は-1より大きい値である必要があります");
            cpp_dec_float result_f = mp::log(cpp_dec_float(1) + arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        // 三角関数
        else if (name_ == "sin") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::sin(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "cos") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::cos(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "tan") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            // tanは (pi/2) + n*pi で定義されていない（無限大）
            // しかし、浮動小数点での完全なチェックは難しいため、
            // 実装に任せて、結果が異常に大きい場合は後で確認する
            cpp_dec_float result_f = mp::tan(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        // 逆三角関数
        else if (name_ == "asin") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            // asinの定義域は[-1, 1]
            if (arg_f < -1 || arg_f > 1)
                throw std::domain_error("asin関数の引数は-1から1の範囲内である必要があります");
            cpp_dec_float result_f = mp::asin(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "acos") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            // acosの定義域は[-1, 1]
            if (arg_f < -1 || arg_f > 1)
                throw std::domain_error("acos関数の引数は-1から1の範囲内である必要があります");
            cpp_dec_float result_f = mp::acos(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "atan") {
            // atanはすべての実数で定義されている
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::atan(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        // 双曲線関数
        else if (name_ == "sinh") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::sinh(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "cosh") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::cosh(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        else if (name_ == "tanh") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::tanh(arg_f);
            return Rational(mp::cpp_int(result_f * cpp_dec_float(precision)), precision);
        }
        // 丸め関数
        else if (name_ == "ceil") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::ceil(arg_f);
            // 整数値として返す
            return Rational(mp::cpp_int(result_f), 1);
        }
        else if (name_ == "floor") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::floor(arg_f);
            // 整数値として返す
            return Rational(mp::cpp_int(result_f), 1);
        }
        else if (name_ == "round") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::round(arg_f);
            // 整数値として返す
            return Rational(mp::cpp_int(result_f), 1);
        }
        else if (name_ == "trunc") {
            cpp_dec_float arg_f = cpp_dec_float(arg_val.numerator()) / cpp_dec_float(arg_val.denominator());
            cpp_dec_float result_f = mp::trunc(arg_f);
            // 整数値として返す
            return Rational(mp::cpp_int(result_f), 1);
        }
        // ユーザー定義関数（1引数）
        else {
            auto it = userFunctions.find(name_);
            if (it != userFunctions.end()) {
                if (it->second.parameters.size() != 1)
                    throw std::runtime_error("関数 " + name_ + " は1引数関数ではありません");
                std::map<std::string, Rational> local_vars = vars;
                local_vars[it->second.parameters[0]] = arg_val;
                return it->second.body->eval(local_vars);
            }
            throw std::runtime_error("未定義の関数: " + name_);
        }
    }
};

// 組み込み複数引数関数ノード
class MultiArgFunctionNode : public ASTNode {
    std::string name_;
    std::vector<std::unique_ptr<ASTNode>> args_;
public:
    MultiArgFunctionNode(const std::string& name, std::vector<std::unique_ptr<ASTNode>>&& args)
        : name_(name), args_(std::move(args)) {}

    Rational eval(std::map<std::string, Rational>& vars) override {
        std::vector<Rational> arg_vals;
        for (const auto& arg : args_) {
            arg_vals.push_back(arg->eval(vars));
        }

        if (name_ == "pow") {
            if (arg_vals.size() != 2)
                throw std::runtime_error("pow関数は2つの引数が必要です");

            // 底が0の場合の特殊処理
            if (arg_vals[0].numerator() == 0) {
                // 底が0で指数が0の場合は1を返す（数学的慣習）
                if (arg_vals[1].numerator() == 0 && arg_vals[1].denominator() == 1) {
                    return Rational(1);
                }
                // 底が0で指数が負の場合はドメインエラー
                if (arg_vals[1] < 0) {
                    throw std::domain_error("0の負のべき乗は定義されていません");
                }
                // 底が0で指数が正の場合は0を返す
                return Rational(0);
            }

            // 底が負で指数が有理数の場合はドメインエラー
            if (arg_vals[0] < 0 && arg_vals[1].denominator() != 1) {
                throw std::domain_error("負の底の有理数べき乗は実数の範囲で定義されていません");
            }

            // pow関数は整数のべき乗の場合、専用の実装を使用
            if (arg_vals[1].denominator() == 1) {
                mp::cpp_int exp = arg_vals[1].numerator();
                bool neg_exp = exp < 0;
                exp = mp::abs(exp);

                // 小さい指数の場合はBoostの特化版を使用
                if (exp <= std::numeric_limits<unsigned int>::max()) {
                    unsigned int uint_exp = static_cast<unsigned int>(exp.convert_to<unsigned int>());
                    Rational result;

                    if (neg_exp) {
                        result = Rational(1) / Rational(
                            mp::pow(arg_vals[0].numerator(), uint_exp),
                            mp::pow(arg_vals[0].denominator(), uint_exp)
                        );
                    } else {
                        result = Rational(
                            mp::pow(arg_vals[0].numerator(), uint_exp),
                            mp::pow(arg_vals[0].denominator(), uint_exp)
                        );
                    }
                    return result;
                } else {
                    // 大きな指数の場合はループ計算
                    Rational result(1);
                    for (mp::cpp_int i = 0; i < exp; ++i) {
                        result *= arg_vals[0];
                    }

                    if (neg_exp) {
                        return Rational(1) / result;
                    }
                    return result;
                }
            } else {
                // 有理数指数の場合は高精度浮動小数点数を使用
                cpp_dec_float base = cpp_dec_float(arg_vals[0].numerator()) / cpp_dec_float(arg_vals[0].denominator());
                cpp_dec_float exponent = cpp_dec_float(arg_vals[1].numerator()) / cpp_dec_float(arg_vals[1].denominator());
                cpp_dec_float result = mp::pow(base, exponent);

                return Rational(mp::cpp_int(result * cpp_dec_float(precision)), precision);
            }
        }

        if (name_ == "max") {
            if (arg_vals.empty())
                throw std::runtime_error("max関数には少なくとも1つの引数が必要です");
            Rational max_val = arg_vals[0];
            for (size_t i = 1; i < arg_vals.size(); ++i)
                if (arg_vals[i] > max_val) max_val = arg_vals[i];
            return max_val;
        }

        if (name_ == "min") {
            if (arg_vals.empty())
                throw std::runtime_error("min関数には少なくとも1つの引数が必要です");
            Rational min_val = arg_vals[0];
            for (size_t i = 1; i < arg_vals.size(); ++i)
                if (arg_vals[i] < min_val) min_val = arg_vals[i];
            return min_val;
        }

        // atan2関数のサポート
        if (name_ == "atan2") {
            if (arg_vals.size() != 2)
                throw std::runtime_error("atan2関数は2つの引数が必要です");

            cpp_dec_float y = cpp_dec_float(arg_vals[0].numerator()) / cpp_dec_float(arg_vals[0].denominator());
            cpp_dec_float x = cpp_dec_float(arg_vals[1].numerator()) / cpp_dec_float(arg_vals[1].denominator());

            // atan2(0,0)は数学的に未定義
            if (y == 0 && x == 0)
                throw std::domain_error("atan2(0,0)は定義されていません");

            cpp_dec_float result = mp::atan2(y, x);

            return Rational(mp::cpp_int(result * cpp_dec_float(precision)), precision);
        }

        // ldexp関数のサポート
        if (name_ == "ldexp") {
            if (arg_vals.size() != 2)
                throw std::runtime_error("ldexp関数は2つの引数が必要です");
            if (arg_vals[1].denominator() != 1)
                throw std::domain_error("ldexpの第2引数は整数である必要があります");

            int exp = static_cast<int>(arg_vals[1].numerator().convert_to<int>());
            cpp_dec_float x = cpp_dec_float(arg_vals[0].numerator()) / cpp_dec_float(arg_vals[0].denominator());
            cpp_dec_float result = mp::ldexp(x, exp);

            return Rational(mp::cpp_int(result * cpp_dec_float(precision)), precision);
        }

        // ユーザー定義関数（複数引数）
        auto it = userFunctions.find(name_);
        if (it != userFunctions.end()) {
            if (it->second.parameters.size() != arg_vals.size())
                throw std::runtime_error("関数 " + name_ + " の引数の数が一致しません");
            std::map<std::string, Rational> local_vars = vars;
            for (size_t i = 0; i < arg_vals.size(); ++i) {
                local_vars[it->second.parameters[i]] = arg_vals[i];
            }
            return it->second.body->eval(local_vars);
        }
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

    Rational eval(std::map<std::string, Rational>& vars) override {
        Rational val = expr_->eval(vars);
        vars[name_] = val;
        return val;
    }
};

// ユーザー定義関数ノード（定義時にグローバルへ登録）
class FunctionDefinitionNode : public ASTNode {
    std::string name_;
    std::vector<std::string> parameters_;
    mutable std::unique_ptr<ASTNode> body_;
public:
    FunctionDefinitionNode(const std::string& name, std::vector<std::string>&& params, std::unique_ptr<ASTNode> body)
        : name_(name), parameters_(std::move(params)), body_(std::move(body)) {}

    Rational eval(std::map<std::string, Rational>& /*vars*/) override {
        userFunctions[name_] = UserFunction{ parameters_, std::move(body_) };
        std::cout << "関数 " << name_ << " を定義しました。" << std::endl;
        return Rational(0);
    }
};

// パーサーの状態
struct ParserState {
    std::vector<std::unique_ptr<ASTNode>> values;  // 値のスタック
    std::vector<std::string> function_names;       // 関数名のスタック
    std::vector<char> ops;                         // 演算子のスタック
    char temp_unary_op = 0;                        // 単項演算子の一時保管

    // 複数引数関数用のスタック
    std::vector<std::vector<std::unique_ptr<ASTNode>>> arg_lists;
    std::string current_func_name;
    std::vector<std::string> current_params;

    // 演算子を追加
    void push_operator(char op) {
        ops.push_back(op);
    }

    // 最後の演算子を取得して削除
    char pop_operator() {
        if (ops.empty())
            throw std::runtime_error("演算子スタックが空です");
        char op = ops.back();
        ops.pop_back();
        return op;
    }

    // 評価結果を取得
    std::unique_ptr<ASTNode> get_result() {
        if (values.size() != 1)
            throw std::runtime_error("構文エラー: 式が正しく構築されていません");
        return std::move(values.back());
    }

    // 現在の引数リストに引数を追加
    void add_argument() {
        if (arg_lists.empty())
            arg_lists.push_back({});
        if (values.empty())
            throw std::runtime_error("関数引数の構文エラー");
        arg_lists.back().push_back(std::move(values.back()));
        values.pop_back();
    }

    // 新しい引数リストを開始
    void start_arg_list() {
        arg_lists.push_back({});
    }

    // 引数リストを取得して削除
    std::vector<std::unique_ptr<ASTNode>> get_arg_list() {
        if (arg_lists.empty())
            return {};
        auto result = std::move(arg_lists.back());
        arg_lists.pop_back();
        return result;
    }
};

// 文法定義
namespace calculator {
    using namespace pegtl;

    // 前方宣言
    struct expression;
    struct statement;
    struct addition;
    struct assignment;

    // 基本ルール
    struct ws : one<' ', '\t'> {};
    struct spaces : star<ws> {};

    struct integer : plus<digit> {};

    // 科学的表記法を含む小数点数
    struct decimal : seq<
        opt<one<'+', '-'>>,
        integer,
        opt<seq<one<'.'>, star<digit>>>,
        opt<seq<
            one<'e', 'E'>,
            opt<one<'+', '-'>>,
            integer
        >>
    > {};

    struct number : decimal {};

    // 識別子（基本ルール）
    struct identifier : seq<ranges<'a', 'z', 'A', 'Z'>, star<ranges<'a', 'z', 'A', 'Z', '0', '9'>>> {};

    // 変数参照用（右辺）
    struct var : identifier {};

    // 代入左辺用（区別のため）
    struct assign_id : identifier {};

    // 関数定義用
    struct func_def_name : identifier {};
    struct param_identifier : identifier {};
    struct parameter_list : seq<param_identifier, star<seq<spaces, one<','>, spaces, param_identifier>>> {};

    // 関数定義：例 f(x)= expression
    struct function_definition : seq<
        func_def_name, spaces,
        one<'('>, spaces, opt<parameter_list>, spaces, one<')'>, spaces,
        one<'='>, spaces, expression
    > {};

    // 関数呼び出し用
    struct func_name;
    struct func_args;
    struct func_arg_list;
    struct function_call;

    // 括弧式
    struct parenthesized : if_must<one<'('>, spaces, expression, spaces, one<')'>> {};

    // 一次式
    struct primary : sor<number, function_call, var, parenthesized> {};

    // 単項演算子および単項式
    struct unary_operator : one<'-', '+'> {};
    struct prefixed_unary : seq<unary_operator, spaces, sor<primary, prefixed_unary>> {};
    struct unary : sor<prefixed_unary, primary> {};

    // べき乗演算子（右結合）用の定義
    // まず、べき乗の開始位置を示すマーカーを入れる
    struct marker : success {};
    // べき乗の右辺部分： '^' と単項式の組
    struct power_tail : star<seq<spaces, one<'^'>, spaces, unary>> {};
    // べき乗は、マーカー、単項式、続く power_tail からなる
    struct power : seq< marker, unary, power_tail > {};

    // 乗除算（明示的な演算子）用
    struct mul_operator : one<'*', '/', '%'> {};
    struct mul_op : seq<spaces, mul_operator, spaces, power> {};

    // 暗黙の乗算ルール：直後が数字、識別子、または '(' で始まる場合に適用
    struct implicit_mul : seq<spaces, at<sor<digit, identifier, one<'('>>>, power> {};

    // 明示的な乗算と暗黙の乗算のいずれかを複数繰り返し可能にする
    struct multiplication : seq<power, star<sor<mul_op, implicit_mul>>> {};

    // 加減算
    struct add_operator : one<'+', '-'> {};
    struct add_op : seq<spaces, add_operator, spaces, multiplication> {};
    struct addition : seq<multiplication, star<add_op>> {};

    // 代入
    struct assignment : seq<assign_id, spaces, one<'='>, spaces, expression> {};

    // expression：代入または加減算
    struct expression : sor<assignment, addition> {};

    // statement：関数定義または expression
    struct statement : sor<function_definition, expression> {};

    // 関数呼び出し
    struct func_arg : expression {};
    struct arg_sep : seq<spaces, one<','>, spaces> {};
    struct func_arg_tail : seq<arg_sep, func_arg> {};
    struct func_arg_list : seq<func_arg, star<func_arg_tail>> {};
    struct func_args : if_must<one<'('>, spaces, opt<func_arg_list>, spaces, one<')'>> {};
    struct function_call : if_must<
        at<seq<identifier, spaces, one<'('>>>,
        seq<func_name, spaces, func_args>
    > {};

    // 関数名
    struct func_name : identifier {};

    // 文法全体
    struct grammar : must<spaces, statement, spaces, eof> {};

    // アクション定義
    template<typename Rule>
    struct action : nothing<Rule> {};

    // 数値：NumberNode を生成
    template<>
    struct action<number> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.values.push_back(std::make_unique<NumberNode>(in.string()));
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

    // 関数呼び出し：関数名をスタックにプッシュ
    template<>
    struct action<func_name> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.function_names.push_back(in.string());
            state.start_arg_list();  // 新しい引数リストを開始
        }
    };

    // 最初の引数の処理
    template<>
    struct action<func_arg> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (!state.arg_lists.empty() && !state.values.empty())
                state.add_argument();
        }
    };

    // 追加の引数の処理
    template<>
    struct action<func_arg_tail> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            // 既に最初の引数は処理済み
        }
    };

    // 関数引数リスト終了処理（空のアクション）
    template<>
    struct action<func_args> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& /*state*/) {}
    };

    // 関数呼び出し全体：FunctionNode または MultiArgFunctionNode を生成
    template<>
    struct action<function_call> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (state.function_names.empty())
                throw std::runtime_error("関数名スタックが空です");

            std::string fname = state.function_names.back();
            state.function_names.pop_back();

            auto args = state.get_arg_list();

            if (args.size() == 1)
                state.values.push_back(std::make_unique<FunctionNode>(fname, std::move(args[0])));
            else
                state.values.push_back(std::make_unique<MultiArgFunctionNode>(fname, std::move(args)));
        }
    };

    // 単項演算子：演算子を記録
    template<>
    struct action<unary_operator> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.temp_unary_op = in.string()[0];
        }
    };

    // 単項演算子付き式：UnaryOpNode を生成
    template<>
    struct action<prefixed_unary> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (state.values.empty())
                throw std::runtime_error("単項演算子の構文エラー");
            auto operand = std::move(state.values.back());
            state.values.pop_back();
            state.values.push_back(std::make_unique<UnaryOpNode>(state.temp_unary_op, std::move(operand)));
        }
    };

    // べき乗演算（右結合）用アクション
    // marker ルール：スタックに nullptr をプッシュ
    template<>
    struct action<marker> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            state.values.push_back(nullptr);
        }
    };

    // power ルール：marker から始まる部分の AST を右結合で畳み込む
    template<>
    struct action<power> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            // state.values に、marker の後に並んだ AST ノードが積まれている
            std::vector<std::unique_ptr<ASTNode>> nodes;
            while (!state.values.empty()) {
                auto node = std::move(state.values.back());
                state.values.pop_back();
                if (!node) { // marker が見つかった
                    break;
                }
                nodes.push_back(std::move(node));
            }
            // nodes は逆順になっているので、元の順に戻す
            std::reverse(nodes.begin(), nodes.end());
            if (nodes.empty()) {
                throw std::runtime_error("べき乗演算子の構文エラー: ノードがありません");
            }
            // 右結合：右端から畳み込む
            std::unique_ptr<ASTNode> result = std::move(nodes.back());
            nodes.pop_back();
            while (!nodes.empty()) {
                result = std::make_unique<BinaryOpNode>('^', std::move(nodes.back()), std::move(result));
                nodes.pop_back();
            }
            state.values.push_back(std::move(result));
        }
    };

    // 加減算：演算子を記録
    template<>
    struct action<add_operator> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.push_operator(in.string()[0]);
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
            state.values.push_back(std::make_unique<BinaryOpNode>(state.pop_operator(), std::move(left), std::move(right)));
        }
    };

    // 乗除算（明示的な演算子）の処理
    template<>
    struct action<mul_operator> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.push_operator(in.string()[0]);
        }
    };

    // 乗除算：左結合で AST を構築（明示的な演算子）
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
            state.values.push_back(std::make_unique<BinaryOpNode>(state.pop_operator(), std::move(left), std::move(right)));
        }
    };

    // 暗黙の乗算：'*' を自動的に挿入する
    template<>
    struct action<implicit_mul> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (state.values.size() < 2)
                throw std::runtime_error("暗黙の乗算の構文エラー");
            auto right = std::move(state.values.back());
            state.values.pop_back();
            auto left = std::move(state.values.back());
            state.values.pop_back();
            state.values.push_back(std::make_unique<BinaryOpNode>('*', std::move(left), std::move(right)));
        }
    };

    // 代入：右辺を取り出し、変数名を抽出して AssignmentNode を生成
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

    // 代入左辺用 assign_id：何も行わない
    template<>
    struct action<assign_id> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {}
    };

    // 関数定義
    template<>
    struct action<func_def_name> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.current_func_name = in.string();
        }
    };

    template<>
    struct action<param_identifier> {
        template<typename ActionInput>
        static void apply(const ActionInput& in, ParserState& state) {
            state.current_params.push_back(in.string());
        }
    };

    template<>
    struct action<function_definition> {
        template<typename ActionInput>
        static void apply(const ActionInput& /*in*/, ParserState& state) {
            if (state.values.empty())
                throw std::runtime_error("関数定義の右辺がありません");
            auto body = std::move(state.values.back());
            state.values.pop_back();
            state.values.push_back(std::make_unique<FunctionDefinitionNode>(
                state.current_func_name, std::move(state.current_params), std::move(body)
            ));
            state.current_func_name.clear();
            state.current_params.clear();
        }
    };
}

int main() {
    std::map<std::string, Rational> variables;

    // 定数の事前定義
    cpp_dec_float pi_value = boost::math::constants::pi<cpp_dec_float>();
    cpp_dec_float e_value = boost::math::constants::e<cpp_dec_float>();
    variables["pi"] = Rational(mp::cpp_int(pi_value * cpp_dec_float(precision)), precision);
    variables["e"]  = Rational(mp::cpp_int(e_value * cpp_dec_float(precision)), precision);

    std::cout << "有理数関数電卓 (exitで終了)" << std::endl;

    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line == "exit") break;
        if (line.empty()) continue;

        try {
            ParserState state;
            pegtl::memory_input input(line, "input");

            if (pegtl::parse<calculator::grammar, calculator::action>(input, state)) {
                auto result = state.get_result();
                Rational value = result->eval(variables);

                // FunctionDefinitionNode の場合は結果を表示しない
                if (dynamic_cast<FunctionDefinitionNode*>(result.get()) == nullptr) {
                    double double_value = boost::rational_cast<double>(value);
                    std::cout << "= " << double_value;

                    // 分母が1の場合（整数の場合）は分数表記を省略
                    if (value.denominator() != 1) {
                        std::cout << " (分数表現: " << value.numerator() 
                                  << "/" << value.denominator() << ")";
                    } else {
                        // 整数値の場合は、cpp_intの値をそのまま表示
                        std::cout << " (整数値: " << value.numerator() << ")";
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cerr << "構文解析エラー：結果が得られませんでした" << std::endl;
            }
        } catch (const pegtl::parse_error& e) {
            std::cerr << "構文エラー: " << e.what() << std::endl;
        } catch (const std::domain_error& e) {
            std::cerr << "定義域エラー: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "エラー: " << e.what() << std::endl;
        }
    }

    return 0;
}
