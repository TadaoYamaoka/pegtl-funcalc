# PEGTLを使った高精度コマンドライン関数電卓

本プロジェクトは、C++と[PEGTL](https://github.com/taocpp/PEGTL) (Parsing Expression Grammar Template Library) を使用して実装した、コマンドラインベースの高精度関数電卓です。

## 概要

- 任意精度の有理数および高精度浮動小数点数（Boost.Multiprecisionを使用）をサポート。
- 数値演算（加減乗除、剰余、べき乗）をサポート。
- 数学関数（`sqrt`, `log`, `exp`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, など）を標準搭載。
- ユーザー定義関数や変数の作成が可能。
- PEGTLを使った柔軟で強力なパーサー実装。

## 主な機能

- 四則演算およびべき乗演算（整数・有理数・浮動小数点数）
- 定数 (`pi`, `e` など) の利用
- 暗黙の乗算をサポート（例：`2pi` → `2 * pi`）
- 数学関数の高精度計算
- ユーザー定義の関数および変数
- エラー処理（ゼロ除算、定義域エラーなど）

## 必要なライブラリ

- [PEGTL (taocpp)](https://github.com/taocpp/PEGTL)
- [Boost.Rational](https://www.boost.org/doc/libs/release/libs/rational/rational.html)
- [Boost.Multiprecision](https://www.boost.org/doc/libs/release/libs/multiprecision/)

## ビルド方法

Mesonビルドシステムを使用しています。

```bash
meson setup build
meson compile -C build
```

実行例:

```bash
./build/funcalc
```

## 使用例

```bash
> 2 + 3 * 4
= 14 (整数値: 14)

> sqrt(2)
= 1.41421 (分数表現: 14142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)

> sin(pi/2)
= 1 (整数値: 1)

> f(x) = x^2 + 2*x + 1
関数 f を定義しました。

> f(3)
= 16 (整数値: 16)

> e^log(1)
= 1 (整数値: 1)

> 1/3 + 1/6
= 0.5 (分数表現: 1/2)
```