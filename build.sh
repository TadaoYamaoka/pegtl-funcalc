#!/bin/bash
set -e

# build が存在しない場合は meson setup を実行
if [ ! -d "build" ]; then
    echo "ビルドディレクトリを初期化しています..."
    meson setup build
else
    echo "既存の build を使用します。"
fi

# プロジェクトをビルド
echo "プロジェクトをビルド中..."
meson compile -C build

echo "ビルドが完了しました。"
echo "実行するには: ./build/funcalc"
