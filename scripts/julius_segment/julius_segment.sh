#!/bin/bash

# 音素アライメントの処理を実行するスクリプト
# 引数に音素アライメントを行うフォルダを指定する
# 例: ./julius_segment.sh ./output
# txt, wavファイルがあるフォルダを指定するとそのフォルダ内にlabファイルが生成される

segment_perl_path="/home/takeshun256/Julius-Test/segmentation-kit/segment_julius.pl"

# 音素アライメントの処理を定義
run_segmentation() {
    local output_dir="$1"
    perl "$segment_perl_path" "$output_dir"
}

# 引数のチェック
if [ $# -lt 1 ]; then
    echo "入力フォルダが指定されていません。"
    exit 1
fi

# 関数の使用例
run_segmentation "$1"
