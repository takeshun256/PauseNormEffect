#!/usr/bin/env bash

# Fail the script if any command fails
set -e
set -o pipefail

# 実行するスクリプトのパス
fix_align_path="/home/takeshun256/PausePrediction/src/julius_segment/fix_align.py"

# 全てのaudiobookディレクトリに対してjulius_segment.pyを実行する
data_dir="/data2/takeshun256/jmac_split_and_added_lab"

# data_dir内のディレクトリのリスト（lsの代わりにfindを使用）
directories=$(find "$data_dir" -maxdepth 1 -type d)

# 各ディレクトリに対してjulius_segment.pyを実行
for directory in $directories; do
    echo "start fix_align.py for $directory"
  
    # lab2ファイルが生成される
    # もしAssertionErrorが出たら、次のディレクトリに移動する
    if python "$fix_align_path" "$directory"; then
        echo "end fix_align.py for $directory"
    else
        echo "AssertionError for $directory. Moving on to the next one."
    fi
done
