#!/usr/bin/env bash

# Fail the script if any command fails
set -e
set -o pipefail

# 全てのaudiobookディレクトリに対してjulius_segment.pyを実行する
julius_segment_path="/home/takeshun256/PausePrediction/src/julius_segment/julius_segment.py"
data_dir="/data2/takeshun256/jmac_split_and_added_lab"

# data_dir内のディレクトリのリスト（lsの代わりにfindを使用）
directories=$(find "$data_dir" -maxdepth 1 -type d)

# 各ディレクトリに対してjulius_segment.pyを実行
for directory in $directories; do
    echo "start julius_segment.py for $directory"
  
    # 300秒でタイムアウト
    if timeout 300 python "$julius_segment_path" "$directory"; then
        echo "end julius_segment.py for $directory"
    else
        echo "Timeout for $directory. Moving on to the next one."
    fi
done
