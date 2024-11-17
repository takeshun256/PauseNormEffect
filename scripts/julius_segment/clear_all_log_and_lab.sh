#!/bin/bash

# This script is used to clear all log and lab files in the julius_segment directory

# Get the data directory
data_dir="/data2/takeshun256/jmac_split_and_added_lab"

# 再起的にファイルを検索し、ファイルを削除する
# 削除対象はlogとlabとlab2ファイル, wavとtxtファイルは削除しない
find $data_dir -type f -name "*.log" -o -name "*.lab" -o -name "*.lab2" | xargs rm -f



