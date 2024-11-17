# ref: https://qiita.com/Syuparn/items/d86e77c39527539b16c5

# Usage: python julius_segment.py <data_dir>
# txt, wavファイルがあるディレクトリを指定すると、そのディレクトリ下にlabファイルが生成される
# 再帰的にディレクトリを探索するので、ディレクトリ下にディレクトリがあってもOK

import os
import subprocess
import sys

model_path = "/home/takeshun256/PausePrediction/segmentation-kit/segment_julius.pl"

args = sys.argv

if len(args) != 2:
    print("Usage: python julius_segment.py <data_dir>")
    sys.exit()

for dirpath, dirnames, filenames in os.walk(args[1]):
    # subprocess.call(['perl', 'segment_julius.pl', dirpath])
    subprocess.call(["perl", model_path, dirpath])
