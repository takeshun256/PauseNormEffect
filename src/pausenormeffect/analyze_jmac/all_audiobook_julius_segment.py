import os
import subprocess
from tqdm import tqdm
import time

# 全てのaudiobookディレクトリに対してjulius_segment.pyを実行する

julius_segment_path = (
    "/home/takeshun256/PausePrediction/src/julius_segment/julius_segment.py"
)

# 適用するディレクトリのリスト
data_dir = "/data2/takeshun256/jmac_split_and_added_lab"

# data_dir内のディレクトリのリスト
directories = os.listdir(data_dir)
directories = [os.path.join(data_dir, directory) for directory in directories]

# 各ディレクトリに対してjulius_segment.pyを実行
for directory in tqdm(directories):
    if directory == "/data2/takeshun256/jmac_split_and_added_lab/audiobook_28":
        continue
    print(f"start julius_segment.py for {directory}")

    try:
        subprocess.run(
            ["python", julius_segment_path, directory], timeout=300
        )  # 300秒でタイムアウト
        print(f"end julius_segment.py for {directory}")
    except subprocess.TimeoutExpired:
        print(f"Timeout for {directory}. Moving on to the next one.")

####--------------------処理が終わらない音声--------------------
# 音声が長い場合、julius_segment.pyが終わらないことがある => タイムアウトを設定
# /data2/takeshun256/jmac_split_and_added_lab/audiobook_28/audiobook_28_148.wav
# /data2/takeshun256/jmac_split_and_added_lab/audiobook_28/audiobook_28_172.log
####--------------------処理が終わらない音声--------------------
