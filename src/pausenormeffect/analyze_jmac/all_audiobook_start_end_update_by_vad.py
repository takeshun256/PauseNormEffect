import csv
import glob
import os
import sys

import yaml

sys.path.append("/home/takeshun256/PausePrediction/src/analyze_jmac")
sys.path.append("/home/takeshun256/PausePrediction/src/vad_tool")
import struct
from pprint import pprint
from typing import Any, Dict, List, Tuple, Union

import japanize_matplotlib
import librosa
import librosa.display
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.io.wavfile
import scipy.ndimage
import scipy.signal
import seaborn as sns
import webrtcvad
from audiobook_dataset_builder import AudiobookDatasetBuilder
from audiobook_script_extractor import AudiobookScriptExtractor
from audiobook_yaml_parser import extract_yaml_data
from IPython.display import Audio
from matplotlib import pyplot as plt
from py_webrtcvad_test import getVadSection
from scipy.io import wavfile
from tqdm import tqdm
from vad_tool import VAD_Segmenter

# 全てのaudiobookの始めと終わりの時間をVADによって更新する

# 1.全てのaudiobookをまとめたyamlファイルを読み込む

jmac_raw_yaml = (
    "/home/takeshun256/PausePrediction/data_pub/j-mac/text_audio_dict_raw.yaml"
)
output_path = (
    "/home/takeshun256/PausePrediction/data_pub/j-mac/text_audio_dict_new.yaml"
)

audiobook = "/data/corpus/J-MAC"
metadata = "/data/corpus/J-MAC/bookdata.csv"

builder = AudiobookDatasetBuilder(audiobook)
builder.build_dataset(metadata, jmac_raw_yaml)

# 2.全てのaudiobookの始めと終わりの時間をVADによって更新する
all_audiobook = builder.text_audio_dict
all_audiobook_id = list(all_audiobook.keys())

builder = AudiobookDatasetBuilder.rebuild_from_yaml(output_path)
# 41以降だけで試す, 41で止まってしまった目。
all_audiobook_id = all_audiobook_id[40:]

for audiobook_id in tqdm(all_audiobook_id):
    # あるaudiobookのtext_audio_dictからvad用のクラスを作成
    vad_segmenter = VAD_Segmenter.from_text_audio_yaml(
        text_audio_dict=all_audiobook, audiobook_id=audiobook_id
    )
    # vadをかける
    vad_segmenter.vad_text_df()
    # vad結果から始めと終わりの時間を更新
    vad_segmenter.update_text_df_vad()
    text_df_with_vad_updated = vad_segmenter.get_text_df_vad_update()
    # sanity check(try-exceptでエラーを出さないようにする)
    # とりあえず、データを作りたいので、エラーが出て止まらないようにコメントアウト
    # try:
    #     assert vad_segmenter.check_start_end_relation(text_df_with_vad_updated)
    # except AssertionError:
    #     print(f"audiobook_id: {audiobook_id} has a start end sec check error.")

    # text_audio_dictを更新
    col = ["character", "sent", "time", "to_whom"]  # 元のtext_dfのカラム
    new_text_yaml_list = AudiobookDatasetBuilder.dataframe2list_of_dicts(
        text_df_with_vad_updated[col]
    )
    builder.update_text_yaml_list(audiobook_id, new_text_yaml_list)

    # 毎回更新を記録する
    builder.to_yaml(output_path)

builder.to_yaml(output_path)
