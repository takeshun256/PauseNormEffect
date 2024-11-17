# from https://github.com/takuya-matsuzaki/Lab/blob/master/utils/j-mac-text-audio.py
import csv
import glob
import os

import yaml
from audiobook_yaml_parser import extract_yaml_data


# テスト用：書き起こしの yaml ファイルから文単位の情報を集める
def collect_leaves(yaml, arr):
    if type(yaml) == list:
        arr += yaml
    else:
        assert type(yaml) == dict
        for v in yaml.values():
            collect_leaves(v, arr)
    return arr


# テスト用：ファイルサイズを取得
def file_size(path):
    size = os.path.getsize(path)
    return size


corpus_dir = "/data/corpus/J-MAC"
# 保存用のdict
text_audio_dict = {}
# 保存先
dict_fname = "/home/takeshun256/PausePrediction/data_pub/j-mac/text_audio_dict.yaml"

with open(f"{corpus_dir}/bookdata.csv") as f:
    reader = csv.DictReader(f)
    for idx, r in enumerate(reader):
        # テキスト（書き起こし＋文アラインメント）ファイル名
        txt_fname = f"{corpus_dir}/txt/{r['author']}/{r['book']}/{r['file']}/all.yaml"

        # # YAML の読み込み
        # with open(txt_fname) as file:
        #     txt = yaml.safe_load(file)

        # # 文の情報を集める
        # ss = collect_leaves(txt, [])
        # print(ss)
        text_yaml_list = extract_yaml_data(txt_fname)

        # 朗読データごとに情報を表示
        print(f"book: {r['book']}")  # 作品名
        print(f"author: {r['author']}")  # 作者
        print(f"url: {r['url']}")  # 音声データの販売サイト URL

        # MP3 ファイルの情報
        if r["mp3"].endswith(".mp3"):
            # mp3 フィールドが .mp3 で終わっているものは単一の MP3 ファイル
            mp3_fname = f"{corpus_dir}/mp3/{r['mp3']}"
            print(f"mp3: {mp3_fname} (size: {file_size(mp3_fname)})")
        else:
            # mp3 フィールドが .mp3 で終わっていないものは MP3 が分割されている
            for mp3_fname in glob.glob(f"{corpus_dir}/mp3/{r['mp3']}/*.mp3"):
                print(f"mp3: {mp3_fname} (size: {file_size(mp3_fname)})")

        # WAVファイル（結合済み）の情報
        wav_fname = f"{corpus_dir}/wav/{r['wav']}"
        print(f"wav: {wav_fname} (size: {file_size(wav_fname)})")

        # dict にまとめる
        text_audio_dict[f"audiobook_{idx}"] = {
            "book": r["book"],
            "author": r["author"],
            "url": r["url"],
            "mp3": mp3_fname,
            "wav": wav_fname,
            "text": text_yaml_list,
        }

    # dict を保存
    with open(dict_fname, "w", encoding="utf-8") as f:
        yaml.dump(text_audio_dict, f, allow_unicode=True)
