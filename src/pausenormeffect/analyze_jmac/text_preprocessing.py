import re
import sys

import jaconv
import pyopenjtalk  # require: python3.10

sys.path.append("/home/takeshun256/PausePrediction/src/analyze_jmac")
from audiobook_script_extractor import AudiobookScriptExtractor


class AudiobookScriptPreprocessor:
    def __init__(self, yaml_filepath: str):
        self.yaml_filepath = yaml_filepath
        self.script_extractor = AudiobookScriptExtractor(self.yaml_filepath)

        self.reading_scripts = None

    def load_audiobook_scripts(self):
        reading_scripts = self.script_extractor.extract_reading_scripts()
        self.reading_scripts = reading_scripts
        return reading_scripts

    def preprocess_audiobook_scripts(self):
        """Preprocess audiobook scripts."""
        pass

    @staticmethod
    def remove_brackets_to_kanji(x):
        """ブラケット内の漢字を取り出す.

        Example:
            入力: "お[菓子|かし]がひとつ" # [kanji|furigana]
            出力: "お菓子がひとつ"
        """
        return re.sub(r"\[(.+?)\|(.+?)\]", r"\1", x)

    @staticmethod
    def remove_brackets_to_furigana(x):
        """ブラケット内のふりがなを取り出す.

        Example:
            入力: "お[菓子|かし]がひとつ" # [kanji|furigana]
            出力: "おかしがひとつ"
        """
        return re.sub(r"\[(.+?)\|(.+?)\]", r"\2", x)

    @staticmethod
    def kanji2julius(x):
        """漢字をjuliusの音素に変換する."""
        katakana = pyopenjtalk.g2p(x, kana=True)
        hira = jaconv.kata2hira(katakana)
        phonemes = jaconv.hiragana2julius(hira)
        return phonemes

    @staticmethod
    def punctuation2space(x):
        """句読点をスペースに変換する."""
        # 、はspへ変換
        # 。は削除
        x = re.sub(r"、", r" sp", x)
        # TODO: これ正規化してから処理した方が良いかも...想定外の削除対象がおそらくあるため
        x = re.sub(r"[『』……——。「」？！\?\!\-\(\)ー]", r"", x)
        return x

    @staticmethod
    def hira2julius(x):
        """ひらがなをjuliusの音素に変換する."""
        phonemes = jaconv.hiragana2julius(x)
        return phonemes

    def __len__(self):
        return len(self.reading_scripts)

    def __getitem__(self, idx):
        return self.reading_scripts[idx]

    def __iter__(self):
        for reading_script in self.reading_scripts:
            yield reading_script

    def __next__(self):
        for reading_script in self.reading_scripts:
            yield reading_script

    def __repr__(self):
        return f"{self.reading_scripts}"


if __name__ == "__main__":
    pass
