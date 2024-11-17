import csv
import glob
import os
from typing import Any, Dict, List

import pandas as pd
import yaml
from audiobook_script_extractor import AudiobookScriptExtractor
from tqdm import tqdm


class AudiobookDatasetBuilder:
    """A class for building a dataset of audiobooks.

    Attributes:
        corpus_dir (str): The directory containing the audiobook corpus.
        text_audio_dict (Dict[str, Any]): A dictionary mapping audiobook IDs to metadata and audio/text file paths.
    """

    def __init__(self, corpus_dir: str):
        print(f"Initializing AudiobookDatasetBuilder from {corpus_dir}...")
        self.corpus_dir = corpus_dir
        self.text_audio_dict = {}

    @staticmethod
    def _file_size(path: str) -> int:
        return os.path.getsize(path)

    def _extract_data_from_row(self, row: Dict[str, str], idx: int):
        txt_fname = f"{self.corpus_dir}/txt/{row['author']}/{row['book']}/{row['file']}/all.yaml"
        # 各朗読データのテキストファイルを読み込む
        script_extractor = AudiobookScriptExtractor(txt_fname)
        text_yaml_list = script_extractor.extract_reading_scripts()

        mp3_files = []
        if row["mp3"].endswith(".mp3"):
            mp3_files.append(f"{self.corpus_dir}/mp3/{row['mp3']}")
        else:
            mp3_files.extend(glob.glob(f"{self.corpus_dir}/mp3/{row['mp3']}/*.mp3"))

        wav_fname = f"{self.corpus_dir}/wav/{row['wav']}"

        self.text_audio_dict[f"audiobook_{idx}"] = {
            "book": row["book"],
            "author": row["author"],
            "url": row["url"],
            "mp3": mp3_files,
            "wav": wav_fname,
            "text": text_yaml_list,
        }

    def build_dataset(self, metadata_file: str, output_path: str):
        print(f"Reading metadata from {metadata_file}...")
        print(f"Building dataset from {self.corpus_dir}...")
        with open(metadata_file) as f:
            reader = csv.DictReader(f)
            for idx, row in tqdm(enumerate(reader)):
                self._extract_data_from_row(row, idx)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.text_audio_dict, f, allow_unicode=True)

        print(f"Saved dataset to {output_path}: {self._file_size(output_path)}.")
        print("Done!")

    @classmethod
    def rebuild_from_yaml(cls, yaml_path: str):
        """outputしたyamlファイルからtext_audio_dictを再構築する."""
        print(f"Rebuilding text_audio_dict from {yaml_path}...")
        with open(yaml_path) as f:
            text_audio_dict = yaml.safe_load(f)
        builder = cls("/data/corpus/J-MAC")
        builder.text_audio_dict = text_audio_dict
        return builder

    def update_text_yaml_list(
        self, audiobook_id: str, new_text_yaml_list: List[Dict[str, Any]]
    ):
        """text_yaml_listを更新する."""
        if isinstance(new_text_yaml_list, list):
            self.text_audio_dict[audiobook_id]["text"] = new_text_yaml_list
        else:
            raise TypeError("new_text_yaml_list must be a list of dicts.")

    def to_yaml(self, output_path: str):
        """text_audio_dictをyamlファイルに保存する."""
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.text_audio_dict, f, allow_unicode=True)
        print(f"Saved dataset to {output_path}: {self._file_size(output_path)}.")

    @staticmethod
    def dataframe2list_of_dicts(df: pd.DataFrame):
        """Convert a dataframe to a list of dicts."""
        return df.to_dict("records")

    @staticmethod
    def list_of_dicts2dataframe(list_of_dicts: List[Dict[str, Any]]):
        """Convert a list of dicts to a dataframe."""
        return pd.DataFrame(list_of_dicts)


if __name__ == "__main__":
    builder = AudiobookDatasetBuilder("/data/corpus/J-MAC")
    builder.build_dataset(
        f"{builder.corpus_dir}/bookdata.csv",
        "/home/takeshun256/PausePrediction/data_pub/j-mac/text_audio_dict_raw.yaml",
    )
