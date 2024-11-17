from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm


class JuliusLabAnalyzer:
    """A class for analyzing the output: lab of Julius Segmentation."""

    def __init__(self, lab_filepath_list: List[str]):
        """Initialize JuliusLabAnalyzer."""
        self.lab_filepath_list = lab_filepath_list

    def _load_lab_file(self, lab_filepath: str) -> pd.DataFrame:
        """Load lab file content."""
        # arr = np.loadtxt(lab_filepath, dtype=str) # これできたらいいな
        arr = []
        with open(lab_filepath, "r") as f:
            for phoneme_idx, line in enumerate(f):
                try:
                    start, end, phoneme = line.split()
                except ValueError:
                    print(f"ValueError for {lab_filepath}, {line}")
                    continue
                arr.append([start, end, phoneme, phoneme_idx])

        df = pd.DataFrame(arr, columns=["start", "end", "phoneme", "phoneme_idx"])
        df["start"] = df["start"].astype(float)
        df["end"] = df["end"].astype(float)
        df["phoneme_idx"] = df["phoneme_idx"].astype(int)
        df["duration"] = df["end"] - df["start"]
        return df

    def load_lab_files(self) -> pd.DataFrame:
        """Load lab files."""
        df = pd.DataFrame()
        count_empty_file = 0
        for idx, lab_filepath in tqdm(enumerate(self.lab_filepath_list)):
            df_temp = self._load_lab_file(lab_filepath)
            if df_temp.empty:
                count_empty_file += 1
                continue
            df_temp["lab_filepath"] = lab_filepath
            # file name: audiobook_1_003.lab
            # audiobook_id: audiobook_1
            # audiobook_id_int: 1
            # chapter_id: 003
            # chapter_id_int: 3
            df_temp["audiobook_id"] = (
                Path(lab_filepath).stem.split("_")[0]
                + "_"
                + Path(lab_filepath).stem.split("_")[1]
            )
            df_temp["audiobook_id_int"] = int(Path(lab_filepath).stem.split("_")[1])
            df_temp["chapter_id"] = Path(lab_filepath).stem.split("_")[2]
            df_temp["chapter_id_int"] = int(Path(lab_filepath).stem.split("_")[2])
            df_temp["lab_idx"] = idx
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)

        df.reset_index(drop=True, inplace=True)
        print(
            f"count_empty_file: {count_empty_file} : {count_empty_file / len(self.lab_filepath_list)}"
        )
        return df

    def set_save_dir(self, save_dir: str):
        """Set save_dir."""
        self.save_dir = save_dir

    # AttributeError: property 'save_dir' of 'JuliusLabAnalyzer' object has no setter
    # @property
    # def save_dir(self):
    #     """Get save_dir."""
    #     if Path(self._save_dir).exists():
    #         return self._save_dir
    #     else:
    #         raise FileNotFoundError(f"{self._save_dir} is not found.")

    def save_df_lab_to_csv(self, df: pd.DataFrame, save_dir: str = None):
        """Save df_lab."""
        if save_dir is None:
            save_dir = self.save_dir
        df.to_csv(Path(save_dir) / "df_lab.csv", index=False)
        print(f"df_lab.csv is saved in {save_dir}")

    def load_df_lab_from_csv(self, save_dir: str = None) -> pd.DataFrame:
        """Load df_lab."""
        if save_dir is None:
            save_dir = self.save_dir
        df = pd.read_csv(Path(save_dir) / "df_lab.csv")
        return df

    def save_df_lab_to_pickle(self, df: pd.DataFrame, save_dir: str = None):
        """Save df_lab."""
        if save_dir is None:
            save_dir = self.save_dir
        df.to_pickle(Path(save_dir) / "df_lab.pkl")
        print(f"df_lab.pkl is saved in {save_dir}")

    def load_df_lab_from_pickle(self, save_dir: str = None) -> pd.DataFrame:
        """Load df_lab."""
        if save_dir is None:
            save_dir = self.save_dir
        df = pd.read_pickle(Path(save_dir) / "df_lab.pkl")
        return df

    # def load_lab_files_only_pause(self) -> pd.DataFrame:
    #     """Load lab files. but df is so big, so only pause"""
    #     df = pd.DataFrame()
    #     for idx, lab_filepath in tqdm(enumerate(self.lab_filepath_list)):
    #         df_temp = self._load_lab_file(lab_filepath)
    #         df_temp["lab_filepath"] = lab_filepath
    #         # file name: audiobook_1_003.lab
    #         # audiobook_id: audiobook_1
    #         # audiobook_id_int: 1
    #         # chapter_id: 003
    #         # chapter_id_int: 3
    #         df_temp["audiobook_id"] = (
    #             Path(lab_filepath).stem.split("_")[0]
    #             + "_"
    #             + Path(lab_filepath).stem.split("_")[1]
    #         )
    #         df_temp["audiobook_id_int"] = int(Path(lab_filepath).stem.split("_")[1])
    #         df_temp["chapter_id"] = Path(lab_filepath).stem.split("_")[2]
    #         df_temp["chapter_id_int"] = int(Path(lab_filepath).stem.split("_")[2])
    #         df_temp["lab_idx"] = idx
    #         df_pause = self.calc_pause_duration(df_temp)
    #         df = pd.concat([df, df_temp], axis=0, ignore_index=True)

    #     df.reset_index(drop=True, inplace=True)
    # return df

    @staticmethod
    def attach_author_and_book_info(yaml_filepath: str, df: pd.DataFrame):
        """Attach author and book name to the dataframe."""
        with open(yaml_filepath, "r") as file:
            yaml_content = yaml.safe_load(file)

        for idx, row in df.iterrows():
            audiobook_id = row["audiobook_id"]
            book = yaml_content[audiobook_id]["book"]
            author = yaml_content[audiobook_id]["author"]
            df.loc[idx, "book"] = book
            df.loc[idx, "author"] = author

        return df

    @staticmethod
    def attach_narrative_info(yaml_filepath: str, df: pd.DataFrame):
        """Attach author and book name to the dataframe."""
        with open(yaml_filepath, "r") as file:
            yaml_content = yaml.safe_load(file)

        for idx, row in df.iterrows():
            audiobook_id = row["audiobook_id"]
            # book = yaml_content[audiobook_id]["book"]
            # author = yaml_content[audiobook_id]["author"]
            chapter_id_int = row["chapter_id_int"]
            text = yaml_content[audiobook_id]["text"][chapter_id_int]["sent"]
            character = yaml_content[audiobook_id]["text"][chapter_id_int]["character"]
            to_whom = yaml_content[audiobook_id]["text"][chapter_id_int]["to_whom"]

            df.loc[idx, "text"] = text
            df.loc[idx, "character"] = character
            df.loc[idx, "to_whom"] = to_whom

        return df

    def load_speaker_info(self) -> pd.DataFrame:
        """Load speaker info."""
        # TODO: これは毎回読み込む必要ないので、一度読み込んだら保存しておく
        lab_filepath_speaker_info = (
            "/home/takeshun256/PausePrediction/data_pub/jmac/bookdata-speaker.csv"
        )
        df_speaker = pd.read_csv(lab_filepath_speaker_info)
        df_speaker = df_speaker[["mp3", "speaker"]]
        df_speaker["speaker"] = df_speaker["speaker"].apply(lambda x: x.split(",")[0])

        speaker_gender_info = (
            "/home/takeshun256/PausePrediction/data_pub/jmac/speaker_gender.csv"
        )
        df_gender = pd.read_csv(speaker_gender_info)
        df_speaker = df_speaker.merge(df_gender, how="left", on="speaker")
        return df_speaker

    def attach_speaker_info(self, df: pd.DataFrame):
        """Attach speaker info to the dataframe."""
        df_speaker = self.load_speaker_info()
        yaml_filepath = (
            "/home/takeshun256/PausePrediction/data_pub/jmac/text_audio_dict_new.yaml"
        )
        with open(yaml_filepath, "r") as file:
            yaml_content = yaml.safe_load(file)
        audiobook_id_list = list(yaml_content.keys())
        audiobook_id_int_list = [
            int(audiobook_id.split("_")[1]) for audiobook_id in audiobook_id_list
        ]
        mp3_list = [
            str(Path(yaml_content[id]["mp3"][0]).name) for id in audiobook_id_list
        ]  # TODO: 直す
        df_mp3 = pd.DataFrame(
            {"mp3": mp3_list, "audiobook_id_int": audiobook_id_int_list}
        )
        display(df_mp3.head())
        display(df_speaker.head())
        df_speaker = df_speaker.merge(df_mp3, how="inner", on="mp3")
        df_speaker["audiobook_id_int"] = df_speaker["audiobook_id_int"].astype(int)
        df = df.merge(df_speaker, how="left", on="audiobook_id_int")
        return df

    def calc_pause_duration(self, df: pd.DataFrame):
        """Calculate pause duration."""
        df_temp = df.query("phoneme == 'sp'")  # 読点はspへ変換済み, 文章間はsilB, silE
        df_temp_grouped = df_temp.groupby("lab_idx").agg(
            pause_duration=pd.NamedAgg(column="duration", aggfunc=np.mean)
        )  # Updated syntax
        df_pause = df.copy()
        df_pause = df_pause.merge(
            df_temp_grouped, how="left", left_on="lab_idx", right_index=True
        )
        df_pause.fillna(0, inplace=True)
        return df_pause

    def plot_phoneme_duration(self, df: pd.DataFrame):
        """Plot phoneme duration."""
        plt.figure(figsize=(20, 10))
        sns.boxplot(x="phoneme", y="duration", data=df)
        plt.xticks(rotation=90)
        plt.show()

    def plot_phoneme_duration_by_(self, df: pd.DataFrame, hue_col: str):
        """Plot phoneme duration by hue_col."""
        # hue_col: audiobook_id, chapter_id, author, book
        plt.figure(figsize=(20, 10))
        sns.boxplot(x="phoneme", y="duration", hue=hue_col, data=df)
        plt.xticks(rotation=90)
        plt.show()

    def plot_pause_duration_by_(self, df: pd.DataFrame, hue_col: str):
        """Plot pause duration by hue_col."""
        # hue_col: audiobook_id, chapter_id, author, book
        df_pause = self.calc_pause_duration(df)
        plt.figure(figsize=(20, 10))
        sns.boxplot(x="lab_idx", y="duration", hue=hue_col, data=df_pause)
        plt.xticks(rotation=90)
        plt.show()
