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
from audiobook_yaml_parser import extract_yaml_data
from IPython.display import Audio
from matplotlib import pyplot as plt
from py_webrtcvad_test import getVadSection
from scipy.io import wavfile
from tqdm import tqdm


class VAD_Segmenter:
    def __init__(
        self,
        text_df: pd.DataFrame,
        y: np.ndarray,
        sr: int = 16000,
        buffer_sec: float = 2.0,
        aggressiveness: int = 3,
        window_duration: float = 0.01,
    ):
        print("VAD_Segmenter init")
        self.y = y
        self.sr = sr
        self.buffer_sec = buffer_sec
        self.aggressiveness = aggressiveness
        self.buffer_size = self.sr * self.buffer_sec
        self.wav_sec_start = 0
        self.wav_sec_end = len(self.y) / self.sr

        self.segment_df = None

        self.y_int = self.convert_to_int(self.y)

        self.text_df = text_df
        self.text_df_buffer = None
        self.get_text_df_with_buffer()  # text_df_bufferを作成する

        # vadの結果を保存するdf
        self.text_df_vad = None

        # VAD結果を元にtext_dfを更新したdf
        self.update_margin = 0.03  # 終了時刻を更新する際のマージン
        self.text_df_vad_update = None

        # define webrtcvad VAD
        self.vad = webrtcvad.Vad(self.aggressiveness)  # set aggressiveness from 0 to 3
        # window_duration = 0.03 # duration in seconds
        self.window_duration = window_duration  # duration in seconds
        self.samples_per_window = int(self.window_duration * self.sr + 0.5)
        self.bytes_per_sample = 2  # for int16

        print("VAD_Segmenter init end")

    @classmethod
    def from_text_audio_yaml(
        cls,
        text_audio_dict: str,
        audiobook_id: str,
        buffer_sec: float = 2.0,
        aggressiveness: int = 3,
        window_duration: float = 0.01,
    ):
        """text_audio_dictからVAD_Segmenterを作成する."""
        if audiobook_id not in text_audio_dict.keys():
            raise Exception(f"{audiobook_id}がtext_audio_dictに存在しません。")

        text_df = pd.DataFrame(text_audio_dict[audiobook_id]["text"])
        y, sr = librosa.load(text_audio_dict[audiobook_id]["wav"], sr=16000)
        return cls(
            text_df,
            y,
            sr=sr,
            buffer_sec=buffer_sec,
            aggressiveness=aggressiveness,
            window_duration=window_duration,
        )

    def __call__(self):
        self.segment_df = self.get_vad_section_df()
        return self.segment_df

    # @classmethod
    # def load_wav(self, wav_path):
    #     y, sr = librosa.load(wav_path, sr=self.sr)
    #     return y, sr

    def convert_to_int(self, y):
        """floatの場合はintに変換する."""
        if y.dtype.kind == "f":
            # convert to int16
            y = np.array([int(s * 32768) for s in y])
            # bound
            y[y > 32767] = 32767
            y[y < -32768] = -32768
        return y

    def convert_to_float(self, y):
        """intの場合はfloatに変換する."""
        if y.dtype.kind == "i":
            y = np.array([float(s) / 32768.0 for s in y])
        return y

    def struct_pack(self, y):
        """音声データをバイナリに変換する."""
        raw_samples = struct.pack("%dh" % len(y), *y)
        return raw_samples

    def get_y_sample(self, sample_idx):
        """sample_idxで指定した音声を切り出す."""
        y_sample_int = self.y_int[
            int(self.text_df.at[sample_idx, "time"][0] * self.sr) : int(
                self.text_df.at[sample_idx, "time"][1] * self.sr
            )
        ]
        return y_sample_int

    def get_y_sample_with_buffer(self, sample_idx):
        """sample_idxで指定したバッファありの音声を切り出す."""
        # print("切り取った音声範囲(s)", self.text_df_buffer.at[sample_idx, "time_buffer"])
        y_sample_int = self.y_int[
            int(self.text_df_buffer.at[sample_idx, "time_buffer"][0] * self.sr) : int(
                self.text_df_buffer.at[sample_idx, "time_buffer"][1] * self.sr
            )
        ]
        return y_sample_int

    def get_vad_section_df(self, sample_idx):
        y_sample_int = self.get_y_sample_with_buffer(sample_idx)
        self.y_sample_int = y_sample_int
        # create raw sample in bit
        raw_samples = self.struct_pack(y_sample_int)

        # Start classifying chunks of samples
        # var to hold segment wise report
        segments = []
        # iterate over the audio samples
        for i, start in enumerate(
            np.arange(0, len(y_sample_int), self.samples_per_window)
        ):
            stop = min(start + self.samples_per_window, len(y_sample_int))
            loc_raw_sample = raw_samples[
                start * self.bytes_per_sample : stop * self.bytes_per_sample
            ]
            try:
                is_speech = self.vad.is_speech(loc_raw_sample, sample_rate=self.sr)
                segments.append(dict(start=start, stop=stop, is_speech=is_speech))
            except Exception as e:
                print(f"Failed for step {i}, reason: {e}")

        # convert to dataframe
        self.segment_df = pd.DataFrame(segments)
        return self.segment_df

    def get_text_df_with_buffer(self):
        # 2秒ずつ前後にバッファを取って分割する
        # print("音声の秒数範囲", self.wav_sec_start, "~", self.wav_sec_end)
        self.text_df_buffer = self.text_df.copy()
        self.text_df_buffer["time_buffer"] = self.text_df_buffer["time"].apply(
            lambda x: [
                max(x[0] - self.buffer_sec, self.wav_sec_start),
                min(x[1] + self.buffer_sec, self.wav_sec_end),
            ]
        )

    def is_speech_at_time(self, segment_df, time):
        for index, row in segment_df.iterrows():
            if row["start"] <= time <= row["stop"]:
                return row["is_speech"]
        return False

    # 音声の初めと終わりがVADでTrueかどうかを判定する
    def is_speech_at_start_end(self, sample_idx):
        segment_df = self.get_vad_section_df(sample_idx)
        # bufferを考慮する必要あり
        is_speech_at_start = self.is_speech_at_time(segment_df, 0 + self.buffer_size)
        is_speech_at_end = self.is_speech_at_time(
            segment_df, len(self.y_sample_int) - self.buffer_size
        )

        return is_speech_at_start, is_speech_at_end

    # 終了時刻が発声(is_speech_end=True)の場合、終了時刻を、次の発声終了時刻(is_speech=Falseが始まる時刻)を取得する
    def update_end_time(self, sample_idx):
        segment_df = self.get_vad_section_df(sample_idx)
        # bufferを考慮する必要あり
        is_speech_at_end = self.is_speech_at_time(
            segment_df, len(self.y_sample_int) - self.buffer_size
        )
        if is_speech_at_end:
            for index, row in segment_df.iterrows():
                if row["start"] > len(self.y_sample_int) - self.buffer_size:
                    if not row["is_speech"]:
                        # print(row['start'], len(self.y_sample_int)-self.buffer_size)
                        return True, row["start"] - (
                            len(self.y_sample_int) - self.buffer_size
                        )
            # raise Exception(
            #     "終了時刻が発声(is_speech_end=True)の場合、終了時刻を、次の発声終了時刻(is_speech=Falseが始まる時刻)を取得できませんでした。"
            # )
            print(
                "終了時刻が発声(is_speech_end=True)の場合、終了時刻を、次の発声終了時刻(is_speech=Falseが始まる時刻)を取得できませんでした。"
            )
            return False, 0  # 一旦、Falseを返す実装にしておく...
        else:
            return False, 0

    # VAD結果をtext_dfに反映する
    def vad_text_df(self):
        is_speech_at_start_end_list = []
        for sample_idx in tqdm(range(self.num_samples)):
            is_speech_at_start, is_speech_at_end = self.is_speech_at_start_end(
                sample_idx
            )
            is_speech_at_start_end_list.append([is_speech_at_start, is_speech_at_end])
        is_speech_at_start_end_df = pd.DataFrame(
            is_speech_at_start_end_list,
            columns=["is_speech_at_start", "is_speech_at_end"],
        )

        if len(self.text_df_buffer) != len(is_speech_at_start_end_df):
            raise Exception("text_df_bufferとis_speech_at_start_end_dfの長さが一致しません。")
        if self.text_df_vad is not None:
            print("text_df_vadがNoneではありません。上書きします。")
        self.text_df_vad = pd.concat(
            [self.text_df_buffer, is_speech_at_start_end_df], axis=1
        )

    # VAD結果をtext_df_vadに反映する
    def update_text_df_vad(self):
        if self.text_df_vad is None:
            raise Exception("text_df_vadがNoneです。")
        if self.text_df_vad_update is not None:
            print("text_df_vad_updateがNoneではありません。上書きします。")

        # time列を更新する
        text_df_vad_update = self.text_df_vad.copy()
        # 終了時刻が発声(is_speech_end=True)の場合、終了時刻を、次の発声終了時刻(is_speech=Falseが始まる時刻)+update_marginを取得する, 次の開始時刻も更新する
        for index, row in tqdm(self.text_df_vad.iterrows()):
            updated_flag, updated_end_time_delta = self.update_end_time(index)
            if updated_flag:
                text_df_vad_update.at[index, "time"][1] += (
                    updated_end_time_delta / self.sr + self.update_margin
                )
            if (
                index + 1 < len(self.text_df_vad)
                and text_df_vad_update.at[index, "time"][1]
                > text_df_vad_update.at[index + 1, "time"][0]
            ):
                text_df_vad_update.at[index + 1, "time"][0] = (
                    text_df_vad_update.at[index, "time"][1] + self.update_margin
                )

        drop_col = ["time_buffer", "is_speech_at_start", "is_speech_at_end"]
        self.text_df_vad_update = text_df_vad_update.drop(drop_col, axis=1)

    # VAD結果と音声波形をプロットする
    def plot_vad_segment(self, sample_idx, save_path=None):
        y_sample_float = self.convert_to_float(
            self.get_y_sample_with_buffer(sample_idx)
        )
        segment_df = self.get_vad_section_df(sample_idx)

        plt.figure(figsize=(20, 5))
        # librosa.display.waveshow(y_sample_float, sr=sr) # 秒数単位で表示する
        # 音声波形をプロット
        plt.plot(y_sample_float, label="waveform")
        # VAD結果を横線をプロット
        for index, row in segment_df.iterrows():
            if row["is_speech"]:
                if index == 0:
                    plt.hlines(
                        1,
                        row["start"],
                        row["stop"],
                        colors="tab:orange",
                        linewidth=5,
                        label="active speech",
                    )
                else:
                    plt.hlines(
                        1, row["start"], row["stop"], colors="tab:orange", linewidth=5
                    )
        # 縦線をプロット
        plt.vlines(
            self.buffer_size,
            -1,
            1,
            colors="tab:red",
            linewidth=2,
            label="speech start end",
        )
        plt.vlines(
            (len(y_sample_float) - self.buffer_size),
            -1,
            1,
            colors="tab:red",
            linewidth=2,
        )
        # 始終端の縦線をプロット
        plt.vlines(
            0,
            -1,
            1,
            colors="tab:green",
            linewidth=2,
            label=f"speech start end with buffer {self.buffer_sec}s",
        )
        plt.vlines(len(y_sample_float), -1, 1, colors="tab:green", linewidth=2)

        plt.title(f"Waveform with VAD: idx: {sample_idx}")
        plt.xlabel(
            f"Time [sec * {self.sr}], Text: {self.text_df_buffer.at[sample_idx, 'sent']}"
        )
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend(loc="lower left")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def display_audio_with_buffer(self, sample_idx):
        y_sample_float = self.convert_to_float(
            self.get_y_sample_with_buffer(sample_idx)
        )
        display(Audio(y_sample_float, rate=self.sr))

    @classmethod
    def check_start_end_relation(self, text_df):
        """text_dfのstartとendの関係をチェックする."""
        for index, row in text_df.iterrows():
            if row["time"][0] > row["time"][1]:
                # print(f"startがendより大きい: {index}")
                raise Exception(f"startがendより大きい: {index}")
            if index > 0:
                if row["time"][0] < text_df.at[index - 1, "time"][1]:
                    # print(f"startが前のendより小さい: {index}")
                    raise Exception(f"startが前のendより小さい: {index}")
        print("startとendの関係は正常です")

    def __len__(self):
        return len(self.text_df)

    @property
    def num_samples(self):
        return len(self.text_df)

    def get_text_df_vad(self):
        return self.text_df_vad

    def get_text_df_vad_update(self):
        return self.text_df_vad_update

    # def text_df_to_yaml(self, yaml_path):
    #     """text_dfをyamlに保存する"""
    #     with open(yaml_path, 'w') as f:
    #         yaml.dump(self.text_df.to_dict(), f)
