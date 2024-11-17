import numpy as np
import librosa

# 音声のポーズ区間を閾値を用いて検出する関数

# 使用方法:
# audio_preprocessingからextract_waveform, convert_db, get_pause_rangesをインポートします

# # 使用例
# wav_path = "あなたのオーディオファイルへのパス"
# wav, sr = extract_waveform(wav_path)
# db = convert_db(wav)
# pause_ranges = get_pause_ranges(db)
# print(pause_ranges)

def extract_waveform(audio_file_path, sr=24000):
    waveform, sample_rate = librosa.load(audio_file_path, sr=sr, mono=True)
    return waveform, sample_rate

def convert_db(waveform):
    db = librosa.amplitude_to_db(waveform, ref=np.max)
    return db

def run_length_encoding(arr, min_run_length=3):
    diff = np.diff(arr)
    run_starts = np.where(diff != 0)[0] + 1
    run_lengths = np.diff(np.concatenate(([0], run_starts, [len(arr)])))
    result = np.repeat(run_lengths >= min_run_length, run_lengths)
    return result

def run_length_encoding_range(arr, min_run_length=3):
    diff = np.diff(arr)
    run_starts = np.where(diff != 0)[0] + 1
    run_lengths = np.diff(np.concatenate(([0], run_starts, [len(arr)])))
    runs = np.concatenate(([0], run_starts))
    ranges_with_length = [(start, start + length, length) for start, length in zip(runs, run_lengths) 
                          if length >= min_run_length and arr[start]]
    return ranges_with_length

def detect_pause_position(db_sequence, db_threshold=-50, time_threshold=0.05, sample_rate=24000):
    num_samples_threshold = int(time_threshold * sample_rate)
    pause_mask = db_sequence < db_threshold
    pause_positions = run_length_encoding(pause_mask, min_run_length=num_samples_threshold)
    return pause_positions

def get_pause_ranges(db_sequence, db_threshold=-50, time_threshold=0.05, sample_rate=24000):
    pause_positions = detect_pause_position(db_sequence, db_threshold, time_threshold, sample_rate)
    pause_ranges = run_length_encoding_range(pause_positions, min_run_length=1)
    return pause_ranges
