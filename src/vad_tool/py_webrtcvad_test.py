# https://qiita.com/ikuo0/items/0d5798db824f3df074af を参考に一部改変

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plot
import numpy as np
import python_speech_features as psf
import scipy
import scipy.io
import scipy.io.wavfile
import scipy.ndimage
import scipy.signal


def get_mfcc(file_name, nfft=2048):
    """MFCCを取得する関数"""
    rate, sig = scipy.io.wavfile.read(file_name)
    print(f"Sampling rate: {rate}")
    mfcc = psf.mfcc(sig, rate, nfft=nfft)
    delta = psf.delta(mfcc, 2)
    delta_delta = psf.delta(delta, 2)
    mfcc_feature = np.c_[mfcc, delta, delta_delta]
    return mfcc_feature


def get_vad_fluctuation(mfcc_power, delta_power, filter_width=10):
    """音声活動検出の変動を取得する関数"""
    mfcc_power[mfcc_power < 0] = 0
    delta_power = np.abs(delta_power)
    y = mfcc_power * delta_power
    y = scipy.ndimage.gaussian_filter(y, filter_width)
    min_id = scipy.signal.argrelmin(y, order=1)
    min_peek = np.zeros(len(y))
    min_peek[min_id] = 1
    max_id = scipy.signal.argrelmax(y, order=1)
    max_peek = np.zeros(len(y))
    max_peek[max_id] = 1
    return y, min_peek, max_peek


def get_mora_fluctuation(mfcc_power, delta_power, filter_width=4):
    """モーラの変動を取得する関数"""
    y = mfcc_power * delta_power
    y = scipy.ndimage.gaussian_filter(delta_power, filter_width)
    min_id = scipy.signal.argrelmin(y, order=1)
    min_peek = np.zeros(len(y))
    min_peek[min_id] = 1
    max_id = scipy.signal.argrelmax(y, order=1)
    max_peek = np.zeros(len(y))
    max_peek[max_id] = 1
    return y, min_peek, max_peek


def plot_results(
    data_length,
    mfcc_heatmap,
    mfcc_power,
    delta_power,
    vad,
    vad_peek_min,
    vad_peek_max,
    mora,
    mora_peek_min,
    mora_peek_max,
    vad_section,
    mora_positions,
    vad_threshold,
):
    """結果をプロットする関数"""
    xlim = [0, data_length]
    plot.style.use("classic")
    plot.figure(figsize=(12, 10))

    # heatmap
    plot.subplot(5, 1, 1)
    plot.xlim(xlim)
    plot.pcolor(mfcc_heatmap, cmap=plot.cm.Blues)

    # power, delta
    plot.subplot(5, 1, 2)
    plot.xlim(xlim)
    plot.plot(mfcc_power)
    plot.plot(delta_power)

    # vad
    plot.subplot(5, 1, 3)
    plot.xlim(xlim)
    plot.plot(vad)
    sx = np.where(vad_peek_min == 1)[0]
    sy = vad[sx]
    plot.scatter(sx, sy, c="blue")
    sx = np.where(vad_peek_max == 1)[0]
    sy = vad[sx]
    plot.scatter(sx, sy, c="red")
    yline = [vad_threshold] * data_length
    plot.plot(yline)

    # mora
    plot.subplot(5, 1, 4)
    plot.xlim(xlim)
    plot.plot(mora)
    sx = np.where(mora_peek_min == 1)[0]
    sy = mora[sx]
    plot.scatter(sx, sy, c="blue")
    sx = np.where(mora_peek_max == 1)[0]
    sy = mora[sx]
    plot.scatter(sx, sy, c="red")

    # vad
    plot.subplot(5, 1, 5)
    plot.xlim(xlim)
    plot.plot(vad_section)
    sx = np.where(mora_positions == 1)[0]
    sy = np.ones(len(sx))
    plot.scatter(sx, sy)

    plot.savefig("./fig.png")


def run():
    """メイン実行関数"""
    vad_threshold = 3

    # MFCC取得
    mfcc = get_mfcc("./sample1.wav")
    data_length = len(mfcc)
    mfcc_power = mfcc[:, 0]
    delta_power = mfcc[:, 13]

    # Voice active detection
    vad, vad_peek_min, vad_peek_max = get_vad_fluctuation(mfcc_power, delta_power)

    # mora
    mora, mora_peek_min, mora_peek_max = get_mora_fluctuation(mfcc_power, delta_power)

    # voice active detection
    vad_section = np.zeros(data_length)
    vad_section[vad >= vad_threshold] = 1
    mora_positions = np.zeros(data_length)
    mora_positions[np.where(mora_peek_max == 1)] = 1
    mora_positions[vad <= vad_threshold] = 0

    # plot data
    mfcc_heatmap = mfcc[:, np.arange(1, 13)].T

    plot_results(
        data_length,
        mfcc_heatmap,
        mfcc_power,
        delta_power,
        vad,
        vad_peek_min,
        vad_peek_max,
        mora,
        mora_peek_min,
        mora_peek_max,
        vad_section,
        mora_positions,
        vad_threshold,
    )


def get_vad_section(file_name, vad_threshold=3):
    """音声ファイルからVADセクションを取得する関数"""
    mfcc = get_mfcc(file_name)
    data_length = len(mfcc)
    mfcc_power = mfcc[:, 0]
    delta_power = mfcc[:, 13]

    vad, _, _ = get_vad_fluctuation(mfcc_power, delta_power)

    vad_section = np.zeros(data_length)
    vad_section[vad >= vad_threshold] = 1

    return vad_section


if __name__ == "__main__":
    run()

