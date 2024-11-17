#!/usr/bin/env python

import re
import wave

import numpy as np

# Usage:
# $ fix_align.py <log-file> <wav-file> <lab-file> <out-file>
# OR
# $ fix_align.py -d directory


# 1. ログファイルから削除されたサンプル区間を読み取る
#    - NOTE: ログファイルのサンプル区間表記 b-e は e 番目のサンプルを *含む* 区間を意味するが
#            読み込むときには終了位置として e+1 を保存する
# 2. アラインメントに使われたサンプル数を読み取る
#
# * Julius はおそらく 320000 サンプルごとに区切って処理（ゼロ区間削除）をしており，
#   ログに表示される b-e の表記はときどき b, e が小さい方に戻る．
#   そのため一応ログ読み込み時に WAV のサンプル列を渡し，
#       samples[ 320000 * k + b : 320000 * k + e + 1 ]
#   がゼロ区間であることが確認できるまで k を増やす処理を入れる．
#   （完全な対策ではないが，だいたいそれで済むことを期待する）
#
# INPUT:
# - fhandle: ログファイルのハンドル
# - samples: 音声サンプル列
#
# OUTPUT: (nsamples, ranges)
# - nsamples: ログファイルに書かれている，アラインメントされたサンプル数
# - ranges: 削除された区間: [(begin_sample, end_sample)] ... [begin_sample, end_sample) が削除区間
#
def collect_stripped_ranges(fhandle, samples):
    offset = 0
    frame_len = 320000

    nsamples = None
    ranges = []
    for line in fhandle:
        if m := re.match(
            r"^Warning: strip: sample (\d+)-(\d+) (?:has zero value|is invalid), stripped",
            line,
        ):
            bgn = int(m.group(1)) + offset  # strip された区間の開始位置
            end = int(m.group(2)) + offset  # strip された区間の終了位置

            assert end < len(
                samples
            ), f"strip range exceeds samples: {end} >= {len(samples)}"
            while (ranges and bgn < ranges[0][1]) or any(
                s != 0 for s in samples[bgn : end + 1]
            ):
                # バッファが切り替わったらしいのでバッファ長の分位置を進める
                bgn += frame_len
                end += frame_len
                offset += frame_len
                # assert end < len(samples)
                # 外から捉えるために、assertではなくraiseでエラーを出す
                if end >= len(samples):
                    raise ValueError(
                        f"strip ranges exceeds STAT samples: {end} >= {len(samples)}"
                    )

            # ここにきたときは  ranges[0][1] <= bgn かつ all(s == 0 for s in samples[bgn : end + 1])
            ranges.append((bgn, end + 1))
        elif m := re.match(r"^STAT: (\d+) samples", line):
            if nsamples:
                assert False, "dup sample count"
            nsamples = int(m.group(1))

    assert nsamples, "no sample count"

    return nsamples, ranges


# WAVファイル読み込み； 1ch, 16bit を前提とする
def read_wav(fn):
    with wave.open(fn, "r") as f:
        ch = f.getnchannels()
        width = f.getsampwidth()
        fr = f.getframerate()
        fn = f.getnframes()

        assert ch == 1 and width == 2

        data = f.readframes(f.getnframes())
        return np.frombuffer(data, dtype=np.int16)


# Julius によるアラインメント結果の lab ファイルを読み込む
def read_lab_file(fhandle):
    phonemes = []
    for line in fhandle:
        b, e, p = line.split()
        phonemes.append((float(b), float(e), p))

    # sanity check
    for i, (b, e, _) in enumerate(phonemes):
        if i == 0:
            # 最初の要素の開始時刻は 0
            assert b == 0.0
        else:
            # 第２要素以降の開始時刻はその前の時刻の終了時刻と一致しているべき
            assert b == phonemes[i - 1][1], f"b@{i} = {b}, e@{i-1} = {phonemes[i-1][1]}"

    return phonemes


# lab ファイルの時間幅をサンプル番号に換算
# INPUT:
# - dur_phonemes: [(b, e, p)] .. b, e は時刻（秒）
# - nsamples: アラインメント対象になった全サンプル数
# - srate: サンプリングレート
#
# OUTPUT:
#  [(b_sample, e_sample, phoneme)] ... b_sample から (e_sample-1) までが phoneme のサンプル
def time_to_sample(dur_phonemes, nsamples, srate=16000):
    phonemes = []
    for i, (b, e, p) in enumerate(dur_phonemes):
        b_sample = int(b * srate)
        e_sample = int(e * srate)
        if i == len(dur_phonemes) - 1:
            e_sample = max(e_sample, nsamples)
        phonemes.append((b_sample, e_sample, p))
    return phonemes


# アラインメント結果と削除区間を併合する
# - 削除区間のサンプル数が threshold 未満のとき
#   -> 削除区間（の開始位置）を含む音素の区間に組み入れる
#
# - 削除区間のサンプル数が threshold 以上のとき
#   -> ある音素の区間 p に含まれる削除区間を s1, s2, ..., sn とするとき，p のうち
#      - s1 より前
#      - sk.end 以降で sk+1.begin より前
#      - sn より後
#      のうち最長の区間が正しい p の区間とし，それ以外の部分は sp とする
#
# INPUT:
# - phonemes: [(b_sample, e_sample, phoneme)]
# - silences: [(b_silence, e_silence)] ... 削除区間のリスト
# - threshold: サンプル数 (int)
# OUTPUT:
# - [(b_sample, e_sample, phoneme)]
def restore_silence(phonemes, silences, threshold=1600):  # 1600 sample @ 16kHz = 100ms
    # Phase 1: とりあえず全ての削除区間の長さを音素区間に併合
    merged = []  # 削除区間を音素区間に併合したもの
    offset = 0  # 併合した削除区間長の累積値
    long_silences = []  # 閾値以上の長さの削除区間

    silences = silences.copy()  # pop するのでコピーしておく

    for pb, pe, phoneme in phonemes:
        pb += offset
        pe += offset

        while silences and pb <= silences[0][0] < pe:
            sb, se = silences.pop(0)
            sdur = se - sb

            # ひとまずこの音素区間に削除区間の長さを含めておく
            pe += sdur
            offset += sdur

            # 長い削除区間は後で処理する
            if sdur >= threshold:
                long_silences.append((sb, se))

        # 削除区間を併合した音素区間を保存
        merged.append((pb, pe, phoneme))

    # まだ削除区間が残っていれば silE に併合
    # silE に併合したので削除区間の長さに関係なくそれで処理は終わり
    while silences:
        assert merged and merged[-1][2] == "silE"
        sb, se = silences.pop(0)
        pb, pe, phoneme = merged.pop()
        # assert pb <= sb <= pe, f"{pb} <= {sb} <= {pe}, offset = {offset}"
        if not (pb <= sb <= pe):
            raise ValueError(
                f"{pb} <= {sb} <= {pe}, offset = {offset}, merged = {merged}"
            )

        pe += se - sb
        merged.append((pb, pe, phoneme))

    # Phase 2: 長い削除区間を sp 区間として切り出す
    split = []  # 長く削除区間を sp 区間として含めた音素区間

    for pb, pe, phoneme in merged:
        # この音素区間に含まれている長い削除区間を集める
        sils = []
        while long_silences and pb <= long_silences[0][0] < pe:
            assert long_silences[0][1] <= pe
            sils.append(long_silences.pop(0))

        # 削除区間を含まない音素区間ならそのまま保存
        if len(sils) == 0:
            split.append((pb, pe, phoneme))
            continue

        # 最初の無音区間の前・削除区間に挟まれた区間・最後の無音区間の後
        # のうち最も長いものを探す
        best_b = pb
        best_e = sils[0][0]
        best_dur = best_e - best_b

        for i in range(1, len(sils)):
            dur = sils[i][0] - sils[i - 1][1]
            if dur > best_dur:
                best_b = sils[i - 1][1]
                best_e = sils[i][0]
                dur = best_dur

        if pe - sils[-1][1] > best_dur:
            best_b = sils[-1][1]
            best_e = pe

        assert best_e > best_b

        # 新しい無音区間および縮めた音素区間を記録
        if pb < best_b:  # 音素区間の前に sp 区間を作成
            split.append((pb, best_b, "sp"))

        split.append((best_b, best_e, phoneme))  # 縮めた音素区間

        if best_e < pe:  # 音素区間の後に sp 区間を作成
            split.append((best_e, pe, "sp"))

    # Phase 3: 隣接する sp (silB, silE) 区間は併合する
    output = []
    while split:
        b, e, p = split.pop(0)

        if p in ["silB", "silE", "sp"]:
            while split and split[0][2] in ["silB", "silE", "sp"]:
                b2, e2, p2 = split.pop(0)
                assert e == b2
                e = e2
                if p == "sp":  # p が silB ならそのままにする
                    p = p2

        output.append((b, e, p))

    return output


# lab ファイル形式で書き出し
def write_lab_file(fhandle, phonemes, srate=16000):
    for b, e, p in phonemes:
        print(f"{b / srate:.7f} {e / srate:.7f} {p}", file=fhandle)


# main
def restore(fn_log, fn_wav, fn_lab, fn_out):
    # 音声ファイル読み込み
    samples = read_wav(fn_wav)

    # Julius によるアライン結果
    with open(fn_lab) as f:
        dur_phonemes = read_lab_file(f)

    # JuliusSegmentの結果が上手くっていないと、Labファイルが空になることがある
    # その場合は、元のLabファイルをコピーして終了
    if len(dur_phonemes) == 0:
        print(f"empty lab file: {fn_lab}", file=sys.stderr)
        print(f"then, copying {fn_lab} to {fn_out}", file=sys.stderr)
        with open(fn_lab) as f:
            with open(fn_out, "w") as g:
                for line in f:
                    print(line, end="", file=g)
        return

    # ログファイルから削除区間とアラインされた総サンプル数を取得
    with open(fn_log) as f:
        try:
            n_aligned_samples, silences = collect_stripped_ranges(f, samples)
        except ValueError as e:
            print(f"ValueError: {e}", file=sys.stderr)
            print(f"then, copying {fn_lab} to {fn_out}", file=sys.stderr)
            with open(fn_lab) as f:
                with open(fn_out, "w") as g:
                    for line in f:
                        print(line, end="", file=g)
            return

    # アラインしたサンプル数 + 削除したサンプル数 = 元ファイルのサンプル数のはず
    orig_n_samples = n_aligned_samples + sum((e - b) for b, e in silences)
    assert orig_n_samples == len(samples)

    # # Julius によるアライン結果
    # with open(fn_lab) as f:
    #     dur_phonemes = read_lab_file(f)

    # 時刻をサンプル番号に換算
    phonemes = time_to_sample(dur_phonemes, n_aligned_samples)

    # 削除区間を併合
    try:
        merged = restore_silence(phonemes, silences)
    except ValueError as e:
        print(f"ValueError: {e}", file=sys.stderr)
        print(f"then, copying {fn_lab} to {fn_out}", file=sys.stderr)
        with open(fn_lab) as f:
            with open(fn_out, "w") as g:
                for line in f:
                    print(line, end="", file=g)
        return

    # Lab ファイルとして書き出し
    with open(fn_out, "w") as f:
        write_lab_file(f, merged)


# ---------------------------------------------------------------

# Usage:
# $ fix_align.py <log-file> <wav-file> <lab-file> <out-file>
# OR
# $ fix_align.py <directory>

import os
import sys
from glob import glob

if len(sys.argv) == 2:  # directory mode
    dir_name = sys.argv[1]

    # *.wav に対して *.lab, *.log があれば *.lab2 を作成
    for fn_wav in glob(f"{dir_name}/*.wav"):
        fn_lab = fn_wav[:-4] + ".lab"
        fn_log = fn_wav[:-4] + ".log"
        fn_lab2 = fn_wav[:-4] + ".lab2"

        if os.path.isfile(fn_lab) and os.path.isfile(fn_log):
            print(f"processing {fn_log} .. ", end="", flush=True, file=sys.stderr)
            restore(fn_log, fn_wav, fn_lab, fn_lab2)
            print("done", flush=True, file=sys.stderr)

elif len(sys.argv) == 5:
    fn_log = sys.argv[1]
    fn_wav = sys.argv[2]
    fn_lab = sys.argv[3]
    fn_out = sys.argv[4]

    restore(fn_log, fn_wav, fn_lab, fn_out)
else:
    print(
        "Usage:\n"
        + f"{sys.argv[0]} <log-file> <wav-file> <lab-file> <out-file>\n"
        + "or\n"
        f"{sys.argv[0]} <directory>",
        file=sys.stderr,
    )
    sys.exit(1)
