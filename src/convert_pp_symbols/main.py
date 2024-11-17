# ttslearnのコードを参考にしました
# https://github.com/r9y9/ttslearn

import re
import sys

from pyopenjtalk import extract_fullcontext, g2p


def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    # 未定義 (xx) の場合、コンテキストの取りうる値以外の適当な値
    if match is None:
        return -50
    return int(match.group(1))


def pp_symbols(labels, drop_unvoiced_vowels=True):
    PP = []
    N = len(labels)

    # 各音素毎に順番に処理
    for n in range(N):
        lab_curr = labels[n]

        # 当該音素
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # 無声化母音を通常の母音として扱う
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # 先頭と末尾の sil のみ例外対応
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                PP.append("^")
            elif n == N - 1:
                # 疑問系かどうか
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("$")
                elif e3 == 1:
                    PP.append("?")
            continue
        elif p3 == "pau":
            PP.append("_")
            continue
        else:
            PP.append(p3)

        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        # アクセント句境界
        if a3 == 1 and a2_next == 1:
            PP.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            PP.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            PP.append("[")

    return PP


def main(text: str) -> list[str]:
    phonemes = g2p(text)
    lablels = extract_fullcontext(text)
    PP = pp_symbols(lablels)
    return phonemes, lablels, PP


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <text>")
        sys.exit(1)
    text = sys.argv[1]
    phonemes, _, PP = main(text)
    print("入力文字列:", text)
    print("音素列:", phonemes)
    print("韻律記号付き音素列:", " ".join(PP))
