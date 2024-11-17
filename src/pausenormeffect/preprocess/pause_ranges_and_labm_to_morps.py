import yaml
from tqdm import tqdm


sr = 24000


def insert_no_pause(ss):
    # [NO_PAUSE]を挿入する morp_pause_clip => morp_pause_clip_no_pause
    result = []
    for i in range(len(ss)):
        # リストの先頭と最後にも[NO_PAUSE]を入れるための条件分岐
        if i == 0 and "PAUSE" not in ss[i]:
            result.append("[NO_PAUSE]")
        elif i > 0 and "PAUSE" not in ss[i-1] and "PAUSE" not in ss[i]:
            result.append("[NO_PAUSE]")
        result.append(ss[i])
        if i == len(ss) - 1 and "PAUSE" not in ss[i]:
            result.append("[NO_PAUSE]")
    return result


def process_morp_and_pause_ranges(morp_lab, pause_ranges_str, sr=24000):
    """主にポーズ区間を考慮した形態素情報を返す関数.調整中"""
    pause_ranges_str = [s.split(" ") for s in pause_ranges_str]
    pause_ranges = [
        [float(start) / sr, float(end) / sr, float(length) / sr]
        for start, end, length in pause_ranges_str
    ]

    morp_pause_list = []
    continue_break = False

    # 、を除いたmorpsの数が1以上の場合は、assertする。
    for pause_start, pause_end, pause_length in pause_ranges:
        for i, s in enumerate(morp_lab):
            start, end, morp = s.strip().split(" ")
            start = float(start)
            end = float(end)
            if pause_start <= start <= pause_end and pause_start <= end <= pause_end:
                if morp not in ["、", "silB", "silE"]:
                    print(f"pause_start: {pause_start}, pause_end: {pause_end}, morp: {morp}")
                    continue_break = True
    if continue_break:
        print(f"[INFO] Skipping due to morp assertion.")
        return None

    for i, s in enumerate(morp_lab):
        start, end, morp = s.strip().split(" ")
        start = float(start)
        end = float(end)

        for pause_start, pause_end, pause_length in pause_ranges:
            if pause_start <= start <= pause_end:
                morp_pause_list.append(f"[PAUSE {pause_length}]")

        if morp == "、":
            for pause_start, pause_end, pause_length in pause_ranges:
                if start <= pause_start <= end and start <= pause_end <= end:
                    morp_pause_list.append(f"[PAUSE {pause_length}]")
                    break

        morp_pause_list.append(morp)
        for pause_start, pause_end, pause_length in pause_ranges:
            if pause_start <= end <= pause_end:
                morp_pause_list.append(f"[PAUSE {pause_length}]")

    morp_pause_clip = []
    for i, s in enumerate(morp_pause_list):
        if "PAUSE" in s:
            if len(morp_pause_clip) != 0:
                if "PAUSE" in morp_pause_clip[-1]:
                    continue
        if s == "silB" or s == "silE":
            continue
        morp_pause_clip.append(s)

    morp_pause_clip_new = []
    for i, s in enumerate(morp_pause_clip):
        if ("PAUSE" in s) and (i != 0):
            if (morp_pause_clip[i-1][-1] == "っ" or morp_pause_clip[i-1][-1] == "ッ"):
                print(f"[INFO] {morp_pause_clip[i-1]} is a sokuon, removing PAUSE.")
            else:
                morp_pause_clip_new.append(s)
        else:
            morp_pause_clip_new.append(s)

    morp_pause_str = "".join(morp_pause_clip_new)
    return {
        "morp_pause_str": morp_pause_str,
        "morp_pause_clip": morp_pause_clip_new
    }


def extract_phonemes_with_pause(phoneme_list, pause_list):
    result = []
    phoneme_iter = iter(phoneme_list)
    pause_iter = iter(pause_list)
    
    for item in pause_iter:
        if item.startswith('[PAUSE'):
            result.append(item)
        elif item == '[NO_PAUSE]':
            continue
        else:
            phoneme = next(phoneme_iter, None)
            if phoneme[0] != item:
                raise ValueError(f"phoneme: {phoneme[0]}, item: {item}")
            result.extend(phoneme[1])
    return result



# Usage
def main():
    pass



# # 入力データ
# phoneme_list = [
#     ["これ", ["k", "o", "r", "e"]],
#     ["は", ["w", "a"]],
#     ["、", ["sp"]],
#     ["私", ["w", "a", "t", "a", "sh", "i"]],
#     ["が", ["g", "a"]],
#     ["小さい", ["ch", "i:", "s", "a", "i"]],
#     ["とき", ["t", "o", "k", "i"]],
#     ["に", ["n", "i"]],
#     ["、", ["sp"]],
#     ["村", ["m", "u", "r", "a"]],
#     ["の", ["n", "o"]],
#     ["茂平", ["m", "o", "h", "e", "i"]],
#     ["と", ["t", "o"]],
#     ["いう", ["i", "u"]],
#     ["おじいさん", ["o", "j", "i:", "s", "a", "N"]],
#     ["から", ["k", "a", "r", "a"]],
#     ["きい", ["k", "i:"]],
#     ["た", ["t", "a"]],
#     ["お話", ["o", "h", "a", "n", "a", "sh", "i"]],
#     ["です", ["d", "e", "s", "u"]],
#     ["、", ["sp"]],
# ]

# pause_list = [
#     '[PAUSE 0.434875]', 'これ', '[NO_PAUSE]', 'は', '[NO_PAUSE]', '私', '[NO_PAUSE]', 'が', '[NO_PAUSE]', '小さい', 
#     '[NO_PAUSE]', 'とき', '[NO_PAUSE]', 'に', '[PAUSE 0.6646666666666666]', '、', '[NO_PAUSE]', '村', '[NO_PAUSE]', 
#     'の', '[NO_PAUSE]', '茂平', '[NO_PAUSE]', 'と', '[NO_PAUSE]', 'いう', '[NO_PAUSE]', 'おじいさん', '[NO_PAUSE]', 
#     'から', '[NO_PAUSE]', 'きい', '[NO_PAUSE]', 'た', '[NO_PAUSE]', 'お話', '[NO_PAUSE]', 'です', '[PAUSE 0.41425]'
# ]

# # 関数の呼び出し
# result = extract_phonemes_with_pause(phoneme_list, pause_list)
# print(result)



if __name__ == "__main__":
    main()

