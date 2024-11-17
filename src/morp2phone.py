import pyopenjtalk

# 子音（または無し） + 以下の音素で１モーラ
PH_OF_MORA = set("a i u e o A I U E O N cl".split())


# 発話中の形態素を表す辞書のリストに音素列を追加して返す
def morphemes_with_phonemes(utterance):
    # NJD 形式の形態素のリスト
    morphemes = pyopenjtalk.run_frontend(utterance)

    # 音素列を入れるフィールドを各形態素に追加
    for morpheme in morphemes:
        morpheme["phonemes"] = []

    # mopheme_ix_of[mora_ix] = mora_ix 番目のモーラが属する形態素の番号
    # e.g.,       sentence = "わたし は  カモメ" のとき
    #       morpheme_ix_of = [0,0,0, 1, 2,2,2]
    morpheme_ix_of = []
    for i, morpheme in enumerate(morphemes):
        morpheme_ix_of += [i] * morpheme["mora_size"]

    # g2p の結果 = フルコンテキストラベルから取り出した音素列 を該当する形態素に追加していく
    mora_ix = 0
    for phoneme in pyopenjtalk.g2p(utterance, join=False):
        assert mora_ix < len(morpheme_ix_of)
        morphemes[morpheme_ix_of[mora_ix]]["phonemes"].append(phoneme)
        if phoneme in PH_OF_MORA:
            mora_ix += 1

    # 全てのモーラの音素が入ったはず
    assert len(morpheme_ix_of) == mora_ix

    return morphemes

from pprint import pprint
pprint(morphemes_with_phonemes("わたしは、カモメ"))
