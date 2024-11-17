from collections import namedtuple

import MeCab


def mecab_wakati_generator(
    text: str,
    mecab_args: str = "-r /dev/null -d /usr/local/lib/mecab/dic/ipadic -Owakati",
):
    """mecabで分かち書きを行うジェネレータ.

    Parameters:
    - text (str): The text to analyze.
    - mecab_args (str): Optional arguments for MeCab Tagger.

    Yields:
        str: 分かち書きされたテキスト.
    """
    tagger = MeCab.Tagger(mecab_args)
    words = tagger.parse(text).split()
    for word in words:
        yield word


def mecab_detailed_generator(
    text: str, mecab_args: str = "-r /dev/null -d /usr/local/lib/mecab/dic/ipadic"
):
    """Perform detailed morphological analysis on the given Japanese text using MeCab.

    Parameters:
    - text (str): The text to analyze.
    - mecab_args (str): Optional arguments for MeCab Tagger.

    Yields:
    - IPAWord: A namedtuple containing the following fields:
        - surface (str): Surface form of the word.
        - pos0 (str): Part-of-speech, large category (e.g., Noun).
        - pos1 (str): Part-of-speech, small category (e.g., Proper noun).
        - pos2 (str): Unused field (always None).
        - pos3 (str): Unused field (always None).
        - infl_type (str): Inflection type.
        - infl_form (str): Inflection form.
        - base (str): Base form.
        - yomi (str): Reading.
        - pron (str): Pronunciation.
    """
    IPAWord = namedtuple(
        "IPAWord",
        "surface pos0 pos1 pos2 pos3 infl_type infl_form base yomi pron".split(),
        defaults=(None,) * 10,
    )

    tagger = MeCab.Tagger(mecab_args)

    for line in tagger.parse(text).split("\n"):
        if line == "EOS":
            break
        surface, features = line.split("\t")
        word = IPAWord(surface, *features.split(","))

        yield word


def main(text: str):
    print("text:")
    print(text)
    print()
    print("mecab_wakati_generator:")
    for word in mecab_wakati_generator(text):
        print(word)
    print()
    print("mecab_detailed_generator:")
    for word in mecab_detailed_generator(text):
        print(word)


if __name__ == "__main__":
    text = "すもももももももものうち"
    text = "「お菓子がひとつあります。"
    text = "親子の銀狐は、洞穴から出ました。"
    main(text)
