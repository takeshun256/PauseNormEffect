import re
import jaconv
import pyopenjtalk
from pyopenjtalk import run_frontend
from pauseprediction.analyze_jmac.mecab import mecab_wakati_generator, mecab_detailed_generator

class TextPreprocessor:
    def __init__(self):
        self.mecab_wakati_generator = mecab_wakati_generator
        self.mecab_detailed_generator = mecab_detailed_generator

    @staticmethod
    def remove_brackets_to_kanji(x):
        return re.sub(r"\[(.+?)\|(.+?)\]", r"\1", x)

    @staticmethod
    def remove_brackets_to_furigana(x):
        return re.sub(r"\[(.+?)\|(.+?)\]", r"\2", x)

    def normalize(self, text):
        return jaconv.normalize(text)

    def get_jmac_blacket_dict(self, text):
        blacket_dict = {}
        for kanji, furigana in re.findall(r"\[(.+?)\|(.+?)\]", text):
            blacket_dict[kanji] = furigana
        blacket_dict = {
            k: "".join([njd["pron"] for njd in run_frontend(self.normalize(v))])
            for k, v in blacket_dict.items()
        }
        return blacket_dict

    def preprocess_text(self, text):
        self.original_text = text
        self.blacket_removed_kanji_text = self.remove_brackets_to_kanji(text)
        self.normalized_text = self.normalize(self.blacket_removed_kanji_text)
        self.njd_features = run_frontend(self.normalized_text)
        self.jmac_blacket_dict = self.get_jmac_blacket_dict(self.original_text)

        self.morp_pron_list = []  # [[morphome, katakana], ...]
        for njd_feature in self.njd_features:
            if njd_feature["pos"] == "記号":
                if njd_feature["string"] == "、":
                    pron = "、"
                else:
                    continue
            else:
                orig = njd_feature["string"]
                pron = njd_feature["pron"]
                if orig in self.jmac_blacket_dict and self.jmac_blacket_dict[orig] != pron:
                    pron = self.jmac_blacket_dict[orig]
            pron = pron.replace("’", "")
            self.morp_pron_list.append([njd_feature["string"], pron])

        self.morp_phons_list = []
        for m, p in self.morp_pron_list:
            if p == "、":
                self.morp_phons_list.append([m, ["sp"]])
            else:
                self.morp_phons_list.append([m, jaconv.hiragana2julius(jaconv.kata2hira(p)).split(" ")])

        morp_join = "".join([m for m, p in self.morp_phons_list])
        phons_join = " ".join([p for _, phons in self.morp_phons_list for p in phons])

        output_dict = {
            "morp_join": morp_join,
            "phons_join": phons_join,
            "morp_phons_list": self.morp_phons_list,
        }
        return output_dict

    def __call__(self, text):
        return self.preprocess_text(text)


def main():
    text_preprocessor = TextPreprocessor()
    s = "これは、[私|わたし]が小さいときに、村の[茂平|もへい]という...おじいさん!?からきいたお話です。"
    print(text_preprocessor(s))


if __name__ == "__main__":
    main()
