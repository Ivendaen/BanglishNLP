
from banglishnlp.language_detector import BanglishLID


class MockModel:
    """Simulate IndicLID without downloading model files."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, text, k=-1):
        labels = [
            "__label__" + label
            for label in self.scores
        ]
        probabilities = list(self.scores.values())
        return labels, probabilities


def test_english_sentence_stays_english():
    native = MockModel({
        "eng_Latn": 0.98,
        "other": 0.02
    })

    roman = MockModel({
        "eng_Latn": 0.999,
        "ben_Latn": 0.0005
    })

    lid = BanglishLID(
        native_model=native,
        roman_model=roman
    )

    text = (
        "i am feeling so stunned and sad about "
        "the earthquake in christchurch new zealand yesterday"
    )

    assert lid.predict(text) == "ENGLISH"


def test_english_conjunction_in_banglish():
    native = MockModel({
        "eng_Latn": 0.40,
        "other": 0.60
    })

    roman = MockModel({
        "ben_Latn": 0.80,
        "eng_Latn": 0.10
    })

    lid = BanglishLID(
        native_model=native,
        roman_model=roman
    )

    text = "onek valo but akto akto problem kore"

    assert lid.predict(text) == "CODE_MIXED"


def test_borrowed_words_remain_banglish():
    native = MockModel({
        "eng_Latn": 0.40,
        "other": 0.60
    })

    roman = MockModel({
        "ben_Latn": 0.80,
        "eng_Latn": 0.10
    })

    lid = BanglishLID(
        native_model=native,
        roman_model=roman
    )

    text = "Apu apnar watch kotha thaka kinsan"

    assert lid.predict(text) == "BANGLISH"
