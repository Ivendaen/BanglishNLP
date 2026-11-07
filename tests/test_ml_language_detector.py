
import pytest

from banglishnlp.ml_language_detector import MLBanglishLID


class MockModel:
    """Test ML prediction rules without loading model files."""

    classes_ = [
        "BANGLA",
        "BANGLISH",
        "CODE_MIXED",
        "ENGLISH",
    ]

    def __init__(self, label, scores):
        self.label = label
        self.scores = scores

    def predict(self, texts):
        return [self.label for _ in texts]

    def predict_proba(self, texts):
        return [self.scores for _ in texts]


def test_code_mixed_threshold():
    model = MockModel(
        "BANGLISH",
        [0.01, 0.50, 0.40, 0.09]
    )

    lid = MLBanglishLID(model=model, mixed_ratio=0.75)

    assert lid.predict("ami today bari jabo") == "CODE_MIXED"


def test_banglish_remains_banglish():
    model = MockModel(
        "BANGLISH",
        [0.01, 0.65, 0.25, 0.09]
    )

    lid = MLBanglishLID(model=model)

    assert lid.predict("ami ajke bari jabo") == "BANGLISH"


def test_english_remains_english():
    model = MockModel(
        "ENGLISH",
        [0.01, 0.49, 0.45, 0.05]
    )

    lid = MLBanglishLID(model=model)

    assert lid.predict("I am going home") == "ENGLISH"


def test_invalid_empty_text():
    model = MockModel(
        "BANGLISH",
        [0.01, 0.65, 0.25, 0.09]
    )

    lid = MLBanglishLID(model=model)

    with pytest.raises(ValueError):
        lid.predict("   ")


def test_invalid_threshold():
    model = MockModel(
        "BANGLISH",
        [0.01, 0.65, 0.25, 0.09]
    )

    with pytest.raises(ValueError):
        MLBanglishLID(model=model, mixed_ratio=0)


def test_batch_prediction():
    model = MockModel(
        "BANGLISH",
        [0.01, 0.65, 0.25, 0.09]
    )

    lid = MLBanglishLID(model=model)

    predictions = lid.batch_predict([
        "ami bari jabo",
        "tumi kemon acho",
    ])

    assert predictions == ["BANGLISH", "BANGLISH"]
