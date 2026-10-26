import pytest

from banglishnlp.language_detector import BanglishLID


class DummyModel:
    """Mock IndicLID model for unit tests."""

    def predict(self, text, k=-1):
        return ["_label_eng_Latn"], [0.99]


@pytest.mark.parametrize(
    "text,baseline,expected",
    [
        (
            "Sam vai redmi 12 review chay",
            "ENGLISH",
            "BANGLISH",
        ),
        (
            "Shariatpur e bari koi?",
            "ENGLISH",
            "BANGLISH",
        ),
        (
            "Offer ti onek Valo.. Apnara Ki Akmot?",
            "ENGLISH",
            "BANGLISH",
        ),
        (
            "Bai light ta kibave kinbo",
            "REJECTED",
            "BANGLISH",
        ),
        (
            "Kob Valo Ekti Bank Bebhar Kore "
            "Kob Valo Laglo Thanks you Sir",
            "REJECTED",
            "CODE_MIXED",
        ),
    ],
)
def test_banglish_cue_improvements(
    monkeypatch, text, baseline, expected
):
    # Supply mock models rather than file paths.
    lid = BanglishLID(
        native_model=DummyModel(),
        roman_model=DummyModel(),
    )

    # Simulate the original detector's predictions.
    monkeypatch.setattr(
        lid,
        "_roman_code_mix",
        lambda text: False,
    )

    def fake_latin_label(text):
        if baseline == "REJECTED":
            raise ValueError("Uncertain language")
        return baseline

    monkeypatch.setattr(
        lid,
        "_latin_label",
        fake_latin_label,
    )

    # Check the new Banglish detection rules.
    assert lid.predict(text) == expected