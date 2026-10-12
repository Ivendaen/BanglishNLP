
import pytest
from banglishnlp.cleaner import clean_text


def test_remove_url():
    assert clean_text(
        "Ami jabo https://example.com ajke"
    ) == "Ami jabo ajke"


def test_remove_mentions():
    assert clean_text(
        "@rahim Ami @sakib ajke jabo"
    ) == "Ami ajke jabo"


def test_spaces_in_middle():
    assert clean_text(
        "Ami     ajke      university     te jabo"
    ) == "Ami ajke university te jabo"


def test_repeated_punctuation():
    assert clean_text(
        "Ami!!! ajke??? jabo..."
    ) == "Ami! ajke? jabo."


def test_remove_punctuation():
    assert clean_text(
        "Ami!!! ajke??? jabo...",
        punctuation="remove"
    ) == "Ami ajke jabo"


def test_keep_punctuation():
    assert clean_text(
        "Ami!!! ajke??? jabo...",
        punctuation="keep"
    ) == "Ami!!! ajke??? jabo..."


def test_bangla_english_mix():
    assert clean_text(
        "আমি     ajke     university তে    যাব!!!"
    ) == "আমি ajke university তে যাব!"


def test_invalid_input():
    with pytest.raises(TypeError):
        clean_text(123)
