
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

def test_bengali_mentions():
    text = "@রহিম Ami    ajke @সাকিব১২৩ university te jabo"
    expected = "Ami ajke university te jabo"

    assert clean_text(text) == expected


def test_preserve_email():
    text = "Contact test@example.com for help"
    expected = "Contact test@example.com for help"

    assert clean_text(text) == expected


def test_url_without_protocol():
    text = "Ami ajke example.com e jabo"
    expected = "Ami ajke e jabo"

    assert clean_text(text) == expected


def test_url_with_path_without_protocol():
    text = "Visit example.org/about now"
    expected = "Visit now"

    assert clean_text(text) == expected


def test_preserve_email_remove_bare_url():
    text = "Contact test@example.com or visit example.org"
    expected = "Contact test@example.com or visit"

    assert clean_text(text) == expected


def test_mixed_question_exclamation():
    text = "Ami?!?!? jabo"
    expected = "Ami?! jabo"

    assert clean_text(text) == expected


def test_mixed_exclamation_question():
    text = "Ami!?!!?? jabo"
    expected = "Ami!? jabo"

    assert clean_text(text) == expected


def test_mixed_period_exclamation():
    text = "Ami...!!! jabo"
    expected = "Ami! jabo"

    assert clean_text(text) == expected


def test_mixed_punctuation_at_end():
    text = "Hello!!!???"
    expected = "Hello!?"

    assert clean_text(text) == expected
