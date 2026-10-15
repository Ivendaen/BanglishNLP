
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


def test_space_before_comma():
    text = "Hello , world !"
    expected = "Hello, world!"
    assert clean_text(text) == expected


def test_space_before_question_mark():
    text = "Ami    valo achi ?"
    expected = "Ami valo achi?"
    assert clean_text(text) == expected


def test_space_before_bangla_punctuation():
    text = "আমি    ভালো আছি ।"
    expected = "আমি ভালো আছি।"
    assert clean_text(text) == expected


def test_missing_space_after_comma():
    assert clean_text("chilo,ajke") == "chilo, ajke"
    assert clean_text("miaa,freefire") == "miaa, freefire"


def test_missing_space_after_period():
    assert clean_text("hoy.so") == "hoy. so"
    assert clean_text("Hi.Every one.tumder") == "Hi. Every one. tumder"


def test_mixed_comma_period():
    assert clean_text("petam ,..") == "petam."
    assert clean_text("howq. ...love") == "howq. love"


def test_preserve_email_and_numbers():
    text = "user.name@gmail.com paid 1,000 for 3.14 items"
    assert clean_text(text) == text


def test_email_followed_by_missing_comma_space():
    text = "Email user.name@gmail.com,please"
    expected = "Email user.name@gmail.com, please"
    assert clean_text(text) == expected
