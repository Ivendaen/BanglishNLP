
import pytest

from banglishnlp.tokenizer import (
    tokenize,
    tokenize_with_spans,
)


def test_bangla():
    assert tokenize("আমি আজকে বাড়ি যাব।") == [
        "আমি", "আজকে", "বাড়ি", "যাব", "।"
    ]


def test_banglish():
    assert tokenize("Ami ajke bari jabo") == [
        "Ami", "ajke", "bari", "jabo"
    ]


def test_code_mixed():
    assert tokenize("আমি today বাড়ি যাব!") == [
        "আমি", "today", "বাড়ি", "যাব", "!"
    ]


def test_punctuation():
    assert tokenize("Hello, world!") == [
        "Hello", ",", "world", "!"
    ]


def test_numbers():
    assert tokenize("3.14 টাকা ১২.৫") == [
        "3.14", "টাকা", "১২.৫"
    ]


def test_hashtag_emoji():
    assert tokenize("Hi #বাংলা 😊!") == [
        "Hi", "#বাংলা", "😊", "!"
    ]


def test_spans():
    text = "আমি  today যাব!"

    for token, start, end in tokenize_with_spans(text):
        assert text[start:end] == token


def test_invalid_input():
    with pytest.raises(TypeError):
        tokenize(123)


def test_email_tokenization():
    assert tokenize("Email test.user@example.com!") == [
        "Email",
        "test.user@example.com",
        "!",
    ]


def test_url_tokenization():
    assert tokenize("Visit https://example.com/path now!") == [
        "Visit",
        "https://example.com/path",
        "now",
        "!",
    ]


def test_complex_emoji():
    assert tokenize("Hello 👨‍👩‍👧‍👦!") == [
        "Hello",
        "👨‍👩‍👧‍👦",
        "!",
    ]

def test_url_trailing_punctuation():
    assert tokenize("Visit https://example.com/path, now!") == [
        "Visit",
        "https://example.com/path",
        ",",
        "now",
        "!",
    ]


def test_email_with_subdomain():
    assert tokenize("Mail hello.world+test@example.co.uk.") == [
        "Mail",
        "hello.world+test@example.co.uk",
        ".",
    ]


def test_complex_emoji_variations():
    assert tokenize("🇧🇩 👍🏽 1️⃣") == [
        "🇧🇩",
        "👍🏽",
        "1️⃣",
    ]


def test_token_offsets_with_url_and_emoji():
    text = "Ami  https://example.com/path, 😊!"

    tokens_with_spans = tokenize_with_spans(text)

    assert [token for token, _, _ in tokens_with_spans] == [
        "Ami",
        "https://example.com/path",
        ",",
        "😊",
        "!",
    ]

    for token, start, end in tokens_with_spans:
        assert text[start:end] == token
