
import pytest

from banglishnlp.token_language_tagger import TokenLanguageTagger


LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]


class FakeVectorizer:
    def transform(self, features):
        return features


class FakeClassifier:
    classes_ = LABELS

    def predict(self, features):
        results = []

        for item in features:
            word = item["word"]
            script = item["script"]

            if script == "BENGALI":
                results.append("BAN")
            elif script == "OTHER":
                results.append("OTHER")
            elif word == "today":
                results.append("ENG")
            else:
                results.append("ROM_BAN")

        return results


def make_tagger():
    return TokenLanguageTagger(
        model={
            "vectorizer": FakeVectorizer(),
            "classifier": FakeClassifier(),
        }
    )


def test_token_tags():
    tagger = make_tagger()

    result = tagger.tag("ami today বাড়ি যাব!")

    assert [item["label"] for item in result] == [
        "ROM_BAN",
        "ENG",
        "BAN",
        "BAN",
        "OTHER",
    ]


def test_token_text():
    tagger = make_tagger()

    result = tagger.tag("ami today বাড়ি যাব!")

    assert [item["token"] for item in result] == [
        "ami",
        "today",
        "বাড়ি",
        "যাব",
        "!",
    ]


def test_character_offsets():
    tagger = make_tagger()
    text = "ami today বাড়ি যাব!"

    for item in tagger.tag(text):
        assert text[item["start"]:item["end"]] == item["token"]


def test_empty_text():
    assert make_tagger().tag("") == []


def test_whitespace_text():
    assert make_tagger().tag("   ") == []


def test_invalid_input():
    with pytest.raises(TypeError):
        make_tagger().tag(123)


def test_batch_tagging():
    tagger = make_tagger()

    results = tagger.batch_tag([
        "ami today",
        "আমি যাব",
    ])

    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 2


def test_batch_rejects_string():
    with pytest.raises(TypeError):
        make_tagger().batch_tag("ami today")


def test_missing_model():
    with pytest.raises(FileNotFoundError):
        TokenLanguageTagger(
            model_path="nonexistent_token_model.joblib"
        )
