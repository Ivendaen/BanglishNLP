
from pathlib import Path

import joblib

from banglishnlp.tokenizer import tokenize_with_spans
from banglishnlp.token_features import token_features


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = ROOT / "models" / "token_tagger_v1.joblib"

LABELS = {"BAN", "ENG", "ROM_BAN", "OTHER", "UNK"}


class TokenLanguageTagger:
    """ML-based token-level language identification."""

    def __init__(self, model_path=DEFAULT_MODEL, *, model=None):

        if model is None:
            model_path = Path(model_path)

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Token tagging model not found: {model_path}"
                )

            # Load only trusted models produced by our team.
            model = joblib.load(model_path)

        self.vectorizer = model["vectorizer"]
        self.classifier = model["classifier"]

        if not LABELS.issubset(
            set(self.classifier.classes_)
        ):
            raise ValueError(
                "Model is missing required token labels"
            )

    def tag(self, text: str):
        """
        Identify the language of each token.

        Returns a list of dictionaries containing:
            token
            start
            end
            label
        """

        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        spans = tokenize_with_spans(text)

        if not spans:
            return []

        tokens = [
            token
            for token, start, end in spans
        ]

        features = []

        for i, token in enumerate(tokens):

            previous = (
                tokens[i - 1]
                if i > 0
                else ""
            )

            following = (
                tokens[i + 1]
                if i + 1 < len(tokens)
                else ""
            )

            features.append(
                token_features(
                    token,
                    previous,
                    following,
                )
            )

        vectors = self.vectorizer.transform(features)

        predictions = self.classifier.predict(vectors)

        results = []

        for (token, start, end), label in zip(
            spans,
            predictions,
        ):

            results.append({
                "token": token,
                "start": start,
                "end": end,
                "label": str(label),
            })

        return results

    def batch_tag(self, texts):
        """Tag multiple sentences."""

        if isinstance(texts, str):
            raise TypeError(
                "Pass a list or iterable of sentences"
            )

        return [
            self.tag(text)
            for text in texts
        ]
