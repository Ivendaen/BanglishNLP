
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = ROOT / "models" / "lid_ml_v1.joblib"

LABELS = (
    "BANGLA",
    "ENGLISH",
    "BANGLISH",
    "CODE_MIXED",
)


class MLBanglishLID:
    """TF-IDF ML classifier with a code-mixed decision threshold."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        *,
        model=None,
        mixed_ratio=0.75,
    ):
        if not 0 < mixed_ratio <= 1:
            raise ValueError("mixed_ratio must be between 0 and 1")

        self.mixed_ratio = mixed_ratio

        if model is None:
            path = Path(model_path)

            if not path.is_file():
                raise FileNotFoundError(
                    f"ML model not found: {path}"
                )

            # Load only a trusted model created by our team.
            model = joblib.load(path)

        self.model = model

        self.classes = list(self.model.classes_)

        if not set(LABELS).issubset(self.classes):
            raise ValueError("Model is missing required LID classes")

        self.banglish_idx = self.classes.index("BANGLISH")
        self.mixed_idx = self.classes.index("CODE_MIXED")

    def predict(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = " ".join(text.split())

        if not text or not any(c.isalpha() for c in text):
            raise ValueError("No language text found")

        prediction = str(self.model.predict([text])[0])

        if prediction not in LABELS:
            raise ValueError("Unsupported prediction")

        if prediction == "BANGLISH":
            probabilities = self.model.predict_proba([text])[0]

            banglish_score = probabilities[self.banglish_idx]
            mixed_score = probabilities[self.mixed_idx]

            if mixed_score >= self.mixed_ratio * banglish_score:
                return "CODE_MIXED"

        return prediction

    def batch_predict(self, texts):
        if isinstance(texts, str):
            raise TypeError("Pass an iterable of texts")

        return [self.predict(text) for text in texts]
