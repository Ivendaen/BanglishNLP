
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from banglishnlp.ml_language_detector import MLBanglishLID

LABELS = ["BANGLA", "ENGLISH", "BANGLISH", "CODE_MIXED"]

dev = pd.read_csv("data/dev/lid_dev.csv", keep_default_na=False)

assert len(dev) == 100
expected_counts = {
    "BANGLA": 25,
    "ENGLISH": 25,
    "BANGLISH": 24,
    "CODE_MIXED": 26,
}

assert dev["label"].value_counts().to_dict() == expected_counts

print("Loading calibrated ML detector...")
lid = MLBanglishLID(mixed_ratio=0.75)

predictions = lid.batch_predict(dev["text"].tolist())

print("\n=== CALIBRATED ML DEVELOPMENT EVALUATION ===")

print(
    f"Accuracy: {accuracy_score(dev['label'], predictions):.2%}"
)

print(
    "Macro-F1:",
    round(
        f1_score(
            dev["label"],
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        3,
    ),
)

print("\n=== CONFUSION MATRIX ===")

matrix = confusion_matrix(
    dev["label"],
    predictions,
    labels=LABELS,
)

print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))

print("\nHeld-out test set was not accessed.")
