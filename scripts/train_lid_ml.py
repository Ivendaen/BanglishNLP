
from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

TRAIN = Path("data/train/lid_train.csv")
DEV = Path("data/dev/lid_dev.csv")
TEST = Path("data/test/lid_test.csv")

MODEL = Path("models/lid_ml_v1.joblib")
PREDICTIONS = Path("data/processed/lid_ml_dev_predictions.csv")

LABELS = [
    "BANGLA",
    "ENGLISH",
    "BANGLISH",
    "CODE_MIXED",
]


def normalize(text):
    return " ".join(str(text).split()).casefold()


print("Loading training and development datasets...")

train = pd.read_csv(TRAIN, keep_default_na=False)
dev = pd.read_csv(DEV, keep_default_na=False)
test = pd.read_csv(TEST, keep_default_na=False)

# Validate labels
for name, df in [("train", train), ("dev", dev)]:
    if not set(df["label"]).issubset(set(LABELS)):
        raise ValueError(f"Invalid labels in {name}")

if set(train["label"]) != set(LABELS):
    raise ValueError("Training data must contain all four classes")

if len(dev) != 100:
    raise ValueError("Expected 100 development sentences")

# Prevent exact-text leakage
train_keys = set(train["text"].map(normalize))
dev_keys = set(dev["text"].map(normalize))
test_keys = set(test["text"].map(normalize))

if train_keys & dev_keys:
    raise ValueError("Training and development data overlap")

if train_keys & test_keys:
    raise ValueError("Training and held-out test data overlap")

if dev_keys & test_keys:
    raise ValueError("Development and test data overlap")

print("\nTraining distribution:")
print(train["label"].value_counts())

print("\nBuilding TF-IDF + Logistic Regression model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=60000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

# Train only on the training dataset
model.fit(train["text"], train["label"])

print("\nTraining completed.")

# Evaluate only on the development dataset
predictions = model.predict(dev["text"])

accuracy = accuracy_score(dev["label"], predictions)
macro_f1 = f1_score(
    dev["label"],
    predictions,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

print("\n=== ML DEVELOPMENT EVALUATION ===")
print("Total development sentences:", len(dev))
print(f"Accuracy: {accuracy:.2%}")
print(f"Macro-F1: {macro_f1:.3f}")

print("\n=== PER-CLASS METRICS ===")
print(
    classification_report(
        dev["label"],
        predictions,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )
)

print("\n=== CONFUSION MATRIX ===")
matrix = confusion_matrix(
    dev["label"],
    predictions,
    labels=LABELS,
)

print(pd.DataFrame(
    matrix,
    index=LABELS,
    columns=LABELS,
))

# Save predictions for error analysis
dev["predicted_label"] = predictions
dev["correct"] = dev["label"] == dev["predicted_label"]

PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
dev.to_csv(PREDICTIONS, index=False, encoding="utf-8-sig")

# Save the trained model
MODEL.parent.mkdir(parents=True, exist_ok=True)

if MODEL.exists():
    raise FileExistsError(
        f"{MODEL} already exists. Back it up before retraining."
    )

joblib.dump(model, MODEL)

print("\nModel saved:", MODEL)
print("Predictions saved:", PREDICTIONS)
print("\nHeld-out test set was NOT evaluated.")
