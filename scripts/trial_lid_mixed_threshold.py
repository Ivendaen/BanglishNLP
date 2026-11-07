
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

MODEL = Path("models/lid_ml_v1.joblib")
DEV = Path("data/dev/lid_dev.csv")

LABELS = [
    "BANGLA", "ENGLISH",
    "BANGLISH", "CODE_MIXED"
]

print("Loading trained ML classifier...")
model = joblib.load(MODEL)

dev = pd.read_csv(DEV, keep_default_na=False)

assert len(dev) == 100
assert all(
    (dev["label"] == label).sum() == 25
    for label in LABELS
)

gold = dev["label"].tolist()
texts = dev["text"].tolist()

original = list(model.predict(texts))
probabilities = model.predict_proba(texts)

classes = list(model.classes_)
banglish_idx = classes.index("BANGLISH")
mixed_idx = classes.index("CODE_MIXED")


def evaluate(predictions):
    accuracy = accuracy_score(gold, predictions)

    macro_f1 = f1_score(
        gold,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    banglish_correct = sum(
        a == "BANGLISH" and p == "BANGLISH"
        for a, p in zip(gold, predictions)
    )

    mixed_correct = sum(
        a == "CODE_MIXED" and p == "CODE_MIXED"
        for a, p in zip(gold, predictions)
    )

    return (
        accuracy,
        macro_f1,
        banglish_correct,
        mixed_correct
    )


def apply_threshold(ratio):
    predictions = original.copy()

    for i, prediction in enumerate(original):
        if prediction != "BANGLISH":
            continue

        banglish_score = probabilities[i][banglish_idx]
        mixed_score = probabilities[i][mixed_idx]

        if mixed_score >= ratio * banglish_score:
            predictions[i] = "CODE_MIXED"

    return predictions


print("\n=== BASELINE ===")
base_acc, base_f1, base_bn, base_cm = evaluate(original)

print(f"Accuracy: {base_acc:.2%}")
print(f"Macro-F1: {base_f1:.3f}")
print(f"BANGLISH correct: {base_bn}/25")
print(f"CODE_MIXED correct: {base_cm}/25")

print("\n=== THRESHOLD EXPERIMENT ===")
print("Ratio | Accuracy | Macro-F1 | Banglish | Mixed | Changes")

for ratio in [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35]:
    trial = apply_threshold(ratio)

    acc, f1, bn, cm = evaluate(trial)

    changed = sum(
        a != b for a, b in zip(original, trial)
    )

    print(
        f"{ratio:.2f}  | "
        f"{acc:.2%}  | "
        f"{f1:.3f}  | "
        f"{bn}/25  | "
        f"{cm}/25  | "
        f"{changed}"
    )

print("\n=== MISSED CODE_MIXED CONFIDENCE ===")

for i, row in dev.iterrows():
    if (
        row["label"] == "CODE_MIXED"
        and original[i] == "BANGLISH"
    ):
        bn = probabilities[i][banglish_idx]
        cm = probabilities[i][mixed_idx]

        print(
            f"{row['id']}: "
            f"BANGLISH={bn:.3f}, "
            f"CODE_MIXED={cm:.3f}"
        )

print("\nNo model files were modified.")
print("Held-out test dataset was not accessed.")
