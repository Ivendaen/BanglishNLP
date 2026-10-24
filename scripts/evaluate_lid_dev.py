
from pathlib import Path
from collections import Counter
import pandas as pd

from banglishnlp.language_detector import BanglishLID

LABELS = ["BANGLA", "ENGLISH", "BANGLISH", "CODE_MIXED"]
REJECTED = "REJECTED"

input_path = Path("data/dev/lid_dev.csv")
output_path = Path("data/processed/lid_dev_predictions.csv")

# Load the development dataset
df = pd.read_csv(input_path, keep_default_na=False)

assert len(df) == 100, "Expected 100 development sentences"
assert set(df["label"]) == set(LABELS)
assert all((df["label"] == label).sum() == 25 for label in LABELS)
assert df["review_status"].str.lower().eq("reviewed").all()

print("Loading BanglishLID models...")
lid = BanglishLID()

predictions = []
errors = []

print("Evaluating development sentences...")

for text in df["text"]:
    try:
        prediction = lid.predict(text)
        if prediction not in LABELS:
            raise RuntimeError(f"Unexpected label: {prediction}")
        predictions.append(prediction)
        errors.append("")
    except ValueError as exc:
        predictions.append(REJECTED)
        errors.append(str(exc))

df["predicted_label"] = predictions
df["correct"] = df["label"] == df["predicted_label"]
df["error_message"] = errors

# Overall accuracy: rejections count as incorrect
total = len(df)
correct = int(df["correct"].sum())
rejected = predictions.count(REJECTED)

print("\n=== DEVELOPMENT EVALUATION ===")
print("Total sentences:", total)
print("Correct predictions:", correct)
print("Rejected predictions:", rejected)
print(f"Accuracy: {correct / total:.2%}")
print(f"Coverage: {(total - rejected) / total:.2%}")

if total > rejected:
    print(
        f"Accuracy on accepted predictions: "
        f"{correct / (total - rejected):.2%}"
    )

# Calculate class-level precision, recall, F1
print("\n=== PER-CLASS METRICS ===")

metrics = []

for label in LABELS:
    tp = sum(
        (df["label"] == label)
        & (df["predicted_label"] == label)
    )
    fp = sum(
        (df["label"] != label)
        & (df["predicted_label"] == label)
    )
    fn = sum(
        (df["label"] == label)
        & (df["predicted_label"] != label)
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    metrics.append(f1)

    print(
        f"{label:12} "
        f"Precision={precision:.3f} "
        f"Recall={recall:.3f} "
        f"F1={f1:.3f}"
    )

macro_f1 = sum(metrics) / len(metrics)
print(f"\nMacro-F1: {macro_f1:.3f}")

# Confusion matrix
print("\n=== CONFUSION MATRIX ===")

matrix = pd.crosstab(
    pd.Categorical(df["label"], categories=LABELS),
    pd.Categorical(
        df["predicted_label"],
        categories=LABELS + [REJECTED]
    ),
    dropna=False,
)

matrix.index.name = "Actual"
matrix.columns.name = "Predicted"

print(matrix.to_string())

# Save predictions for error analysis
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nSaved predictions to:", output_path)
print("\n=== PREDICTION DISTRIBUTION ===")
print(Counter(predictions))
