
from pathlib import Path
import pandas as pd

input_path = Path("data/processed/lid_dev_predictions.csv")
output_path = Path("data/processed/lid_dev_errors.csv")

df = pd.read_csv(input_path, keep_default_na=False)

# Select incorrectly classified sentences
errors = df[
    df["label"] != df["predicted_label"]
].copy()

print("=== LID ERROR ANALYSIS ===")
print("Total development sentences:", len(df))
print("Incorrect predictions:", len(errors))

print("\n=== ERROR TYPES ===")
print(
    errors.groupby(
        ["label", "predicted_label"]
    ).size().to_string()
)

print("\n=== BANGLISH ERRORS ===")
banglish_errors = errors[
    errors["label"] == "BANGLISH"
]

for _, row in banglish_errors.iterrows():
    print(f"\nID: {row['id']}")
    print(f"Text: {row['text']}")
    print(f"Actual: {row['label']}")
    print(f"Predicted: {row['predicted_label']}")

print("\n=== CODE_MIXED ERRORS ===")
mixed_errors = errors[
    errors["label"] == "CODE_MIXED"
]

for _, row in mixed_errors.iterrows():
    print(f"\nID: {row['id']}")
    print(f"Text: {row['text']}")
    print(f"Actual: {row['label']}")
    print(f"Predicted: {row['predicted_label']}")

# Save all incorrect predictions
output_path.parent.mkdir(parents=True, exist_ok=True)

errors.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved error report to:", output_path)
