
from pathlib import Path
import pandas as pd

source = Path("data/processed/lid_ml_dev_predictions.csv")
output = Path("data/processed/lid_ml_errors.csv")

df = pd.read_csv(source, keep_default_na=False)

errors = df[
    df["label"] != df["predicted_label"]
].copy()

print("=== ML ERROR ANALYSIS ===")
print("Total errors:", len(errors))

print("\nError distribution:")
print(
    errors.groupby(["label", "predicted_label"])
    .size()
)

print("\n=== INCORRECT SENTENCES ===")

for _, row in errors.iterrows():
    print("\nID:", row["id"])
    print("Text:", row["text"])
    print("Correct:", row["label"])
    print("Predicted:", row["predicted_label"])

output.parent.mkdir(parents=True, exist_ok=True)
errors.to_csv(output, index=False, encoding="utf-8-sig")

print("\nSaved:", output)
