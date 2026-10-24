
from pathlib import Path
import pandas as pd

source = Path("data/annotations/lid_final_200.csv")
dev_path = Path("data/dev/lid_dev.csv")
test_path = Path("data/test/lid_test.csv")

labels = ["BANGLA", "ENGLISH", "BANGLISH", "CODE_MIXED"]

df = pd.read_csv(source, keep_default_na=False)

# Validate input dataset
assert len(df) == 200, "Expected 200 sentences"
assert df["label"].value_counts().to_dict() == {
    label: 50 for label in labels
}, "Expected 50 examples for each language"

assert df["id"].is_unique, "Duplicate IDs found"
assert df["review_status"].str.lower().eq("reviewed").all(), (
    "Some sentences are not reviewed"
)

# Never overwrite an existing split
if dev_path.exists() or test_path.exists():
    raise FileExistsError(
        "A LID split already exists. "
        "Do not overwrite a frozen evaluation dataset."
    )

dev_parts = []
test_parts = []

for label in labels:
    group = df[df["label"] == label]

    # Reproducible random split
    shuffled = group.sample(frac=1, random_state=42)

    dev_parts.append(shuffled.iloc[:25])
    test_parts.append(shuffled.iloc[25:])

dev = pd.concat(dev_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

# Confirm there is no overlap
assert set(dev["id"]).isdisjoint(set(test["id"]))

def normalize(text):
    return " ".join(str(text).split()).casefold()

assert set(dev["text"].map(normalize)).isdisjoint(
    set(test["text"].map(normalize))
)

dev_path.parent.mkdir(parents=True, exist_ok=True)
test_path.parent.mkdir(parents=True, exist_ok=True)

dev.to_csv(dev_path, index=False, encoding="utf-8-sig")
test.to_csv(test_path, index=False, encoding="utf-8-sig")

print("=== DEVELOPMENT DATASET ===")
print(dev["label"].value_counts())
print("Total:", len(dev))

print("\n=== TEST DATASET ===")
print(test["label"].value_counts())
print("Total:", len(test))

print("\nNo overlap between datasets.")
print("Development:", dev_path)
print("Test:", test_path)
