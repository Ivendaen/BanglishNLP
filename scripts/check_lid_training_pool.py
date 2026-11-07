
from pathlib import Path
import pandas as pd

base = Path("data/annotations")

files = [
    base / "lid_bangla_candidates.csv",
    base / "lid_english_candidates.csv",
    base / "lid_banglish_candidates.csv",
    base / "lid_code_mixed_extra_candidates.csv",
]

labels = ["BANGLA", "ENGLISH", "BANGLISH", "CODE_MIXED"]

def normalize(text):
    return " ".join(str(text).split()).casefold()

# Load all previously annotated examples
frames = []

for path in files:
    df = pd.read_csv(path, keep_default_na=False)
    frames.append(df)

pool = pd.concat(frames, ignore_index=True)

pool["text_key"] = pool["text"].map(normalize)
pool["label"] = pool["label"].str.strip().str.upper()
pool["review_status"] = (
    pool["review_status"].str.strip().str.lower()
)

# Keep only reviewed annotations
pool = pool[
    (pool["review_status"] == "reviewed")
    & (pool["label"].isin(labels))
    & (pool["text_key"] != "")
].copy()

# Exclude conflicting annotations
conflicts = pool.groupby("text_key")["label"].nunique()
conflict_keys = set(conflicts[conflicts > 1].index)
pool = pool[~pool["text_key"].isin(conflict_keys)]

pool = pool.drop_duplicates("text_key")

# Exclude all development and held-out test texts
used = set()

for path in [
    Path("data/dev/lid_dev.csv"),
    Path("data/test/lid_test.csv")
]:
    df = pd.read_csv(path, keep_default_na=False)
    used.update(df["text"].map(normalize))

unused = pool[~pool["text_key"].isin(used)]

print("=== AVAILABLE TRAINING EXAMPLES ===")

for label in labels:
    count = (unused["label"] == label).sum()
    print(f"{label}: {count}")

print("\nTotal unused reviewed examples:", len(unused))
print("Conflicting texts excluded:", len(conflict_keys))
print("No files were modified.")
