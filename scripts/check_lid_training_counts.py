
from pathlib import Path
import pandas as pd

BASE = Path("data/annotations")

FILES = {
    "BANGLA": BASE / "lid_bangla_train_candidates.csv",
    "ENGLISH": BASE / "lid_english_train_candidates.csv",
}

def normalize(text):
    return " ".join(str(text).split()).casefold()

# Load development and test sentences to prevent overlap
used = set()

for path in [
    Path("data/dev/lid_dev.csv"),
    Path("data/test/lid_test.csv"),
]:
    df = pd.read_csv(path, keep_default_na=False)
    used.update(df["text"].map(normalize))

print("=== TRAINING DATA VERIFICATION ===")

for label, path in FILES.items():
    df = pd.read_csv(path, keep_default_na=False)

    df["label"] = df["label"].str.strip().str.upper()
    df["review_status"] = (
        df["review_status"].str.strip().str.lower()
    )
    df["text_key"] = df["text"].map(normalize)

    reviewed = df[
        (df["label"] == label)
        & (df["review_status"] == "reviewed")
        & (df["text_key"] != "")
    ].copy()

    reviewed = reviewed.drop_duplicates("text_key")

    overlap = reviewed["text_key"].isin(used)
    eligible = reviewed[~overlap].copy()

    print(f"\n{label}")
    print("Total candidates:", len(df))
    print("Unique reviewed:", len(reviewed))
    print("Evaluation overlap:", int(overlap.sum()))
    print("Eligible for training:", len(eligible))
    print("Target:", 200)

    used.update(eligible["text_key"])

    if len(eligible) >= 200:
        print("Status: READY")
    else:
        print("Status: NEED MORE DATA")

print("\nOriginal files were not modified.")
