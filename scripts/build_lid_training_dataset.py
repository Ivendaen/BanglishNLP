
from pathlib import Path
import pandas as pd

BASE = Path("data/annotations")
OUTPUT = Path("data/train/lid_train.csv")

LABELS = [
    "BANGLA",
    "ENGLISH",
    "BANGLISH",
    "CODE_MIXED"
]

FILES = [
    BASE / "lid_bangla_train_candidates.csv",
    BASE / "lid_english_train_candidates.csv",
    BASE / "lid_banglish_candidates.csv",
    BASE / "lid_code_mixed_extra_candidates.csv",
    BASE / "lid_roman_train_candidates.csv",
]

def normalize(text):
    return " ".join(str(text).split()).casefold()

# Collect reviewed training annotations
frames = []

for path in FILES:
    df = pd.read_csv(path, keep_default_na=False)

    required = {"text", "label", "review_status"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}")

    df["label"] = df["label"].str.strip().str.upper()
    df["review_status"] = (
        df["review_status"].str.strip().str.lower()
    )
    df["text_key"] = df["text"].map(normalize)

    df = df[
        df["label"].isin(LABELS)
        & (df["review_status"] == "reviewed")
        & (df["text_key"] != "")
    ].copy()

    df["source_file"] = path.name

    frames.append(df)

train = pd.concat(frames, ignore_index=True)

# Stop if the same text has conflicting labels
conflicts = train.groupby("text_key")["label"].nunique()

if (conflicts > 1).any():
    raise ValueError(
        "Conflicting labels detected. "
        "Resolve them before training."
    )

# Remove duplicate training sentences
train = train.drop_duplicates("text_key").copy()

# Exclude development and test examples
evaluation_texts = set()

for path in [
    Path("data/dev/lid_dev.csv"),
    Path("data/test/lid_test.csv"),
]:
    df = pd.read_csv(path, keep_default_na=False)
    evaluation_texts.update(df["text"].map(normalize))

overlap = train["text_key"].isin(evaluation_texts)
print("Evaluation overlaps excluded:", int(overlap.sum()))

train = train[~overlap].copy()

# Reproducible shuffle
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep relevant columns
train = train[
    ["text", "label", "source_file"]
]

# Never overwrite an existing training dataset
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. "
        "Back it up before rebuilding."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

train.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== FINAL TRAINING DATASET ===")
print(train["label"].value_counts())

print("\nTotal training sentences:", len(train))
print("Duplicate sentences: 0")
print("Saved to:", OUTPUT)
