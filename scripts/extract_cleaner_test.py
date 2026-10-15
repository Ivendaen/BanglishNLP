
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    ROOT / "data/raw/collected/"
    "train-00000-of-00001.parquet"
)

DEVELOPMENT_FILE = (
    ROOT / "data/annotations/cleaner_gold.csv"
)

OUTPUT_FILE = (
    ROOT / "data/annotations/cleaner_test_candidates.csv"
)


def comparison_key(text):
    """Normalize text only for duplicate detection."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


# 1. Load original dataset
df = pd.read_parquet(SOURCE_FILE, columns=["text"])
df["source_row"] = df.index

# 2. Load the previous 100 development examples
development = pd.read_csv(
    DEVELOPMENT_FILE,
    encoding="utf-8-sig"
)

# 3. Get previous sentence IDs and text
used_rows = set(
    pd.to_numeric(
        development["source_row"],
        errors="coerce"
    ).dropna().astype(int)
)

used_texts = set(
    development["noisy_text"]
    .dropna()
    .map(comparison_key)
)

# 4. Remove missing and empty sentences
df = df.dropna(subset=["text"]).copy()

df["_key"] = df["text"].map(comparison_key)
df = df[df["_key"] != ""]

# 5. Remove duplicate sentences
df = df.drop_duplicates(subset=["_key"])

# 6. Exclude ALL previous development examples
df = df[
    ~df["source_row"].isin(used_rows)
    & ~df["_key"].isin(used_texts)
].copy()

print("Eligible unseen sentences:", len(df))

# 7. Randomly select 100 new sentences
if len(df) < 100:
    raise ValueError("Not enough unseen sentences available.")

sample = df.sample(
    n=100,
    random_state=2026
).reset_index(drop=True)

# 8. Prepare the annotation file
result = pd.DataFrame({
    "id": [
        f"t{i:04d}"
        for i in range(1, 101)
    ],
    "source_row": sample["source_row"],
    "noisy_text": sample["text"],
    "expected_clean_text": "",
    "source": "bn_code_mix_sentiment_dataset"
})

# 9. Save without overwriting an existing test set
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Test candidates already exist: {OUTPUT_FILE}"
    )

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Selected test sentences:", len(result))
print("Saved to:", OUTPUT_FILE)
print("Extraction completed!")
