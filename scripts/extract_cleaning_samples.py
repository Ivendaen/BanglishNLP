
from pathlib import Path
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Original downloaded dataset
INPUT_FILE = (
    ROOT / "data/raw/collected/"
    "train-00000-of-00001.parquet"
)

# Output for manual annotation
OUTPUT_FILE = (
    ROOT / "data/annotations/cleaner_candidates.csv"
)

# STEP 1: Load the dataset
df = pd.read_parquet(INPUT_FILE)

print("Original records:", len(df))

# STEP 2: Keep only text and original row number
df = df[["text"]].copy()
df["source_row"] = df.index

# STEP 3: Remove missing and empty text
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].copy()

# STEP 4: Identify duplicates using whitespace-normalized text
# Original text is not changed.
df["_key"] = (
    df["text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df.drop_duplicates(subset=["_key"])

print("Unique records:", len(df))

# STEP 5: Randomly select up to 100 sentences
sample = df.sample(
    n=min(100, len(df)),
    random_state=42
).reset_index(drop=True)

# STEP 6: Prepare the annotation file
result = pd.DataFrame({
    "id": [
        f"c{i:04d}"
        for i in range(1, len(sample) + 1)
    ],
    "source_row": sample["source_row"],
    "noisy_text": sample["text"],
    "expected_clean_text": "",
    "source": "bn_code_mix_sentiment_dataset"
})

# STEP 7: Save the candidate examples
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Selected sentences:", len(result))
print("Saved to:", OUTPUT_FILE)
print("Extraction completed!")
