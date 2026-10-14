
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

input_file = (
    ROOT / "data/raw/collected/"
    "train-00000-of-00001.parquet"
)

output_file = ROOT / "data/raw/pilot_sentences.csv"

df = pd.read_parquet(input_file)

print("Available columns:", df.columns.tolist())
print("Total rows:", len(df))

# Keep the original text, not the sentiment labels.
df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str)

# Remove empty examples and exact duplicates.
df = df[df["text"].str.strip() != ""]
df = df.drop_duplicates(subset=["text"])

# Select 200 examples reproducibly.
pilot = df.sample(
    n=min(200, len(df)),
    random_state=42
)[["text"]].copy()

pilot.insert(
    0, "id",
    [f"p{i:04d}" for i in range(1, len(pilot) + 1)]
)

pilot["source"] = "bn_code_mix_sentiment_dataset"

pilot.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("Pilot rows:", len(pilot))
print("Saved to:", output_file)
