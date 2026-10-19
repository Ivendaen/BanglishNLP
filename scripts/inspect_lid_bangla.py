
import pandas as pd

dataset_path = "data/raw/collected/train-00000-of-00001.parquet"

df = pd.read_parquet(dataset_path)

def bengali_ratio(text):
    text = str(text)
    bengali = sum("\u0980" <= char <= "\u09FF" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)

    total = bengali + latin
    return bengali / total if total else 0

# Potential Bengali-script examples, not confirmed language labels.
candidates = df[
    df["text"].apply(
        lambda text: bengali_ratio(text) >= 0.8
        and sum("\u0980" <= c <= "\u09FF" for c in str(text)) >= 5
    )
].drop_duplicates(subset=["text"])

print("Total dataset sentences:", len(df))
print("Potential Bangla sentences:", len(candidates))

print("\nFirst 10 candidates:")
for i, text in enumerate(candidates["text"].head(10), start=1):
    print(f"{i}. {text}")
