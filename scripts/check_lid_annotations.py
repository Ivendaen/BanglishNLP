
import pandas as pd

path = "data/annotations/lid_banglish_candidates.csv"

df = pd.read_csv(path, keep_default_na=False)

df["label"] = df["label"].str.strip().str.upper()
df["review_status"] = df["review_status"].str.strip().str.lower()

print("Total sentences:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False))

reviewed = df[df["review_status"] == "reviewed"]

print("\nReviewed annotations:")
print(reviewed["label"].value_counts())

print("\nDuplicate sentences:",
      df["text"].str.strip().str.casefold().duplicated().sum())

for label in ["BANGLISH", "CODE_MIXED"]:
    count = (reviewed["label"] == label).sum()
    print(f"{label}: {count}/50 reviewed examples")
