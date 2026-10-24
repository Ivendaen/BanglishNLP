
import pandas as pd

path = "data/annotations/lid_banglish_candidates.csv"

df = pd.read_csv(path, keep_default_na=False)

df["text_key"] = df["text"].apply(
    lambda x: " ".join(str(x).split()).casefold()
)

df["label"] = df["label"].str.strip().str.upper()
df["review_status"] = (
    df["review_status"].str.strip().str.lower()
)

# Identify duplicates
duplicates = df[df.duplicated("text_key", keep=False)]

print("\n=== DUPLICATES ===")
print(duplicates[["id", "text", "label"]].to_string(index=False))

# Identify unlabeled rows
unlabeled = df[df["label"] == ""]

print("\n=== UNLABELED ===")
print(unlabeled[["id", "text"]].to_string(index=False))

# Count unique, reviewed examples
reviewed = df[
    (df["review_status"] == "reviewed")
    & (df["label"].isin(["BANGLISH", "CODE_MIXED"]))
]

unique = reviewed.drop_duplicates("text_key")

print("\n=== UNIQUE REVIEWED COUNTS ===")
for label in ["BANGLISH", "CODE_MIXED"]:
    count = (unique["label"] == label).sum()
    print(f"{label}: {count}/50")

# Check conflicting labels
conflicts = reviewed.groupby("text_key")["label"].nunique()

print("\nConflicting duplicate labels:", (conflicts > 1).sum())
