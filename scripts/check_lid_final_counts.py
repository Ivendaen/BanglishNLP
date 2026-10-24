
import pandas as pd

old_path = "data/annotations/lid_banglish_candidates.csv"
new_path = "data/annotations/lid_code_mixed_extra_candidates.csv"

old = pd.read_csv(old_path, keep_default_na=False)
new = pd.read_csv(new_path, keep_default_na=False)

def key(text):
    return " ".join(str(text).split()).casefold()

for df in [old, new]:
    df["text_key"] = df["text"].map(key)
    df["label"] = df["label"].str.strip().str.upper()
    df["review_status"] = (
        df["review_status"].str.strip().str.lower()
    )

# Only count manually reviewed rows
old_reviewed = old[old["review_status"] == "reviewed"]
new_reviewed = new[new["review_status"] == "reviewed"]

# Exclude new sentences appearing anywhere in old data
old_texts = set(old["text_key"])

extra = new_reviewed[
    ~new_reviewed["text_key"].isin(old_texts)
]

combined = pd.concat([old_reviewed, extra])
combined = combined.drop_duplicates("text_key")

print("=== FINAL REVIEWED COUNTS ===")
print(combined["label"].value_counts())

banglish = (combined["label"] == "BANGLISH").sum()
mixed = (combined["label"] == "CODE_MIXED").sum()

print(f"\nBANGLISH: {banglish}/50")
print(f"CODE_MIXED: {mixed}/50")

if banglish >= 50 and mixed >= 50:
    print("\nREADY FOR FINAL DATASET MERGE!")
else:
    print("\nMore reviewed examples are needed.")
