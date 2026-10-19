
from pathlib import Path
import shutil
import pandas as pd

source = Path("data/annotations/lid_english_candidates.csv")
target = Path("data/annotations/lid_candidates.csv")

english = pd.read_csv(source, keep_default_na=False)
main = pd.read_csv(target, keep_default_na=False)

# Validate annotations
assert len(english) == 50, "Expected 50 English sentences"
assert (english["label"] == "ENGLISH").all(), "Check labels"
assert (english["review_status"] == "Reviewed").all(), "Review all rows"
assert not english["text"].str.strip().str.casefold().duplicated().any()
assert set(english.columns) == set(main.columns)

# Avoid duplicate sentences and IDs
existing_texts = set(main["text"].str.strip().str.casefold())
new_rows = english[
    ~english["text"].str.strip().str.casefold().isin(existing_texts)
].copy()

assert not set(new_rows["id"]).intersection(set(main["id"])), \
    "Duplicate IDs found"

# Preserve a backup before updating
shutil.copy2(target, target.with_suffix(".backup.csv"))

combined = pd.concat(
    [main, new_rows[main.columns]], ignore_index=True
)

combined.to_csv(target, index=False, encoding="utf-8-sig")

print("New English sentences added:", len(new_rows))
print("Total LID sentences:", len(combined))
print("\nLabel distribution:")
print(combined["label"].value_counts())

