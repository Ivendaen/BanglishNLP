
from pathlib import Path
import shutil
import pandas as pd

source = Path("data/annotations/lid_banglish_candidates.csv")
target = Path("data/annotations/lid_candidates.csv")

candidates = pd.read_csv(source, keep_default_na=False)
main = pd.read_csv(target, keep_default_na=False)

# Select only manually reviewed Banglish sentences
banglish = candidates[
    (candidates["label"].str.strip() == "BANGLISH")
    & (candidates["review_status"].str.strip().str.lower() == "reviewed")
].copy()

print("Reviewed Banglish examples available:", len(banglish))

# Check sufficient annotations before changing the main dataset
if len(banglish) < 50:
    raise ValueError("Need at least 50 reviewed Banglish examples.")

# Exclude examples already included in the main dataset
existing_texts = set(main["text"].str.strip().str.casefold())
banglish = banglish[
    ~banglish["text"].str.strip().str.casefold().isin(existing_texts)
]

banglish = banglish.drop_duplicates(subset=["text"]).head(50)

if len(banglish) != 50:
    raise ValueError(
        "Fewer than 50 new unique Banglish examples are available."
    )

if set(banglish["id"]) & set(main["id"]):
    raise ValueError("Duplicate sentence IDs detected.")

# Backup existing annotations
backup = target.with_name("lid_candidates_before_banglish.csv")
if not backup.exists():
    shutil.copy2(target, backup)

# Merge without changing existing rows
combined = pd.concat(
    [main, banglish[main.columns]],
    ignore_index=True
)

combined.to_csv(target, index=False, encoding="utf-8-sig")

print("Banglish examples added:", len(banglish))
print("Total LID sentences:", len(combined))
print("\nLabel distribution:")
print(combined["label"].value_counts())
