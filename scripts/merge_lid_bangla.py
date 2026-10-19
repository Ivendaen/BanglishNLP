
from pathlib import Path
import pandas as pd

source = Path("data/annotations/lid_bangla_candidates.csv")
target = Path("data/annotations/lid_candidates.csv")

bangla = pd.read_csv(source, keep_default_na=False)
main = pd.read_csv(target, keep_default_na=False)

# Verify annotations before merging
assert len(bangla) == 50, "Expected 50 Bangla sentences"
assert (bangla["label"] == "BANGLA").all(), "Check language labels"
assert not bangla["text"].duplicated().any(), "Duplicate Bangla sentences found"

# Avoid adding sentences already present
existing = set(main["text"].str.strip())
new_rows = bangla[~bangla["text"].str.strip().isin(existing)]

# Ensure the columns match
new_rows = new_rows[main.columns]

# Save the combined dataset
combined = pd.concat([main, new_rows], ignore_index=True)
combined.to_csv(target, index=False, encoding="utf-8-sig")

print("New sentences added:", len(new_rows))
print("Total LID sentences:", len(combined))
print("Label distribution:")
print(combined["label"].value_counts())
