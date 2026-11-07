
from pathlib import Path
import shutil
import pandas as pd

target_text = (
    "App ti valo laglo...tobe aro "
    "user interface koro uchit."
)

paths = [
    Path("data/annotations/lid_banglish_candidates.csv"),
    Path("data/annotations/lid_final_200.csv"),
    Path("data/dev/lid_dev.csv"),
]

def normalize(text):
    return " ".join(str(text).split()).casefold()

# Validate all files before modifying anything
files = {}

for path in paths:
    df = pd.read_csv(path, keep_default_na=False)
    mask = df["text"].map(normalize) == normalize(target_text)

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one matching sentence in {path}; "
            f"found {mask.sum()}"
        )

    files[path] = (df, mask)

# Update the one confirmed annotation correction
for path, (df, mask) in files.items():
    old_label = df.loc[mask, "label"].iloc[0]

    backup = path.with_name(
        path.stem + "_before_correction.csv"
    )

    if not backup.exists():
        shutil.copy2(path, backup)

    df.loc[mask, "label"] = "CODE_MIXED"

    if "notes" in df.columns:
        old_note = df.loc[mask, "notes"].iloc[0]
        correction = (
            f"Post-evaluation annotation correction; "
            f"previous label: {old_label}"
        )
        if correction not in old_note:
            df.loc[mask, "notes"] = (
                old_note + "; " + correction
            ).strip("; ")

    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"{path}: {old_label} -> CODE_MIXED")

print("\nCorrection recorded successfully.")
print("Original files backed up.")
print("Held-out test dataset was not modified.")
