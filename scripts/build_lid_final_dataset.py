
from pathlib import Path
import pandas as pd

BASE = Path("data/annotations")
OUTPUT = BASE / "lid_final_200.csv"

COLUMNS = [
    "id", "text", "label", "source",
    "annotator", "review_status", "notes"
]

SOURCES = {
    "BANGLA": [
        BASE / "lid_bangla_candidates.csv"
    ],
    "ENGLISH": [
        BASE / "lid_english_candidates.csv"
    ],
    "BANGLISH": [
        BASE / "lid_banglish_candidates.csv",
        BASE / "lid_code_mixed_extra_candidates.csv",
    ],
    "CODE_MIXED": [
        BASE / "lid_banglish_candidates.csv",
        BASE / "lid_code_mixed_extra_candidates.csv",
    ],
}

def normalize(text):
    return " ".join(str(text).split()).casefold()

def load_file(path):
    df = pd.read_csv(path, keep_default_na=False)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing {missing}")

    df = df[COLUMNS].copy()
    df["label"] = df["label"].str.strip().str.upper()
    df["review_status"] = (
        df["review_status"].str.strip().str.lower()
    )
    df["text_key"] = df["text"].map(normalize)

    return df

# Load each source only once
paths = sorted({
    path
    for group in SOURCES.values()
    for path in group
})

loaded = {path: load_file(path) for path in paths}

# Check whether any reviewed text has conflicting labels
all_rows = pd.concat(loaded.values(), ignore_index=True)
reviewed = all_rows[
    (all_rows["review_status"] == "reviewed")
    & (all_rows["label"].isin(SOURCES))
]

conflicts = (
    reviewed.groupby("text_key")["label"]
    .nunique()
)
if (conflicts > 1).any():
    raise ValueError(
        "Some reviewed sentences have conflicting labels. "
        "Resolve these before building the dataset."
    )

selected = []
used_texts = set()

for label, source_paths in SOURCES.items():
    pool = pd.concat(
        [loaded[path] for path in source_paths],
        ignore_index=True
    )

    pool = pool[
        (pool["label"] == label)
        & (pool["review_status"] == "reviewed")
        & (pool["text_key"] != "")
    ].copy()

    pool = pool.drop_duplicates("text_key")
    pool = pool[
        ~pool["text_key"].isin(used_texts)
    ]

    print(f"{label}: {len(pool)} eligible examples")

    if len(pool) < 50:
        raise ValueError(
            f"{label} needs {50 - len(pool)} "
            "more unique reviewed examples."
        )

    # Reproducible selection
    sample = pool.sample(n=50, random_state=42).copy()

    # Preserve original candidate ID in the notes
    sample["notes"] = sample.apply(
        lambda row: (
            f"Original ID: {row['id']}; "
            f"{row['notes']}"
        ).strip(),
        axis=1
    )

    prefix = {
        "BANGLA": "bn",
        "ENGLISH": "en",
        "BANGLISH": "bl",
        "CODE_MIXED": "cm"
    }[label]

    sample["id"] = [
        f"{prefix}{i:04d}"
        for i in range(1, 51)
    ]
    sample["review_status"] = "Reviewed"

    used_texts.update(sample["text_key"])
    selected.append(sample[COLUMNS])

final = pd.concat(selected, ignore_index=True)

assert len(final) == 200
assert final["id"].is_unique
assert not final["text"].map(normalize).duplicated().any()

# Avoid accidentally overwriting an existing final dataset
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Back it up before rebuilding."
    )

final.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== FINAL DATASET ===")
print(final["label"].value_counts())
print("Total:", len(final))
print("Duplicate texts: 0")
print("Saved to:", OUTPUT)
