
from pathlib import Path
import pandas as pd

BASE = Path("data/annotations")

source = Path(
    "data/raw/collected/train-00000-of-00001.parquet"
)

output = BASE / "lid_roman_train_candidates.csv"

def normalize(text):
    return " ".join(str(text).split()).casefold()

# Exclude previously collected and evaluated sentences
used = set()

previous_files = [
    "data/dev/lid_dev.csv",
    "data/test/lid_test.csv",
    "data/annotations/lid_banglish_candidates.csv",
    "data/annotations/lid_code_mixed_extra_candidates.csv",
    "data/annotations/lid_bangla_train_candidates.csv",
    "data/annotations/lid_english_train_candidates.csv",
]

for filename in previous_files:
    path = Path(filename)
    df = pd.read_csv(path, keep_default_na=False)
    used.update(df["text"].map(normalize))

# Load original dataset
df = pd.read_parquet(source)

if "text" not in df.columns:
    raise ValueError("Missing text column")

df = df.sample(frac=1, random_state=2026)

candidates = []

for value in df["text"].dropna():
    sentence = " ".join(str(value).split())
    key = normalize(sentence)

    latin_count = sum(
        c.isascii() and c.isalpha()
        for c in sentence
    )

    bangla_count = sum(
        "\u0980" <= c <= "\u09FF"
        for c in sentence
    )

    if (
        20 <= len(sentence) <= 220
        and latin_count >= 12
        and bangla_count == 0
        and key not in used
    ):
        used.add(key)

        candidates.append({
            "id": f"romantrain{len(candidates)+1:04d}",
            "text": sentence,
            "label": "",
            "source": "Swarnadeep-28/bn_code_mix_sentiment_dataset",
            "annotator": "",
            "review_status": "Pending",
            "notes": "",
        })

    if len(candidates) == 850:
        break

if output.exists():
    raise FileExistsError(
        "Training candidates already exist. "
        "Do not overwrite annotations."
    )

output.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame(candidates).to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("Romanized candidates:", len(candidates))
print("Saved:", output)
