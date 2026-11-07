
from pathlib import Path
import pandas as pd

source = Path("data/raw/lid/english/emotion_english.parquet")
output = Path("data/annotations/lid_english_train_candidates.csv")

def normalize(text):
    return " ".join(str(text).split()).casefold()

# Exclude all development and test sentences
used = set()

for path in [
    Path("data/dev/lid_dev.csv"),
    Path("data/test/lid_test.csv"),
]:
    df = pd.read_csv(path, keep_default_na=False)
    used.update(df["text"].map(normalize))

# Load English dataset
df = pd.read_parquet(source)

if "text" not in df.columns:
    raise ValueError("Dataset must contain a text column")

df = df.sample(frac=1, random_state=123)

candidates = []

for value in df["text"].dropna():
    sentence = " ".join(str(value).split())
    key = normalize(sentence)

    latin_count = sum(
        c.isascii() and c.isalpha() for c in sentence
    )

    bangla_count = sum(
        "\u0980" <= c <= "\u09FF" for c in sentence
    )

    if (
        20 <= len(sentence) <= 220
        and latin_count >= 15
        and bangla_count == 0
        and key not in used
    ):
        used.add(key)

        candidates.append({
            "id": f"entrain{len(candidates)+1:04d}",
            "text": sentence,
            "label": "",
            "source": "dair-ai/emotion",
            "annotator": "",
            "review_status": "Pending",
            "notes": "",
        })

    if len(candidates) == 250:
        break

if output.exists():
    raise FileExistsError(
        "Training CSV already exists. "
        "Do not overwrite previous annotations."
    )

output.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame(candidates).to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("English training candidates:", len(candidates))
print("Saved to:", output)
