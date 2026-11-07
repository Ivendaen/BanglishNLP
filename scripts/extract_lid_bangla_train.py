
from pathlib import Path
import pandas as pd

source = Path("data/raw/lid/bangla/xlsum_bengali.parquet")
output = Path("data/annotations/lid_bangla_train_candidates.csv")

def normalize(text):
    return " ".join(str(text).split()).casefold()

# Prevent evaluation-set contamination
used = set()

for path in [
    Path("data/dev/lid_dev.csv"),
    Path("data/test/lid_test.csv"),
]:
    df = pd.read_csv(path, keep_default_na=False)
    used.update(df["text"].map(normalize))

# Read source corpus
df = pd.read_parquet(source)

if "summary" not in df.columns:
    raise ValueError("Expected XL-Sum summary column")

df = df.sample(frac=1, random_state=123)

candidates = []

for value in df["summary"].dropna():
    sentence = normalize(str(value).split("।")[0])

    bangla_chars = sum(
        "\u0980" <= char <= "\u09FF"
        for char in sentence
    )

    latin_chars = sum(
        char.isascii() and char.isalpha()
        for char in sentence
    )

    if (
        40 <= len(sentence) <= 220
        and bangla_chars >= 30
        and latin_chars == 0
        and sentence not in used
    ):
        used.add(sentence)

        candidates.append({
            "id": f"bntrain{len(candidates)+1:04d}",
            "text": sentence,
            "label": "",
            "source": "csebuetnlp/xlsum",
            "annotator": "",
            "review_status": "Pending",
            "notes": "",
        })

    if len(candidates) >= 250:
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

print("Bangla training candidates:", len(candidates))
print("Saved:", output)
