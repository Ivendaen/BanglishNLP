
from pathlib import Path
import pandas as pd

source = Path(
    "data/raw/collected/train-00000-of-00001.parquet"
)

existing_path = Path(
    "data/annotations/lid_banglish_candidates.csv"
)

main_path = Path(
    "data/annotations/lid_candidates.csv"
)

output = Path(
    "data/annotations/lid_code_mixed_extra_candidates.csv"
)

def text_key(text):
    return " ".join(str(text).split()).casefold()

# Load the original dataset
df = pd.read_parquet(source)

# Exclude previously annotated sentences
existing = pd.read_csv(existing_path, keep_default_na=False)
main = pd.read_csv(main_path, keep_default_na=False)

used = set(existing["text"].map(text_key))
used.update(main["text"].map(text_key))

candidates = []

# New reproducible random sample
df = df.sample(frac=1, random_state=43)

for value in df["text"].dropna():
    sentence = " ".join(str(value).split())
    key = text_key(sentence)

    if (
        20 <= len(sentence) <= 220
        and key not in used
    ):
        used.add(key)

        candidates.append({
            "id": f"cm{len(candidates) + 1:04d}",
            "text": sentence,
            "label": "",
            "source": "Swarnadeep-28/bn_code_mix_sentiment_dataset",
            "annotator": "",
            "review_status": "Pending",
            "notes": "",
        })

    if len(candidates) == 50:
        break

if output.exists():
    raise FileExistsError(
        "Extra candidates already exist. "
        "Do not overwrite your annotations."
    )

pd.DataFrame(candidates).to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("Additional candidates:", len(candidates))
print("Saved to:", output)
