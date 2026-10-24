
from pathlib import Path
import pandas as pd

input_path = Path(
    "data/raw/collected/train-00000-of-00001.parquet"
)
output_path = Path(
    "data/annotations/lid_banglish_candidates.csv"
)
main_path = Path("data/annotations/lid_candidates.csv")

df = pd.read_parquet(input_path)

# Exclude sentences already in our main LID dataset
main = pd.read_csv(main_path, keep_default_na=False)
existing = set(main["text"].str.strip().str.casefold())

candidates = []
seen = set()

# Random, reproducible sampling
df = df.sample(frac=1, random_state=42)

for value in df["text"].dropna():
    text = " ".join(str(value).split())
    key = text.casefold()

    latin_count = sum(
        c.isascii() and c.isalpha() for c in text
    )
    bangla_count = sum(
        "\u0980" <= c <= "\u09FF" for c in text
    )

    if (
        20 <= len(text) <= 220
        and latin_count >= 15
        and bangla_count == 0
        and key not in seen
        and key not in existing
    ):
        seen.add(key)

        candidates.append({
            "id": f"bl{len(candidates) + 1:04d}",
            "text": text,
            "label": "",
            "source": "Swarnadeep-28/bn_code_mix_sentiment_dataset",
            "annotator": "",
            "review_status": "Pending",
            "notes": "",
        })

    if len(candidates) == 150:
        break

output_path.parent.mkdir(parents=True, exist_ok=True)

if output_path.exists():
    raise FileExistsError(
        f"{output_path} already exists. "
        "Review it instead of overwriting annotations."
    )

pd.DataFrame(candidates).to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("Banglish candidates:", len(candidates))
print("Saved to:", output_path)

print("\nFirst 5 candidates:")
for item in candidates[:5]:
    print(item["id"], item["text"])

