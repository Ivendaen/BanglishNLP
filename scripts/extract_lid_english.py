
from pathlib import Path
import pandas as pd

input_path = Path("data/raw/lid/english/emotion_english.parquet")
output_path = Path("data/annotations/lid_english_candidates.csv")

df = pd.read_parquet(input_path)

print("Total records:", len(df))
print("Available columns:", list(df.columns))

if "text" not in df.columns:
    raise ValueError("Dataset does not contain a text column")

candidates = []
seen = set()

# Reproducible random selection
df = df.sample(frac=1, random_state=42)

for value in df["text"].dropna():
    sentence = " ".join(str(value).split())

    latin_letters = sum(c.isascii() and c.isalpha() for c in sentence)
    bengali_letters = sum("\u0980" <= c <= "\u09FF" for c in sentence)

    if (
        20 <= len(sentence) <= 220
        and latin_letters >= 15
        and bengali_letters == 0
        and sentence.casefold() not in seen
    ):
        seen.add(sentence.casefold())

        candidates.append({
            "id": f"en{len(candidates) + 1:04d}",
            "text": sentence,
            "label": "",
            "source": "dair-ai/emotion (English train)",
            "annotator": "",
            "review_status": "Pending",
            "notes": "",
        })

    if len(candidates) == 50:
        break

output_path.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame(candidates).to_csv(
    output_path, index=False, encoding="utf-8-sig"
)

print("Candidates extracted:", len(candidates))
print("Saved to:", output_path)

print("\nFirst 5 candidates:")
for item in candidates[:5]:
    print(item["id"], item["text"])
