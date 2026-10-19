
from pathlib import Path
import pandas as pd

# Dataset locations
input_path = Path("data/raw/lid/bangla/xlsum_bengali.parquet")
output_path = Path("data/annotations/lid_bangla_candidates.csv")

# Load dataset
df = pd.read_parquet(input_path)

print("Total records:", len(df))
print("Available columns:", list(df.columns))

# Prefer summaries, which are shorter than full news articles
column = "summary" if "summary" in df.columns else "text"

candidates = []
seen = set()

# Shuffle reproducibly
df = df.sample(frac=1, random_state=42)

for value in df[column].dropna():
    # Extract the first Bengali sentence
    sentence = str(value).split("।")[0].strip()

    # Keep manageable Bengali-script examples
    bengali_chars = sum(
        "\u0980" <= char <= "\u09FF" for char in sentence
    )

    has_english = any(
        char.isascii() and char.isalpha()
        for char in sentence
    )

    if (
        40 <= len(sentence) <= 220
        and bengali_chars >= 30
        and not has_english
        and sentence not in seen
    ):
        seen.add(sentence)

        candidates.append({
            "id": f"bn{len(candidates) + 1:04d}",
            "text": sentence,
            "label": "",
            "source": "csebuetnlp/xlsum (Bengali test)",
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
