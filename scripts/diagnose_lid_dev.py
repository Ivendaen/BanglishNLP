
from pathlib import Path
import pandas as pd

from banglishnlp.language_detector import BanglishLID

source = Path("data/processed/lid_dev_errors.csv")
output = Path("data/processed/lid_dev_diagnostics.csv")

df = pd.read_csv(source, keep_default_na=False)

print("Loading IndicLID models...")
lid = BanglishLID()

results = []

for _, row in df.iterrows():
    text = row["text"]

    # Raw scores from both pretrained models
    ftn = lid._scores(lid.native_model, text)
    ftr = lid._scores(lid.roman_model, text)

    ben = ftr.get("ben_Latn", 0.0)
    eng = ftr.get("eng_Latn", 0.0)

    relative_ben = (
        ben / (ben + eng)
        if ben + eng > 0
        else 0.0
    )

    results.append({
        "id": row["id"],
        "text": text,
        "actual": row["label"],
        "predicted": row["predicted_label"],
        "ftn_top": max(ftn, key=ftn.get, default=""),
        "ftr_top": max(ftr, key=ftr.get, default=""),
        "ftn_english": round(
            ftn.get("eng_Latn", 0.0), 4
        ),
        "ftr_banglish": round(ben, 4),
        "ftr_english": round(eng, 4),
        "relative_banglish": round(relative_ben, 4),
    })

diagnostics = pd.DataFrame(results)

output.parent.mkdir(parents=True, exist_ok=True)
diagnostics.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== DIAGNOSTIC SUMMARY ===")
print("Analyzed errors:", len(diagnostics))

print("\nFTR top predictions:")
print(diagnostics["ftr_top"].value_counts())

print("\nFTN top predictions:")
print(diagnostics["ftn_top"].value_counts())

print("\nSaved:", output)
