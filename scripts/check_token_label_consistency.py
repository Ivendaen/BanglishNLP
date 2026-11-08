
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/prepared/token_tagging")

WORDS = [
    "university",
    "assignment",
    "class",
    "exam",
    "phone",
    "bro",
    "rafsan",
    "dhaka",
]


def main():
    for split in ["train", "dev"]:
        path = DATA_DIR / f"{split}.csv"

        df = pd.read_csv(
            path,
            keep_default_na=False,
        )

        df["word_lower"] = (
            df["token"]
            .astype(str)
            .str.casefold()
        )

        print(f"\n=== {split.upper()} ===")

        for word in WORDS:
            matches = df[
                df["word_lower"] == word
            ]

            print(f"\nWORD: {word}")
            print("Occurrences:", len(matches))

            if matches.empty:
                continue

            print("Label distribution:")
            print(
                matches["label"]
                .value_counts()
                .to_string()
            )

            print("Example sentences:")

            for text in matches["text"].head(3):
                print("-", str(text)[:130])


if __name__ == "__main__":
    main()
