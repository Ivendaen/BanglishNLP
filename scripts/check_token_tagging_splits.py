
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/prepared/token_tagging")

LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]


def normalize(text):
    return " ".join(str(text).split()).casefold()


def main():
    datasets = {}
    text_sets = {}
    row_sets = {}

    for name in ["train", "dev", "test"]:
        path = DATA_DIR / f"{name}.csv"

        df = pd.read_csv(
            path,
            keep_default_na=False,
        )

        required = {
            "row_index",
            "text",
            "token",
            "start",
            "end",
            "label",
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing columns in {name}"
            )

        if not set(df["label"]).issubset(LABELS):
            raise ValueError(
                f"Invalid language labels in {name}"
            )

        datasets[name] = df
        text_sets[name] = set(
            df["text"].map(normalize)
        )
        row_sets[name] = set(df["row_index"])

        print(f"\n=== {name.upper()} ===")
        print(
            "Sentences:",
            df["row_index"].nunique(),
        )
        print("Tokens:", len(df))

        print("\nLabel distribution:")

        counts = (
            df["label"]
            .value_counts()
            .reindex(LABELS, fill_value=0)
        )

        for label, count in counts.items():
            print(
                f"{label}: {count} "
                f"({count / len(df):.2%})"
            )

    print("\n=== DATA LEAKAGE CHECK ===")

    pairs = [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]

    for first, second in pairs:
        text_overlap = (
            text_sets[first] & text_sets[second]
        )

        row_overlap = (
            row_sets[first] & row_sets[second]
        )

        print(
            f"{first} vs {second}: "
            f"{len(text_overlap)} exact-text overlaps, "
            f"{len(row_overlap)} source-row overlaps"
        )

        if text_overlap or row_overlap:
            raise ValueError(
                "Dataset leakage detected"
            )

    total_tokens = sum(
        len(df) for df in datasets.values()
    )

    print("\nTotal tokens:", total_tokens)
    print("Exact-text split separation verified.")


if __name__ == "__main__":
    main()
