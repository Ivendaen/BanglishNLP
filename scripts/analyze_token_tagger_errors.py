
from pathlib import Path

import pandas as pd


SOURCE = Path(
    "data/processed/token_tagger_dev_predictions.csv"
)

OUTPUT = Path(
    "data/processed/token_tagger_dev_errors.csv"
)


def main():
    df = pd.read_csv(SOURCE, keep_default_na=False)

    errors = df[
        df["label"] != df["predicted_label"]
    ].copy()

    print("=== TOKEN TAGGER ERROR ANALYSIS ===")
    print("Development tokens:", len(df))
    print("Incorrect predictions:", len(errors))

    print("\n=== MOST COMMON CONFUSIONS ===")

    confusion = (
        errors.groupby(
            ["label", "predicted_label"]
        )
        .size()
        .sort_values(ascending=False)
    )

    print(confusion.to_string())

    print("\n=== ENGLISH VS ROMANIZED BANGLA ERRORS ===")

    mixed_errors = errors[
        (
            (errors["label"] == "ENG")
            & (errors["predicted_label"] == "ROM_BAN")
        )
        |
        (
            (errors["label"] == "ROM_BAN")
            & (errors["predicted_label"] == "ENG")
        )
    ]

    print("Total:", len(mixed_errors))

    print(
        mixed_errors[
            ["token", "label", "predicted_label", "text"]
        ]
        .head(20)
        .to_string(index=False, max_colwidth=100)
    )

    print("\n=== UNK ERRORS ===")

    unk_errors = errors[errors["label"] == "UNK"]

    if unk_errors.empty:
        print("No UNK errors.")
    else:
        print(
            unk_errors[
                ["token", "label", "predicted_label", "text"]
            ].to_string(index=False, max_colwidth=100)
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    errors.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
