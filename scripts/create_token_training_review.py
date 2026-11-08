
from pathlib import Path

import pandas as pd


SOURCE = Path("data/prepared/token_tagging/train.csv")

OUTPUT = Path(
    "data/prepared/token_tagging/train_label_review.csv"
)

TARGETS = {
    "university": {"ROM_BAN": 10, "ENG": 3},
    "phone": {"ROM_BAN": 10, "ENG": 3},
    "bro": {"ROM_BAN": 10, "ENG": 3},
    "rafsan": {"ROM_BAN": 4, "ENG": 4},
    "dhaka": {"ROM_BAN": 4, "ENG": 4},
}


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "Review file already exists. "
            "Do not overwrite previous reviews."
        )

    df = pd.read_csv(SOURCE, keep_default_na=False)

    df["word_lower"] = (
        df["token"].astype(str).str.casefold()
    )

    samples = []

    for word, label_limits in TARGETS.items():
        for label, limit in label_limits.items():

            candidates = df[
                (df["word_lower"] == word)
                & (df["label"] == label)
            ]

            if candidates.empty:
                continue

            sample = candidates.sample(
                n=min(limit, len(candidates)),
                random_state=42,
            )

            samples.append(sample)

    if not samples:
        raise ValueError("No matching training tokens")

    audit = pd.concat(
        samples,
        ignore_index=True,
    )

    audit = audit[
        [
            "row_index",
            "text",
            "token",
            "start",
            "end",
            "label",
        ]
    ].copy()

    audit = audit.rename(
        columns={"label": "original_label"}
    )

    audit["reviewed_label"] = ""
    audit["review_status"] = "Needs Review"
    audit["review_notes"] = ""

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== TRAINING LABEL REVIEW ===")
    print("Selected tokens:", len(audit))

    print("\nTokens by word:")
    print(
        audit["token"]
        .str.casefold()
        .value_counts()
        .to_string()
    )

    print("\nSaved:", OUTPUT)
    print("Original training data unchanged.")


if __name__ == "__main__":
    main()
