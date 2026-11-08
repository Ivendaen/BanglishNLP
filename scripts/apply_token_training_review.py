
from pathlib import Path

import pandas as pd


SOURCE = Path("data/prepared/token_tagging/train.csv")
REVIEW = Path("data/prepared/token_tagging/train_label_review.csv")
OUTPUT = Path("data/prepared/token_tagging/train_reviewed.csv")

LABELS = {"BAN", "ENG", "ROM_BAN", "OTHER", "UNK"}
KEYS = ["row_index", "start", "end"]


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "Corrected training file already exists. "
            "Do not overwrite it."
        )

    train = pd.read_csv(SOURCE, keep_default_na=False)
    review = pd.read_csv(REVIEW, keep_default_na=False)

    required = {
        "row_index",
        "text",
        "token",
        "start",
        "end",
        "original_label",
        "reviewed_label",
        "review_status",
    }

    if not required.issubset(review.columns):
        raise ValueError("Review file is missing required columns")

    if train.duplicated(KEYS).any():
        raise ValueError("Duplicate training token positions")

    if review.duplicated(KEYS).any():
        raise ValueError("Duplicate review token positions")

    review["reviewed_label"] = (
        review["reviewed_label"].astype(str).str.strip()
    )

    review["review_status"] = (
        review["review_status"].astype(str).str.strip()
    )

    if not review["review_status"].eq("Reviewed").all():
        raise ValueError("Not all annotations are reviewed")

    if not review["reviewed_label"].isin(LABELS).all():
        raise ValueError("Invalid reviewed labels found")

    paired = review.merge(
        train[
            KEYS + ["text", "token", "label"]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_review", "_train"),
    )

    if not paired["_merge"].eq("both").all():
        raise ValueError("Some reviewed tokens are missing")

    if not (
        paired["token_review"] == paired["token_train"]
    ).all():
        raise ValueError("Token text mismatch")

    if not (
        paired["text_review"] == paired["text_train"]
    ).all():
        raise ValueError("Sentence text mismatch")

    if not (
        paired["original_label"] == paired["label"]
    ).all():
        raise ValueError("Original label mismatch")

    corrected = train.copy()
    changed = 0

    for row in review.itertuples(index=False):
        mask = (
            (corrected["row_index"] == row.row_index)
            & (corrected["start"] == row.start)
            & (corrected["end"] == row.end)
        )

        old_label = corrected.loc[mask, "label"].iloc[0]

        if old_label != row.reviewed_label:
            corrected.loc[mask, "label"] = row.reviewed_label
            changed += 1

    if len(corrected) != len(train):
        raise ValueError("Training token count changed")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    corrected.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== CORRECTED TOKEN TRAINING DATA ===")
    print("Original tokens:", len(train))
    print("Reviewed annotations:", len(review))
    print("Labels changed:", changed)
    print("Tokens after correction:", len(corrected))

    print("\n=== UPDATED LABEL DISTRIBUTION ===")
    print(corrected["label"].value_counts().to_string())

    print("\nSaved:", OUTPUT)
    print("Original train.csv unchanged.")
    print("Development and test sets unchanged.")


if __name__ == "__main__":
    main()
