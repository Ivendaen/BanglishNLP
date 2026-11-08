
from pathlib import Path

import pandas as pd


SOURCE = Path("data/prepared/token_tagging/test.csv")
PREVIOUS_AUDIT = Path(
    "data/prepared/banglamix_label_audit.csv"
)
OUTPUT = Path(
    "data/prepared/token_tagging/blind_test_review.csv"
)

SAMPLE_SIZE = 40
SEED = 42


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "Review file already exists. Do not overwrite it."
        )

    df = pd.read_csv(SOURCE, keep_default_na=False)

    # Exclude test sentences that appeared in
    # our earlier cross-dataset annotation audit.
    excluded = set()

    if PREVIOUS_AUDIT.exists():
        audit = pd.read_csv(
            PREVIOUS_AUDIT,
            keep_default_na=False,
        )
        excluded = set(audit["row_index"])

    available = df[
        ~df["row_index"].isin(excluded)
    ].copy()

    sentences = available[
        ["row_index", "text"]
    ].drop_duplicates(subset=["row_index"])

    if len(sentences) < SAMPLE_SIZE:
        raise ValueError(
            "Not enough unaudited test sentences."
        )

    selected = sentences.sample(
        n=SAMPLE_SIZE,
        random_state=SEED,
    )

    review = available[
        available["row_index"].isin(
            selected["row_index"]
        )
    ].copy()

    review = review.sort_values(
        ["row_index", "start"],
        kind="stable",
    )

    # Never expose source labels or predictions
    # to the human annotators.
    review = review[
        ["row_index", "text", "token", "start", "end"]
    ].copy()

    for row in review.itertuples(index=False):
        if row.text[row.start:row.end] != row.token:
            raise ValueError("Token offset mismatch")

    review["human_label"] = ""
    review["review_status"] = "Needs Review"
    review["review_notes"] = ""

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== BLIND TOKEN EVALUATION SAMPLE ===")
    print("Sentences:", review["row_index"].nunique())
    print("Tokens requiring review:", len(review))
    print("Saved:", OUTPUT)
    print("No model predictions were generated.")
    print("Original test dataset unchanged.")


if __name__ == "__main__":
    main()
