
from pathlib import Path

import pandas as pd


SOURCE = Path("data/prepared/banglamix_safe_tokens.csv")
OUTPUT = Path("data/prepared/banglamix_label_audit.csv")

LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "Audit file already exists. "
            "Do not overwrite existing reviews."
        )

    df = pd.read_csv(SOURCE, keep_default_na=False)

    samples = []
    selected = set()

    def add_sample(candidates, count, seed):
        candidates = candidates.loc[
            ~candidates.index.isin(selected)
        ]

        if candidates.empty:
            return

        sampled = candidates.sample(
            n=min(count, len(candidates)),
            random_state=seed,
        )

        samples.append(sampled)
        selected.update(sampled.index.tolist())

    # Inspect each language and alignment type.
    for i, label in enumerate(LABELS):
        for j, alignment in enumerate(
            ["exact", "safe_split"]
        ):
            candidates = df[
                (df["label"] == label)
                & (df["alignment_type"] == alignment)
            ]

            add_sample(
                candidates,
                count=10,
                seed=42 + i * 10 + j,
            )

    # Investigate alphabetic tokens labeled OTHER.
    suspicious_other = df[
        (df["label"] == "OTHER")
        & df["token"].map(
            lambda token: any(
                char.isalpha() for char in str(token)
            )
        )
    ]

    add_sample(suspicious_other, 20, 300)

    # Include additional examples of the rare UNK label.
    add_sample(df[df["label"] == "UNK"], 15, 301)

    # Include a general random sample.
    add_sample(df, 30, 302)

    audit = pd.concat(
        samples,
        ignore_index=True,
    )

    audit = audit.rename(
        columns={"label": "suggested_label"}
    )

    audit["reviewed_label"] = ""
    audit["review_status"] = "Needs Review"
    audit["review_notes"] = ""

    columns = [
        "comment_id",
        "row_index",
        "text",
        "token",
        "start",
        "end",
        "suggested_label",
        "alignment_type",
        "source_token",
        "source_label",
        "reviewed_label",
        "review_status",
        "review_notes",
    ]

    audit = audit[columns]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    audit.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== HUMAN ANNOTATION AUDIT ===")
    print("Total audit tokens:", len(audit))

    print("\n=== SUGGESTED LABEL DISTRIBUTION ===")
    print(
        audit["suggested_label"]
        .value_counts()
        .to_string()
    )

    print("\n=== ALIGNMENT DISTRIBUTION ===")
    print(
        audit["alignment_type"]
        .value_counts()
        .to_string()
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
