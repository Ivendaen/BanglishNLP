
from pathlib import Path

import pandas as pd


SOURCE = Path(
    "data/processed/token_tagger_dev_predictions.csv"
)

OUTPUT = Path(
    "data/prepared/token_tagging/error_audit.csv"
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "Audit file already exists. Do not overwrite it."
        )

    df = pd.read_csv(SOURCE, keep_default_na=False)

    errors = df[
        df["label"] != df["predicted_label"]
    ].copy()

    selected = []
    used = set()

    def select_sample(candidates, count, seed):
        available = candidates.loc[
            ~candidates.index.isin(used)
        ]

        if available.empty:
            return

        sample = available.sample(
            n=min(count, len(available)),
            random_state=seed,
        )

        selected.append(sample)
        used.update(sample.index.tolist())

    select_sample(
        errors[
            (errors["label"] == "ENG")
            & (errors["predicted_label"] == "ROM_BAN")
        ],
        15,
        42,
    )

    select_sample(
        errors[
            (errors["label"] == "ROM_BAN")
            & (errors["predicted_label"] == "ENG")
        ],
        15,
        43,
    )

    select_sample(
        errors[errors["label"] == "UNK"],
        7,
        44,
    )

    select_sample(
        errors,
        13,
        45,
    )

    audit = pd.concat(
        selected,
        ignore_index=True,
    )

    # Hide both original and predicted labels to
    # reduce confirmation bias during human review.
    audit = audit[
        [
            "row_index",
            "text",
            "token",
            "start",
            "end",
        ]
    ].copy()

    audit.insert(
        0,
        "audit_id",
        range(1, len(audit) + 1),
    )

    audit["human_label"] = ""
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

    print("=== TOKEN ERROR AUDIT ===")
    print("Total development errors:", len(errors))
    print("Selected review tokens:", len(audit))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
