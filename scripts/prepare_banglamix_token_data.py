
import json
import re
from pathlib import Path

import pandas as pd

from banglishnlp.tokenizer import tokenize_with_spans
from scripts.check_banglamix_safe_alignment import (
    safe_split_labels,
)


SOURCE = Path("data/raw/banglamix/BanglaMix.csv")
OUTPUT_DIR = Path("data/prepared")

SAFE_OUTPUT = OUTPUT_DIR / "banglamix_safe_tokens.csv"
REVIEW_OUTPUT = OUTPUT_DIR / "banglamix_review_tokens.csv"


def main():
    df = pd.read_csv(SOURCE, keep_default_na=False)

    safe_records = []
    review_records = []

    safe_rows = 0
    review_rows = 0

    for row_index, row in df.iterrows():
        text = str(row["cleaned_text"])
        annotations = str(row["token_labels"]).split()

        source_matches = list(re.finditer(r"\S+", text))
        target_tokens = tokenize_with_spans(text)

        if len(source_matches) != len(annotations):
            raise ValueError(
                f"Annotation mismatch in row {row_index}"
            )

        row_records = []
        row_reviews = []
        target_index = 0

        for annotation, source_match in zip(
            annotations,
            source_matches,
        ):
            source_token, source_label = annotation.rsplit("/", 1)
            start, end = source_match.span()

            if source_token != source_match.group():
                raise ValueError(
                    f"Source token mismatch in row {row_index}"
                )

            pieces = []

            while (
                target_index < len(target_tokens)
                and target_tokens[target_index][1] < end
            ):
                token, token_start, token_end = target_tokens[
                    target_index
                ]

                if token_start < start or token_end > end:
                    raise ValueError(
                        f"Cross-boundary token in row {row_index}"
                    )

                pieces.append(
                    (token, token_start, token_end)
                )
                target_index += 1

            piece_texts = [item[0] for item in pieces]

            labels = safe_split_labels(
                source_token,
                source_label,
                piece_texts,
            )

            if labels is None:
                row_reviews.append({
                    "row_index": row_index,
                    "comment_id": str(row["id"]),
                    "text": text,
                    "source_token": source_token,
                    "source_label": source_label,
                    "target_pieces": json.dumps(
                        piece_texts,
                        ensure_ascii=False,
                    ),
                    "review_status": "Needs Review",
                })
                continue

            if len(labels) != len(pieces):
                raise ValueError(
                    f"Label count mismatch in row {row_index}"
                )

            alignment_type = (
                "exact" if len(pieces) == 1 else "safe_split"
            )

            for (token, token_start, token_end), label in zip(
                pieces,
                labels,
            ):
                if text[token_start:token_end] != token:
                    raise ValueError(
                        f"Offset mismatch in row {row_index}"
                    )

                row_records.append({
                    "row_index": row_index,
                    "comment_id": str(row["id"]),
                    "text": text,
                    "token": token,
                    "start": token_start,
                    "end": token_end,
                    "label": label,
                    "source_token": source_token,
                    "source_label": source_label,
                    "alignment_type": alignment_type,
                    "review_status": "Auto Aligned",
                })

        if target_index != len(target_tokens):
            raise ValueError(
                f"Unconsumed target tokens in row {row_index}"
            )

        if row_reviews:
            review_rows += 1
            review_records.extend(row_reviews)
        else:
            safe_rows += 1
            safe_records.extend(row_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(safe_records).to_csv(
        SAFE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(review_records).to_csv(
        REVIEW_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== BANGLAMIX DATA PREPARATION ===")
    print("Original sentences:", len(df))
    print("Auto-aligned sentences:", safe_rows)
    print("Sentences needing review:", review_rows)
    print("Auto-aligned output tokens:", len(safe_records))
    print("Source tokens needing review:", len(review_records))

    print("\n=== AUTO-ALIGNED LABEL DISTRIBUTION ===")
    print(
        pd.Series(
            [record["label"] for record in safe_records]
        ).value_counts().to_string()
    )

    print("\nSaved:", SAFE_OUTPUT)
    print("Saved:", REVIEW_OUTPUT)


if __name__ == "__main__":
    main()
