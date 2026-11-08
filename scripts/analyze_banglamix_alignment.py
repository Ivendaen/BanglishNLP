
import re
from collections import Counter

import pandas as pd

from banglishnlp.tokenizer import tokenize_with_spans


DATASET = "data/raw/banglamix/BanglaMix.csv"


def main():
    df = pd.read_csv(DATASET, keep_default_na=False)

    exact_tokens = 0
    split_tokens = 0
    unaligned_tokens = 0

    split_labels = Counter()
    examples = []

    for row in df.itertuples(index=False):
        text = str(row.cleaned_text)
        annotations = str(row.token_labels).split()

        source_spans = list(re.finditer(r"\S+", text))
        target_spans = tokenize_with_spans(text)

        if len(annotations) != len(source_spans):
            raise ValueError("Source annotation count mismatch")

        for annotation, source_match in zip(
            annotations, source_spans
        ):
            source_token, label = annotation.rsplit("/", 1)

            if source_token != source_match.group():
                raise ValueError("Source token mismatch")

            start, end = source_match.span()

            pieces = [
                token
                for token, token_start, token_end in target_spans
                if token_start >= start and token_end <= end
            ]

            if pieces == [source_token]:
                exact_tokens += 1

            elif len(pieces) > 1 and "".join(pieces) == source_token:
                split_tokens += 1
                split_labels[label] += 1

                if len(examples) < 15:
                    examples.append(
                        (source_token, label, pieces)
                    )

            else:
                unaligned_tokens += 1

    print("=== TOKEN ALIGNMENT ANALYSIS ===")
    print("Total source tokens:", (
        exact_tokens + split_tokens + unaligned_tokens
    ))
    print("Exact tokens:", exact_tokens)
    print("Split tokens:", split_tokens)
    print("Unaligned tokens:", unaligned_tokens)

    print("\n=== SPLIT TOKENS BY LABEL ===")
    for label, count in split_labels.most_common():
        print(label, ":", count)

    print("\n=== FIRST 15 SPLIT EXAMPLES ===")
    for token, label, pieces in examples:
        print(
            ascii(token),
            "->",
            [ascii(piece) for piece in pieces],
            "| Label:",
            label,
        )


if __name__ == "__main__":
    main()
