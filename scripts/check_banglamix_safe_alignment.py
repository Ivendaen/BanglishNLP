
import re
from collections import Counter

import pandas as pd

from banglishnlp.tokenizer import tokenize_with_spans


DATASET = "data/raw/banglamix/BanglaMix.csv"

LANGUAGE_LABELS = {"BAN", "ENG", "ROM_BAN"}


def is_word(token):
    return any(char.isalpha() for char in token)


def script_matches(token, label):
    has_bengali = any(
        "\u0980" <= char <= "\u09FF" and char.isalpha()
        for char in token
    )

    has_latin = any(
        ("a" <= char.lower() <= "z")
        for char in token
    )

    if label == "BAN":
        return has_bengali and not has_latin

    if label in {"ENG", "ROM_BAN"}:
        return has_latin and not has_bengali

    return False


def safe_split_labels(source_token, label, pieces):
    if "".join(pieces) != source_token:
        return None

    if len(pieces) == 1 and pieces[0] == source_token:
        return [label]

    if label == "OTHER":
        if all(not is_word(piece) for piece in pieces):
            return ["OTHER"] * len(pieces)

        return None

    if label not in LANGUAGE_LABELS:
        return None

    word_indices = [
        i for i, piece in enumerate(pieces)
        if is_word(piece)
    ]

    if len(word_indices) != 1:
        return None

    word_index = word_indices[0]
    word = pieces[word_index]

    if not script_matches(word, label):
        return None

    return [
        label if i == word_index else "OTHER"
        for i in range(len(pieces))
    ]


def main():
    df = pd.read_csv(DATASET, keep_default_na=False)

    counts = Counter()
    review_examples = []

    for row in df.itertuples(index=False):
        text = str(row.cleaned_text)
        annotations = str(row.token_labels).split()

        source_matches = list(re.finditer(r"\S+", text))
        target = tokenize_with_spans(text)

        if len(source_matches) != len(annotations):
            raise ValueError("Annotation count mismatch")

        target_index = 0
        row_safe = True

        for annotation, match in zip(
            annotations, source_matches
        ):
            source_token, label = annotation.rsplit("/", 1)
            start, end = match.span()

            if source_token != match.group():
                raise ValueError("Source text mismatch")

            pieces = []

            while (
                target_index < len(target)
                and target[target_index][1] < end
            ):
                piece, piece_start, piece_end = target[target_index]

                if piece_start < start or piece_end > end:
                    raise ValueError("Cross-boundary token detected")

                pieces.append(piece)
                target_index += 1

            assigned = safe_split_labels(
                source_token,
                label,
                pieces,
            )

            if assigned is None:
                row_safe = False
                counts["review_tokens"] += 1

                if len(review_examples) < 15:
                    review_examples.append(
                        (source_token, label, pieces)
                    )

            elif len(pieces) == 1:
                counts["exact_tokens"] += 1

            else:
                counts["safe_split_tokens"] += 1

        if target_index != len(target):
            raise ValueError("Some tokenizer tokens were not aligned")

        if row_safe:
            counts["safe_rows"] += 1
        else:
            counts["review_rows"] += 1

    print("=== SAFE ALIGNMENT SUMMARY ===")
    print("Total rows:", len(df))
    print("Safe rows:", counts["safe_rows"])
    print("Rows needing review:", counts["review_rows"])
    print("Exact tokens:", counts["exact_tokens"])
    print("Safely split source tokens:", counts["safe_split_tokens"])
    print("Source tokens needing review:", counts["review_tokens"])

    print("\n=== FIRST 15 REVIEW EXAMPLES ===")

    for token, label, pieces in review_examples:
        print(
            ascii(token),
            "| Label:", label,
            "| Pieces:", [ascii(p) for p in pieces],
        )


if __name__ == "__main__":
    main()
