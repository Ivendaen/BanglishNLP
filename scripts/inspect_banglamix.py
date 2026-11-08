
from collections import Counter

import pandas as pd

from banglishnlp.tokenizer import tokenize


DATASET = "data/raw/banglamix/BanglaMix.csv"

LABELS = {"BAN", "ENG", "ROM_BAN", "OTHER", "UNK"}


def main():
    df = pd.read_csv(DATASET, keep_default_na=False)

    label_counts = Counter()
    invalid_rows = 0
    whitespace_matches = 0
    tokenizer_matches = 0
    mismatch_examples = []

    for _, row in df.iterrows():
        text = str(row["cleaned_text"])
        annotations = str(row["token_labels"]).split()

        tokens = []
        labels = []
        valid = True

        for annotation in annotations:
            if "/" not in annotation:
                valid = False
                break

            token, label = annotation.rsplit("/", 1)

            if not token or label not in LABELS:
                valid = False
                break

            tokens.append(token)
            labels.append(label)

        if not valid or not tokens:
            invalid_rows += 1
            continue

        label_counts.update(labels)

        if tokens == text.split():
            whitespace_matches += 1

        if tokens == tokenize(text):
            tokenizer_matches += 1

        elif len(mismatch_examples) < 3:
            mismatch_examples.append(
                (text, tokens, tokenize(text))
            )

    print("=== BANGLAMIX INSPECTION ===")
    print("Total rows:", len(df))
    print("Invalid annotation rows:", invalid_rows)

    print("\n=== TOKEN LABEL COUNTS ===")
    for label in sorted(LABELS):
        print(label, ":", label_counts[label])

    print("\nTotal parsed tokens:", sum(label_counts.values()))

    print("\n=== ALIGNMENT ===")
    print("Whitespace-exact rows:", whitespace_matches)
    print("BanglishNLP tokenizer-exact rows:", tokenizer_matches)

    print("\n=== FIRST 3 TOKENIZER MISMATCHES ===")

    for text, original, generated in mismatch_examples:
        print("\nText:", ascii(text[:150]))
        print("Dataset tokens:", ascii(original[:15]))
        print("Our tokens:", ascii(generated[:15]))


if __name__ == "__main__":
    main()
