
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE = Path("data/prepared/banglamix_safe_tokens.csv")
OUT = Path("data/prepared/token_tagging")
SEED = 42


def normalize(text):
    return " ".join(str(text).split()).casefold()


def main():
    df = pd.read_csv(SOURCE, keep_default_na=False)

    # Group identical sentences together.
    df["text_key"] = df["text"].map(normalize)

    unique_sentences = df["text_key"].astype("object").drop_duplicates().tolist()

    train_dev, test = train_test_split(
        unique_sentences,
        test_size=0.10,
        random_state=SEED,
    )

    train, dev = train_test_split(
        train_dev,
        test_size=1 / 9,
        random_state=SEED,
    )

    groups = {
        "train": set(train),
        "dev": set(dev),
        "test": set(test),
    }

    # Check for exact-text leakage.
    assert not groups["train"] & groups["dev"]
    assert not groups["train"] & groups["test"]
    assert not groups["dev"] & groups["test"]

    OUT.mkdir(parents=True, exist_ok=True)

    for name, keys in groups.items():
        part = df[df["text_key"].isin(keys)].copy()
        part = part.drop(columns=["text_key"])

        output = OUT / f"{name}.csv"
        part.to_csv(
            output,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"{name.upper()}: "
            f"{part['row_index'].nunique()} sentences, "
            f"{len(part)} tokens"
        )

    print("\nDataset splitting complete.")
    print("Random seed:", SEED)


if __name__ == "__main__":
    main()
