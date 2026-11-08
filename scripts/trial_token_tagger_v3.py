
from pathlib import Path

import joblib

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from scripts.train_token_tagger_baseline import prepare_data


TRAIN = Path("data/prepared/token_tagging/train.csv")
DEV = Path("data/prepared/token_tagging/dev.csv")
V1_MODEL = Path("models/token_tagger_v1.joblib")

LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
CORE_LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER"]


def evaluate(name, gold, predictions):
    accuracy = accuracy_score(gold, predictions)

    macro_f1 = f1_score(
        gold,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    core_f1 = f1_score(
        gold,
        predictions,
        labels=CORE_LABELS,
        average="macro",
        zero_division=0,
    )

    mixed_errors = sum(
        (
            true == "ENG" and pred == "ROM_BAN"
        ) or (
            true == "ROM_BAN" and pred == "ENG"
        )
        for true, pred in zip(gold, predictions)
    )

    unk_total = sum(label == "UNK" for label in gold)

    unk_correct = sum(
        true == "UNK" and pred == "UNK"
        for true, pred in zip(gold, predictions)
    )

    unk_recall = (
        unk_correct / unk_total
        if unk_total
        else 0.0
    )

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1 (5 classes): {macro_f1:.3f}")
    print(f"Macro-F1 (4 core classes): {core_f1:.3f}")
    print("ENG / ROM_BAN errors:", mixed_errors)
    print(f"UNK recall: {unk_recall:.2%}")


def main():
    print("Loading training and development data...")

    train_df, train_features = prepare_data(TRAIN)
    dev_df, dev_features = prepare_data(DEV)

    gold = dev_df["label"].tolist()
    y_train = train_df["label"].tolist()

    print("Loading V1 baseline...")

    bundle = joblib.load(V1_MODEL)

    context_vectorizer = bundle["vectorizer"]
    baseline = bundle["classifier"]

    x_train_context = context_vectorizer.transform(
        train_features
    )

    x_dev_context = context_vectorizer.transform(
        dev_features
    )

    baseline_predictions = baseline.predict(
        x_dev_context
    )

    evaluate(
        "V1 BASELINE",
        gold,
        baseline_predictions,
    )

    print("\nBuilding character n-gram features...")

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    train_tokens = train_df["token"].astype(str).tolist()
    dev_tokens = dev_df["token"].astype(str).tolist()

    x_train_char = char_vectorizer.fit_transform(
        train_tokens
    )

    x_dev_char = char_vectorizer.transform(
        dev_tokens
    )

    print("Combining character and context features...")

    x_train = hstack(
        [x_train_context, x_train_char],
        format="csr",
    )

    x_dev = hstack(
        [x_dev_context, x_dev_char],
        format="csr",
    )

    print("Total feature dimensions:", x_train.shape[1])

    print("\nTraining V3 experiment...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_dev)

    evaluate(
        "V3 CHARACTER + CONTEXT",
        gold,
        predictions,
    )

    print("\n=== V3 CLASSIFICATION REPORT ===")

    print(
        classification_report(
            gold,
            predictions,
            labels=LABELS,
            digits=3,
            zero_division=0,
        )
    )

    print("\nExperiment completed.")
    print("No model files were overwritten.")
    print("Held-out test set was not evaluated.")


if __name__ == "__main__":
    main()
