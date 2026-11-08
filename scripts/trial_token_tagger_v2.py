
from collections import Counter
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from scripts.train_token_tagger_baseline import prepare_data


TRAIN = Path("data/prepared/token_tagging/train.csv")
DEV = Path("data/prepared/token_tagging/dev.csv")

BASELINE_MODEL = Path("models/token_tagger_v1.joblib")

LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]


def evaluate(name, y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)

    macro_f1 = f1_score(
        y_true,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    mixed_errors = sum(
        (
            (gold == "ENG" and pred == "ROM_BAN")
            or
            (gold == "ROM_BAN" and pred == "ENG")
        )
        for gold, pred in zip(y_true, predictions)
    )

    unk_recall = sum(
        gold == "UNK" and pred == "UNK"
        for gold, pred in zip(y_true, predictions)
    ) / max(1, sum(label == "UNK" for label in y_true))

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1: {macro_f1:.3f}")
    print("ENG / ROM_BAN errors:", mixed_errors)
    print(f"UNK recall: {unk_recall:.2%}")

    return accuracy, macro_f1


def main():
    print("Loading datasets...")

    train_df, train_features = prepare_data(TRAIN)
    dev_df, dev_features = prepare_data(DEV)

    print("Loading V1 model...")

    bundle = joblib.load(BASELINE_MODEL)

    vectorizer = bundle["vectorizer"]
    baseline = bundle["classifier"]

    x_train = vectorizer.transform(train_features)
    x_dev = vectorizer.transform(dev_features)

    y_train = train_df["label"].tolist()
    y_dev = dev_df["label"].tolist()

    baseline_predictions = baseline.predict(x_dev)

    evaluate(
        "V1 BASELINE",
        y_dev,
        baseline_predictions,
    )

    print("\nCalculating class weights...")

    counts = Counter(y_train)
    largest = max(counts.values())

    weights = {
        label: min(
            4.0,
            (largest / counts[label]) ** 0.5,
        )
        for label in LABELS
    }

    print("Class weights:", weights)

    print("\nTraining weighted V2 experiment...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight=weights,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_dev)

    evaluate(
        "V2 WEIGHTED EXPERIMENT",
        y_dev,
        predictions,
    )

    print("\n=== V2 CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y_dev,
            predictions,
            labels=LABELS,
            digits=3,
            zero_division=0,
        )
    )

    print("\nExperiment complete.")
    print("V1 model was not changed.")
    print("No V2 model was saved.")
    print("Held-out test data was not evaluated.")


if __name__ == "__main__":
    main()
