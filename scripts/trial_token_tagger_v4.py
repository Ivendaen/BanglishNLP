
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd

from scripts.train_token_tagger_baseline import prepare_data


TRAIN = Path(
    "data/prepared/token_tagging/train_reviewed.csv"
)

DEV = Path(
    "data/prepared/token_tagging/dev.csv"
)

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
        (g == "ENG" and p == "ROM_BAN")
        or (g == "ROM_BAN" and p == "ENG")
        for g, p in zip(gold, predictions)
    )

    unk_total = sum(g == "UNK" for g in gold)
    unk_correct = sum(
        g == "UNK" and p == "UNK"
        for g, p in zip(gold, predictions)
    )

    unk_recall = (
        unk_correct / unk_total if unk_total else 0.0
    )

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1 (5 classes): {macro_f1:.3f}")
    print(f"Macro-F1 (4 core classes): {core_f1:.3f}")
    print("ENG / ROM_BAN errors:", mixed_errors)
    print(f"UNK recall: {unk_recall:.2%}")

    return accuracy, macro_f1


def main():
    print("Loading corrected training data...")

    train_df, train_features = prepare_data(TRAIN)
    dev_df, dev_features = prepare_data(DEV)

    print("Training tokens:", len(train_df))
    print("Development tokens:", len(dev_df))

    print("\nLoading V1 baseline...")

    bundle = joblib.load(V1_MODEL)

    vectorizer = bundle["vectorizer"]
    baseline = bundle["classifier"]

    x_train = vectorizer.transform(train_features)
    x_dev = vectorizer.transform(dev_features)

    y_train = train_df["label"].tolist()
    y_dev = dev_df["label"].tolist()

    v1_predictions = baseline.predict(x_dev)

    evaluate(
        "V1 ORIGINAL BASELINE",
        y_dev,
        v1_predictions,
    )

    print("\nTraining V4 with corrected labels...")

    v4 = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    v4.fit(x_train, y_train)

    v4_predictions = v4.predict(x_dev)

    evaluate(
        "V4 CORRECTED TRAINING LABELS",
        y_dev,
        v4_predictions,
    )

    print("\n=== V4 CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y_dev,
            v4_predictions,
            labels=LABELS,
            digits=3,
            zero_division=0,
        )
    )

    print("\n=== V4 CONFUSION MATRIX ===")

    matrix = confusion_matrix(
        y_dev,
        v4_predictions,
        labels=LABELS,
    )

    print(
        pd.DataFrame(
            matrix,
            index=LABELS,
            columns=LABELS,
        ).to_string()
    )

    changes = sum(
        old != new
        for old, new in zip(
            v1_predictions,
            v4_predictions,
        )
    )

    print("\n=== PREDICTION COMPARISON ===")
    print("Predictions changed from V1:", changes)

    print("\nExperiment completed.")
    print("V1 model was not modified.")
    print("V4 model was not saved.")
    print("Held-out test set was NOT evaluated.")


if __name__ == "__main__":
    main()
