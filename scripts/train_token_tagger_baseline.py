from banglishnlp.token_features import token_features
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


DATA_DIR = Path("data/prepared/token_tagging")
TRAIN = DATA_DIR / "train.csv"
DEV = DATA_DIR / "dev.csv"

MODEL = Path("models/token_tagger_v1.joblib")
OUTPUT = Path("data/processed/token_tagger_dev_predictions.csv")

LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]

def prepare_data(path):
    df = pd.read_csv(path, keep_default_na=False)

    required = {
        "row_index",
        "token",
        "start",
        "label",
    }

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {path}")

    if not df["label"].isin(LABELS).all():
        raise ValueError(f"Invalid labels: {path}")

    df = df.sort_values(
        ["row_index", "start"],
        kind="stable",
    ).reset_index(drop=True)

    features = []

    for _, group in df.groupby("row_index", sort=False):
        tokens = group["token"].astype(str).tolist()

        for i, token in enumerate(tokens):
            previous = tokens[i - 1] if i > 0 else ""
            following = (
                tokens[i + 1]
                if i + 1 < len(tokens)
                else ""
            )

            features.append(
                token_features(token, previous, following)
            )

    if len(features) != len(df):
        raise ValueError("Feature count mismatch")

    return df, features


def main():
    if MODEL.exists():
        raise FileExistsError(
            "Model already exists. Do not overwrite it."
        )

    print("Loading token-level training data...")

    train_df, train_features = prepare_data(TRAIN)
    dev_df, dev_features = prepare_data(DEV)

    print("Training tokens:", len(train_df))
    print("Development tokens:", len(dev_df))

    print("\nExtracting token features...")

    vectorizer = DictVectorizer(sparse=True)

    x_train = vectorizer.fit_transform(train_features)
    x_dev = vectorizer.transform(dev_features)

    y_train = train_df["label"].tolist()
    y_dev = dev_df["label"].tolist()

    print("Feature dimensions:", x_train.shape[1])

    print("\nTraining Logistic Regression classifier...")

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    classifier.fit(x_train, y_train)

    print("\nEvaluating on development data...")

    predictions = classifier.predict(x_dev)

    accuracy = accuracy_score(y_dev, predictions)

    macro_f1 = f1_score(
        y_dev,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print("\n=== TOKEN TAGGING DEVELOPMENT RESULTS ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1: {macro_f1:.3f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_dev,
            predictions,
            labels=LABELS,
            digits=3,
            zero_division=0,
        )
    )

    print("\n=== CONFUSION MATRIX ===")
    matrix = confusion_matrix(
        y_dev,
        predictions,
        labels=LABELS,
    )

    print(
        pd.DataFrame(
            matrix,
            index=LABELS,
            columns=LABELS,
        ).to_string()
    )

    dev_df["predicted_label"] = predictions
    dev_df["correct"] = (
        dev_df["label"] == dev_df["predicted_label"]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)

    dev_df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": LABELS,
            "feature_version": 1,
        },
        MODEL,
    )

    print("\nSaved model:", MODEL)
    print("Saved development predictions:", OUTPUT)
    print("\nHeld-out test data was NOT evaluated.")


if __name__ == "__main__":
    main()

