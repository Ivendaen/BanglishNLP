
from pathlib import Path
import hashlib

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from banglishnlp.token_features import token_features


MODEL = Path("models/token_tagger_v1.joblib")

GOLD = Path(
    "data/prepared/token_tagging/blind_test_review.csv"
)

PREDICTIONS = Path(
    "data/processed/token_tagger_final_predictions.csv"
)

REPORT = Path(
    "docs/token_tagger_final_evaluation.md"
)

LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
CORE_LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if PREDICTIONS.exists() or REPORT.exists():
        raise FileExistsError(
            "Final evaluation already exists. "
            "Do not overwrite previous results."
        )

    print("Loading reviewed evaluation dataset...")

    df = pd.read_csv(GOLD, keep_default_na=False)

    required = {
        "row_index",
        "text",
        "token",
        "start",
        "end",
        "human_label",
        "review_status",
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    df["human_label"] = (
        df["human_label"].astype(str).str.strip()
    )

    df["review_status"] = (
        df["review_status"].astype(str).str.strip()
    )

    if len(df) != 340:
        raise ValueError("Expected 340 reviewed tokens")

    if df["row_index"].nunique() != 40:
        raise ValueError("Expected 40 reviewed sentences")

    if not df["human_label"].isin(LABELS).all():
        raise ValueError("Invalid or missing human labels")

    if not df["review_status"].eq("Reviewed").all():
        raise ValueError("Some tokens are not reviewed")

    keys = ["row_index", "start", "end"]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate token positions")

    for row in df.itertuples(index=False):
        if row.text[row.start:row.end] != row.token:
            raise ValueError(
                f"Invalid token offset in row {row.row_index}"
            )

    df = df.sort_values(
        ["row_index", "start"],
        kind="stable",
    ).reset_index(drop=True)

    print("Loading frozen V1 model...")

    bundle = joblib.load(MODEL)

    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    if not set(LABELS).issubset(
        set(classifier.classes_)
    ):
        raise ValueError("Model classes do not match")

    print("Extracting token features...")

    features = []

    for _, group in df.groupby(
        "row_index",
        sort=False,
    ):
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

    print("Running final predictions...")

    x_test = vectorizer.transform(features)

    predictions = classifier.predict(x_test)

    gold = df["human_label"].tolist()

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

    details = classification_report(
        gold,
        predictions,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )

    matrix = pd.DataFrame(
        confusion_matrix(
            gold,
            predictions,
            labels=LABELS,
        ),
        index=LABELS,
        columns=LABELS,
    )

    print("\n=== FINAL TOKEN TAGGING EVALUATION ===")
    print("Sentences:", df["row_index"].nunique())
    print("Tokens:", len(df))
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1 (5 classes): {macro_f1:.3f}")
    print(f"Macro-F1 (4 core classes): {core_f1:.3f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(details)

    print("\n=== CONFUSION MATRIX ===")
    print(matrix.to_string())

    df["predicted_label"] = predictions
    df["correct"] = (
        df["human_label"] == df["predicted_label"]
    )

    PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    report = (
        "# BanglishNLP Token Tagger - Final Evaluation\n\n"
        "## Model\n"
        "- Token-level Logistic Regression classifier\n"
        "- Contextual token, script, prefix, and suffix features\n"
        "- Frozen V1 model; no retraining for final evaluation\n\n"
        "## Human-reviewed evaluation\n"
        "- Sentences: 40\n"
        "- Tokens: 340\n"
        f"- Accuracy: {accuracy:.2%}\n"
        f"- Macro-F1 (5 classes): {macro_f1:.3f}\n"
        f"- Macro-F1 (4 core classes): {core_f1:.3f}\n\n"
        "## Classification report\n\n"
        "```\n"
        f"{details}\n"
        "```\n\n"
        "## Confusion matrix\n\n"
        "```\n"
        f"{matrix.to_string()}\n"
        "```\n\n"
        "## Reproducibility\n"
        f"- Model SHA-256: `{sha256(MODEL)}`\n"
        f"- Reviewed data SHA-256: `{sha256(GOLD)}`\n\n"
        "## Limitations\n"
        "- Small evaluation sample of 40 sentences\n"
        "- Human-reviewed labels have not been independently "
        "adjudicated by a second annotator\n"
        "- Proper-name and loanword annotation decisions "
        "remain policy-dependent\n"
        "- UNK has limited representation\n"
        "- Training data originated from silver-standard annotations\n"
        "- This result should not be generalized to all "
        "real-world code-mixed text\n"
    )

    REPORT.write_text(
        report,
        encoding="utf-8",
    )

    print("\nSaved predictions:", PREDICTIONS)
    print("Saved report:", REPORT)
    print("\nFinal evaluation complete.")


if __name__ == "__main__":
    main()
