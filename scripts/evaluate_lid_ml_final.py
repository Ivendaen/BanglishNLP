
from pathlib import Path
import hashlib

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from banglishnlp.ml_language_detector import MLBanglishLID


LABELS = ["BANGLA", "ENGLISH", "BANGLISH", "CODE_MIXED"]

TRAIN = Path("data/train/lid_train.csv")
DEV = Path("data/dev/lid_dev.csv")
TEST = Path("data/test/lid_test.csv")
MODEL = Path("models/lid_ml_v1.joblib")

OUTPUT = Path("data/processed/lid_ml_test_predictions.csv")
REPORT = Path("docs/lid_ml_final_evaluation.md")


def normalize(text):
    return " ".join(str(text).split()).casefold()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError(
            "Final evaluation results already exist. "
            "Do not overwrite the frozen test results."
        )

    print("Checking dataset integrity...")

    train = pd.read_csv(TRAIN, keep_default_na=False)
    dev = pd.read_csv(DEV, keep_default_na=False)
    test = pd.read_csv(TEST, keep_default_na=False)

    if len(test) != 100:
        raise ValueError("Expected 100 held-out test sentences")

    expected = {label: 25 for label in LABELS}

    if test["label"].value_counts().to_dict() != expected:
        raise ValueError("Expected 25 test sentences per category")

    if not test["review_status"].str.lower().eq("reviewed").all():
        raise ValueError("Some test annotations are not reviewed")

    train_keys = set(train["text"].map(normalize))
    dev_keys = set(dev["text"].map(normalize))
    test_keys = set(test["text"].map(normalize))

    if len(test_keys) != len(test):
        raise ValueError("Duplicate test sentences detected")

    if train_keys & dev_keys:
        raise ValueError("Training and development data overlap")

    if train_keys & test_keys:
        raise ValueError("Training and test data overlap")

    if dev_keys & test_keys:
        raise ValueError("Development and test data overlap")

    model_hash = sha256(MODEL)
    test_hash = sha256(TEST)

    print("Loading frozen ML model...")

    lid = MLBanglishLID(
        model_path=MODEL,
        mixed_ratio=0.75,
    )

    print("Evaluating held-out test set...")

    predictions = lid.batch_predict(test["text"].tolist())

    accuracy = accuracy_score(test["label"], predictions)

    macro_f1 = f1_score(
        test["label"],
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    details = classification_report(
        test["label"],
        predictions,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )

    matrix = pd.DataFrame(
        confusion_matrix(
            test["label"],
            predictions,
            labels=LABELS,
        ),
        index=LABELS,
        columns=LABELS,
    )

    print("\n=== FINAL HELD-OUT EVALUATION ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1: {macro_f1:.3f}")

    print("\n=== PER-CLASS METRICS ===")
    print(details)

    print("\n=== CONFUSION MATRIX ===")
    print(matrix.to_string())

    test["predicted_label"] = predictions
    test["correct"] = test["label"] == test["predicted_label"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    test.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    report = (
        "# BanglishNLP - Final ML LID Evaluation\n\n"
        "## Model\n"
        "- TF-IDF character n-grams + Logistic Regression\n"
        f"- Model file: `{MODEL}`\n"
        "- CODE_MIXED/BANGLISH ratio threshold: 0.75\n"
        "- Threshold selected using development data\n\n"
        "## Held-out test results\n"
        f"- Test sentences: {len(test)}\n"
        f"- Accuracy: {accuracy:.2%}\n"
        f"- Macro-F1: {macro_f1:.3f}\n\n"
        "## Per-class metrics\n\n"
        "```\n"
        f"{details}\n"
        "```\n\n"
        "## Confusion matrix\n\n"
        "```\n"
        f"{matrix.to_string()}\n"
        "```\n\n"
        "## Reproducibility\n"
        f"- Model SHA-256: `{model_hash}`\n"
        f"- Test SHA-256: `{test_hash}`\n"
        "- No exact-text overlap among training, development "
        "and held-out test sets\n\n"
        "## Limitations\n"
        "- Small held-out test set of 100 sentences\n"
        "- AI-assisted training annotations require quality auditing\n"
        "- Near-duplicate and source-level overlap not fully assessed\n"
        "- Loanword distinctions can make CODE_MIXED labels ambiguous\n"
        "- The development set was used to select the threshold\n"
        "- Post-evaluation annotation corrections must be disclosed\n"
    )

    REPORT.write_text(report, encoding="utf-8")

    print("\nSaved predictions:", OUTPUT)
    print("Saved report:", REPORT)
    print("\nFinal evaluation complete.")


if __name__ == "__main__":
    main()
