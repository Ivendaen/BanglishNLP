import re
import pandas as pd

from banglishnlp.language_detector import BanglishLID

SOURCE = "data/dev/lid_dev.csv"

LABELS = [
    "BANGLA",
    "ENGLISH",
    "BANGLISH",
    "CODE_MIXED"
]

# Common Romanized Bangla words and spelling variants.
# Experimental vocabulary, not a complete dictionary.
BANGLISH_CUES = {
    "ami", "amar", "amr",
    "tumi", "tomar",
    "apu", "vai", "bhai",
    "apnar", "apnader",
    "bari", "koi", "chay",
    "kotha", "thaka",
    "kinbo", "kinsan",
    "kivabe", "kibave",
    "keno", "ajke",
    "onek", "valo", "bhalo",
    "laglo", "jabo", "korbo",
    "ektu", "aktu",
    "hocche", "hobe"
}


def predict_original(lid, text):
    try:
        return lid.predict(text)
    except ValueError:
        return "REJECTED"


def predict_trial(lid, text, original):
    words = set(
        re.findall(r"[a-z]+", text.lower())
    )

    banglish_matches = words & BANGLISH_CUES

    # Experimental indicators of multiword English phrases.
    english_phrases = [
        r"\bthank\s+you\b",
        r"\bthanks\s+you\b",
        r"\bgood\s+luck\b",
        r"\blove\s+you\b",
        r"\bsorry\s+to\s+say\b",
        r"\bto\s+be\s+honest\b",
        r"\bvery\s+disappointed\b",
    ]

    has_english_phrase = any(
        re.search(pattern, text.lower())
        for pattern in english_phrases
    )

    # Only test changing English or rejected predictions.
    if (
        original in {"ENGLISH", "REJECTED"}
        and len(banglish_matches) >= 2
    ):
        # Banglish clues plus a recognizable English phrase:
        # possible code-mixing.
        if has_english_phrase:
            return "CODE_MIXED"

        return "BANGLISH"

    return original


def calculate_metrics(gold, predicted):
    correct = sum(
        actual == pred
        for actual, pred in zip(gold, predicted)
    )

    accuracy = correct / len(gold)

    f1_scores = []

    for label in LABELS:
        tp = sum(
            a == label and p == label
            for a, p in zip(gold, predicted)
        )

        fp = sum(
            a != label and p == label
            for a, p in zip(gold, predicted)
        )

        fn = sum(
            a == label and p != label
            for a, p in zip(gold, predicted)
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        f1_scores.append(f1)

    return accuracy, sum(f1_scores) / len(f1_scores)


df = pd.read_csv(SOURCE, keep_default_na=False)

assert len(df) == 100
assert all(
    (df["label"] == label).sum() == 25
    for label in LABELS
)

print("Loading current BanglishLID...")
lid = BanglishLID()

original_predictions = []
trial_predictions = []

for text in df["text"]:
    original = predict_original(lid, text)
    trial = predict_trial(lid, text, original)

    original_predictions.append(original)
    trial_predictions.append(trial)

gold = df["label"].tolist()

original_acc, original_f1 = calculate_metrics(
    gold, original_predictions
)

trial_acc, trial_f1 = calculate_metrics(
    gold, trial_predictions
)

print("\n=== DEVELOPMENT COMPARISON ===")
print(f"Original accuracy: {original_acc:.2%}")
print(f"Trial accuracy:    {trial_acc:.2%}")
print(f"Original macro-F1: {original_f1:.3f}")
print(f"Trial macro-F1:    {trial_f1:.3f}")

print("\n=== PER-CLASS RECALL ===")

for label in LABELS:
    original_correct = sum(
        a == label and p == label
        for a, p in zip(gold, original_predictions)
    )

    trial_correct = sum(
        a == label and p == label
        for a, p in zip(gold, trial_predictions)
    )

    print(
        f"{label}: "
        f"Before {original_correct}/25, "
        f"After {trial_correct}/25"
    )

print("\n=== CHANGED PREDICTIONS ===")

changed_count = 0

for i, (before, after) in enumerate(
    zip(original_predictions, trial_predictions)
):
    if before != after:
        changed_count += 1

        print("\nID:", df.iloc[i]["id"])
        print("Text:", df.iloc[i]["text"])
        print("Gold:", gold[i])
        print("Before:", before)
        print("Trial:", after)

print("\nTotal changed:", changed_count)
print("Production detector was not modified.")