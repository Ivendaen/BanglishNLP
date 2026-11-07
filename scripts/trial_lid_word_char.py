
from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

TRAIN = Path("data/train/lid_train.csv")
DEV = Path("data/dev/lid_dev.csv")
OLD_MODEL = Path("models/lid_ml_v1.joblib")

LABELS = [
    "BANGLA", "ENGLISH",
    "BANGLISH", "CODE_MIXED"
]

train = pd.read_csv(TRAIN, keep_default_na=False)
dev = pd.read_csv(DEV, keep_default_na=False)

assert len(dev) == 100
assert all((dev["label"] == x).sum() == 25 for x in LABELS)

print("Loading previous ML model...")
old_model = joblib.load(OLD_MODEL)
old_pred = old_model.predict(dev["text"])

print("Training experimental word + character model...")

trial_model = Pipeline([
    ("features", FeatureUnion([
        ("characters", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=60000,
            sublinear_tf=True
        )),
        ("words", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        ))
    ])),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

trial_model.fit(train["text"], train["label"])
new_pred = trial_model.predict(dev["text"])

gold = dev["label"]

print("\n=== DEVELOPMENT COMPARISON ===")
print("Original accuracy:",
      f"{accuracy_score(gold, old_pred):.2%}")
print("Trial accuracy:",
      f"{accuracy_score(gold, new_pred):.2%}")

print("Original macro-F1:",
      f"{f1_score(gold, old_pred, labels=LABELS, average='macro'):.3f}")
print("Trial macro-F1:",
      f"{f1_score(gold, new_pred, labels=LABELS, average='macro'):.3f}")

print("\n=== CORRECT PER CATEGORY ===")

for label in LABELS:
    mask = gold == label
    before = sum(old_pred[mask] == label)
    after = sum(new_pred[mask] == label)
    print(f"{label}: {before}/25 -> {after}/25")

print("\n=== CHANGED PREDICTIONS ===")

changed = 0

for i in range(len(dev)):
    if old_pred[i] != new_pred[i]:
        changed += 1
        row = dev.iloc[i]

        print("\nID:", row["id"])
        print("Text:", row["text"])
        print("Gold:", row["label"])
        print("Before:", old_pred[i])
        print("Trial:", new_pred[i])

print("\nTotal changed:", changed)
print("Original model was not modified.")
print("Held-out test dataset was not used.")
