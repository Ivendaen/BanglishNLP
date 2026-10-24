
from pathlib import Path
import pandas as pd

from banglishnlp.language_detector import BanglishLID

SOURCE = Path("data/processed/lid_dev_predictions.csv")


class TrialLID(BanglishLID):
    """Experimental rule: do not modify the production detector."""

    def _roman_code_mix(self, text):
        ftn = self._scores(self.native_model, text)
        ftr = self._scores(self.roman_model, text)

        strong_english = (
            ftn.get("eng_Latn", 0.0) >= 0.95
            and ftr.get("eng_Latn", 0.0) >= 0.995
            and ftr.get("ben_Latn", 0.0) <= 0.001
        )

        if strong_english:
            return False

        return super()._roman_code_mix(text)


df = pd.read_csv(SOURCE, keep_default_na=False)

assert len(df) == 100, "Expected 100 development examples"

print("Loading experimental detector...")
lid = TrialLID()

new_predictions = []

for text in df["text"]:
    try:
        predicted = lid.predict(text)
    except ValueError:
        predicted = "REJECTED"

    new_predictions.append(predicted)

df["trial_prediction"] = new_predictions

original_correct = (
    df["label"] == df["predicted_label"]
).sum()

trial_correct = (
    df["label"] == df["trial_prediction"]
).sum()

changed = df[
    df["predicted_label"] != df["trial_prediction"]
]

print("\n=== COMPARISON ===")
print(f"Original accuracy: {original_correct}%")
print(f"Trial accuracy:    {trial_correct}%")
print("Changed predictions:", len(changed))

print("\n=== CHANGED SENTENCES ===")

for _, row in changed.iterrows():
    print(f"\nID: {row['id']}")
    print("Text:", row["text"])
    print("Gold:", row["label"])
    print("Before:", row["predicted_label"])
    print("Trial:", row["trial_prediction"])

print("\nProduction detector was not modified.")
