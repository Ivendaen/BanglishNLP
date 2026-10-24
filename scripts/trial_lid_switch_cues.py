
import re
import pandas as pd

from banglishnlp.language_detector import BanglishLID

SOURCE = "data/processed/lid_dev_predictions.csv"

# Small, general-purpose list of Romanized Bangla clues
BANGLA_CUES = {
    "ami", "amar", "amr", "tumi", "tomar",
    "apnar", "apnader", "onek", "valo",
    "bhalo", "ektu", "aktu", "kintu",
    "ache", "ase", "nai", "hobe", "hoto",
    "laglo", "lage", "korar", "kore",
    "chilo", "jonno", "diye", "kharap",
    "meyad", "abong", "daam", "er",
    "ta", "te", "ki", "na"
}

ENGLISH_SWITCH_CUES = {
    "but", "because", "though",
    "although", "however"
}


class TrialLID(BanglishLID):

    def _roman_code_mix(self, text):
        words = set(
            re.findall(r"[a-z]+", text.lower())
        )

        # Preserve the English guard from Step 25
        ftn = self._scores(self.native_model, text)
        ftr = self._scores(self.roman_model, text)

        strong_english = (
            ftn.get("eng_Latn", 0) >= 0.95
            and ftr.get("eng_Latn", 0) >= 0.995
            and ftr.get("ben_Latn", 0) <= 0.001
        )

        if strong_english:
            return False

        # Experimental English-switch detection
        bangla_matches = words & BANGLA_CUES
        english_matches = words & ENGLISH_SWITCH_CUES

        if len(bangla_matches) >= 2 and english_matches:
            return True

        return super()._roman_code_mix(text)


df = pd.read_csv(SOURCE, keep_default_na=False)

assert len(df) == 100

print("Loading experimental detector...")
lid = TrialLID()

predictions = []

for text in df["text"]:
    try:
        prediction = lid.predict(text)
    except ValueError:
        prediction = "REJECTED"

    predictions.append(prediction)

df["trial_prediction"] = predictions

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
print(f"Trial accuracy: {trial_correct}%")
print("Changed predictions:", len(changed))

print("\n=== CHANGED SENTENCES ===")

for _, row in changed.iterrows():
    print("\nID:", row["id"])
    print("Text:", row["text"])
    print("Gold:", row["label"])
    print("Before:", row["predicted_label"])
    print("Trial:", row["trial_prediction"])

print("\nProduction detector was not modified.")
