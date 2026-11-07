# BanglishNLP

A Python library for processing Bangla-English code-mixed text.

## Language Identification (LID)

BanglishNLP supports four language identification categories:

- **BANGLA:** Bangla written in Bengali script.
- **ENGLISH:** English-language text.
- **BANGLISH:** Romanized Bangla, including commonly borrowed words.
- **CODE_MIXED:** Meaningful switching between Bangla and English.

### Installation

Clone the repository and install the package:

```bash
git clone https://github.com/Ivendaen/BanglishNLP.git
cd BanglishNLP
python -m pip install -e .
```

### Using the ML Language Detector

The trained model file must be available at `models/lid_ml_v1.joblib`.

```python
from banglishnlp.ml_language_detector import MLBanglishLID

lid = MLBanglishLID()

print(lid.predict("ami ajke bari jabo"))
print(lid.predict("I am going home"))
print(lid.predict("আমি আজ বাড়ি যাব"))
```

The detector also supports batch prediction:

```python
sentences = [
    "ami ajke bari jabo",
    "I am going home",
    "আমি আজ বাড়ি যাব",
]

predictions = lid.batch_predict(sentences)
print(predictions)
```

### Model

The ML language detector uses:

- Character-level TF-IDF features (2–5 character n-grams)
- Logistic Regression with balanced class weights
- A calibrated CODE_MIXED/BANGLISH probability ratio of 0.75

### Final Evaluation

Evaluated on 100 held-out sentences.

| Metric | Result |
|---|---|
| Accuracy | 94.00% |
| Macro-F1 | 0.939 |
| BANGLA Recall | 100% |
| ENGLISH Recall | 100% |
| BANGLISH Recall | 100% |
| CODE_MIXED Recall | 76% |

See `docs/lid_ml_final_evaluation.md` for the complete report.

### Important Notes

The trained `.joblib` file and the evaluation datasets are not included in the GitHub repository. Teammates must obtain the trusted trained model separately or reproduce it using the approved training data.

The reported performance comes from a small held-out benchmark and may differ on new real-world text. Training annotations included AI-assisted labeling and require continued quality auditing.

Only load trusted `.joblib` model files.

### Project Status

- Text Cleaning — Implemented
- Language Identification — Implemented and evaluated
- Tokenization — Next development phase

### Acknowledgments

The project also contains an experimental language detector adapted from AI4Bharat IndicLID. See `docs/indiclid_attribution.md` for details.