# BanglishNLP — Token-Level Language Tagging

## 1. Overview

The Token-Level Language Tagging module identifies the language of individual tokens in Bangla-English code-mixed text.

It combines BanglishNLP's rule-based tokenizer with a machine learning classifier to assign language labels while preserving token positions.

## 2. Supported Language Labels

| Label | Description |
|---|---|
| BAN | Bengali written in Bengali script |
| ENG | English-language tokens |
| ROM_BAN | Romanized Bengali (Banglish) |
| OTHER | Punctuation, numbers, symbols, and emojis |
| UNK | Uncertain or unsupported language tokens |

## 3. Implementation

The module uses a Logistic Regression classifier trained with dictionary-based token features.

Features include:

- Token identity and lowercase representation
- Character prefixes and suffixes
- Bengali and Latin script identification
- Previous and following token context
- Token length and numerical characteristics

The trained classifier is stored as `token_tagger_v1.joblib`.

## 4. Python Interface

The `TokenLanguageTagger` class provides two main methods:

| Method | Description |
|---|---|
| `tag(text)` | Identifies the language of each token in a sentence |
| `batch_tag(texts)` | Processes multiple sentences |

Each tagged token includes its text, start character offset, end character offset, and predicted language label.

## 5. Dataset

**Dataset:** BanglaMix (2026)

**Source:** https://zenodo.org/records/22086156

The original dataset contained 12,576 comments with 120,862 parsed token annotations.

Following token-boundary alignment, 9,152 sentences containing 90,510 output tokens were selected for model development.

The dataset was divided into:

| Partition | Sentences | Tokens |
|---|---:|---:|
| Training | 7,320 | 71,995 |
| Development | 916 | 9,262 |
| Test | 916 | 9,253 |

The V1 classifier was trained on the original automatically aligned training partition. Targeted annotation corrections were investigated separately and did not replace the frozen V1 model.

## 6. Model Evaluation

### Development Results

| Metric | Result |
|---|---|
| Accuracy | 97.86% |
| Macro-F1 (5 classes) | 0.779 |
| Macro-F1 (4 core classes) | 0.974 |

### Final Reviewed Evaluation

The frozen V1 classifier was evaluated on 340 reviewed tokens from 40 held-out sentences.

| Metric | Result |
|---|---|
| Accuracy | **97.65%** |
| Correct predictions | 332/340 |
| Macro-F1 (5 classes) | 0.779 |
| Macro-F1 (4 core classes) | 0.973 |

### Per-Class Performance

| Language | Precision | Recall | F1-score |
|---|---:|---:|---:|
| BAN | 1.000 | 1.000 | 1.000 |
| ENG | 0.928 | 0.970 | 0.948 |
| ROM_BAN | 0.945 | 0.945 | 0.945 |
| OTHER | 1.000 | 1.000 | 1.000 |
| UNK | 0.000 | 0.000 | 0.000 |

The complete classification report and confusion matrix are available in `docs/token_tagger_final_evaluation.md`.

## 7. Limitations

- The final evaluation contains only 340 tokens from 40 sentences.
- Human-reviewed labels were not independently adjudicated by a second annotator.
- The original training data contains silver-standard annotations.
- English and Romanized Bengali words may be confused in certain contexts.
- Annotation conventions for borrowed words and proper names remain language-policy dependent.
- The classifier does not reliably recognize the UNK category.
- Performance on other real-world datasets may differ.

## 8. Reproducibility

The frozen V1 model and the reviewed evaluation dataset are identified by SHA-256 checksums in the final evaluation report.

Training datasets and serialized model files are maintained separately from the public source-code repository.

## 9. Conclusion

BanglishNLP's Token-Level Language Tagging module achieved 97.65% accuracy on the human-reviewed evaluation sample.

The results demonstrate strong performance for BAN, ENG, ROM_BAN, and OTHER classification under the project's annotation conventions. Reliable detection of uncertain or unsupported tokens remains an area for improvement.