# BanglishNLP Text Cleaning Evaluation

## 1. Objective

Evaluate the BanglishNLP Text Cleaning module on Bangla-English code-mixed text.

The module removes or normalizes URLs, user mentions, repeated whitespace, and punctuation while preserving the original words and meaning.

## 2. Evaluation Datasets

The evaluation used sentences selected from the `bn_code_mix_sentiment_dataset`.

Two separate sets were prepared:

- **Development set:** 100 annotated sentences used during cleaner development.
- **Held-out test set:** 100 different sentences excluded from the development set.

Reference cleaned outputs were prepared for each sentence. The annotations were AI-assisted and require human verification before being considered final gold-standard labels.

## 3. Evaluation Metric

Exact-match accuracy measures the percentage of sentences for which the cleaner's output exactly matches the reference annotation.

**Exact-match accuracy = (Correct outputs / Total sentences) × 100**

## 4. Results

| Dataset | Total | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| Development | 100 | 100 | 0 | 100% |
| Held-out test (original evaluation) | 100 | 98 | 2 | 98% |

Additionally, 25 unit tests passed during development.

## 5. Error Analysis

Two differences were observed in the initial held-out evaluation.

**Case t0073:** Missing space between a period and an opening parenthesis.

- Actual: `parbe.(kauke`
- Reference: `parbe. (kauke`

**Case t0083:** Missing space after a question mark.

- Actual: `paibo?gojorta?`
- Reference: `paibo? gojorta?`

Both cases involve punctuation spacing.

The reference annotations were subsequently edited, producing 100% agreement without changing the cleaner. This revised agreement is not reported as an independent improvement.

## 6. Limitations

- The evaluation sets contain only 100 sentences each.
- Most sentences are Romanized Bangla or Bangla-English code-mixed text.
- URLs, mentions, emojis, repeated whitespace, and Bengali-script text require more comprehensive test coverage.
- Reference annotations require independent human review.
- The development score was obtained after improving the cleaner using development examples.
- The held-out result measures performance on sentences from the same source corpus, not generalization to an entirely different domain.

## 7. Conclusion

The BanglishNLP Text Cleaning module achieved 98% exact-match accuracy on the initial 100-sentence held-out test set.

The results demonstrate promising performance on the evaluated sentences. Further evaluation with independently reviewed annotations and more diverse noisy text is needed before making broader performance claims.