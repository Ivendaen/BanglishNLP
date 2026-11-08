# BanglishNLP — Token-Level Language Annotation Guidelines

## 1. Purpose

These guidelines define how BanglishNLP assigns language labels to individual tokens in Bangla-English code-mixed text.

The tagging module will use the existing BanglishNLP tokenizer to identify token boundaries.

## 2. Language Labels

| Label | Description |
|---|---|
| BAN | Bengali word written in Bengali script |
| ENG | English-language word |
| ROM_BAN | Romanized Bengali word |
| OTHER | Punctuation, emoji, numbers, or other non-linguistic tokens |
| UNK | Linguistic token with uncertain or ambiguous language identity |

## 3. Annotation Rules

1. Assign one label to each token.
2. Use `BAN` for Bengali-script Bengali words.
3. Use `ENG` for words clearly functioning as English.
4. Use `ROM_BAN` for Bengali words written in Latin script.
5. Use `OTHER` for standalone punctuation, symbols, emojis, and numbers.
6. Use `UNK` only when the language of a linguistic token cannot reasonably be determined from its context.
7. Consider surrounding words when distinguishing English from Romanized Bengali.
8. Do not automatically classify every borrowed English word as `ENG`.
9. Keep the original token text and character offsets unchanged.
10. Record uncertain annotation decisions for later review.

## 4. Example

Input:

`ami today বাড়ি যাব!`

Expected annotation:

| Token | Label |
|---|---|
| ami | ROM_BAN |
| today | ENG |
| বাড়ি | BAN |
| যাব | BAN |
| ! | OTHER |

## 5. Ambiguous Words

Tokens such as `class`, `exam`, and `school` may occur in otherwise Bengali sentences.

Their classification should follow a consistent contextual annotation policy rather than relying only on their spelling.

Disagreements should be recorded and reviewed by a second annotator.

## 6. Dataset

Primary candidate:

**BanglaMix (2026)**

Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC13685450/

Dataset: https://zenodo.org/records/22086156

Research code: https://github.com/smahmuddz/BanglaMix

The original dataset labels must be examined before adapting them to BanglishNLP.

## 7. Evaluation Rules

- Split data at the sentence/comment level, not the individual-token level.
- Keep training, development, and final test sets separate.
- Check for duplicate or near-duplicate sentences across splits.
- Report token-level accuracy, macro-F1, and per-class precision and recall.
- Do not tune the model using the final held-out test set.

## 8. Important Limitation

BanglaMix uses whitespace-based token annotations, while BanglishNLP's tokenizer separates additional punctuation and special tokens.

Token boundaries must be aligned and verified before using BanglaMix labels for training or evaluation.

Unaligned or ambiguous examples should be reviewed rather than silently assigned labels.