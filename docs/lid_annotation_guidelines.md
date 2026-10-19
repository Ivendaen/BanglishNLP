# BanglishNLP — Language Identification Annotation Guidelines

## 1. Objective

Prepare a manually labeled dataset to evaluate BanglishNLP's Language Identification (LID) module.

The module identifies four categories:

- `BANGLA`
- `ENGLISH`
- `BANGLISH`
- `CODE_MIXED`

Each sentence receives one language label.

## 2. Label Definitions

### BANGLA

Text written in Bengali script, expressing Bangla language.

**Example:**

- `আমি আজকে বাড়ি যাব` → `BANGLA`
- `তুমি কেমন আছো?` → `BANGLA`

### ENGLISH

Text written in English, using English vocabulary and grammar.

**Example:**

- `I am going home today` → `ENGLISH`
- `How are you doing?` → `ENGLISH`

### BANGLISH

Bangla language written entirely in Roman (English) letters, without clear English-language switching.

**Example:**

- `ami ajke bari jabo` → `BANGLISH`
- `tumi kemon acho` → `BANGLISH`

### CODE_MIXED

Text containing meaningful elements of both Bangla and English, regardless of whether they use Bengali or Roman script.

**Example:**

- `ami today bari jabo` → `CODE_MIXED`
- `আমি today বাড়ি যাব` → `CODE_MIXED`
- `Tomorrow amar exam ache` → `CODE_MIXED`

English-origin words commonly borrowed into everyday Bangla may be ambiguous. Do not automatically classify every such word as language switching.

## 3. Annotation Rules

1. Read the complete sentence before assigning a label.
2. Identify the actual language, not merely the writing script.
3. Ignore punctuation, emojis, numbers, and extra whitespace when determining language.
4. Proper names and brand names alone do not establish code-mixing.
5. A sentence mixing Bengali and Roman scripts is not necessarily mixing Bangla and English.
6. Do not use the LID model's predictions to decide the correct reference label.
7. If the language is unclear or the sentence is too short to classify reliably, flag it for manual review instead of guessing.

## 4. Ambiguous Examples

| Sentence | Annotation decision |
|---|---|
| `ami ajke university te jabo` | Review whether *university* functions as English switching or an established loanword |
| `ami ajke class e jabo` | Review: *class* is commonly borrowed into Bangla |
| `আমি ajke বাড়ি যাব` | Review: same language expressed in two scripts |
| `okay` | Review if contextual meaning is insufficient |

`REVIEW` is an annotation workflow status, **not** a fifth LID model output category. Resolve or exclude such cases from the four-class evaluation.

## 5. Annotation Process

- Collect sentences from documented sources.
- Assign labels independently of model predictions.
- Have another team member review ambiguous or disputed labels.
- Record the source and final annotation.
- Keep development and held-out evaluation sets separate.
- Do not change reference labels simply to match predictions.

## 6. Initial Dataset Target

Collect approximately **200 sentences**:

| Label | Target |
|---|---:|
| BANGLA | 50 |
| ENGLISH | 50 |
| BANGLISH | 50 |
| CODE_MIXED | 50 |
| **Total** | **200** |

The initial target is for a small evaluation benchmark, not a comprehensive training dataset.

After annotation, divide the examples into development and held-out test sets without duplicate or near-duplicate sentences across them.

## 7. Evaluation

Measure:

- Overall accuracy
- Precision, recall, and F1-score for each category
- Macro-F1
- Confusion matrix

Analyze classification errors before modifying the detector.

The held-out test set must remain untouched during model improvement.