
# BanglishNLP Text Cleaning Guidelines

## 1. Objective

Remove or normalize noise from Bangla-English code-mixed text
without changing the original meaning.

The cleaning module preserves Bangla, English, Romanized Bangla
(Banglish), numbers, and emojis.

## 2. Cleaning Rules

### 2.1 URL Removal

- Remove URLs beginning with `http://` or `https://`.
- Remove URLs beginning with `www.`.
- Remove bare domains with supported extensions,
  such as `example.com` and `example.org`.
- Preserve surrounding meaningful text and punctuation.
- Preserve email addresses.

### 2.2 User Mention Removal

- Remove English mentions such as `@rahim`.
- Remove Bengali mentions such as `@রহিম`.
- Support mentions containing numbers.
- Do not remove email addresses.

### 2.3 Whitespace Normalization

- Replace repeated spaces with a single space.
- Replace tabs and line breaks with a single space.
- Remove leading and trailing whitespace.

### 2.4 Punctuation Normalization

- Convert repeated punctuation:
  - `!!!` to `!`
  - `???` to `?`
  - `...` to `.`
  - `?!?!?` to `?!`
- Normalize supported mixed punctuation sequences.
- Remove unnecessary spaces before standard punctuation.
- Insert missing spaces after commas and periods
  where appropriate.
- Preserve decimal numbers and email addresses.
- Preserve meaningful punctuation.

### 2.5 Unicode Punctuation

Normalize supported Unicode punctuation variants,
including curly quotation marks and ellipses.

### 2.6 Language Preservation

- Preserve Bengali-script words.
- Preserve English words.
- Preserve Romanized Bangla words.
- Preserve numbers and emojis.
- Preserve original capitalization.

## 3. Punctuation Modes

The `clean_text()` function supports three modes:

- `normalize`: Normalize supported punctuation (default).
- `remove`: Remove Unicode punctuation.
- `keep`: Preserve punctuation while applying other
  cleaning operations.

## 4. Operations Not Included

The cleaning module does not perform:

- Spelling correction
- Banglish-to-Bangla transliteration
- Language identification
- Tokenization
- Semantic rewriting

These operations belong to separate BanglishNLP modules.

## 5. Evaluation

Text Cleaning was initially evaluated using 100
annotated code-mixed sentences.

Results:

- Initial development exact-match accuracy: 93%
- Improved development exact-match accuracy: 100%
- Automated unit tests: 25 passed

These are development results because the same examples
were used to identify and fix errors.

A separate unseen test dataset is required for an
independent evaluation.

## 6. Known Limitations

- Bare URL detection uses a supported set of common
  domain extensions.
- Unusual URLs and punctuation patterns may require
  additional handling.
- Current development samples do not cover every
  cleaning rule equally.
- Ambiguous annotations should be reviewed manually.
