# BanglishNLP — Tokenization Module

## Overview

The Tokenization module splits Bangla, English, Romanized Bangla (Banglish), and code-mixed text into individual tokens.

It uses a Unicode-aware, rule-based approach implemented in Python.

## Features

- Bengali-script word tokenization
- English and Romanized Bangla tokenization
- Bangla-English code-mixed text support
- Separation of punctuation
- Preservation of decimal numbers
- Email address and URL preservation
- Hashtag and mention handling
- Support for common emojis, including joined family emojis
- Original character-offset preservation

## Usage

```python
from banglishnlp.tokenizer import tokenize

text = "Ami today বাড়ি যাব!"

tokens = tokenize(text)

print(tokens)
```

Expected output:

```python
["Ami", "today", "বাড়ি", "যাব", "!"]
```

## Tokenization with Character Offsets

```python
from banglishnlp.tokenizer import tokenize_with_spans

text = "Ami jabo!"

result = tokenize_with_spans(text)

print(result)
```

Expected output:

```python
[
    ("Ami", 0, 3),
    ("jabo", 4, 8),
    ("!", 8, 9)
]
```

Offsets refer to Python string character positions in the original input, with the end position exclusive.

## Implementation

The module uses Python's `re` library and ordered regular-expression rules.

Special token types, including URLs, email addresses, and common emoji sequences, are recognized before ordinary words and punctuation.

The tokenizer does not require a machine learning model or a training dataset.

## Relationship with Text Cleaning

Tokenization can operate directly on the original text.

When exact character offsets, URLs, or user mentions must be preserved, tokenization should be performed before destructive text cleaning.

## Relationship with Language Identification

Tokenization only identifies token boundaries. It does not determine whether a token is Bangla, English, or Romanized Bangla.

Token-level language tagging will be developed as a separate module.

## Testing

The project test suite passed 60 tests after adding tokenizer coverage.

Tokenizer tests cover basic Bangla, Banglish, code-mixed text, punctuation, numbers, URLs, email addresses, emojis, and character offsets.

To run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

## Limitations

- The tokenizer is rule-based and has not been evaluated against a large, independently annotated tokenization benchmark.
- Some uncommon Unicode grapheme clusters may be split incorrectly.
- Certain URL formats, social-media conventions, and unusual punctuation patterns may require additional rules.
- The tokenizer does not perform morphological segmentation or subword tokenization.

## Future Work

- Evaluate on real Bangla-English code-mixed datasets.
- Expand regression tests for difficult Unicode and social-media cases.
- Integrate token-level language identification.
- Evaluate token-level tagging using independently annotated data.