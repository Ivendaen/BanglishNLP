# IndicLID Attribution

## Original Project

- **Project:** IndicLID
- **Organization:** AI4Bharat
- **Source code:** https://github.com/AI4Bharat/IndicLID
- **Research paper:** Bhasa-Abhijnaanam: Native-script and Romanized Language Identification for 22 Indic Languages (2023)
- **License:** MIT License, as stated in the original project's README.

## Usage in BanglishNLP

BanglishNLP's Language Identification module is adapted from the IndicLID language-identification implementation.

The module uses the pretrained IndicLID-FTN and IndicLID-FTR models for native-script and Romanized language identification.

## BanglishNLP Modifications

The implementation has been adapted to:

- Identify Bangla, English, Banglish, and code-mixed text.
- Map IndicLID language predictions to BanglishNLP's supported output labels.
- Introduce experimental code-mixed detection logic.
- Provide an interface integrated into the BanglishNLP Python library.

The code-mixed identification logic is an adaptation and should not be interpreted as an original IndicLID feature.

## Acknowledgment

We acknowledge AI4Bharat and the IndicLID authors for making their research, implementation, and pretrained models publicly available.

BanglishNLP is an independent project and is not an official AI4Bharat implementation.