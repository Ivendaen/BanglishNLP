"""BanglishNLP four-label language detector (experimental).

Adapted from the IndicLID FTN/FTR inference approach by AI4Bharat.
Original project: https://github.com/AI4Bharat/IndicLID
Preserve the original MIT LICENSE and attribution when redistributing.

Pretrained IndicLID recognizes many languages. This adapter only exposes
BANGLA, ENGLISH, BANGLISH and CODE_MIXED. Inputs outside its supported
scope raise ValueError instead of receiving a knowingly misleading label.

Heuristics, thresholds, and mixing rules are NOT validated accuracy claims.
"""

import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FTN = ROOT / "models" / "indiclid-ftn" / "model_baseline_roman.bin"
DEFAULT_FTR = ROOT / "models" / "indiclid-ftr" / "model_baseline_roman.bin"
LATIN_WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LABELS = {"BANGLA", "ENGLISH", "BANGLISH", "CODE_MIXED"}


class BanglishLID:
    """Use existing IndicLID weights with BanglishNLP-specific decision rules.

    `native_model`/`roman_model` allow small mock models in unit tests.
    The model weights are required for real predictions.
    """

    def __init__(self, native_model_path=DEFAULT_FTN, roman_model_path=DEFAULT_FTR,
                 *, native_model=None, roman_model=None,
                 min_banglish_score=0.05, min_relative_banglish=0.65,
                 english_ftn_threshold=0.90):
        if (native_model is None) != (roman_model is None):
            raise ValueError("Provide both model objects or neither")
        for value in (min_banglish_score, min_relative_banglish, english_ftn_threshold):
            if not 0 <= value <= 1:
                raise ValueError("Thresholds must be between 0 and 1")
        self.min_banglish_score = min_banglish_score
        self.min_relative_banglish = min_relative_banglish
        self.english_ftn_threshold = english_ftn_threshold

        if native_model is not None:
            self.native_model, self.roman_model = native_model, roman_model
            return
        paths = (Path(native_model_path), Path(roman_model_path))
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"IndicLID model file not found: {path}")
        try:
            import fasttext  # Provided by fasttext-community on Python 3.14
        except ImportError as exc:
            raise ImportError("Install fasttext-community first") from exc
        self.native_model = fasttext.load_model(str(paths[0]))
        self.roman_model = fasttext.load_model(str(paths[1]))

    @staticmethod
    def _has_bangla(text):
        return any('\u0980' <= c <= '\u09ff' and
                   unicodedata.category(c).startswith('L') for c in text)

    @staticmethod
    def _has_latin(text):
        return bool(re.search(r"[A-Za-z]", text))

    @staticmethod
    def _scores(model, text):
        # Keep multilingual scores internally. Never return unsupported labels.
        labels, probs = model.predict(text, k=-1)
        return {str(label).removeprefix('__label__'): float(prob)
                for label, prob in zip(labels, probs)}

    def _latin_label(self, text):
        """Provisional English-vs-Banglish routing based on both pretrained models.

        These thresholds are diagnostic defaults, not fitted to a gold set.
        An unsupported/ambiguous sample is rejected rather than force-labelled.
        """
        ftr = self._scores(self.roman_model, text)
        ftn = self._scores(self.native_model, text)
        ben = ftr.get('ben_Latn', 0.0)
        eng = ftr.get('eng_Latn', 0.0)
        relative_ben = ben / (ben + eng) if ben + eng else 0.0

        # A strong romanized-Bengali signal can override the incorrect
        # multilingual top-1 label (e.g. snd_Latn on 'ami ajke bari jabo').
        if ben >= self.min_banglish_score and relative_ben >= self.min_relative_banglish:
            return 'BANGLISH'
        # FTN recognized ordinary English better in our first diagnostic tests.
        if ftn.get('eng_Latn', 0.0) >= self.english_ftn_threshold:
            return 'ENGLISH'
        raise ValueError('Uncertain or unsupported Latin-script language')

    def _roman_code_mix(self, text):
        """Detect some multiword English/Banglish spans; not token-level LID.

        Restrict to boundaries with >=3 words on each side to reduce false
        positives from isolated loanwords. May miss interleaved code-mixing.
        """
        
        # Avoid false code-mixing predictions
        # on strongly identified English sentences.
        ftn = self._scores(self.native_model, text)
        ftr = self._scores(self.roman_model, text)

        strong_english = (
            ftn.get("eng_Latn", 0.0) >= 0.95
            and ftr.get("eng_Latn", 0.0) >= 0.995
            and ftr.get("ben_Latn", 0.0) <= 0.001
        )

        if strong_english:
            return False

        # Experimental Romanized Bangla + English switch cues.
        bangla_cues = {
            "ami", "amar", "amr", "tumi", "tomar",
            "apnar", "apnader", "onek", "valo",
            "bhalo", "ektu", "aktu", "kintu",
            "ache", "ase", "nai", "hobe", "hoto",
            "laglo", "lage", "korar", "kore",
            "chilo", "jonno", "diye", "kharap",
            "meyad", "abong", "daam", "er",
            "ta", "te", "ki", "na"
        }

        english_cues = {
            "but", "because", "though",
            "although", "however"
        }

        word_set = set(
            re.findall(r"[a-z]+", text.lower())
        )

        if (
            len(word_set & bangla_cues) >= 2
            and word_set & english_cues
        ):
            return True

        words = LATIN_WORDS.findall(text)[:40]
        if len(words) < 6:
            return False
        for split in range(3, len(words) - 2):
            left, right = ' '.join(words[:split]), ' '.join(words[split:])
            try:
                labels = {self._latin_label(left), self._latin_label(right)}
            except ValueError:
                continue
            if labels == {'ENGLISH', 'BANGLISH'}:
                return True
        return False

    def predict(self, text):
        """Return BANGLA, ENGLISH, BANGLISH or CODE_MIXED.

        Raises ValueError for empty, unsupported, or uncertain inputs.
        """
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        text = ' '.join(text.split())
        if not text:
            raise ValueError('Cannot classify empty text')
        bengali, latin = self._has_bangla(text), self._has_latin(text)
        if not (bengali or latin):
            raise ValueError('No Bangla or Latin text found')
        if bengali and latin:
            # Mixed script is a useful signal, but not proof of language mixing.
            # This preliminary rule must be benchmarked on named entities, etc.
            return 'CODE_MIXED'
        if bengali:
            scores = self._scores(self.native_model, text)
            if max(scores, key=scores.get, default=None) == 'ben_Beng':
                return 'BANGLA'
            raise ValueError('Unsupported or uncertain Bengali-script language')
        if self._roman_code_mix(text):
            return 'CODE_MIXED'
        return self._latin_label(text)

    def batch_predict(self, texts):
        if isinstance(texts, str):
            raise TypeError('Pass a list of texts to batch_predict')
        return [self.predict(text) for text in texts]
