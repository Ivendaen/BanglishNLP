"""Unit tests for classification *rules*, not pretrained-model accuracy."""
import pytest
from banglishnlp.language_detector import BanglishLID

class MockModel:
    def __init__(self, mapping):
        self.mapping = mapping
    def predict(self, text, k=-1):
        data = self.mapping.get(text, self.mapping.get('*', {}))
        labels = tuple('__label__' + k for k in data)
        return labels, list(data.values())


def test_bangla():
    ftn = MockModel({'*': {'ben_Beng': 0.99, 'asm_Beng': 0.01}})
    ftr = MockModel({'*': {'ben_Latn': 0.8, 'eng_Latn': 0.2}})
    assert BanglishLID(native_model=ftn, roman_model=ftr).predict('আমি বাড়ি যাব') == 'BANGLA'


def test_english():
    ftn = MockModel({'*': {'eng_Latn': 0.99, 'other': 0.01}})
    ftr = MockModel({'*': {'kok_Latn': 0.9, 'ben_Latn': 0.005, 'eng_Latn': 0.003}})
    assert BanglishLID(native_model=ftn, roman_model=ftr).predict('I will go home') == 'ENGLISH'


def test_banglish_multilingual_top_prediction_ignored():
    ftn = MockModel({'*': {'eng_Latn': 0.61, 'other': 0.39}})
    ftr = MockModel({'*': {'snd_Latn': 0.819, 'ben_Latn': 0.136, 'eng_Latn': 0.009}})
    assert BanglishLID(native_model=ftn, roman_model=ftr).predict('ami ajke bari jabo') == 'BANGLISH'


def test_mixed_script():
    ftn, ftr = MockModel({'*': {}}), MockModel({'*': {}})
    assert BanglishLID(native_model=ftn, roman_model=ftr).predict('আমি today যাব') == 'CODE_MIXED'


def test_unsupported_rejected():
    ftn = MockModel({'*': {'eng_Latn': 0.3, 'other': 0.7}})
    ftr = MockModel({'*': {'snd_Latn': 0.95, 'ben_Latn': 0.01, 'eng_Latn': 0.01}})
    with pytest.raises(ValueError):
        BanglishLID(native_model=ftn, roman_model=ftr).predict('xyzxyzzzz')


def test_empty_rejected():
    ftn, ftr = MockModel({'*': {}}), MockModel({'*': {}})
    with pytest.raises(ValueError):
        BanglishLID(native_model=ftn, roman_model=ftr).predict(' ')
