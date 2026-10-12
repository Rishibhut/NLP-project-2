"""
Unit tests for span language identification and pair classification.
"""

import pytest
from src.language_identifier import SpanLanguageIdentifier


@pytest.fixture
def lang_id():
    return SpanLanguageIdentifier()


def test_english_span(lang_id):
    pred = lang_id.identify_span("battery backup")
    assert pred.predicted_language == "English"
    assert pred.confidence >= 0.70


def test_romanized_hindi_span(lang_id):
    pred = lang_id.identify_span("bahut kharab hai")
    assert pred.predicted_language == "Hindi"
    assert pred.confidence >= 0.70


def test_devanagari_span(lang_id):
    pred = lang_id.identify_span("बहुत अच्छा है")
    assert pred.predicted_language == "Hindi"
    assert pred.devanagari_token_count > 0


def test_mixed_span(lang_id):
    pred = lang_id.identify_span("display acha hai")
    assert pred.predicted_language == "Mixed"


def test_ambiguous_short_token(lang_id):
    # 'is' or 'to' by itself
    pred = lang_id.identify_span("is")
    assert pred.predicted_language in {"English", "Uncertain"}


def test_empty_span(lang_id):
    pred = lang_id.identify_span("")
    assert pred.predicted_language == "Uncertain"
    assert pred.confidence == 0.0


def test_pair_classification(lang_id):
    assert lang_id.classify_pair_type("English", "English") == "same-language"
    assert lang_id.classify_pair_type("Hindi", "Hindi") == "same-language"
    assert lang_id.classify_pair_type("English", "Hindi") == "cross-lingual"
    assert lang_id.classify_pair_type("Hindi", "English") == "cross-lingual"
    assert lang_id.classify_pair_type("Uncertain", "English") == "Uncertain"
