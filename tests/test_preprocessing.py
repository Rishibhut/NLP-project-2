"""
Unit tests for transliteration normalization, character flooding reduction, and conservative canonical mapping.
"""

import pytest
import pandas as pd
from src.preprocessing import TransliterationNormalizer, NormalizationConfig


def test_character_flooding_reduction():
    normalizer = TransliterationNormalizer(NormalizationConfig(reduce_repeated_chars=True, max_repeated_chars=2))

    res1 = normalizer.normalize("acccchhaaa")
    assert res1.is_modified is True
    # 'cccc' reduced to 'cc', 'aaa' reduced to 'aa'
    assert "c" * 3 not in res1.normalized_text
    assert "a" * 3 not in res1.normalized_text

    res2 = normalizer.normalize("sooooo good")
    assert res2.normalized_text == "soo good"


def test_phonetic_spelling_canonicalization():
    normalizer = TransliterationNormalizer(NormalizationConfig(normalize_phonetic_spelling=True))

    res1 = normalizer.normalize("Phone bohot accha hai")
    assert "bahut" in res1.normalized_text

    res2 = normalizer.normalize("Battery backup khraab hai")
    assert "kharab" in res2.normalized_text

    res3 = normalizer.normalize("Mic clarity bekar hai")
    assert "bekaar" in res3.normalized_text


def test_preserves_negation():
    normalizer = TransliterationNormalizer()
    res = normalizer.normalize("Display bilkul accha nahi hai")
    # 'nahi' must never be removed or translated
    assert "nahi" in res.normalized_text
    assert res.normalized_text.startswith("Display")


def test_empty_string_normalization():
    normalizer = TransliterationNormalizer()
    res = normalizer.normalize("")
    assert res.normalized_text == ""
    assert res.is_modified is False


def test_dataframe_normalization():
    df = pd.DataFrame({
        "review_id": [1],
        "review_text": ["Phone bohot accha hai"],
        "aspect_term": ["Phone"],
        "opinion_term": ["bohot accha"],
    })
    normalizer = TransliterationNormalizer()
    norm_df, stats = normalizer.normalize_dataframe(df)

    assert "normalized_review_text" in norm_df.columns
    assert "normalized_opinion_term" in norm_df.columns
    assert "bahut" in norm_df["normalized_review_text"].iloc[0]
    assert stats["changed_records"] == 1
