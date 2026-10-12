"""
Unit tests for data validation, schema enforcement, and span containment checks.
"""

import pytest
import pandas as pd
from src.data_validation import (
    validate_absa_dataframe,
    derive_pair_type,
    REQUIRED_COLUMNS,
)


def test_valid_dataframe():
    data = {
        "review_id": [1, 2],
        "review_text": [
            "Camera quality achhi hai, but battery backup bahut kharab hai.",
            "Display is clear and bright.",
        ],
        "aspect_term": ["Camera quality", "Display"],
        "opinion_term": ["achhi hai", "clear and bright"],
        "sentiment": ["Positive", "Positive"],
        "aspect_language": ["English", "English"],
        "opinion_language": ["Hindi", "English"],
        "pair_type": ["cross-lingual", "same-language"],
        "product_category": ["Smartphones", "Smartphones"],
    }
    df = pd.DataFrame(data)
    result = validate_absa_dataframe(df)

    assert result.is_valid is True
    assert result.total_rows == 2
    assert result.valid_rows == 2
    assert len(result.errors) == 0


def test_missing_required_columns():
    df = pd.DataFrame({"review_id": [1], "review_text": ["Good product"]})
    result = validate_absa_dataframe(df)

    assert result.is_valid is False
    assert any("Missing required columns" in err for err in result.errors)


def test_empty_dataframe():
    df = pd.DataFrame()
    result = validate_absa_dataframe(df)

    assert result.is_valid is False
    assert result.total_rows == 0


def test_invalid_sentiment_label():
    data = {
        "review_id": [1],
        "review_text": ["Nice display"],
        "aspect_term": ["display"],
        "opinion_term": ["nice"],
        "sentiment": ["SuperAwesome"],  # Invalid label
        "aspect_language": ["English"],
        "opinion_language": ["English"],
        "pair_type": ["same-language"],
        "product_category": ["Electronics"],
    }
    df = pd.DataFrame(data)
    result = validate_absa_dataframe(df)

    assert result.is_valid is False
    assert any("invalid sentiment labels" in err.lower() for err in result.errors)


def test_invalid_language_label():
    data = {
        "review_id": [1],
        "review_text": ["Nice display"],
        "aspect_term": ["display"],
        "opinion_term": ["nice"],
        "sentiment": ["Positive"],
        "aspect_language": ["Klingon"],  # Invalid
        "opinion_language": ["English"],
        "pair_type": ["same-language"],
        "product_category": ["Electronics"],
    }
    df = pd.DataFrame(data)
    result = validate_absa_dataframe(df)

    assert result.is_valid is False
    assert any("invalid aspect_language labels" in err.lower() for err in result.errors)


def test_span_containment_warning():
    data = {
        "review_id": [1],
        "review_text": ["Camera is great"],
        "aspect_term": ["Battery"],  # Not in review text
        "opinion_term": ["great"],
        "sentiment": ["Positive"],
        "aspect_language": ["English"],
        "opinion_language": ["English"],
        "pair_type": ["same-language"],
        "product_category": ["Electronics"],
    }
    df = pd.DataFrame(data)
    result = validate_absa_dataframe(df, strict=False)

    assert any("span alignment mismatches" in w.lower() for w in result.warnings)


def test_derive_pair_type():
    assert derive_pair_type("English", "English") == "same-language"
    assert derive_pair_type("Hindi", "Hindi") == "same-language"
    assert derive_pair_type("English", "Hindi") == "cross-lingual"
    assert derive_pair_type("Hindi", "English") == "cross-lingual"
    assert derive_pair_type("Uncertain", "Hindi") == "Uncertain"
    assert derive_pair_type("Mixed", "English") == "Uncertain"


def test_multiple_aspects_per_review():
    data = {
        "review_id": [1, 1],
        "review_text": [
            "Sound is good but mic is bad",
            "Sound is good but mic is bad",
        ],
        "aspect_term": ["Sound", "mic"],
        "opinion_term": ["good", "bad"],
        "sentiment": ["Positive", "Negative"],
        "aspect_language": ["English", "English"],
        "opinion_language": ["English", "English"],
        "pair_type": ["same-language", "same-language"],
        "product_category": ["Audio", "Audio"],
    }
    df = pd.DataFrame(data)
    result = validate_absa_dataframe(df)

    assert result.is_valid is True
    assert result.stats["unique_reviews"] == 1
    assert result.valid_rows == 2
