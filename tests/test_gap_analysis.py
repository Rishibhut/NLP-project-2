"""
Unit tests for the Cross-Lingual Aspect–Opinion Alignment Gap calculation.
"""

import pytest
import pandas as pd
from src.gap_analysis import analyze_alignment_gap
from src.evaluation import compute_cross_lingual_gap


def test_alignment_gap_calculation_positive():
    # Same-language F1 is 1.0, cross-lingual is 0.5 -> positive gap
    data = {
        "review_id": list(range(12)),
        "aspect_term": ["camera"] * 12,
        "opinion_term": ["good"] * 12,
        "aspect_language": ["English"] * 6 + ["English"] * 6,
        "opinion_language": ["English"] * 6 + ["Hindi"] * 6,
        "pair_type": ["same-language"] * 6 + ["cross-lingual"] * 6,
        "sentiment": ["Positive", "Negative", "Positive", "Negative", "Positive", "Negative"] * 2,
        # Same language: all correct
        # Cross lingual: some incorrect
        "predicted_sentiment": [
            "Positive", "Negative", "Positive", "Negative", "Positive", "Negative",
            "Positive", "Positive", "Positive", "Positive", "Positive", "Positive",
        ],
    }
    df = pd.DataFrame(data)
    detail = analyze_alignment_gap(df, model_name="TestModel", min_samples=3)

    assert detail.f1_same_lang is not None
    assert detail.f1_cross_lang is not None
    assert detail.alignment_gap_pp > 0  # Demonstrates positive gap
    assert "Alignment Gap of +" in detail.interpretation


def test_alignment_gap_insufficient_samples():
    # Only 2 same-language samples (below min_samples=5)
    data = {
        "review_id": [1, 2, 3],
        "aspect_term": ["camera", "display", "battery"],
        "opinion_term": ["good", "kharab", "average"],
        "aspect_language": ["English", "English", "English"],
        "opinion_language": ["English", "Hindi", "English"],
        "pair_type": ["same-language", "cross-lingual", "same-language"],
        "sentiment": ["Positive", "Negative", "Neutral"],
        "predicted_sentiment": ["Positive", "Negative", "Neutral"],
    }
    df = pd.DataFrame(data)
    detail = analyze_alignment_gap(df, model_name="TestModel", min_samples=5)

    assert detail.alignment_gap_pp is None
    assert "Insufficient annotated pairs" in detail.interpretation
