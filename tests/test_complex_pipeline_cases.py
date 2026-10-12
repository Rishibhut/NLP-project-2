"""
Automated unit tests for complex pipeline edge cases:
- Multi-aspect reviews with contrast words (lekin, but)
- Negation handling in opinion extraction
- Missing checkpoints and missing dataset graceful error handling
- Normalization toggle behavior
- Zero-division and empty subset evaluation
"""

import os
import pytest
import pandas as pd
from src.aspect_extractor import AspectExtractor
from src.opinion_extractor import OpinionExtractor
from src.relation_extractor import RelationExtractor
from src.preprocessing import TransliterationNormalizer, NormalizationConfig
from src.evaluation import compute_sentiment_metrics, compute_cross_lingual_gap
from src.classical_models import ClassicalABSAModel


def test_multi_aspect_contrast_binding():
    """
    Validates that in:
    'Camera accha hai, lekin battery bekaar hai.'
    Camera links to 'accha hai' (Positive),
    and Battery links to 'bekaar hai' (Negative).
    Cross-talk must NOT bind Camera to bekaar hai!
    """
    text = "Camera accha hai, lekin battery bekaar hai."
    asp_ext = AspectExtractor()
    op_ext = OpinionExtractor()
    rel_ext = RelationExtractor()

    asps = asp_ext.extract_aspects(text)
    ops = op_ext.extract_opinions(text)
    rels = rel_ext.extract_relations(text, asps, ops)

    assert len(rels) >= 2

    # Map aspect to linked opinion
    asp_to_op = {r.aspect_text.lower(): r.opinion_text.lower() for r in rels}

    assert "camera" in asp_to_op
    assert "battery" in asp_to_op

    assert "accha" in asp_to_op["camera"]
    assert "bekaar" in asp_to_op["battery"]


def test_negation_in_delivery_opinion():
    text = "Product achha hai lekin delivery time par nahi hui."
    op_ext = OpinionExtractor()
    opinions = op_ext.extract_opinions(text)
    op_texts = [o.text.lower() for o in opinions]

    assert any("time par nahi hui" in o for o in op_texts)


def test_normalization_toggle():
    text = "Phone bohot accha hai"
    # Condition A: Normalization disabled (identity)
    norm_disabled = TransliterationNormalizer(NormalizationConfig(
        reduce_repeated_chars=False,
        normalize_phonetic_spelling=False,
        normalize_whitespace=False,
    ))
    res_a = norm_disabled.normalize(text)
    assert res_a.normalized_text == text
    assert res_a.is_modified is False

    # Condition B: Normalization enabled
    norm_enabled = TransliterationNormalizer(NormalizationConfig(
        reduce_repeated_chars=True,
        normalize_phonetic_spelling=True,
        normalize_whitespace=True,
    ))
    res_b = norm_enabled.normalize(text)
    assert "bahut" in res_b.normalized_text
    assert res_b.is_modified is True


def test_missing_checkpoint_handling():
    # Attempting to load non-existent checkpoint raises FileNotFoundError or RuntimeError gracefully
    with pytest.raises((FileNotFoundError, Exception)):
        ClassicalABSAModel.load("models/saved_checkpoints/non_existent_checkpoint_xyz.joblib")


def test_empty_subsets_and_zero_division():
    # Dataset with 0 same-language samples
    df = pd.DataFrame({
        "review_id": [1, 2],
        "aspect_term": ["camera", "display"],
        "opinion_term": ["achha", "kharab"],
        "aspect_language": ["English", "English"],
        "opinion_language": ["Hindi", "Hindi"],
        "pair_type": ["cross-lingual", "cross-lingual"],
        "sentiment": ["Positive", "Negative"],
        "predicted_sentiment": ["Positive", "Negative"],
    })
    gap = compute_cross_lingual_gap(df, min_subset_size=5)
    assert gap.is_valid_comparison is False
    assert gap.same_lang_f1 is None
    assert "Insufficient sample size" in gap.status_note


def test_empty_predictions_metrics():
    # Zero sample evaluation does not crash
    metrics = compute_sentiment_metrics([], [], compute_ci=False)
    assert metrics.accuracy == 0.0
    assert metrics.macro_f1 == 0.0
    assert metrics.sample_count == 0
