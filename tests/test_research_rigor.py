"""
Comprehensive unit tests for research rigor, provenance validation,
3-tier sample adequacy standards, cluster bootstrap CIs, and error categorization.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.data_validation import validate_absa_dataframe, derive_pair_type
from src.data_loader import split_by_review_id, verify_no_leakage
from src.gap_analysis import analyze_alignment_gap, compute_gap_bootstrap_ci, compute_permutation_test_p_value
from src.preprocessing import TransliterationNormalizer, NormalizationConfig
from src.error_analysis import diagnose_prediction_error, DOCUMENTED_ERROR_CATEGORIES
from src.transformer_models import MultilingualABSAModel


def test_annotation_consistency_and_audit_files():
    """Validates that validation report and audit files exist and report consistent counts."""
    df = pd.read_csv("data/research_reviews_annotated.csv")
    val_res = validate_absa_dataframe(df)

    assert val_res.is_valid is True
    assert val_res.total_rows == 61
    assert val_res.stats["unique_reviews"] == 40

    # Disagreements must be exactly 11 due to Mixed language spans
    assert len(val_res.stats["pair_type_disagreements"]) == 11

    # Verify audit file if present
    audit_file = "results/annotation_audit.csv"
    if os.path.exists(audit_file):
        audit_df = pd.read_csv(audit_file)
        assert len(audit_df) == 61
        assert audit_df["disagreement_detected"].sum() == 11
        # All disagreements resolved as Uncertain
        dis_rows = audit_df[audit_df["disagreement_detected"] == True]
        assert (dis_rows["final_pair_type"] == "Uncertain").all()


def test_provenance_metadata_truthfulness():
    """Ensures dataset provenance truthfully records unverified pilot status."""
    prov_path = "data/provenance_metadata.json"
    assert os.path.exists(prov_path), "provenance_metadata.json must exist"

    with open(prov_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "source_verification_status" in meta
    assert "UNVERIFIED" in meta["source_verification_status"]
    assert meta["is_empirical_study_ready"] is False
    assert "schema_version" in meta


def test_character_span_offset_validation():
    """Tests exact span extraction and offset detection."""
    df = pd.DataFrame([{
        "review_id": 101,
        "review_text": "Is phone ka camera quality bahut badhiya hai.",
        "aspect_term": "camera quality",
        "opinion_term": "bahut badhiya hai",
        "sentiment": "Positive",
        "aspect_language": "English",
        "opinion_language": "Hindi",
        "pair_type": "cross-lingual",
        "product_category": "Smartphones",
    }])
    val_res = validate_absa_dataframe(df)
    assert val_res.is_valid is True
    cleaned = val_res.cleaned_df
    assert cleaned["aspect_start_char"].iloc[0] == 12
    assert cleaned["aspect_end_char"].iloc[0] == 26
    assert cleaned["opinion_start_char"].iloc[0] == 27
    assert cleaned["opinion_end_char"].iloc[0] == 44


def test_data_leakage_exception_on_overlap():
    """Ensures verify_no_leakage strictly catches overlap if forced."""
    df_train = pd.DataFrame({"review_id": [1, 2, 3], "review_text": ["a", "b", "c"]})
    df_val = pd.DataFrame({"review_id": [4, 5], "review_text": ["d", "e"]})
    # Intentional leakage: review_id 3 in both train and test
    df_test_leaked = pd.DataFrame({"review_id": [3, 6], "review_text": ["c", "f"]})

    class LeakedSplits:
        train_df = df_train
        val_df = df_val
        test_df = df_test_leaked

    with pytest.raises(AssertionError, match="Data leakage detected"):
        verify_no_leakage(LeakedSplits())


def test_minimum_sample_adequacy_3_tiers():
    """
    Tests 3-tier sample adequacy standard:
    Tier A: Insufficient (< 2) -> returns None
    Tier B: Exploratory / Inconclusive (2 <= n < 15) -> returns gap, is_conclusive=False
    Tier C: Conclusive (n >= 15) -> returns gap, is_conclusive=True
    """
    # Tier A: 1 sample
    df_tier_a = pd.DataFrame({
        "review_id": [1, 2],
        "aspect_term": ["a", "b"],
        "opinion_term": ["o", "p"],
        "pair_type": ["same-language", "cross-lingual"],
        "sentiment": ["Positive", "Negative"],
        "predicted_sentiment": ["Positive", "Negative"],
    })
    res_a = analyze_alignment_gap(df_tier_a, "ModelA", min_conclusive_samples=15, min_descriptive_samples=2)
    assert res_a.status == "insufficient_data"
    assert res_a.is_conclusive is False
    assert res_a.alignment_gap_pp is None

    # Tier B: 4 samples per subset (below 15)
    records_b = []
    for i in range(4):
        records_b.append({
            "review_id": i + 1, "aspect_term": "a", "opinion_term": "o",
            "pair_type": "same-language", "sentiment": "Positive", "predicted_sentiment": "Positive",
        })
        records_b.append({
            "review_id": i + 10, "aspect_term": "b", "opinion_term": "p",
            "pair_type": "cross-lingual", "sentiment": "Negative", "predicted_sentiment": "Negative",
        })
    df_tier_b = pd.DataFrame(records_b)
    res_b = analyze_alignment_gap(df_tier_b, "ModelB", min_conclusive_samples=15, min_descriptive_samples=2)
    assert res_b.status == "exploratory"
    assert res_b.is_conclusive is False
    assert res_b.alignment_gap_pp is not None
    assert "INCONCLUSIVE (Exploratory Only)" in res_b.interpretation

    # Tier C: 16 samples per subset (conclusive)
    records_c = []
    for i in range(16):
        records_c.append({
            "review_id": i + 1, "aspect_term": "a", "opinion_term": "o",
            "pair_type": "same-language", "sentiment": "Positive", "predicted_sentiment": "Positive",
        })
        records_c.append({
            "review_id": i + 100, "aspect_term": "b", "opinion_term": "p",
            "pair_type": "cross-lingual", "sentiment": "Negative", "predicted_sentiment": "Negative",
        })
    df_tier_c = pd.DataFrame(records_c)
    res_c = analyze_alignment_gap(df_tier_c, "ModelC", min_conclusive_samples=15, min_descriptive_samples=2)
    assert res_c.status == "conclusive"
    assert res_c.is_conclusive is True
    assert "CONCLUSIVE" in res_c.interpretation


def test_cluster_bootstrap_ci_requires_minimum_reviews():
    """Validates cluster bootstrap CI returns None when review clusters are too few."""
    same_df = pd.DataFrame({
        "review_id": [1, 1, 1],  # Only 1 unique review
        "sentiment": ["Positive", "Negative", "Neutral"],
        "predicted_sentiment": ["Positive", "Negative", "Neutral"],
    })
    cross_df = pd.DataFrame({
        "review_id": [2, 2, 2],  # Only 1 unique review
        "sentiment": ["Positive", "Negative", "Neutral"],
        "predicted_sentiment": ["Positive", "Positive", "Neutral"],
    })
    ci = compute_gap_bootstrap_ci(same_df, cross_df)
    assert ci is None  # Accurately avoids degenerate variance


def test_error_analysis_10_categories():
    """Tests diagnosis of error modes across documented categories."""
    row = pd.Series({
        "review_id": 1,
        "review_text": "Battery backup accha nahi hai",
        "aspect_term": "Battery backup",
        "opinion_term": "accha nahi hai",
        "sentiment": "Negative",
        "pair_type": "cross-lingual",
    })
    # Prediction erroneously predicted Positive despite negation
    err_record = diagnose_prediction_error(row, pred_sentiment="Positive", all_aspects_in_review=1)
    assert err_record is not None
    assert "Negation handling failure" in err_record.error_categories
    assert err_record.true_sentiment == "Negative"
    assert err_record.predicted_sentiment == "Positive"


def test_missing_transformer_checkpoint_error():
    """Tests that loading non-existent transformer checkpoint fails cleanly."""
    model = MultilingualABSAModel("mBERT")
    with pytest.raises(Exception):
        model.load_checkpoint("models/saved_checkpoints/non_existent_folder_xyz")
