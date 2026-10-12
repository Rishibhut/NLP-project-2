"""
Unit tests for evaluation metrics, span F1, link F1, multiclass metrics, and bootstrap CI.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation import compute_sentiment_metrics, bootstrap_metric_ci
from src.aspect_extractor import evaluate_span_extraction
from src.relation_extractor import evaluate_relation_extraction


def test_sentiment_metrics_perfect():
    y_true = ["Positive", "Negative", "Neutral"]
    y_pred = ["Positive", "Negative", "Neutral"]

    metrics = compute_sentiment_metrics(y_true, y_pred, compute_ci=False)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.sample_count == 3


def test_sentiment_metrics_imperfect():
    y_true = ["Positive", "Positive", "Negative", "Neutral"]
    y_pred = ["Positive", "Negative", "Negative", "Neutral"]

    metrics = compute_sentiment_metrics(y_true, y_pred, compute_ci=False)
    assert metrics.accuracy == 0.75
    assert metrics.macro_f1 > 0.0


def test_bootstrap_ci_sample_size_guard():
    # Less than 10 samples must return None without crashing
    y_true = ["Positive", "Negative", "Neutral"]
    y_pred = ["Positive", "Negative", "Positive"]

    ci = bootstrap_metric_ci(y_true, y_pred, lambda yt, yp: 1.0)
    assert ci is None


def test_span_extraction_evaluation():
    pred_spans = [["camera quality"], ["battery life", "display"]]
    gold_spans = [["camera quality"], ["battery life", "touch response"]]

    p, r, f1 = evaluate_span_extraction(pred_spans, gold_spans, match_mode="exact")
    assert p == pytest.approx(2 / 3, abs=1e-3)
    assert r == pytest.approx(2 / 3, abs=1e-3)
    assert f1 == pytest.approx(2 / 3, abs=1e-3)


def test_relation_extraction_evaluation():
    pred_pairs = [[("camera", "good")], [("display", "bright")]]
    gold_pairs = [[("camera", "good")], [("display", "bright")]]

    p, r, f1 = evaluate_relation_extraction(pred_pairs, gold_pairs)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0
