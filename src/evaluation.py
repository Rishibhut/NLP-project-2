"""
Evaluation module for Aspect-Based Sentiment Analysis and Cross-Lingual Benchmarks.
Computes span F1, link F1, sentiment metrics, paired bootstrap confidence intervals,
and confusion matrices with strict research integrity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score


@dataclass
class SentimentMetrics:
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class: Dict[str, Dict[str, float]]
    confusion_mat: List[List[int]]
    labels: List[str]
    sample_count: int
    bootstrap_ci_f1: Optional[Tuple[float, float]] = None


@dataclass
class CrossLingualGapMetrics:
    same_lang_f1: Optional[float]
    cross_lang_f1: Optional[float]
    alignment_gap_percentage_points: Optional[float]
    same_lang_count: int
    cross_lang_count: int
    uncertain_count: int
    is_valid_comparison: bool
    status_note: str


@dataclass
class PipelineEvaluationReport:
    model_name: str
    overall_sentiment: SentimentMetrics
    cross_lingual_gap: CrossLingualGapMetrics
    aspect_span_f1: Optional[float] = None
    opinion_span_f1: Optional[float] = None
    relation_link_f1: Optional[float] = None
    eval_timestamp: str = ""
    notes: List[str] = field(default_factory=list)


def bootstrap_metric_ci(
    y_true: List[str],
    y_pred: List[str],
    metric_fn: Callable[[List[str], List[str]], float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Optional[Tuple[float, float]]:
    """
    Computes empirical bootstrap confidence interval (1 - alpha)% for any metric.
    Returns None if sample size is too small (< 10) to avoid spurious statistical claims.
    """
    n = len(y_true)
    if n < 10:
        return None

    rng = np.random.RandomState(random_state)
    boot_scores = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_true = y_true_arr[idx]
        sample_pred = y_pred_arr[idx]
        # Skip if single class present in sample
        if len(np.unique(sample_true)) < 2:
            continue
        try:
            score = metric_fn(sample_true, sample_pred)
            boot_scores.append(score)
        except Exception:
            continue

    if len(boot_scores) < 100:
        return None

    lower = float(np.percentile(boot_scores, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


def compute_sentiment_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    compute_ci: bool = True,
) -> SentimentMetrics:
    """
    Computes multiclass sentiment classification metrics.
    """
    if labels is None:
        labels = ["Negative", "Neutral", "Positive"]

    sample_count = len(y_true)
    if sample_count == 0:
        return SentimentMetrics(
            accuracy=0.0,
            macro_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            per_class={},
            confusion_mat=[],
            labels=labels,
            sample_count=0,
            bootstrap_ci_f1=None,
        )

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    p_per, r_per, f_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {}
    for lbl, pr, rec, f, s in zip(labels, p_per, r_per, f_per, sup_per):
        per_class[lbl] = {
            "precision": round(float(pr), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    ci = None
    if compute_ci:
        ci = bootstrap_metric_ci(
            y_true,
            y_pred,
            lambda yt, yp: f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
        )

    return SentimentMetrics(
        accuracy=round(acc, 4),
        macro_f1=round(float(f1), 4),
        macro_precision=round(float(p), 4),
        macro_recall=round(float(r), 4),
        per_class=per_class,
        confusion_mat=cm,
        labels=labels,
        sample_count=sample_count,
        bootstrap_ci_f1=ci,
    )


def compute_cross_lingual_gap(
    df: pd.DataFrame,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
    min_subset_size: int = 5,
) -> CrossLingualGapMetrics:
    """
    Computes Alignment Gap = F1_same_language - F1_cross_language.
    Validates sample sizes before drawing conclusions.
    """
    same_df = df[df["pair_type"] == "same-language"]
    cross_df = df[df["pair_type"] == "cross-lingual"]
    uncertain_df = df[df["pair_type"] == "Uncertain"]

    n_same = len(same_df)
    n_cross = len(cross_df)
    n_unc = len(uncertain_df)

    if n_same < min_subset_size or n_cross < min_subset_size:
        # Insufficient sample size for rigorous comparison
        return CrossLingualGapMetrics(
            same_lang_f1=None,
            cross_lang_f1=None,
            alignment_gap_percentage_points=None,
            same_lang_count=n_same,
            cross_lang_count=n_cross,
            uncertain_count=n_unc,
            is_valid_comparison=False,
            status_note=(
                f"Insufficient sample size: same-language has {n_same} samples, "
                f"cross-lingual has {n_cross} samples. Minimum required is {min_subset_size}."
            ),
        )

    labels = ["Negative", "Neutral", "Positive"]
    same_f1 = float(f1_score(same_df[y_true_col], same_df[y_pred_col], labels=labels, average="macro", zero_division=0))
    cross_f1 = float(f1_score(cross_df[y_true_col], cross_df[y_pred_col], labels=labels, average="macro", zero_division=0))

    # Gap in percentage points: (F1_same - F1_cross) * 100
    gap = round((same_f1 - cross_f1) * 100.0, 2)

    return CrossLingualGapMetrics(
        same_lang_f1=round(same_f1, 4),
        cross_lang_f1=round(cross_f1, 4),
        alignment_gap_percentage_points=gap,
        same_lang_count=n_same,
        cross_lang_count=n_cross,
        uncertain_count=n_unc,
        is_valid_comparison=True,
        status_note="Rigorous comparison computed over validated subsets.",
    )
