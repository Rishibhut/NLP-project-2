"""
Cross-Lingual Gap Analysis module.
Quantifies the Cross-Lingual Aspect–Opinion Alignment Gap:
Alignment Gap = F1_same_language - F1_cross_language
Computes subset breakdowns, bootstrap confidence intervals on the gap, and exports results.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


@dataclass
class GapAnalysisDetail:
    model_name: str
    f1_same_lang: Optional[float]
    f1_cross_lang: Optional[float]
    alignment_gap_pp: Optional[float]  # Percentage points
    accuracy_same_lang: Optional[float]
    accuracy_cross_lang: Optional[float]
    n_same: int
    n_cross: int
    n_uncertain: int
    gap_ci_95: Optional[Tuple[float, float]]
    pair_breakdown: Dict[str, Any]
    interpretation: str


def compute_gap_bootstrap_ci(
    same_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Optional[Tuple[float, float]]:
    """
    Computes 95% bootstrap confidence interval for (F1_same - F1_cross) in percentage points.
    """
    if len(same_df) < 10 or len(cross_df) < 10:
        return None

    rng = np.random.RandomState(random_state)
    labels = ["Negative", "Neutral", "Positive"]
    diffs = []

    y_true_same = same_df[y_true_col].values
    y_pred_same = same_df[y_pred_col].values
    y_true_cross = cross_df[y_true_col].values
    y_pred_cross = cross_df[y_pred_col].values

    n_s = len(same_df)
    n_c = len(cross_df)

    for _ in range(n_bootstraps):
        idx_s = rng.randint(0, n_s, size=n_s)
        idx_c = rng.randint(0, n_c, size=n_c)

        s_true_sample = y_true_same[idx_s]
        s_pred_sample = y_pred_same[idx_s]
        c_true_sample = y_true_cross[idx_c]
        c_pred_sample = y_pred_cross[idx_c]

        if len(np.unique(s_true_sample)) < 2 or len(np.unique(c_true_sample)) < 2:
            continue

        try:
            f1_s = f1_score(s_true_sample, s_pred_sample, labels=labels, average="macro", zero_division=0)
            f1_c = f1_score(c_true_sample, c_pred_sample, labels=labels, average="macro", zero_division=0)
            diff_pp = (f1_s - f1_c) * 100.0
            diffs.append(diff_pp)
        except Exception:
            continue

    if len(diffs) < 100:
        return None

    lower = float(np.percentile(diffs, 2.5))
    upper = float(np.percentile(diffs, 97.5))
    return round(lower, 2), round(upper, 2)


def analyze_alignment_gap(
    df: pd.DataFrame,
    model_name: str,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
    min_samples: int = 5,
) -> GapAnalysisDetail:
    """
    Performs full cross-lingual alignment gap analysis for a model.
    """
    same_df = df[df["pair_type"] == "same-language"]
    cross_df = df[df["pair_type"] == "cross-lingual"]
    uncertain_df = df[df["pair_type"] == "Uncertain"]

    n_same = len(same_df)
    n_cross = len(cross_df)
    n_unc = len(uncertain_df)

    labels = ["Negative", "Neutral", "Positive"]

    if n_same < min_samples or n_cross < min_samples:
        return GapAnalysisDetail(
            model_name=model_name,
            f1_same_lang=None,
            f1_cross_lang=None,
            alignment_gap_pp=None,
            accuracy_same_lang=None,
            accuracy_cross_lang=None,
            n_same=n_same,
            n_cross=n_cross,
            n_uncertain=n_unc,
            gap_ci_95=None,
            pair_breakdown={},
            interpretation=f"Insufficient annotated pairs (same: {n_same}, cross: {n_cross}) to compute gap.",
        )

    f1_same = float(f1_score(same_df[y_true_col], same_df[y_pred_col], labels=labels, average="macro", zero_division=0))
    f1_cross = float(f1_score(cross_df[y_true_col], cross_df[y_pred_col], labels=labels, average="macro", zero_division=0))
    acc_same = float(accuracy_score(same_df[y_true_col], same_df[y_pred_col]))
    acc_cross = float(accuracy_score(cross_df[y_true_col], cross_df[y_pred_col]))

    gap_pp = round((f1_same - f1_cross) * 100.0, 2)
    ci_95 = compute_gap_bootstrap_ci(same_df, cross_df, y_true_col, y_pred_col)

    # Detailed linguistic subgroup breakdown
    df["lang_pair_combo"] = df["aspect_language"].astype(str) + " -> " + df["opinion_language"].astype(str)
    pair_breakdown = {}
    for combo, group in df.groupby("lang_pair_combo"):
        if len(group) >= 2:
            combo_f1 = float(f1_score(group[y_true_col], group[y_pred_col], labels=labels, average="macro", zero_division=0))
            combo_acc = float(accuracy_score(group[y_true_col], group[y_pred_col]))
            pair_breakdown[combo] = {
                "count": len(group),
                "f1_macro": round(combo_f1, 4),
                "accuracy": round(combo_acc, 4),
                "pair_type": group["pair_type"].iloc[0],
            }

    if gap_pp > 0:
        interp = (
            f"The model exhibits an Alignment Gap of +{gap_pp:.2f} percentage points. "
            f"Performance drops when the opinion expression is cross-lingual relative to the aspect term."
        )
    elif gap_pp < 0:
        interp = (
            f"The model scored {abs(gap_pp):.2f} percentage points higher on cross-lingual pairs than same-language pairs."
        )
    else:
        interp = "No performance disparity detected between same-language and cross-lingual pairs."

    return GapAnalysisDetail(
        model_name=model_name,
        f1_same_lang=round(f1_same, 4),
        f1_cross_lang=round(f1_cross, 4),
        alignment_gap_pp=gap_pp,
        accuracy_same_lang=round(acc_same, 4),
        accuracy_cross_lang=round(acc_cross, 4),
        n_same=n_same,
        n_cross=n_cross,
        n_uncertain=n_unc,
        gap_ci_95=ci_95,
        pair_breakdown=pair_breakdown,
        interpretation=interp,
    )
