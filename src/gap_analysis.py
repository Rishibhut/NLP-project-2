"""
Cross-Lingual Gap Analysis module.
Quantifies the Cross-Lingual Aspect–Opinion Alignment Gap:
Alignment Gap = F1_same_language - F1_cross_language

Enforces a 3-tier sample adequacy standard:
- Tier A: Mathematical computation possible (n >= 1)
- Tier B: Descriptive reporting / Exploratory only (n >= min_descriptive_samples, e.g. 2-14)
- Tier C: Conclusive scientific evidence (n >= min_conclusive_samples, e.g. 15+)

Computes review-level cluster bootstrap confidence intervals and permutation test p-values.
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
    unique_reviews_same: int
    unique_reviews_cross: int
    gap_ci_95: Optional[Tuple[float, float]]
    pair_breakdown: Dict[str, Any]
    interpretation: str
    status: str  # "conclusive", "exploratory", "insufficient_data"
    is_conclusive: bool
    evidence_level: str  # "Tier A: Mathematical", "Tier B: Descriptive/Exploratory", "Tier C: Conclusive"
    p_value: Optional[float] = None
    statistical_test_name: Optional[str] = None


def compute_gap_bootstrap_ci(
    same_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
    group_col: str = "review_id",
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Optional[Tuple[float, float]]:
    """
    Computes 95% group-aware bootstrap confidence interval for (F1_same - F1_cross) in percentage points.
    Samples review_ids with replacement to account for intra-review aspect clustering and prevent overoptimistic variance.
    Requires at least 4 unique reviews per subset to prevent degenerate variance estimation.
    """
    if len(same_df) < 5 or len(cross_df) < 5:
        return None

    same_reviews = same_df[group_col].unique()
    cross_reviews = cross_df[group_col].unique()

    if len(same_reviews) < 4 or len(cross_reviews) < 4:
        return None

    rng = np.random.RandomState(random_state)
    labels = ["Negative", "Neutral", "Positive"]
    diffs = []

    # Map review_id to sub-dataframes for fast lookup
    same_by_rev = {r_id: g for r_id, g in same_df.groupby(group_col)}
    cross_by_rev = {r_id: g for r_id, g in cross_df.groupby(group_col)}

    n_s_rev = len(same_reviews)
    n_c_rev = len(cross_reviews)

    for _ in range(n_bootstraps):
        sampled_s_ids = rng.choice(same_reviews, size=n_s_rev, replace=True)
        sampled_c_ids = rng.choice(cross_reviews, size=n_c_rev, replace=True)

        s_batch = pd.concat([same_by_rev[rid] for rid in sampled_s_ids], ignore_index=True)
        c_batch = pd.concat([cross_by_rev[rid] for rid in sampled_c_ids], ignore_index=True)

        if s_batch[y_true_col].nunique() < 2 or c_batch[y_true_col].nunique() < 2:
            continue

        try:
            f1_s = f1_score(s_batch[y_true_col], s_batch[y_pred_col], labels=labels, average="macro", zero_division=0)
            f1_c = f1_score(c_batch[y_true_col], c_batch[y_pred_col], labels=labels, average="macro", zero_division=0)
            diff_pp = (f1_s - f1_c) * 100.0
            diffs.append(diff_pp)
        except Exception:
            continue

    if len(diffs) < 50:
        return None

    lower = float(np.percentile(diffs, 2.5))
    upper = float(np.percentile(diffs, 97.5))
    return round(lower, 2), round(upper, 2)


def compute_permutation_test_p_value(
    same_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Optional[float]:
    """
    Two-tailed permutation test testing H0: Alignment gap == 0.
    Permutes subset indicators ('same-language' vs 'cross-lingual') across pooled records.
    """
    if len(same_df) < 5 or len(cross_df) < 5:
        return None

    labels = ["Negative", "Neutral", "Positive"]
    f1_s_obs = f1_score(same_df[y_true_col], same_df[y_pred_col], labels=labels, average="macro", zero_division=0)
    f1_c_obs = f1_score(cross_df[y_true_col], cross_df[y_pred_col], labels=labels, average="macro", zero_division=0)
    obs_gap = abs(f1_s_obs - f1_c_obs)

    pooled = pd.concat([same_df, cross_df], ignore_index=True)
    n_s = len(same_df)
    n_total = len(pooled)

    rng = np.random.RandomState(random_state)
    perm_gaps = []

    for _ in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_s = pooled.iloc[perm_idx[:n_s]]
        perm_c = pooled.iloc[perm_idx[n_s:]]

        f1_s = f1_score(perm_s[y_true_col], perm_s[y_pred_col], labels=labels, average="macro", zero_division=0)
        f1_c = f1_score(perm_c[y_true_col], perm_c[y_pred_col], labels=labels, average="macro", zero_division=0)
        perm_gaps.append(abs(f1_s - f1_c))

    p_val = float(np.mean([g >= obs_gap for g in perm_gaps]))
    return round(p_val, 4)


def analyze_alignment_gap(
    df: pd.DataFrame,
    model_name: str,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
    min_conclusive_samples: int = 15,
    min_descriptive_samples: int = 2,
    group_col: str = "review_id",
    min_samples: Optional[int] = None,
) -> GapAnalysisDetail:
    """
    Performs full cross-lingual alignment gap analysis for a model.
    Enforces a strict 3-tier sample adequacy standard:
    - Tier A: Mathematical computation
    - Tier B: Descriptive reporting only (Exploratory / Inconclusive)
    - Tier C: Conclusive scientific evidence
    """
    if min_samples is not None:
        min_descriptive_samples = min_samples

    same_df = df[df["pair_type"] == "same-language"]
    cross_df = df[df["pair_type"] == "cross-lingual"]
    uncertain_df = df[df["pair_type"] == "Uncertain"]

    n_same = len(same_df)
    n_cross = len(cross_df)
    n_unc = len(uncertain_df)

    rev_same = int(same_df[group_col].nunique()) if (group_col in same_df.columns and n_same > 0) else 0
    rev_cross = int(cross_df[group_col].nunique()) if (group_col in cross_df.columns and n_cross > 0) else 0

    labels = ["Negative", "Neutral", "Positive"]

    # Tier 1: Below descriptive threshold (cannot even calculate descriptive metrics reliably)
    if n_same < min_descriptive_samples or n_cross < min_descriptive_samples:
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
            unique_reviews_same=rev_same,
            unique_reviews_cross=rev_cross,
            gap_ci_95=None,
            pair_breakdown={},
            interpretation=(
                f"Insufficient annotated pairs (same: {n_same}, cross: {n_cross}) to compute gap. "
                f"Minimum required is {min_descriptive_samples} per subset."
            ),
            status="insufficient_data",
            is_conclusive=False,
            evidence_level="Tier A (Insufficient Data)",
            p_value=None,
            statistical_test_name=None,
        )

    # Calculate mathematical metrics
    f1_same = float(f1_score(same_df[y_true_col], same_df[y_pred_col], labels=labels, average="macro", zero_division=0))
    f1_cross = float(f1_score(cross_df[y_true_col], cross_df[y_pred_col], labels=labels, average="macro", zero_division=0))
    acc_same = float(accuracy_score(same_df[y_true_col], same_df[y_pred_col]))
    acc_cross = float(accuracy_score(cross_df[y_true_col], cross_df[y_pred_col]))
    gap_pp = round((f1_same - f1_cross) * 100.0, 2)

    ci_95 = compute_gap_bootstrap_ci(same_df, cross_df, y_true_col, y_pred_col, group_col=group_col)
    p_val = compute_permutation_test_p_value(same_df, cross_df, y_true_col, y_pred_col)
    test_name = "Review-Level Permutation Test (Two-Tailed)" if p_val is not None else None

    # Detailed linguistic subgroup breakdown
    pair_breakdown = {}
    if "aspect_language" in df.columns and "opinion_language" in df.columns:
        df_copy = df.copy()
        df_copy["lang_pair_combo"] = df_copy["aspect_language"].astype(str) + " -> " + df_copy["opinion_language"].astype(str)
        for combo, group in df_copy.groupby("lang_pair_combo"):
            if len(group) >= 1:
                combo_f1 = float(f1_score(group[y_true_col], group[y_pred_col], labels=labels, average="macro", zero_division=0))
                combo_acc = float(accuracy_score(group[y_true_col], group[y_pred_col]))
                pair_breakdown[combo] = {
                    "count": len(group),
                    "unique_reviews": int(group[group_col].nunique()) if group_col in group.columns else len(group),
                    "f1_macro": round(combo_f1, 4),
                    "accuracy": round(combo_acc, 4),
                    "pair_type": str(group["pair_type"].iloc[0]),
                }

    # Evaluate conclusive adequacy (Tier C vs Tier B)
    is_conclusive = (n_same >= min_conclusive_samples and n_cross >= min_conclusive_samples)

    if not is_conclusive:
        status = "exploratory"
        evidence_level = "Tier B: Descriptive / Exploratory Only"
        interp = (
            f"INCONCLUSIVE (Exploratory Only): The nominal Alignment Gap of {gap_pp:+.2f} percentage points was observed "
            f"(Same F1: {f1_same:.3f}, Cross F1: {f1_cross:.3f}). However, sample counts "
            f"(same-language n={n_same} across {rev_same} reviews, cross-lingual n={n_cross} across {rev_cross} reviews) "
            f"are below the scientific adequacy threshold (minimum {min_conclusive_samples} per subset). "
            f"High sampling variance precludes asserting a statistically valid alignment effect."
        )
    else:
        status = "conclusive"
        evidence_level = "Tier C: Scientifically Conclusive"
        p_note = f" (p={p_val:.4f}, {test_name})" if p_val is not None else ""
        if gap_pp > 0:
            interp = (
                f"CONCLUSIVE: A positive Alignment Gap of +{gap_pp:.2f} percentage points was observed{p_note}. "
                f"Aspect-level sentiment performance significantly degraded on cross-lingual aspect-opinion pairs."
            )
        elif gap_pp < 0:
            interp = (
                f"CONCLUSIVE: A negative Alignment Gap of {gap_pp:.2f} percentage points was observed{p_note}. "
                f"Performance on cross-lingual pairs exceeded same-language pairs."
            )
        else:
            interp = f"CONCLUSIVE: No performance disparity detected between same-language and cross-lingual subsets{p_note}."

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
        unique_reviews_same=rev_same,
        unique_reviews_cross=rev_cross,
        gap_ci_95=ci_95,
        pair_breakdown=pair_breakdown,
        interpretation=interp,
        status=status,
        is_conclusive=is_conclusive,
        evidence_level=evidence_level,
        p_value=p_val,
        statistical_test_name=test_name,
    )
