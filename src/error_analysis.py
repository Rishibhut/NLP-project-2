"""
Error Analysis module for Hinglish ABSA and Cross-Lingual Evaluation.
Automatically classifies observed errors into the 10 documented research categories:
1. Aspect extraction failure
2. Opinion extraction failure
3. Aspect-opinion binding failure
4. Transliteration recognition failure
5. Sentiment classification failure
6. Negation handling failure
7. Language-identification failure
8. Annotation disagreement
9. Implicit aspect or opinion
10. Other / unresolved
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import os
import re
import pandas as pd

DOCUMENTED_ERROR_CATEGORIES = [
    "Aspect extraction failure",
    "Opinion extraction failure",
    "Aspect-opinion binding failure",
    "Transliteration recognition failure",
    "Sentiment classification failure",
    "Negation handling failure",
    "Language-identification failure",
    "Annotation disagreement",
    "Implicit aspect or opinion",
    "Other / unresolved",
]


@dataclass
class ErrorRecord:
    review_id: Any
    review_text: str
    aspect_term: str
    opinion_term: str
    true_sentiment: str
    predicted_sentiment: str
    true_pair_type: str
    predicted_pair_type: str
    error_categories: List[str]
    explanation: str
    model_name: str


def diagnose_prediction_error(
    row: pd.Series,
    pred_sentiment: str,
    pred_pair_type: Optional[str] = None,
    pred_aspect: Optional[str] = None,
    pred_opinion: Optional[str] = None,
    model_name: str = "Model",
    all_aspects_in_review: int = 1,
) -> Optional[ErrorRecord]:
    """
    Diagnoses model prediction errors according to the 10 documented categories.
    """
    gold_sent = str(row["sentiment"]).strip()
    pred_sent = str(pred_sentiment).strip()
    true_pair = str(row.get("pair_type", "Uncertain"))
    pred_pair = str(pred_pair_type or true_pair)

    # Check if there is an error in sentiment or extraction
    is_sent_err = (gold_sent != pred_sent)
    is_pair_err = (pred_pair != true_pair and pred_pair_type is not None)
    is_asp_err = (pred_aspect is not None and str(pred_aspect).lower() != str(row["aspect_term"]).lower())
    is_op_err = (pred_opinion is not None and str(pred_opinion).lower() != str(row["opinion_term"]).lower())

    if not (is_sent_err or is_pair_err or is_asp_err or is_op_err):
        return None

    review_text = str(row["review_text"])
    op_term = str(row["opinion_term"])
    asp_term = str(row["aspect_term"])

    assigned_categories = []
    explanations = []

    if is_asp_err:
        assigned_categories.append("Aspect extraction failure")
        explanations.append(f"Predicted aspect '{pred_aspect}' does not match gold span '{asp_term}'.")

    if is_op_err:
        assigned_categories.append("Opinion extraction failure")
        explanations.append(f"Predicted opinion '{pred_opinion}' does not match gold span '{op_term}'.")

    if is_pair_err:
        assigned_categories.append("Language-identification failure")
        explanations.append(f"Predicted pair type '{pred_pair}' disagrees with reference '{true_pair}'.")

    if is_sent_err:
        # Check negation handling
        neg_pattern = re.compile(r"\b(?:nahi|nhi|nahin|not|mat|never|no|without|kabhi\s+nahi)\b", re.IGNORECASE)
        has_neg = bool(neg_pattern.search(review_text) or neg_pattern.search(op_term))

        if has_neg and ((gold_sent == "Negative" and pred_sent == "Positive") or (gold_sent == "Positive" and pred_sent == "Negative")):
            assigned_categories.append("Negation handling failure")
            explanations.append("Failed to invert polarity in presence of negation operator.")
        elif all_aspects_in_review > 1:
            assigned_categories.append("Aspect-opinion binding failure")
            explanations.append(f"Multiple aspects ({all_aspects_in_review}) in review caused sentiment context bleed.")
        else:
            assigned_categories.append("Sentiment classification failure")
            explanations.append(f"Misclassified true '{gold_sent}' as '{pred_sent}'.")

        # Check transliteration variants
        translit_variants = ["bohot", "achha", "khraab", "bekar", "ghatya", "ekdm", "blkl", "accccha", "soooo"]
        if any(var in review_text.lower() for var in translit_variants):
            assigned_categories.append("Transliteration recognition failure")
            explanations.append("Review contains informal non-standard Romanized spelling variants.")

        # Check annotation ambiguity
        if row.get("pair_type_disagreement", False):
            assigned_categories.append("Annotation disagreement")
            explanations.append("Dataset record has an existing annotation/derived pair disagreement.")

    if not assigned_categories:
        assigned_categories.append("Other / unresolved")
        explanations.append("Uncategorized prediction discrepancy.")

    return ErrorRecord(
        review_id=row["review_id"],
        review_text=review_text,
        aspect_term=asp_term,
        opinion_term=op_term,
        true_sentiment=gold_sent,
        predicted_sentiment=pred_sent,
        true_pair_type=true_pair,
        predicted_pair_type=pred_pair,
        error_categories=assigned_categories,
        explanation=" | ".join(explanations),
        model_name=model_name,
    )


def run_error_analysis(
    df: pd.DataFrame,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
    pred_pair_col: Optional[str] = None,
    model_name: str = "Model",
    output_dir: Optional[str] = "results",
) -> Tuple[List[ErrorRecord], pd.DataFrame, Dict[str, Any]]:
    """
    Runs diagnostic analysis over model outputs, categorizing errors into 10 classes
    and saving results/error_analysis_<model_name>.csv.
    """
    aspect_counts = df["review_id"].value_counts().to_dict()
    error_records: List[ErrorRecord] = []
    category_counts = {cat: 0 for cat in DOCUMENTED_ERROR_CATEGORIES}

    for _, row in df.iterrows():
        n_aspects = aspect_counts.get(row["review_id"], 1)
        pred_sent = str(row[y_pred_col])
        pred_pair = str(row[pred_pair_col]) if (pred_pair_col and pred_pair_col in row) else None

        err = diagnose_prediction_error(
            row=row,
            pred_sentiment=pred_sent,
            pred_pair_type=pred_pair,
            model_name=model_name,
            all_aspects_in_review=n_aspects,
        )
        if err:
            error_records.append(err)
            for c in err.error_categories:
                category_counts[c] = category_counts.get(c, 0) + 1

    total_samples = len(df)
    total_errors = len(error_records)
    error_rate = round((total_errors / max(1, total_samples)) * 100, 2)

    # Subgroup breakdown
    same_df = df[df["pair_type"] == "same-language"]
    cross_df = df[df["pair_type"] == "cross-lingual"]

    same_err_cnt = sum(1 for e in error_records if e.true_pair_type == "same-language")
    cross_err_cnt = sum(1 for e in error_records if e.true_pair_type == "cross-lingual")

    stats = {
        "model_name": model_name,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "overall_error_rate_pct": error_rate,
        "same_lang_errors": same_err_cnt,
        "same_lang_error_rate_pct": round((same_err_cnt / max(1, len(same_df))) * 100, 2),
        "cross_lang_errors": cross_err_cnt,
        "cross_lang_error_rate_pct": round((cross_err_cnt / max(1, len(cross_df))) * 100, 2),
        "category_counts": category_counts,
    }

    # Build CSV export DataFrame
    rows_data = []
    for e in error_records:
        rows_data.append({
            "review_id": e.review_id,
            "review_text": e.review_text,
            "aspect_term": e.aspect_term,
            "opinion_term": e.opinion_term,
            "true_sentiment": e.true_sentiment,
            "predicted_sentiment": e.predicted_sentiment,
            "true_pair_type": e.true_pair_type,
            "predicted_pair_type": e.predicted_pair_type,
            "error_category": ", ".join(e.error_categories),
            "explanation": e.explanation,
            "model_name": e.model_name,
        })

    error_df = pd.DataFrame(rows_data)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_model_name = model_name.replace(" ", "_").replace("+", "_").lower()
        out_csv = os.path.join(output_dir, f"error_analysis_{safe_model_name}.csv")
        error_df.to_csv(out_csv, index=False)
        stats["saved_csv_path"] = out_csv

    return error_records, error_df, stats
