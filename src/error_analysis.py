"""
Error Analysis module for Hinglish ABSA and Cross-Lingual Evaluation.
Automatically diagnoses, categorizes, and provides inspectable records of model prediction failures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import re

ERROR_CATEGORIES = [
    "Incorrect sentiment polarity",
    "Negation-handling error",
    "Multiple-aspect confusion",
    "Transliteration-related error",
    "Cross-lingual alignment error",
    "Ambiguous or neutral boundary",
]


@dataclass
class ErrorRecord:
    review_id: Any
    review_text: str
    aspect_term: str
    opinion_term: str
    pair_type: str
    gold_sentiment: str
    predicted_sentiment: str
    error_categories: List[str]
    diagnosis_detail: str


def diagnose_prediction_error(
    row: pd.Series,
    pred_sentiment: str,
    all_aspects_in_review: int = 1,
) -> Optional[ErrorRecord]:
    """
    Diagnoses the underlying reason for a model prediction failure.
    """
    gold_sentiment = str(row["sentiment"]).strip()
    pred_sentiment = str(pred_sentiment).strip()

    if gold_sentiment == pred_sentiment:
        return None

    review_text = str(row["review_text"])
    op_term = str(row["opinion_term"])
    pair_type = str(row.get("pair_type", "Uncertain"))
    categories = ["Incorrect sentiment polarity"]
    explanations = []

    # 1. Negation handling error check
    neg_pattern = re.compile(r"\b(?:nahi|nhi|nahin|not|mat|never|no)\b", re.IGNORECASE)
    if neg_pattern.search(review_text) or neg_pattern.search(op_term):
        if (gold_sentiment == "Negative" and pred_sentiment == "Positive") or \
           (gold_sentiment == "Positive" and pred_sentiment == "Negative"):
            categories.append("Negation-handling error")
            explanations.append("Contains negation operator that model may have inverted or missed.")

    # 2. Multiple-aspect confusion
    if all_aspects_in_review > 1:
        categories.append("Multiple-aspect confusion")
        explanations.append(f"Review contains {all_aspects_in_review} aspects; potential context bleed.")

    # 3. Transliteration variant influence
    translit_variants = ["bohot", "achha", "khraab", "bekar", "ghatya", "ekdm", "blkl"]
    if any(var in review_text.lower() for var in translit_variants):
        categories.append("Transliteration-related error")
        explanations.append("Contains non-standard Romanized spelling variants.")

    # 4. Cross-lingual alignment error
    if pair_type == "cross-lingual":
        categories.append("Cross-lingual alignment error")
        explanations.append("Aspect and opinion are in different languages.")

    # 5. Neutral boundary ambiguity
    if gold_sentiment == "Neutral" or pred_sentiment == "Neutral":
        categories.append("Ambiguous or neutral boundary")
        explanations.append("Boundary confusion involving neutral sentiment polarity.")

    detail = " | ".join(explanations) if explanations else "Misclassified sentiment polarity."

    return ErrorRecord(
        review_id=row["review_id"],
        review_text=review_text,
        aspect_term=row["aspect_term"],
        opinion_term=op_term,
        pair_type=pair_type,
        gold_sentiment=gold_sentiment,
        predicted_sentiment=pred_sentiment,
        error_categories=categories,
        diagnosis_detail=detail,
    )


def run_error_analysis(
    df: pd.DataFrame,
    y_true_col: str = "sentiment",
    y_pred_col: str = "predicted_sentiment",
) -> Tuple[List[ErrorRecord], pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes all misclassifications in a DataFrame.
    Returns:
        - List of ErrorRecord objects
        - DataFrame of errors for export and display
        - Summary statistics dictionary
    """
    # Count aspects per review_id
    aspect_counts = df["review_id"].value_counts().to_dict()

    error_records: List[ErrorRecord] = []
    category_counts: Dict[str, int] = {cat: 0 for cat in ERROR_CATEGORIES}

    for _, row in df.iterrows():
        n_aspects = aspect_counts.get(row["review_id"], 1)
        err = diagnose_prediction_error(row, str(row[y_pred_col]), all_aspects_in_review=n_aspects)
        if err:
            error_records.append(err)
            for cat in err.error_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1

    total_samples = len(df)
    total_errors = len(error_records)
    error_rate = round((total_errors / max(1, total_samples)) * 100, 2)

    # Subgroup error rates
    same_df = df[df["pair_type"] == "same-language"]
    cross_df = df[df["pair_type"] == "cross-lingual"]

    same_errors = sum(1 for e in error_records if e.pair_type == "same-language")
    cross_errors = sum(1 for e in error_records if e.pair_type == "cross-lingual")

    same_error_rate = round((same_errors / max(1, len(same_df))) * 100, 2) if len(same_df) > 0 else 0.0
    cross_error_rate = round((cross_errors / max(1, len(cross_df))) * 100, 2) if len(cross_df) > 0 else 0.0

    stats = {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "overall_error_rate": error_rate,
        "same_lang_errors": same_errors,
        "same_lang_error_rate": same_error_rate,
        "cross_lang_errors": cross_errors,
        "cross_lang_error_rate": cross_error_rate,
        "category_counts": category_counts,
    }

    # Format into DataFrame
    rows_data = []
    for e in error_records:
        rows_data.append({
            "Review ID": e.review_id,
            "Review Text": e.review_text,
            "Aspect": e.aspect_term,
            "Opinion": e.opinion_term,
            "Pair Type": e.pair_type,
            "Gold Sentiment": e.gold_sentiment,
            "Predicted": e.predicted_sentiment,
            "Error Categories": ", ".join(e.error_categories),
            "Diagnosis Detail": e.diagnosis_detail,
        })

    error_df = pd.DataFrame(rows_data)
    return error_records, error_df, stats
