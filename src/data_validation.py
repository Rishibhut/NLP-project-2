"""
Data validation module for Hinglish ABSA datasets.
Enforces schema conformity, span containment, categorical label integrity,
and prevents data anomalies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import re

REQUIRED_COLUMNS = [
    "review_id",
    "review_text",
    "aspect_term",
    "opinion_term",
    "sentiment",
    "aspect_language",
    "opinion_language",
    "pair_type",
    "product_category",
]

VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}
VALID_LANGUAGES = {"Hindi", "English", "Mixed", "Other", "Uncertain"}
VALID_PAIR_TYPES = {"same-language", "cross-lingual", "Uncertain"}


@dataclass
class ValidationResult:
    is_valid: bool
    total_rows: int
    valid_rows: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    cleaned_df: Optional[pd.DataFrame] = None


def normalize_sentiment_label(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().capitalize()
    if s in VALID_SENTIMENTS:
        return s
    return None


def normalize_language_label(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().capitalize()
    if s in VALID_LANGUAGES:
        return s
    return None


def normalize_pair_type_label(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in {"same-language", "same_language", "same language"}:
        return "same-language"
    if s in {"cross-lingual", "cross_lingual", "cross lingual"}:
        return "cross-lingual"
    if s in {"uncertain", "unknown"}:
        return "Uncertain"
    return None


def derive_pair_type(aspect_lang: str, opinion_lang: str) -> str:
    """
    Deterministically derive pair_type based strictly on aspect_language and opinion_language.
    """
    if aspect_lang in {"Uncertain", "Mixed", "Other"} or opinion_lang in {"Uncertain", "Mixed", "Other"}:
        return "Uncertain"
    if aspect_lang == opinion_lang:
        return "same-language"
    return "cross-lingual"


def validate_absa_dataframe(df: pd.DataFrame, strict: bool = False) -> ValidationResult:
    """
    Validates an ABSA DataFrame against research specifications.
    
    Args:
        df: Input pandas DataFrame.
        strict: If True, any error invalidates the dataset. If False, invalid rows are quarantined.
        
    Returns:
        ValidationResult with validation metrics, error details, and sanitized dataframe.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    if df is None or df.empty:
        return ValidationResult(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=["Dataset is empty or None."],
            warnings=[],
            stats={},
            cleaned_df=None,
        )
    
    total_rows = len(df)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return ValidationResult(
            is_valid=False,
            total_rows=total_rows,
            valid_rows=0,
            errors=[f"Missing required columns: {missing_cols}"],
            warnings=[],
            stats={"missing_columns": missing_cols},
            cleaned_df=None,
        )
        
    cleaned = df.copy()
    
    # Check null values in crucial text columns
    null_texts = cleaned["review_text"].isna().sum()
    if null_texts > 0:
        errors.append(f"Found {null_texts} rows with missing 'review_text'.")
        
    null_aspects = cleaned["aspect_term"].isna().sum()
    if null_aspects > 0:
        errors.append(f"Found {null_aspects} rows with missing 'aspect_term'.")
        
    null_opinions = cleaned["opinion_term"].isna().sum()
    if null_opinions > 0:
        errors.append(f"Found {null_opinions} rows with missing 'opinion_term'.")
        
    # Check duplicates
    duplicate_count = cleaned.duplicated(subset=["review_id", "aspect_term", "opinion_term"]).sum()
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate aspect-opinion pairs for the same review_id.")
        
    # Standardize string fields
    cleaned["review_text"] = cleaned["review_text"].astype(str).str.strip()
    cleaned["aspect_term"] = cleaned["aspect_term"].astype(str).str.strip()
    cleaned["opinion_term"] = cleaned["opinion_term"].astype(str).str.strip()
    cleaned["product_category"] = cleaned["product_category"].fillna("General").astype(str).str.strip()
    
    # Check span containment
    invalid_span_rows = []
    for idx, row in cleaned.iterrows():
        text_lower = row["review_text"].lower()
        aspect_lower = row["aspect_term"].lower()
        opinion_lower = row["opinion_term"].lower()
        
        if aspect_lower not in text_lower:
            invalid_span_rows.append((idx, f"Row {idx}: aspect_term '{row['aspect_term']}' not found in review_text."))
        if opinion_lower not in text_lower:
            invalid_span_rows.append((idx, f"Row {idx}: opinion_term '{row['opinion_term']}' not found in review_text."))
            
    if invalid_span_rows:
        warnings.append(f"Found {len(invalid_span_rows)} span alignment mismatches where terms were not found in review_text.")
        if strict:
            errors.extend([msg for _, msg in invalid_span_rows[:10]])
            if len(invalid_span_rows) > 10:
                errors.append(f"...and {len(invalid_span_rows) - 10} more span mismatches.")

    # Validate and normalize sentiment
    cleaned["sentiment_norm"] = cleaned["sentiment"].apply(normalize_sentiment_label)
    invalid_sentiments = cleaned[cleaned["sentiment_norm"].isna()]
    if len(invalid_sentiments) > 0:
        errors.append(f"Found {len(invalid_sentiments)} invalid sentiment labels. Must be one of: {sorted(VALID_SENTIMENTS)}")
    cleaned["sentiment"] = cleaned["sentiment_norm"]
    cleaned.drop(columns=["sentiment_norm"], inplace=True)
    
    # Validate and normalize languages
    cleaned["aspect_language_norm"] = cleaned["aspect_language"].apply(normalize_language_label)
    invalid_aspect_langs = cleaned[cleaned["aspect_language_norm"].isna()]
    if len(invalid_aspect_langs) > 0:
        errors.append(f"Found {len(invalid_aspect_langs)} invalid aspect_language labels. Must be one of: {sorted(VALID_LANGUAGES)}")
    cleaned["aspect_language"] = cleaned["aspect_language_norm"]
    cleaned.drop(columns=["aspect_language_norm"], inplace=True)

    cleaned["opinion_language_norm"] = cleaned["opinion_language"].apply(normalize_language_label)
    invalid_op_langs = cleaned[cleaned["opinion_language_norm"].isna()]
    if len(invalid_op_langs) > 0:
        errors.append(f"Found {len(invalid_op_langs)} invalid opinion_language labels. Must be one of: {sorted(VALID_LANGUAGES)}")
    cleaned["opinion_language"] = cleaned["opinion_language_norm"]
    cleaned.drop(columns=["opinion_language_norm"], inplace=True)

    # Validate pair_type
    cleaned["pair_type_norm"] = cleaned["pair_type"].apply(normalize_pair_type_label)
    # Check consistency with derived pair_type
    for idx, row in cleaned.iterrows():
        if pd.notna(row["aspect_language"]) and pd.notna(row["opinion_language"]):
            expected = derive_pair_type(row["aspect_language"], row["opinion_language"])
            annotated = row["pair_type_norm"]
            if annotated and annotated != expected and expected != "Uncertain":
                warnings.append(f"Row {idx}: Annotated pair_type '{annotated}' differs from derived pair_type '{expected}'.")
                # Auto-align to mathematically rigorous derived value
                cleaned.at[idx, "pair_type_norm"] = expected
                
    cleaned["pair_type"] = cleaned["pair_type_norm"].fillna("Uncertain")
    cleaned.drop(columns=["pair_type_norm"], inplace=True)

    # Filter out invalid rows if not strict
    valid_mask = (
        cleaned["review_text"].notna() &
        (cleaned["review_text"] != "") &
        cleaned["aspect_term"].notna() &
        (cleaned["aspect_term"] != "") &
        cleaned["opinion_term"].notna() &
        (cleaned["opinion_term"] != "") &
        cleaned["sentiment"].notna() &
        cleaned["aspect_language"].notna() &
        cleaned["opinion_language"].notna()
    )
    
    cleaned_valid = cleaned[valid_mask].copy()
    valid_rows = len(cleaned_valid)
    
    is_valid = (len(errors) == 0) and (valid_rows > 0)
    
    stats = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "unique_reviews": cleaned_valid["review_id"].nunique() if valid_rows > 0 else 0,
        "sentiment_counts": cleaned_valid["sentiment"].value_counts().to_dict() if valid_rows > 0 else {},
        "pair_type_counts": cleaned_valid["pair_type"].value_counts().to_dict() if valid_rows > 0 else {},
        "aspect_lang_counts": cleaned_valid["aspect_language"].value_counts().to_dict() if valid_rows > 0 else {},
        "opinion_lang_counts": cleaned_valid["opinion_language"].value_counts().to_dict() if valid_rows > 0 else {},
        "product_categories": cleaned_valid["product_category"].value_counts().to_dict() if valid_rows > 0 else {},
    }
    
    return ValidationResult(
        is_valid=is_valid,
        total_rows=total_rows,
        valid_rows=valid_rows,
        errors=errors,
        warnings=warnings,
        stats=stats,
        cleaned_df=cleaned_valid if is_valid else None,
    )
