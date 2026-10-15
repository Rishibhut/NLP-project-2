"""
Data validation module for Hinglish ABSA datasets.
Enforces schema conformity, span containment, categorical label integrity,
preserves human annotations, flags pair-type disagreements, and outputs audit reports.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os
import re
import pandas as pd

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

    def save_report(self, output_dir: str = "results") -> str:
        """Saves structured JSON and text/markdown validation reports to disk."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "data_validation_report.json")
        txt_path = os.path.join(output_dir, "data_validation_report.txt")
        legacy_json_path = os.path.join(output_dir, "validation_report.json")
        legacy_md_path = os.path.join(output_dir, "validation_report.md")

        report_dict = {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        with open(legacy_json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        # Detailed text and markdown report content
        txt_content = f"""================================================================================
DATASET VALIDATION & ANNOTATION AUDIT REPORT
================================================================================
Validation Status: {"PASS" if self.is_valid else "FAIL"}
Total Records Inspected: {self.total_rows}
Valid Usable Records: {self.valid_rows}
Unique Customer Reviews: {self.stats.get("unique_reviews", 0)}

INCLUSION & EXCLUSION SUMMARY:
- Records included in Sentiment Model Training: {self.valid_rows}
- Records included in Primary Alignment Comparison: {self.stats.get("pair_type_counts", {}).get("same-language", 0) + self.stats.get("pair_type_counts", {}).get("cross-lingual", 0)}
  * Same-Language Subset: {self.stats.get("pair_type_counts", {}).get("same-language", 0)}
  * Cross-Lingual Subset: {self.stats.get("pair_type_counts", {}).get("cross-lingual", 0)}
- Records excluded from Primary Alignment Comparison: {self.stats.get("pair_type_counts", {}).get("Uncertain", 0)}
  * Exclusion Reason: Quarantined under uncertainty policy due to intra-span code-mixing (Mixed/Uncertain language tags).

QUALITY & INTEGRITY CHECKS:
- Missing Value Counts: {self.stats.get("missing_value_counts", {})}
- Duplicate Annotation Records: {self.stats.get("duplicate_count", 0)}
- Span Alignment Mismatches: {self.stats.get("span_mismatch_count", 0)}
- Pair-Type Disagreements Detected: {len(self.stats.get("pair_type_disagreements", []))}
- Review ID Text Inconsistencies: {self.stats.get("review_text_inconsistencies", 0)}
- Category Inconsistencies per Review ID: {self.stats.get("category_inconsistencies", 0)}

SENTIMENT CLASS DISTRIBUTION:
{json.dumps(self.stats.get("sentiment_counts", {}), indent=2)}

PAIR TYPE DISTRIBUTION:
{json.dumps(self.stats.get("pair_type_counts", {}), indent=2)}

PAIR-TYPE DISAGREEMENTS & RESOLUTION STATUS:
"""
        disagreements = self.stats.get("pair_type_disagreements", [])
        if disagreements:
            for d in disagreements:
                txt_content += (
                    f"- Row {d.get('row_index')} (review_id {d.get('review_id')}): "
                    f"Original '{d.get('annotated_pair_type')}' vs Derived '{d.get('derived_pair_type')}' "
                    f"({d.get('aspect_language')} + {d.get('opinion_language')}) | "
                    f"Status: Resolved as Uncertain (quarantined)\n"
                )
        else:
            txt_content += "None detected.\n"

        txt_content += "\nERRORS (Prevents Execution):\n"
        if self.errors:
            for err in self.errors:
                txt_content += f"- {err}\n"
        else:
            txt_content += "None (All integrity constraints satisfied)\n"

        txt_content += "\nWARNINGS (Flagged for Inspection):\n"
        if self.warnings:
            for w in self.warnings[:30]:
                txt_content += f"- {w}\n"
            if len(self.warnings) > 30:
                txt_content += f"- ...and {len(self.warnings) - 30} more warnings.\n"
        else:
            txt_content += "None.\n"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        with open(legacy_md_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        return json_path


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
    if s in {"Hindi-roman", "Hindi_roman", "Romanized hindi", "Romanized-hindi", "Romanized_hindi"}:
        return "Hindi"
    if s in VALID_LANGUAGES:
        return s
    return None


def normalize_pair_type_label(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in {"same-language", "same_language", "same language", "aligned"}:
        return "same-language"
    if s in {"cross-lingual", "cross_lingual", "cross lingual", "crosslingual"}:
        return "cross-lingual"
    if s in {"uncertain", "unknown", "ambiguous"}:
        return "Uncertain"
    return None


def derive_pair_type(aspect_lang: str, opinion_lang: str) -> str:
    """
    Deterministically derive pair_type based strictly on aspect_language and opinion_language.
    Annotation policy:
    - English + English -> same-language
    - Hindi + Hindi -> same-language
    - English + Hindi / Hindi + English -> cross-lingual
    - If either is Uncertain, Mixed, or Other -> Uncertain (must not be silently assigned to primary evaluation groups)
    """
    if aspect_lang in {"Uncertain", "Mixed", "Other"} or opinion_lang in {"Uncertain", "Mixed", "Other"}:
        return "Uncertain"
    if aspect_lang == opinion_lang:
        return "same-language"
    return "cross-lingual"


def find_span_offsets(full_text: str, span_term: str) -> Optional[tuple[int, int]]:
    """Finds start and end character offsets of a span in the review text."""
    if not full_text or not span_term:
        return None
    pattern = re.escape(span_term.strip())
    match = re.search(pattern, full_text, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def validate_absa_dataframe(df: pd.DataFrame, strict: bool = False) -> ValidationResult:
    """
    Validates an ABSA DataFrame against research specifications.
    Preserves human annotations while recording derived pair types and disagreements.
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

    # Track missing values
    missing_counts = {col: int(cleaned[col].isna().sum()) for col in REQUIRED_COLUMNS}

    if missing_counts["review_text"] > 0:
        errors.append(f"Found {missing_counts['review_text']} rows with missing 'review_text'.")
    if missing_counts["aspect_term"] > 0:
        errors.append(f"Found {missing_counts['aspect_term']} rows with missing 'aspect_term'.")
    if missing_counts["opinion_term"] > 0:
        errors.append(f"Found {missing_counts['opinion_term']} rows with missing 'opinion_term'.")

    # Track duplicates
    duplicate_count = int(cleaned.duplicated(subset=["review_id", "aspect_term", "opinion_term"]).sum())
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate aspect-opinion pairs for the same review_id.")

    # Standardize string fields
    cleaned["review_text"] = cleaned["review_text"].astype(str).str.strip()
    cleaned["aspect_term"] = cleaned["aspect_term"].astype(str).str.strip()
    cleaned["opinion_term"] = cleaned["opinion_term"].astype(str).str.strip()
    cleaned["product_category"] = cleaned["product_category"].fillna("General").astype(str).str.strip()

    # Check review_id consistency: one review_id must map to one unique review_text and category
    rev_text_inconsistencies = 0
    cat_inconsistencies = 0
    for r_id, g in cleaned.groupby("review_id"):
        if g["review_text"].nunique() > 1:
            rev_text_inconsistencies += 1
            errors.append(f"Inconsistent review_text for review_id '{r_id}': found {g['review_text'].nunique()} distinct texts.")
        if g["product_category"].nunique() > 1:
            cat_inconsistencies += 1
            warnings.append(f"Inconsistent product_category for review_id '{r_id}': found {g['product_category'].nunique()} distinct categories.")

    # Check for empty strings in critical fields
    empty_rev_mask = (cleaned["review_text"] == "")
    if empty_rev_mask.sum() > 0:
        errors.append(f"Found {empty_rev_mask.sum()} rows with empty review_text.")
    empty_asp_mask = (cleaned["aspect_term"] == "")
    if empty_asp_mask.sum() > 0:
        errors.append(f"Found {empty_asp_mask.sum()} rows with empty aspect_term.")
    empty_op_mask = (cleaned["opinion_term"] == "")
    if empty_op_mask.sum() > 0:
        errors.append(f"Found {empty_op_mask.sum()} rows with empty opinion_term.")

    # Compute character offsets and check containment
    asp_starts, asp_ends = [], []
    op_starts, op_ends = [], []
    invalid_span_rows = []

    for idx, row in cleaned.iterrows():
        rev_t = row["review_text"]
        asp_t = row["aspect_term"]
        op_t = row["opinion_term"]

        asp_off = find_span_offsets(rev_t, asp_t)
        if asp_off:
            asp_starts.append(asp_off[0])
            asp_ends.append(asp_off[1])
        else:
            asp_starts.append(-1)
            asp_ends.append(-1)
            invalid_span_rows.append((idx, f"Row {idx}: aspect_term '{asp_t}' not found in review_text."))

        op_off = find_span_offsets(rev_t, op_t)
        if op_off:
            op_starts.append(op_off[0])
            op_ends.append(op_off[1])
        else:
            op_starts.append(-1)
            op_ends.append(-1)
            invalid_span_rows.append((idx, f"Row {idx}: opinion_term '{op_t}' not found in review_text."))

    cleaned["aspect_start_char"] = asp_starts
    cleaned["aspect_end_char"] = asp_ends
    cleaned["opinion_start_char"] = op_starts
    cleaned["opinion_end_char"] = op_ends

    if invalid_span_rows:
        warnings.append(f"Found {len(invalid_span_rows)} span alignment mismatches where terms were not found in review_text.")
        if strict:
            errors.extend([msg for _, msg in invalid_span_rows[:10]])

    # Normalize sentiments
    cleaned["sentiment_norm"] = cleaned["sentiment"].apply(normalize_sentiment_label)
    invalid_sentiments = cleaned[cleaned["sentiment_norm"].isna()]
    if len(invalid_sentiments) > 0:
        errors.append(f"Found {len(invalid_sentiments)} invalid sentiment labels. Must be one of: {sorted(VALID_SENTIMENTS)}")
    cleaned["sentiment"] = cleaned["sentiment_norm"]
    cleaned.drop(columns=["sentiment_norm"], inplace=True)

    # Normalize languages
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

    # Validate pair_type: preserve human annotation, compute derived pair_type, flag disagreements
    if "annotated_pair_type" in df.columns:
        cleaned["annotated_pair_type"] = df["annotated_pair_type"]
    else:
        cleaned["annotated_pair_type"] = cleaned["pair_type"]
    cleaned["pair_type_norm"] = cleaned["annotated_pair_type"].apply(normalize_pair_type_label)

    derived_types = []
    disagreements = []

    for idx, row in cleaned.iterrows():
        a_lang = row["aspect_language"]
        o_lang = row["opinion_language"]
        if pd.notna(a_lang) and pd.notna(o_lang):
            exp = derive_pair_type(a_lang, o_lang)
            derived_types.append(exp)
            ann = row["pair_type_norm"]
            if ann and ann != exp:
                disagreements.append({
                    "row_index": int(idx),
                    "review_id": str(row["review_id"]),
                    "annotated_pair_type": ann,
                    "derived_pair_type": exp,
                    "aspect_language": a_lang,
                    "opinion_language": o_lang,
                })
                warnings.append(
                    f"Row {idx} (review_id {row['review_id']}): Annotated pair_type '{ann}' "
                    f"disagrees with derived pair_type '{exp}' ({a_lang} + {o_lang})."
                )
        else:
            derived_types.append("Uncertain")

    cleaned["derived_pair_type"] = derived_types
    cleaned["pair_type_disagreement"] = [
        (cleaned.at[i, "pair_type_norm"] != cleaned.at[i, "derived_pair_type"])
        for i in cleaned.index
    ]

    # For evaluation consistency, use derived_pair_type as the primary pair_type while preserving original
    cleaned["pair_type"] = cleaned["derived_pair_type"]
    cleaned.drop(columns=["pair_type_norm"], inplace=True)

    # Filter out empty or null rows
    valid_mask = (
        cleaned["review_text"].notna() & (cleaned["review_text"] != "") &
        cleaned["aspect_term"].notna() & (cleaned["aspect_term"] != "") &
        cleaned["opinion_term"].notna() & (cleaned["opinion_term"] != "") &
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
        "unique_reviews": int(cleaned_valid["review_id"].nunique()) if valid_rows > 0 else 0,
        "missing_value_counts": missing_counts,
        "duplicate_count": duplicate_count,
        "span_mismatch_count": len(invalid_span_rows),
        "pair_type_disagreements": disagreements,
        "review_text_inconsistencies": rev_text_inconsistencies,
        "category_inconsistencies": cat_inconsistencies,
        "sentiment_counts": {str(k): int(v) for k, v in cleaned_valid["sentiment"].value_counts().items()} if valid_rows > 0 else {},
        "pair_type_counts": {str(k): int(v) for k, v in cleaned_valid["pair_type"].value_counts().items()} if valid_rows > 0 else {},
        "aspect_lang_counts": {str(k): int(v) for k, v in cleaned_valid["aspect_language"].value_counts().items()} if valid_rows > 0 else {},
        "opinion_lang_counts": {str(k): int(v) for k, v in cleaned_valid["opinion_language"].value_counts().items()} if valid_rows > 0 else {},
        "product_categories": {str(k): int(v) for k, v in cleaned_valid["product_category"].value_counts().items()} if valid_rows > 0 else {},
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
