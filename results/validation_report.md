================================================================================
DATASET VALIDATION & ANNOTATION AUDIT REPORT
================================================================================
Validation Status: PASS
Total Records Inspected: 61
Valid Usable Records: 61
Unique Customer Reviews: 40

INCLUSION & EXCLUSION SUMMARY:
- Records included in Sentiment Model Training: 61
- Records included in Primary Alignment Comparison: 50
  * Same-Language Subset: 16
  * Cross-Lingual Subset: 34
- Records excluded from Primary Alignment Comparison: 11
  * Exclusion Reason: Quarantined under uncertainty policy due to intra-span code-mixing (Mixed/Uncertain language tags).

QUALITY & INTEGRITY CHECKS:
- Missing Value Counts: {'review_id': 0, 'review_text': 0, 'aspect_term': 0, 'opinion_term': 0, 'sentiment': 0, 'aspect_language': 0, 'opinion_language': 0, 'pair_type': 0, 'product_category': 0}
- Duplicate Annotation Records: 0
- Span Alignment Mismatches: 0
- Pair-Type Disagreements Detected: 11
- Review ID Text Inconsistencies: 0
- Category Inconsistencies per Review ID: 0

SENTIMENT CLASS DISTRIBUTION:
{
  "Positive": 35,
  "Negative": 22,
  "Neutral": 4
}

PAIR TYPE DISTRIBUTION:
{
  "cross-lingual": 34,
  "same-language": 16,
  "Uncertain": 11
}

PAIR-TYPE DISAGREEMENTS & RESOLUTION STATUS:
- Row 3 (review_id 2): Original 'cross-lingual' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 11 (review_id 7): Original 'cross-lingual' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 25 (review_id 15): Original 'same-language' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 26 (review_id 15): Original 'same-language' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 33 (review_id 21): Original 'cross-lingual' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 34 (review_id 21): Original 'cross-lingual' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 39 (review_id 24): Original 'same-language' vs Derived 'Uncertain' (Mixed + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 41 (review_id 26): Original 'same-language' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 42 (review_id 26): Original 'same-language' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 46 (review_id 30): Original 'cross-lingual' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)
- Row 47 (review_id 30): Original 'same-language' vs Derived 'Uncertain' (English + Mixed) | Status: Resolved as Uncertain (quarantined)

ERRORS (Prevents Execution):
None (All integrity constraints satisfied)

WARNINGS (Flagged for Inspection):
- Row 3 (review_id 2): Annotated pair_type 'cross-lingual' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 11 (review_id 7): Annotated pair_type 'cross-lingual' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 25 (review_id 15): Annotated pair_type 'same-language' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 26 (review_id 15): Annotated pair_type 'same-language' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 33 (review_id 21): Annotated pair_type 'cross-lingual' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 34 (review_id 21): Annotated pair_type 'cross-lingual' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 39 (review_id 24): Annotated pair_type 'same-language' disagrees with derived pair_type 'Uncertain' (Mixed + Mixed).
- Row 41 (review_id 26): Annotated pair_type 'same-language' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 42 (review_id 26): Annotated pair_type 'same-language' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 46 (review_id 30): Annotated pair_type 'cross-lingual' disagrees with derived pair_type 'Uncertain' (English + Mixed).
- Row 47 (review_id 30): Annotated pair_type 'same-language' disagrees with derived pair_type 'Uncertain' (English + Mixed).
