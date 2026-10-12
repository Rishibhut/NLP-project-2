# Dataset Validation & Annotation Audit Report

- **Validation Status:** ✅ PASS
- **Total Records:** 61
- **Valid Usable Records:** 61
- **Unique Reviews:** 40

## Summary Statistics
- **Missing Values:** {'review_id': 0, 'review_text': 0, 'aspect_term': 0, 'opinion_term': 0, 'sentiment': 0, 'aspect_language': 0, 'opinion_language': 0, 'pair_type': 0, 'product_category': 0}
- **Duplicate Records:** 0
- **Span Mismatch Count:** 0
- **Pair-Type Disagreements:** 11

### Subset Record Counts
- **Same-Language:** 16
- **Cross-Lingual:** 34
- **Uncertain / Mixed:** 11

### Sentiment Class Distribution
{
  "Positive": 35,
  "Negative": 22,
  "Neutral": 4
}

### Warnings & Disagreements
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
