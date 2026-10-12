# Dataset Documentation & Annotation Guidelines

## Overview
This directory contains dataset resources for Aspect-Based Sentiment Analysis (ABSA) of Hindi-English code-mixed (Hinglish) product reviews.

> **Research Integrity Notice:**
> The file `sample_reviews.csv` is an **illustrative demonstration dataset** designed for reproducing and testing the end-to-end ABSA pipeline, language identification, and cross-lingual gap analysis. It is not empirical evidence from an uncurated production dump. Real research evaluations must be run on validated, annotated datasets adhering to the guidelines below.

---

## Schema Specification

Every review entry must adhere to the following schema:

| Column Name | Type | Allowed Values / Format | Description |
|---|---|---|---|
| `review_id` | Integer / String | e.g., `1`, `review_042` | Unique review identifier. Repeated across rows when a review has multiple aspects. |
| `review_text` | String | UTF-8 text | The complete Hinglish customer review text. |
| `aspect_term` | String | Substring of `review_text` | The product feature/aspect mentioned (e.g. `Camera quality`). |
| `opinion_term` | String | Substring of `review_text` | The subjective/evaluative expression describing the aspect (e.g. `achhi hai`). |
| `sentiment` | Category | `Positive`, `Negative`, `Neutral` | Polarity of the aspect-specific sentiment. |
| `aspect_language` | Category | `English`, `Hindi`, `Mixed`, `Other`, `Uncertain` | Language identity of the aspect term. |
| `opinion_language` | Category | `Hindi`, `English`, `Mixed`, `Other`, `Uncertain` | Language identity of the opinion term. |
| `pair_type` | Category | `same-language`, `cross-lingual`, `Uncertain` | Classified directly by comparing `aspect_language` and `opinion_language`. |
| `product_category` | Category | e.g., `Smartphones`, `Audio`, `Footwear` | The e-commerce product domain. |

---

## Pair Type Determination Rule

- **same-language**: `aspect_language` == `opinion_language` (e.g. both English or both Hindi).
- **cross-lingual**: `aspect_language` != `opinion_language` (e.g., English aspect + Hindi opinion, or Hindi aspect + English opinion).
- **Uncertain**: If either term is flagged as `Uncertain` or `Mixed`.

*Rule of Scientific Integrity:* `pair_type` is determined strictly from the annotated language of the specific `aspect_term` and `opinion_term`, **never** inferred merely from the presence of both languages somewhere in the overall sentence.
