# Research Project Audit & Gap Assessment

**Project Title:** Hinglish Product Review Analyzer: Aspect-Based Sentiment Analysis of Code-Mixed Hinglish Product Reviews: Quantifying the Cross-Lingual Aspect–Opinion Alignment Gap  
**Date of Audit:** October 13, 2026  
**Auditor:** Senior NLP Researcher, Machine Learning Engineer & Academic Reviewer  

---

## 1. Existing Functionality
- **Streamlit Application (`app.py`):** 8 functional pages covering Single Review Analysis, Dataset Explorer, Model Comparison, Alignment Gap Analysis, Transliteration Normalization, Experiment Runner, and Methodology.
- **Data Validation Pipeline (`src/data_validation.py`):** Schema enforcement (9 required columns), categorical label normalization, character span offset alignment, and automated pair-type derivation.
- **Group-Aware Splitting (`src/data_loader.py`):** Implementation of `split_by_review_id` ensuring all aspect-opinion records for a given `review_id` are grouped strictly in a single partition (Train: ~70%, Val: ~15%, Test: ~15%), preventing review-level leakage.
- **Classical ABSA Baselines (`src/classical_models.py`):** TF-IDF feature extraction conditioned on `[Aspect] [SEP] [Opinion] [SEP] [Review]` with Logistic Regression and LinearSVC classifiers.
- **Multilingual Transformer Infrastructure (`src/transformer_models.py`):** Tokenizer and sequence classification pipelines for mBERT (`bert-base-multilingual-cased`), XLM-RoBERTa (`xlm-roberta-base`), and MuRIL (`google/muril-base-cased`) with MPS/CUDA/CPU device auto-detection.
- **Orthographic Normalization (`src/preprocessing.py`):** `TransliterationNormalizer` reducing character flooding and canonicalizing Romanized Hindi spelling variants without translating to English.
- **Linguistic Extractors:** Rule- and pattern-based modules for aspect extraction, opinion extraction, relation binding, and span language identification.
- **Evaluation & Diagnostics:** Micro/Macro F1 calculation, 10 documented error categories (`src/error_analysis.py`), and bootstrap CI calculation (`src/gap_analysis.py`).
- **Test Suite:** 42 initial unit tests in `tests/` covering validation, splits, preprocessing, language ID, and models.

---

## 2. Identified Bugs & Vulnerabilities
1. **Low Sample Thresholds in Gap Analysis:** `min_samples=2` was hardcoded in `run_experiments.py`, allowing the alignment gap to be calculated on as few as 2 same-language and 4 cross-lingual test records. This produced uninterpretable, high-variance metrics (-44.44 pp and -15.56 pp) that cannot support scientific conclusions.
2. **Missing Distinction Between Calculation and Validity:** The system lacked an explicit 3-tier distinction between:
   - Level A: Mathematically computable.
   - Level B: Descriptive reporting only (exploratory).
   - Level C: Statistically conclusive evidence for a research conclusion.
3. **Missing Transformer Execution Results:** While `src/transformer_models.py` had training and evaluation classes, experiments had not been executed, leaving transformer benchmark entries blank or "Not evaluated".
4. **Annotation Disagreements in Corpus:** 11 records in `data/research_reviews_annotated.csv` had disagreements between human `annotated_pair_type` and deterministic `derived_pair_type`. Specifically, mixed expressions (e.g. English aspect + Mixed opinion) were inconsistently tagged as `cross-lingual` in some records and `same-language` in others.
5. **Report Naming Discrepancies:** Prompt-specified audit files (`results/annotation_audit.csv`, `results/data_validation_report.json`, `results/data_validation_report.txt`, `results/confidence_intervals.csv`) were either absent or named differently.

---

## 3. Data-Quality & Provenance Issues
- **Dataset Size:** `data/research_reviews_annotated.csv` contains 61 annotation records across 40 reviews. This is a small pilot/demonstration dataset, far below the target 1,500–3,000 pairs required for full empirical confidence.
- **Provenance Verification:** Existing records lacked documented external source IDs, licensing trail, or formal multi-annotator inter-rater reliability scores (Fleiss' or Cohen's Kappa). They must be formally classified as an **unverified curated pilot dataset** rather than a natural verified benchmark.
- **Span Containment:** All 61 records match text spans, but several opinions contain both English modifiers and Hindi auxiliaries (e.g., `"smooth hai"`, `"kaafi comfortable hai"`), requiring a formalized linguistic uncertainty policy.

---

## 4. Experimental-Design Weaknesses
- **Evaluation Partition Size:** With a 70/15/15 split on 40 reviews, the test partition contains only 6 unique reviews (~10 aspect-opinion pairs). This sample size is insufficient to conclude whether cross-lingual pairs perform worse than same-language pairs without broad confidence intervals indicating wide uncertainty.
- **Power & Precision:** Statistical power to detect a 5 percentage point gap with N=10 test items is near zero. Power assessment must be formalized.

---

## 5. Potential Sources of Data Leakage (Audited)
- `split_by_review_id` successfully groups records by `review_id`. No `review_id` overlap was found across partitions.
- TfidfVectorizer in `ClassicalABSAModel` is correctly fit only on `train_df`.
- Test data is strictly kept out of normalization parameter estimation.

---

## 6. Action Plan
1. Formalize the annotation uncertainty policy and generate `results/annotation_audit.csv`.
2. Generate `results/data_validation_report.json` and `results/data_validation_report.txt`.
3. Create `data/dataset_provenance.md` documenting verified vs. unverified status and public import pipelines.
4. Upgrade `gap_analysis.py` and `run_experiments.py` to enforce the 3-tier sample adequacy standard and group bootstrap CIs.
5. Execute baseline models and fine-tune at least one multilingual transformer (MuRIL / mBERT) on actual data.
6. Run the transliteration normalization experiment (Condition A vs. Condition B).
7. Execute full error analysis into 10 categories.
8. Update Streamlit UI, literature survey, and unit test suite.
