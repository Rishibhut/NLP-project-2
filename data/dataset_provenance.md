# Dataset Provenance, Scientific Audit & Ingestion Protocol

**Document Version:** 1.0.0  
**Last Updated:** October 13, 2026  
**Status:** Audited & Verified for Academic Integrity  

---

## 1. Provenance of the Current Pilot Dataset (`data/research_reviews_annotated.csv`)

| Metadata Attribute | Declared Status | Verification Audit |
|---|---|---|
| **Dataset Title** | Hinglish ABSA Pilot Dataset | Curated pilot benchmark (N=61 records, 40 reviews) |
| **Origin / Source** | Manually curated illustrative Hinglish e-commerce reviews | **UNVERIFIED as real-world platform crawl.** Constructed/curated for controlled pilot testing. |
| **Source URL / Citation** | None available | **UNVERIFIED** (No external repository DOI/URL exists for this specific initial subset) |
| **Data Nature** | Semi-synthetic / Manually constructed illustrative records | **NOT a natural organic e-commerce crawl.** Constructed to model realistic code-mixing patterns across 9 product categories. |
| **Collection Date** | October 2026 | Documented in project creation log |
| **License / Usage** | Academic & Educational Evaluation License (Non-Commercial) | Open for internal research benchmarking and educational demonstration |
| **Annotator Information** | Single-expert curation with deterministic policy reconciliation | **UNVERIFIED for formal inter-annotator agreement (IAA)**. No double-blind multi-annotator Kappa score exists. |
| **Known Limitations** | 1. Small sample size (N=61 records, 40 reviews).<br>2. Severe statistical power limitation for held-out splits (N_test ≈ 10 records).<br>3. Dominance of English aspect terms (58/61) with Hindi opinion terms (36/61).<br>4. Unsuitable for definitive conclusions on large-scale e-commerce behavior. | Documented explicitly; all empirical findings from this subset are classified as **preliminary and exploratory**. |

---

## 2. Separation of Demonstration Data and Research Data
- `data/sample_reviews.csv`: Initial 61 illustrative records used for basic Streamlit UI demonstrations.
- `data/research_reviews_annotated.csv`: Validated pilot dataset with deterministic character offsets, derived pair types, and audited disagreements.
- `data/dataset_template.csv`: Standardized schema for importing external research datasets.

---

## 3. Annotation Methodology & Uncertainty Policy
The annotation policy strictly follows deterministic rules based on span-level language tagging:
- **Same-Language (`same-language`):** Both aspect and opinion belong to the same language:
  - English Aspect + English Opinion $\rightarrow$ `same-language`
  - Hindi Aspect + Hindi Opinion $\rightarrow$ `same-language`
- **Cross-Lingual (`cross-lingual`):** Aspect and opinion span belong to different languages:
  - English Aspect + Hindi Opinion $\rightarrow$ `cross-lingual`
  - Hindi Aspect + English Opinion $\rightarrow$ `cross-lingual`
- **Uncertain / Mixed (`Uncertain`):**
  - If either the aspect span or opinion span contains intra-span code-mixing (tagged `Mixed`, e.g., English noun + Hindi auxiliary `"smooth hai"` or `"safe thi"`), or if the language is tagged `Uncertain` or `Other`.
  - **Policy Rule:** Under no circumstances are `Mixed` or `Uncertain` spans silently assigned to `same-language` or `cross-lingual` subsets. They are quarantined into the `Uncertain` subset and excluded from primary alignment gap calculations to avoid systematic bias.

---

## 4. Documented Workflow for Importing Public Benchmark Datasets
To scale from the current pilot dataset (N=61) to the research target (1,500–3,000 annotated pairs), use the following standardized ingestion pipeline without scraping or violating terms of service:

### A. Candidate Public Repositories
1. **SemEval-2020 Task 9 (Patwa et al., 2020):** 17,000 Hinglish social media posts (CC BY 4.0).
2. **GLUECoS Benchmark (Khanuja et al., 2020):** Multi-task code-mixed Hindi-English sentiment and NLI corpora.
3. **L3Cube-Hinglish Corpus (Joshi et al., 2022):** Over 30,000 code-mixed Hindi-English sentences with sentiment annotations.

### B. Standard Ingestion & Schema Alignment Procedure
Execute the public dataset importer using `src/dataset_preparation.py`:
```bash
python3 -m src.dataset_preparation --import-public <source_csv_or_json> --source-name "L3Cube-Hinglish" --license "CC-BY-4.0"
```
The ingestion pipeline automatically:
1. Verifies required columns (`review_id`, `review_text`, `aspect_term`, `opinion_term`, `sentiment`, `aspect_language`, `opinion_language`).
2. Calculates exact character start and end offsets (`aspect_start_char`, etc.).
3. Applies deterministic pair-type derivation and logs disagreements into `results/annotation_audit.csv`.
4. Saves verified provenance metadata with cryptographic SHA-256 hash in `data/provenance_metadata.json`.
