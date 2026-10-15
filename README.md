# Hinglish Product Review Analyzer

### *Aspect-Based Sentiment Analysis of Code-Mixed Hinglish Product Reviews: Quantifying the Cross-Lingual Aspect–Opinion Alignment Gap*

---

## 📌 1. Project Overview & Research Objective

In multilingual and code-mixed digital markets (e.g., India), customer product reviews frequently exhibit **intra-sentential code-mixing**, predominantly between Hindi and English (**Hinglish**). 

A prominent linguistic pattern in code-mixed product reviews is the structural asymmetry between the product entity and its evaluation: consumers frequently specify the product attribute (**Aspect Term**) in English (e.g., *"Camera quality"*, *"Battery backup"*, *"Screen resolution"*), while expressing the subjective appraisal (**Opinion Expression**) in Romanized Hindi (e.g., *"achhi hai"*, *"bahut kharab nikla"*, *"ekdum bekaar"*), or vice versa.

### Central Research Question:
> *"Does aspect-opinion language alignment affect aspect-level sentiment classification performance on Hinglish product reviews, and can transliteration normalization reduce the observed performance difference?"*

The system quantifies the **Cross-Lingual Aspect–Opinion Alignment Gap**:

$$\text{Alignment Gap} = \text{Macro-F1}_{\text{same-language}} - \text{Macro-F1}_{\text{cross-lingual}}$$

- A **positive gap** indicates lower Macro-F1 on cross-lingual pairs than on same-language pairs.
- A **negative gap** indicates the opposite.
- Neither direction is assumed in advance; conclusions are derived strictly from empirical prediction evaluation.

---

## 🏗️ 2. System Architecture

The application decouples the ABSA pipeline into modular, verifiable components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Hinglish Customer Review                           │
│       "Camera quality achhi hai, but battery backup bahut kharab hai."      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
   [Aspect Term Extractor]                       [Opinion Term Extractor]
   - Camera quality (0..14)                      - achhi hai (15..24)
   - battery backup (30..44)                     - bahut kharab hai (45..61)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                     [Aspect-Opinion Relation Extractor]
                     - Clause segmentation & syntactic proximity
                     - Link 1: (Camera quality, achhi hai)
                     - Link 2: (battery backup, bahut kharab hai)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
   [Span Language Identifier]                    [Aspect Sentiment Classifier]
   - Aspect: English                             - Pair 1: Positive (Conf: 0.85)
   - Opinion: Hindi                              - Pair 2: Negative (Conf: 0.90)
   - Pair Type: cross-lingual                    (Lexicon / TF-IDF / MuRIL)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                   [Cross-Lingual Alignment Gap Analyzer]
                   Gap = F1(same-language) - F1(cross-lingual)
                   (Enforces 3-Tier Sample Adequacy Standard)
```

---

## 📊 3. Dataset Schema & Annotation Rules

### Dataset Schema
Each record in `data/research_reviews_annotated.csv` corresponds to a single aspect-opinion pair and contains the following fields:

| Field Name | Type | Description |
|---|---|---|
| `review_id` | Integer | Unique identifier grouping all aspect mentions within the same customer review |
| `review_text` | String | Raw review text containing the code-mixed statement |
| `aspect_term` | String | Extracted target product aspect |
| `opinion_term` | String | Associated sentiment-bearing opinion span |
| `sentiment` | Categorical | Target sentiment polarity (`Positive`, `Negative`, `Neutral`) |
| `aspect_language` | Categorical | Span language of aspect (`English`, `Hindi`, `Mixed`, `Uncertain`) |
| `opinion_language` | Categorical | Span language of opinion (`English`, `Hindi`, `Mixed`, `Uncertain`) |
| `pair_type` | Categorical | Derived linguistic pair alignment (`same-language`, `cross-lingual`, `Uncertain`) |
| `product_category` | Categorical | E-commerce domain (`Smartphones`, `Audio`, `Laptops`, `Footwear`, etc.) |
| `aspect_start_char` | Integer | Exact 0-indexed start character offset in `review_text` |
| `aspect_end_char` | Integer | Exact 0-indexed end character offset in `review_text` |
| `opinion_start_char` | Integer | Exact 0-indexed start character offset in `review_text` |
| `opinion_end_char` | Integer | Exact 0-indexed end character offset in `review_text` |
| `annotated_pair_type`| Categorical | Original human annotator assignment (preserved for auditing) |
| `derived_pair_type`  | Categorical | Deterministically derived pair type based on span language rules |
| `pair_type_disagreement` | Boolean | Flags whether human annotation disagrees with deterministic derivation |

### Annotation Policy & Uncertainty Rules
1. **Same-Language (`same-language`):**
   - English Aspect + English Opinion $\to$ `same-language`
   - Hindi-Roman Aspect + Hindi-Roman Opinion $\to$ `same-language`
2. **Cross-Lingual (`cross-lingual`):**
   - English Aspect + Hindi-Roman Opinion $\to$ `cross-lingual`
   - Hindi-Roman Aspect + English Opinion $\to$ `cross-lingual`
3. **Uncertainty & Quarantine Policy (`Uncertain`):**
   - Spans containing intra-span code-mixing (tagged `Mixed`, e.g., English noun + Hindi auxiliary `"smooth hai"` or `"safe thi"`) are **never silently forced** into either `same-language` or `cross-lingual`.
   - All `Mixed` or ambiguous records are classified as `Uncertain` and quarantined from primary alignment gap calculations to preserve scientific validity.

---

## 🔍 4. Dataset Provenance Status

- **Pilot Benchmark Dataset (`data/research_reviews_annotated.csv`):** Contains 61 annotation records across 40 unique reviews.
- **Verification Status:** Explicitly documented as an **unverified curated pilot dataset** in `data/provenance_metadata.json` and `data/dataset_provenance.md`. It is designed for controlled pipeline validation and pilot testing, not a verified large-scale web crawl.
- **Public Benchmark Ingestion Workflow:** A documented public ingestion protocol is implemented in `src/dataset_preparation.py` to support importing verified public corpora (e.g., SemEval-2020 Task 9, GLUECoS, L3Cube-Hinglish) without scraping or platform violations.

---

## 💻 5. Environment Setup & Dependency Installation

### Prerequisites
- Python 3.10 to 3.14+
- `pip` package manager

### Installation Commands
```bash
# 1. Clone repository
git clone <repo-url>
cd "NLP project 2"

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 🛡️ 6. Data Validation & Leakage Prevention

### Validation Pipeline
Execute automated data validation:
```bash
python3 -c "from src.data_validation import validate_absa_dataframe; import pandas as pd; res = validate_absa_dataframe(pd.read_csv('data/research_reviews_annotated.csv')); res.save_report('results'); print('Valid:', res.is_valid)"
```
This checks:
- Schema completeness (all 9 mandatory columns present).
- Empty string absence and missing value tracking.
- Duplicate record detection.
- Exact character span offset containment.
- Categorical label legitimacy.
- Review ID consistency (ensuring one `review_id` has a single `review_text` and single `product_category`).
- Saves audit logs: `results/annotation_audit.csv`, `results/data_validation_report.json`, `results/data_validation_report.txt`.

### Review-Level Splitting & Leakage Prevention
Multiple aspect records frequently share the same `review_id`. To prevent review-level data leakage:
- `split_by_review_id()` in `src/data_loader.py` performs group-aware splitting on `review_id` (Train: ~70%, Val: ~15%, Test: ~15%).
- `verify_no_leakage()` guarantees disjoint partitions: no `review_id` or customer text is shared between training, validation, or testing partitions.
- Feature extractors (e.g., TF-IDF) are strictly fitted on the training split only.

---

## ⚙️ 7. Model Implementations

### Classical Baselines (`src/classical_models.py`)
1. **TF-IDF + Logistic Regression:** Fitted with `class_weight='balanced'`, L2 regularization, and aspect-conditioned text representations:
   `[Aspect Term] [SEP] [Opinion Term] [SEP] [Review Text]`
2. **TF-IDF + Linear SVM:** LinearSVC with balanced weighting and softmax-approximated decision values.
- Serialized checkpoints are saved to `models/saved_checkpoints/`.

### Multilingual Transformer Models (`src/transformer_models.py`)
- Supported architectures: **MuRIL** (`google/muril-base-cased`), **mBERT** (`bert-base-multilingual-cased`), and **XLM-RoBERTa** (`xlm-roberta-base`).
- Aspect-conditioned sequence input: `Aspect: {aspect} | Opinion: {opinion} | Context: {review}`.
- Checkpoints are saved with complete experiment metadata: model identifier, revision, learning rate, batch size, epoch count, duration, and hardware detected (`mps`/`cuda`/`cpu`).

---

## 🔬 8. Reproducing Experiments via CLI

Execute the complete experimental pipeline:

```bash
# Run baseline models and MuRIL transformer on held-out test split
python3 -m src.run_experiments --dataset data/research_reviews_annotated.csv --seed 42 --transformer MuRIL
```

Or run classical baselines only:
```bash
python3 -m src.run_experiments --dataset data/research_reviews_annotated.csv --seed 42 --no_transformers
```

Expected output artifacts written to `results/`:
- `results/baseline_metrics.csv`
- `results/alignment_gap_results.csv`
- `results/per_class_metrics.csv`
- `results/confidence_intervals.csv`
- `results/normalization_comparison.csv`
- `results/annotation_audit.csv`
- `results/data_validation_report.json`
- `results/data_validation_report.txt`
- `results/confusion_matrices/`
- `results/predictions/`
- `results/experiment_configurations/`
- `results/error_analysis_*.csv`

---

## 📐 9. Alignment Gap & 3-Tier Sample Adequacy Standard

The system formalizes a strict 3-tier sample adequacy standard in `src/gap_analysis.py`:
- **Tier A (Mathematical):** Computable whenever $n \ge 1$, but not reported if below descriptive minimum ($n < 2$).
- **Tier B (Descriptive / Exploratory Only):** When $2 \le n < 15$ per subset, the nominal gap is calculated mathematically for debugging, but explicitly labeled **INCONCLUSIVE (Exploratory Only)** with clear sample counts and wide uncertainty warnings.
- **Tier C (Conclusive Evidence):** Requires at least 15 pairs per subset. Confidence intervals are calculated using **Review-Level Cluster Bootstrap** (1,000 resamples), sampling review clusters with replacement to account for intra-review clustering. Significance is evaluated via review-level permutation testing.

---

## 🔤 10. Transliteration Normalization Comparison

A controlled experiment evaluates the effect of Romanized Hindi transliteration normalization:
- **Condition A:** Original text.
- **Condition B:** Normalized text using `TransliterationNormalizer` (repeated character reduction, phonetic spelling canonicalization).
- Evaluates whether normalization improves Macro-F1 or narrows the alignment gap without altering sentiment semantics or translating words to English.

---

## 🐞 11. Error Analysis Engine (`src/error_analysis.py`)

Every prediction error is categorized into the 10 documented research error modes:
1. Aspect extraction failure
2. Opinion extraction failure
3. Aspect-opinion binding failure
4. Transliteration recognition failure
5. Sentiment classification failure
6. Negation handling failure
7. Language-identification failure
8. Annotation disagreement
9. Implicit aspect or opinion
10. Other / unresolved error

Detailed error logs are exported to `results/error_analysis_<model_name>.csv`.

---

## 🚀 12. Launching the Streamlit Application

Launch the 8-page academic research application:

```bash
python3 -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Page Guide:
- **Page 1: Home / Dashboard:** Research objective, live dataset metrics, architecture roster.
- **Page 2: Single Review Analysis:** Interactive ABSA extraction, relation binding, polarity prediction.
- **Page 3: Dataset Explorer:** Validation report summary, inclusion/exclusion counts, category filters.
- **Page 4: Model Comparison:** Comparison of completed experiments (uncomputed models show "Not evaluated").
- **Page 5: Cross-Lingual Gap Analysis:** Gap breakdown, sample counts, unique review counts, conclusive status banners.
- **Page 6: Transliteration Normalization:** Interactive tester and held-out normalization comparison.
- **Page 7: Experiments & Training:** Live training and test evaluation runner.
- **Page 8: About / Methodology:** Theoretical background and literature survey link.

---

## ✅ 13. Running Automated Tests

Execute the complete test suite:
```bash
python3 -m pytest -v
```
All unit tests run offline using synthetic fixtures and do not require external downloads.

---

## ⚠️ 14. Known Limitations & Scientific Honesty

1. **Pilot Sample Size:** The current dataset (N=61 pairs, 40 reviews) provides $N=10$ test records (2 same-language, 4 cross-lingual, 4 uncertain). As reported in `results/alignment_gap_results.csv`, this sample size is **statistically inconclusive** for definitive claims on the alignment gap.
2. **Dominance of English Aspects:** Most e-commerce reviews feature English technical nouns (*"Camera"*, *"Processor"*) paired with Hindi adjectives, resulting in an asymmetry between English $\to$ Hindi and Hindi $\to$ English pairs.
3. **Hardware Considerations:** Transformer fine-tuning automatically utilizes MPS (Apple Silicon) or CUDA; on CPU, epoch count should be kept small ($E=2$).

---

## 🔧 15. Troubleshooting

- **Port in use:** If port 8501 is occupied, launch on a custom port: `streamlit run app.py --server.port 8502`.
- **PyTorch MPS Device Allocation:** If MPS memory warnings occur, specify CPU fallback: `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Package Conflicts:** Ensure `requirements.txt` is installed in a clean virtual environment.
