# Hinglish Product Review Analyzer

### *Aspect-Based Sentiment Analysis of Code-Mixed Hinglish Product Reviews: Quantifying the Cross-Lingual Aspect–Opinion Alignment Gap*

---

## 📌 1. Project Overview & Research Objective

In multilingual e-commerce markets, consumer reviews frequently exhibit **code-mixing**, predominantly between Hindi and English (**Hinglish**). 

A critical phenomenon in Hinglish product reviews is that consumers often mention the product attribute (**Aspect Term**) in English (e.g., *"Camera quality"*, *"Battery backup"*, *"Screen resolution"*) while expressing subjective sentiment (**Opinion Expression**) in Romanized Hindi (e.g., *"achhi hai"*, *"bahut kharab nikla"*, *"ekdum bekaar"*). 

### Central Research Question:
> *"Does aspect-opinion language alignment affect aspect-level sentiment classification performance on Hinglish product reviews, and can transliteration normalization reduce the observed performance difference?"*

The system quantifies the **Cross-Lingual Aspect–Opinion Alignment Gap**:

$$\text{Alignment Gap} = \text{Macro-F1}_{\text{same-language}} - \text{Macro-F1}_{\text{cross-lingual}}$$

A positive value indicates that NLP models experience degraded performance when the sentiment descriptor crosses linguistic boundaries relative to the target entity.

---

## 🏗️ 2. System Architecture

The application decouples the ABSA pipeline into modular, inspectable components:

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
   - battery backup (34..48)                     - bahut kharab hai (49..65)
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
   - Pair Type: cross-lingual                    (Lexicon / TF-IDF / Transformers)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                   [Cross-Lingual Alignment Gap Analyzer]
                   Gap = F1(same-language) - F1(cross-lingual)
```

---

## 📂 3. Directory Structure

```
hinglish-review-analyzer/
│
├── app.py                          # 8-Page Streamlit Academic Dashboard
├── requirements.txt                # Pinned dependencies
├── README.md                       # Comprehensive setup, methodology, and execution guide
├── .gitignore                      # Git exclusion rules
│
├── data/
│   ├── README.md                   # Dataset guidelines & integrity rules
│   ├── sample_reviews.csv          # Illustrative demonstration dataset
│   ├── research_reviews_annotated.csv # Prepared research benchmark dataset with character offsets
│   ├── dataset_template.csv        # Blank CSV template for annotation workflows
│   └── provenance_metadata.json    # Provenance tracking (source, license, collection method)
│
├── docs/
│   └── LITERATURE_SURVEY.md        # Literature survey of 10 peer-reviewed papers across 13 fields
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data_loader.py              # Data loader with review_id group-aware splitting (no leakage)
│   ├── data_validation.py          # Strict schema validation, span containment, label checks
│   ├── dataset_preparation.py      # Research dataset ingestion, offset derivation, provenance
│   ├── preprocessing.py            # Conservative transliteration normalizer & character flooding
│   ├── language_identifier.py      # Span-level Romanized Hindi, English & Devanagari detection
│   ├── aspect_extractor.py         # Aspect Term Extraction (ATE) and exact span evaluation
│   ├── opinion_extractor.py        # Opinion Term Extraction (OTE) and polar pattern matcher
│   ├── relation_extractor.py       # Aspect-Opinion Relation Extraction (AORE) & clause segmenter
│   ├── sentiment_classifier.py     # Transparent lexicon baseline with negation handling
│   ├── classical_models.py         # TF-IDF + Logistic Regression & Linear Support Vector Machine
│   ├── transformer_models.py       # mBERT, XLM-RoBERTa, MuRIL with lazy loading & GPU/MPS detection
│   ├── evaluation.py               # Multiclass F1, accuracy, review-level clustered bootstrap CIs
│   ├── gap_analysis.py             # Alignment gap quantification & linguistic combo breakdowns
│   ├── error_analysis.py           # 10-category failure mode diagnostic engine
│   ├── run_experiments.py          # Standalone reproducible experiment CLI orchestrator
│   └── visualization.py            # Interactive Plotly & Matplotlib/Seaborn visualization suite
│
├── tests/
│   ├── test_data_validation.py     # Schema, span containment, label validation tests
│   ├── test_preprocessing.py       # Normalization & character flooding reduction tests
│   ├── test_language_identifier.py # Language classification & pair type determination tests
│   ├── test_evaluation.py          # Multiclass metrics, span F1, and relation F1 tests
│   ├── test_gap_analysis.py        # Alignment gap computation & sample size guard tests
│   ├── test_leakage_and_splits.py  # Review ID data leakage prevention tests
│   ├── test_complex_pipeline_cases.py # Contrast words (lekin/but), negation, missing checkpoints
│   └── test_models.py              # Lexicon, ML baselines, and span/relation extractor tests
│
├── models/
│   ├── README.md                   # Checkpoint policies & lazy-loading documentation
│   └── saved_checkpoints/          # Serialized model weights (git-ignored)
│
└── results/
    ├── README.md                   # Results documentation
    ├── baseline_metrics.csv        # Evaluated model accuracies, macro-F1, and sample counts
    ├── alignment_gap_results.csv   # Same-language vs. cross-lingual F1 and alignment gap (pp)
    ├── per_class_metrics.csv       # Precision, recall, F1, support per sentiment class
    ├── normalization_comparison.csv# Condition A (Original) vs. Condition B (Normalized) delta
    ├── validation_report.md        # Dataset audit report with pair-type disagreement flags
    ├── validation_report.json      # Structured validation report JSON
    ├── confusion_matrices/         # Confusion matrix outputs
    ├── predictions/                # Raw test partition model prediction logs
    └── experiment_configurations/  # Traceable configuration JSON files with hashes and seeds
```

---

## 💻 4. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14+
- `pip` package manager

### A. Clone and Virtual Environment Setup

**macOS / Linux:**
```bash
git clone <repo-url>
cd "NLP project 2"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd "NLP project 2"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🧪 5. Reproducing Experiments via CLI

You can reproduce the entire experimental evaluation directly from terminal without launching the UI:

### Step 1: Prepare and Validate the Research Dataset
```bash
python3 -m src.dataset_preparation
```
This generates `data/research_reviews_annotated.csv`, computes character offsets (`aspect_start_char`, `opinion_start_char`), and writes `results/validation_report.md`.

### Step 2: Execute Baseline Models, Alignment Gap, and Normalization Suite
```bash
python3 -m src.run_experiments --dataset data/research_reviews_annotated.csv --seed 42
```
This trains the TF-IDF models on the train split, evaluates on the held-out test split, computes review-level clustered bootstrap intervals, runs 10-category error analysis, and populates:
- `results/baseline_metrics.csv`
- `results/alignment_gap_results.csv`
- `results/per_class_metrics.csv`
- `results/normalization_comparison.csv`
- `results/experiment_configurations/`

### Step 3: Run the Complete Automated Test Suite (42 Tests)
```bash
python3 -m pytest -v
```

---

## 🚀 6. Launching the Streamlit Application

Run the Streamlit server from the project root:

```bash
python3 -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📖 7. User Guide: Page-by-Page Overview

### Page 1: Home / Dashboard
- Displays research title, central question, and core alignment gap formula.
- Shows live dataset metrics (record counts, unique reviews, pair-type breakdown).
- Summarizes available baseline and transformer model architectures.

### Page 2: Single Review Analysis
- Interactive input for arbitrary Hinglish customer reviews.
- Provides sample test reviews via dropdown.
- Automatically extracts aspect terms, opinion expressions, polarities, span languages, and pair types.
- Provides fallback **Annotation-Assisted Inspection Mode** if heuristic extractors find low-confidence boundaries.

### Page 3: Dataset Explorer & Validator
- Upload custom ABSA CSV datasets.
- Enforces strict validation checks:
  1. Required columns presence.
  2. Span containment in original text.
  3. Valid categorical values (`Positive`, `Negative`, `Neutral`, `Hindi`, `English`, `same-language`, `cross-lingual`).
  4. Character offset derivation.
- Interactive filtering by Product Category and Sentiment Polarity.
- Export cleaned/sanitized data via the download button.

### Page 4: Model Comparison
- Compares evaluated models on held-out test splits.
- Reports Accuracy, Macro-F1, Same-Language F1, Cross-Lingual F1, and the Alignment Gap.
- Transparently displays **Not evaluated** for uncomputed models to prevent fabricated results.

### Page 5: Cross-Lingual Gap Analysis
- Deep-dive diagnostics into the Cross-Lingual Aspect–Opinion Alignment Gap:
  $$\text{Alignment Gap} = \text{Macro-F1}_{\text{same-language}} - \text{Macro-F1}_{\text{cross-lingual}}$$
- Displays review-level clustered bootstrap confidence intervals when sample sizes are statistically sufficient ($N \ge 10$).
- Subgroup performance breakdown by directional language combinations:
  - English (Aspect) $\to$ Hindi (Opinion)
  - English (Aspect) $\to$ English (Opinion)
  - Hindi (Aspect) $\to$ Hindi (Opinion)
  - Hindi (Aspect) $\to$ English (Opinion)
- One-click CSV export of gap analysis records.

### Page 6: Transliteration Normalization
- Configurable normalization pipeline:
  - **Repeated Character Reduction:** reduces flooding (e.g., *"acccchhaaa"* $\to$ *"accha"*).
  - **Phonetic Canonicalization:** maps non-standard Romanized spelling variants conservatively (*"bohot"* $\to$ *"bahut"*, *"khraab"* $\to$ *"kharab"*).
  - Preserves original reviews; never translates Hindi to English.
- Corpus Impact Benchmark: tests Condition A (Original) vs. Condition B (Normalized) and measures performance $\Delta$.

### Page 7: Experiments and Model Training
- Execute full automated experiment suite or train individual models on custom seeds.
- **Leakage Prevention:** Strictly splits partitions by `review_id` so that multiple aspects from the same review never leak across train and test sets.
- Displays live epoch progress, confusion matrices, per-class metrics, diagnosed prediction errors, and saved experiment configuration JSONs.

### Page 8: About / Methodology
- Detailed academic documentation explaining the research questions, pipeline decoupling, 10 error categories, statistical validation, and ethical safeguards.
- Direct link to `docs/LITERATURE_SURVEY.md`.

---

## 🔬 8. Hardware & Device Requirements

The application automatically checks for hardware acceleration:
1. **NVIDIA GPUs (CUDA):** Detected automatically via `torch.cuda.is_available()`.
2. **Apple Silicon (MPS):** Detected automatically on macOS via `torch.backends.mps.is_available()`.
3. **CPU:** Fallback default on all platforms.

Transformer fine-tuning can be run on CPU or GPU; the UI uses compact batch sizes ($B=4$) and sequence lengths ($L=128$) to accommodate local workstation testing.

---

## ⚖️ 9. Research Integrity Principles

1. **No Fabricated Metrics:** Uncomputed models strictly report "Not evaluated" with clear reasons.
2. **No Data Leakage:** Stratified grouped splitting ensures reviews are kept isolated in either train, validation, or test partitions.
3. **Conservative Normalization:** Transliteration normalizers never alter sentence semantics, delete negation tokens, or translate words into English.
4. **Empirical Evidence:** The sample dataset `data/sample_reviews.csv` is explicitly marked as an illustrative benchmark for testing the pipeline and replicating the alignment gap methodology.
