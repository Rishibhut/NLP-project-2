# Hinglish Product Review Analyzer

### *Aspect-Based Sentiment Analysis of Code-Mixed Hinglish Product Reviews: Quantifying the Cross-Lingual Aspect–Opinion Alignment Gap*

---

## 📌 1. Project Overview & Research Objective

In multilingual societies such as India, consumer reviews on e-commerce platforms (Amazon, Flipkart, Myntra) frequently exhibit **code-mixing**, predominantly between Hindi and English (**Hinglish**). 

A critical phenomenon in Hinglish product reviews is that consumers often mention the product attribute or entity (**Aspect Term**) in English (e.g., *"Camera quality"*, *"Battery backup"*, *"Screen resolution"*) while expressing subjective sentiment (**Opinion Expression**) in Romanized Hindi (e.g., *"achhi hai"*, *"bahut kharab nikla"*, *"ekdum bekaar"*). 

This research-oriented application implements an end-to-end NLP pipeline to investigate and quantify the **Cross-Lingual Aspect–Opinion Alignment Gap**:

$$\text{Alignment Gap} = \text{Macro-F1}_{\text{same-language}} - \text{Macro-F1}_{\text{cross-lingual}}$$

A positive value indicates that standard NLP models experience degraded performance when the sentiment signal crosses linguistic boundaries within the same sentence.

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
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive research & setup documentation
├── .gitignore                      # Git exclusion rules
│
├── data/
│   ├── README.md                   # Dataset guidelines & integrity rules
│   ├── sample_reviews.csv          # Illustrative Hinglish ABSA benchmark dataset
│   └── dataset_template.csv        # Blank template for user annotations
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data_loader.py              # Data loader with review_id group-aware splitting
│   ├── data_validation.py          # Schema validation, span containment, label checks
│   ├── preprocessing.py            # Conservative transliteration normalizer
│   ├── language_identifier.py      # Span-level Romanized/English/Devanagari identifier
│   ├── aspect_extractor.py         # Aspect Term Extraction & span metrics
│   ├── opinion_extractor.py        # Opinion Term Extraction & polar patterns
│   ├── relation_extractor.py       # Aspect-Opinion Relation Extraction & link metrics
│   ├── sentiment_classifier.py     # Lexicon-based baseline with negation handling
│   ├── classical_models.py         # TF-IDF + Logistic Regression & Linear SVM
│   ├── transformer_models.py       # mBERT, XLM-RoBERTa, MuRIL fine-tuning & inference
│   ├── evaluation.py               # Multiclass F1, accuracy, bootstrap 95% CIs
│   ├── gap_analysis.py             # Alignment gap quantification & subgroup analysis
│   ├── error_analysis.py           # Automated failure mode diagnostic engine
│   └── visualization.py            # Interactive Plotly & Matplotlib/Seaborn figures
│
├── tests/
│   ├── test_data_validation.py     # Schema, containment, and label tests
│   ├── test_preprocessing.py       # Normalization & character flooding tests
│   ├── test_language_identifier.py # Language classification & pair type tests
│   ├── test_evaluation.py          # Multiclass metrics & span F1 tests
│   ├── test_gap_analysis.py        # Alignment gap computation tests
│   └── test_models.py              # Lexicon, ML baselines, and extractor tests
│
├── models/
│   ├── README.md                   # Model architecture notes & checkpoint policies
│   └── saved_checkpoints/          # Serialized weights (git-ignored)
│
└── results/
    └── README.md                   # Results logs & exported CSV files
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

## 🚀 5. Launching the Streamlit Application

Run the Streamlit server from the project root:

```bash
python3 -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📖 6. User Guide: Page-by-Page Overview

### Page 1: Home / Dashboard
- Displays research title, objectives, and core alignment gap formula.
- Shows live dataset metrics (record counts, unique reviews, pair-type breakdown).
- Summarizes available baseline and transformer model architectures.

### Page 2: Single Review Analysis
- Interactive input for arbitrary Hinglish customer reviews.
- Provides sample test reviews via dropdown.
- Automatically extracts aspect terms, opinion expressions, polarity, span languages, and pair types.
- Provides fallback **Annotation-Assisted Inspection Mode** if heuristic extractors find low-confidence boundaries.

### Page 3: Dataset Explorer & Validator
- Upload custom ABSA CSV datasets.
- Enforces strict validation checks:
  1. Required columns presence.
  2. Span containment in original text.
  3. Valid categorical values (`Positive`, `Negative`, `Neutral`, `Hindi`, `English`, `same-language`, `cross-lingual`).
- Interactive filtering by Product Category and Sentiment Polarity.
- Export cleaned/sanitized data via the download button.

### Page 4: Model Comparison
- Compares evaluated models on held-out test splits.
- Reports Accuracy, Macro-F1, Same-Language F1, Cross-Lingual F1, and the Alignment Gap.
- Transparently displays **Not Trained / Not Available** for uncomputed models to prevent fabricated results.

### Page 5: Cross-Lingual Gap Analysis
- Deep-dive diagnostics into the Cross-Lingual Aspect–Opinion Alignment Gap.
- Displays 95% Bootstrap Confidence Intervals for the gap.
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
- Trains Classical ML models (Logistic Regression, Linear Support Vector Machine) or fine-tunes Multilingual Transformers (**mBERT**, **XLM-RoBERTa**, **MuRIL**).
- **Leakage Prevention:** Strictly splits partitions by `review_id` so that multiple aspects from the same review never leak across train and test sets.
- Displays live epoch progress, confusion matrices, per-class metrics, and diagnosed prediction errors.

### Page 8: About / Methodology
- Detailed academic documentation of the research questions, pipeline decoupling, statistical validation, and ethical safeguards.

---

## 🧪 7. Running Unit Tests

The test suite covers schema validation, span containment, normalization rules, language identification, alignment gap calculations, and model inference.

Execute via pytest:

```bash
python3 -m pytest -v
```

All 33 unit tests execute in under 1 second.

---

## 🔬 8. Hardware & Device Requirements

The application automatically checks for hardware acceleration:
1. **NVIDIA GPUs (CUDA):** Detected automatically via `torch.cuda.is_available()`.
2. **Apple Silicon (MPS):** Detected automatically on macOS via `torch.backends.mps.is_available()`.
3. **CPU:** Fallback default on all platforms.

Transformer fine-tuning can be run on CPU or GPU; the UI uses compact batch sizes ($B=4$) and sequence lengths ($L=128$) to accommodate local workstation testing.

---

## ⚖️ 9. Research Integrity Principles

1. **No Fabricated Metrics:** Uncomputed models strictly report "Not Available" with clear reasons.
2. **No Data Leakage:** Stratified grouped splitting ensures reviews are kept isolated in either train, validation, or test partitions.
3. **Conservative Normalization:** Transliteration normalizers never alter sentence semantics, delete negation tokens, or translate words into English.
4. **Empirical Evidence:** The sample dataset `data/sample_reviews.csv` is explicitly marked as an illustrative benchmark for testing the pipeline and replicating the alignment gap methodology.
