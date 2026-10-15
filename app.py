"""
Hinglish Product Review Analyzer
Aspect-Based Sentiment Analysis and Cross-Lingual Performance Evaluation
Streamlit Application
"""

import os
import io
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Internal modules
from src.data_validation import validate_absa_dataframe, derive_pair_type, REQUIRED_COLUMNS
from src.data_loader import load_dataset, split_by_review_id
from src.preprocessing import TransliterationNormalizer, NormalizationConfig
from src.language_identifier import SpanLanguageIdentifier
from src.aspect_extractor import AspectExtractor, evaluate_span_extraction
from src.opinion_extractor import OpinionExtractor
from src.relation_extractor import RelationExtractor, evaluate_relation_extraction
from src.sentiment_classifier import LexiconSentimentClassifier
from src.classical_models import ClassicalABSAModel
from src.transformer_models import MultilingualABSAModel, SUPPORTED_TRANSFORMERS, get_optimal_device
from src.evaluation import compute_sentiment_metrics, compute_cross_lingual_gap
from src.gap_analysis import analyze_alignment_gap
from src.error_analysis import run_error_analysis
from src.visualization import (
    plot_model_comparison,
    plot_same_vs_cross_f1,
    plot_alignment_gap,
    plot_normalization_impact,
    plot_sentiment_distribution,
    plot_language_pair_distribution,
    plot_error_categories,
    plot_confusion_matrix_figure,
)

# Set page configuration
st.set_page_config(
    page_title="Hinglish Product Review Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom academic styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .academic-header {
        background: linear-gradient(135deg, #102a43 0%, #1e3d59 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .academic-header h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .academic-header p {
        color: #d9e2ec;
        font-size: 14px;
        margin: 0;
    }
    
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-cross {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-same {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .metric-box {
        background: #f8fafc;
        border-left: 4px solid #1e3d59;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "dataset" not in st.session_state:
        # Load research dataset if available, otherwise fallback to demonstration dataset
        research_path = "data/research_reviews_annotated.csv"
        default_path = "data/sample_reviews.csv"
        if os.path.exists(research_path):
            df, val_res = load_dataset(research_path)
            st.session_state["dataset"] = df
            st.session_state["validation_result"] = val_res
            st.session_state["is_demo_dataset"] = False
            st.session_state["dataset_name"] = "research_reviews_annotated.csv (Pilot Research Dataset)"
        elif os.path.exists(default_path):
            df, val_res = load_dataset(default_path)
            st.session_state["dataset"] = df
            st.session_state["validation_result"] = val_res
            st.session_state["is_demo_dataset"] = True
            st.session_state["dataset_name"] = "sample_reviews.csv (Demo)"
        else:
            st.session_state["dataset"] = None
            st.session_state["validation_result"] = None
            st.session_state["is_demo_dataset"] = False
            st.session_state["dataset_name"] = "None"

    if "models" not in st.session_state:
        models_dict = {
            "Lexicon Baseline": LexiconSentimentClassifier(),
        }
        # Auto-load saved classical checkpoints if available
        for m_type, m_label in [("logistic_regression", "TF-IDF + Logistic Regression"), ("linear_svm", "TF-IDF + Linear SVM")]:
            cp_path = f"models/saved_checkpoints/{m_type}_seed42.joblib"
            if os.path.exists(cp_path):
                try:
                    models_dict[m_label] = ClassicalABSAModel.load(cp_path)
                except Exception:
                    pass
        # Auto-load saved transformer checkpoint if available
        muril_dir = "models/saved_checkpoints/muril"
        if os.path.exists(os.path.join(muril_dir, "model.safetensors")):
            try:
                muril_m = MultilingualABSAModel(model_name="MuRIL")
                muril_m.load_checkpoint(muril_dir)
                models_dict["MuRIL"] = muril_m
            except Exception:
                pass

        st.session_state["models"] = models_dict

    if "experiments" not in st.session_state:
        st.session_state["experiments"] = []

    if "device_name" not in st.session_state:
        device = get_optimal_device()
        st.session_state["device_name"] = str(device)


init_session_state()

# Sidebar Navigation
st.sidebar.markdown("### 🔬 Navigation")
PAGES = [
    "Page 1: Home / Dashboard",
    "Page 2: Single Review Analysis",
    "Page 3: Dataset Explorer",
    "Page 4: Model Comparison",
    "Page 5: Cross-Lingual Gap Analysis",
    "Page 6: Transliteration Normalization",
    "Page 7: Experiments and Results",
    "Page 8: About / Methodology",
]
selected_page = st.sidebar.radio("Select Research View", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ System Status")
st.sidebar.info(f"**Compute Device:** `{st.session_state['device_name']}`\n\n**Active Dataset:** `{st.session_state['dataset_name']}`")

if st.session_state.get("is_demo_dataset"):
    st.sidebar.warning("⚠️ **Notice:** Demonstration dataset active. Results are illustrative.")


# ==============================================================================
# PAGE 1: HOME / DASHBOARD
# ==============================================================================
if selected_page == "Page 1: Home / Dashboard":
    st.markdown("""
    <div class="academic-header">
        <h1>Hinglish Product Review Analyzer</h1>
        <p>Aspect-Based Sentiment Analysis of Code-Mixed Hinglish Product Reviews: Quantifying the Cross-Lingual Aspect–Opinion Alignment Gap</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 Research Objective")
        st.write(
            "This research system analyzes **Hindi-English code-mixed (Hinglish)** product reviews to perform "
            "fine-grained **Aspect-Based Sentiment Analysis (ABSA)**. Beyond coarse document-level classification, "
            "it addresses explicit span extraction, sentiment assignment, and linguistic alignment."
        )

        st.markdown("#### Key Research Questions:")
        st.markdown(
            """
            1. **Span Identification:** How reliably can aspects and opinion expressions be extracted in code-mixed text?
            2. **Cross-Lingual Alignment Gap:** Does model performance degrade when an aspect term and its opinion expression occur across distinct linguistic boundaries (e.g. English Aspect + Hindi Opinion)?
            3. **Transliteration Sensitivity:** Does conservative phonetic and character-flooding normalization bridge vocabulary mismatch and enhance alignment?
            """
        )

        st.info(
            "**Core Metric — Alignment Gap Formula:**\n\n"
            r"$$\text{Alignment Gap} = \text{Macro-F1}_{\text{same-language}} - \text{Macro-F1}_{\text{cross-lingual}}$$"
            "\n\n*A positive value quantifies model performance degradation on cross-lingual pairings.*"
        )

    with col2:
        st.markdown("### 📊 Dataset Overview")
        df = st.session_state.get("dataset")
        val = st.session_state.get("validation_result")

        if df is not None and val is not None and val.is_valid:
            st.metric("Total Aspect-Opinion Pairs", val.stats.get("valid_rows", 0))
            st.metric("Unique Customer Reviews", val.stats.get("unique_reviews", 0))

            pairs = val.stats.get("pair_type_counts", {})
            st.write("**Pair Type Breakdown:**")
            st.write(f"- 🟢 **Same-Language:** {pairs.get('same-language', 0)}")
            st.write(f"- 🟠 **Cross-Lingual:** {pairs.get('cross-lingual', 0)}")
            st.write(f"- ⚪ **Uncertain / Mixed:** {pairs.get('Uncertain', 0)}")
        else:
            st.warning("No validated dataset loaded. Upload a dataset in the Dataset Explorer.")

    st.markdown("---")
    st.markdown("### 📦 Available Architectures & Benchmark Status")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("#### 1. Baseline Models")
        st.write("- **Lexicon Sentiment Baseline** (Transparent polar dictionary)")
        st.write("- **TF-IDF + Logistic Regression** (Group-split on review_id)")
        st.write("- **TF-IDF + Linear SVM** (Balanced class weighting)")

    with m_col2:
        st.markdown("#### 2. Multilingual Transformers")
        st.write("- **mBERT** (`bert-base-multilingual-cased`)")
        st.write("- **XLM-RoBERTa** (`xlm-roberta-base`)")
        st.write("- **MuRIL** (`google/muril-base-cased` for Indian languages)")

    with m_col3:
        st.markdown("#### 3. Extraction Components")
        st.write("- **Span Extractor:** Domain & pattern matching")
        st.write("- **Relation Extractor:** Clause-aware proximity linking")
        st.write("- **Language Identifier:** Romanized, English & Devanagari")


# ==============================================================================
# PAGE 2: SINGLE REVIEW ANALYSIS
# ==============================================================================
elif selected_page == "Page 2: Single Review Analysis":
    st.markdown("""
    <div class="academic-header">
        <h1>Single Review Analysis</h1>
        <p>Interactive Aspect-Based Sentiment Analysis & Language-Pair Alignment</p>
    </div>
    """, unsafe_allow_html=True)

    EXAMPLE_REVIEWS = [
        "Camera quality achhi hai, but battery backup bahut kharab hai.",
        "Phone ka display bohot vibrant hai aur touch response smooth hai.",
        "Sound quality is crystal clear and bass is really powerful.",
        "Is headphone ka build quality bekaar hai par sound kaafi accha hai.",
        "Fabric soft nahi hai aur stitching bhi kharab nikli.",
        "Keemat ke hisab se cheez bahut badiya hai.",
    ]

    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("##### 💡 Example Reviews")
        selected_example = st.selectbox("Pick an illustrative review:", ["Select an example..."] + EXAMPLE_REVIEWS)

    with col_input:
        default_val = selected_example if selected_example != "Select an example..." else EXAMPLE_REVIEWS[0]
        review_input = st.text_area(
            "Enter a Hinglish Product Review:",
            value=default_val,
            height=110,
            help="Enter a Hindi-English code-mixed customer review.",
        )

    col_btn, col_mode = st.columns([2, 2])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Review", type="primary", use_container_width=True)
    with col_mode:
        model_choice = st.selectbox(
            "Sentiment Classification Model:",
            options=["Lexicon Baseline"] + [m for m in st.session_state["models"].keys() if m != "Lexicon Baseline"],
        )

    if analyze_clicked or review_input:
        if not review_input.strip():
            st.error("Please enter a valid review text.")
        else:
            with st.spinner("Executing extraction and alignment pipeline..."):
                asp_extractor = AspectExtractor()
                op_extractor = OpinionExtractor()
                rel_extractor = RelationExtractor()
                lang_id = SpanLanguageIdentifier()

                # Extract spans
                aspects = asp_extractor.extract_aspects(review_input)
                opinions = op_extractor.extract_opinions(review_input)
                relations = rel_extractor.extract_relations(review_input, aspects, opinions)

                st.markdown("### 🔍 Analysis Results")

                if not relations:
                    st.warning(
                        "⚠️ **Extraction Limitation Notice:**\n"
                        "The automatic rule-based extractor did not identify high-confidence aspect-opinion pairs "
                        "in this text. You may utilize annotation-assisted baseline mode below to inspect specific spans."
                    )

                    with st.expander("🛠️ Annotation-Assisted Inspection Mode", expanded=True):
                        st.write("Specify candidate aspect and opinion terms manually to evaluate the model:")
                        man_col1, man_col2 = st.columns(2)
                        with man_col1:
                            manual_asp = st.text_input("Aspect Term:", value="Camera")
                        with man_col2:
                            manual_op = st.text_input("Opinion Term:", value="achha hai")

                        if st.button("Evaluate Specified Spans"):
                            asp_lang_pred = lang_id.identify_span(manual_asp)
                            op_lang_pred = lang_id.identify_span(manual_op)
                            p_type = lang_id.classify_pair_type(asp_lang_pred.predicted_language, op_lang_pred.predicted_language)

                            # Predict sentiment
                            clf = st.session_state["models"].get(model_choice, LexiconSentimentClassifier())
                            if isinstance(clf, LexiconSentimentClassifier):
                                s_res = clf.predict(review_input, opinion_span=manual_op, aspect_span=manual_asp)
                                pred_sent = s_res.predicted_sentiment
                                conf = s_res.confidence
                            else:
                                pred_sent, conf = clf.predict(manual_asp, review_input, manual_op)

                            st.success(
                                f"**Aspect:** {manual_asp} ({asp_lang_pred.predicted_language}) | "
                                f"**Opinion:** {manual_op} ({op_lang_pred.predicted_language}) | "
                                f"**Pair Type:** `{p_type}` | "
                                f"**Sentiment:** `{pred_sent}` (Conf: {conf})"
                            )
                else:
                    results_data = []
                    clf = st.session_state["models"].get(model_choice, LexiconSentimentClassifier())

                    for rel in relations:
                        asp_lang_res = lang_id.identify_span(rel.aspect_text)
                        op_lang_res = lang_id.identify_span(rel.opinion_text)
                        pair_type_label = lang_id.classify_pair_type(asp_lang_res.predicted_language, op_lang_res.predicted_language)

                        # Sentiment prediction
                        if isinstance(clf, LexiconSentimentClassifier):
                            s_res = clf.predict(review_input, opinion_span=rel.opinion_text, aspect_span=rel.aspect_text)
                            sentiment_label = s_res.predicted_sentiment
                            conf_val = s_res.confidence
                            explanation = s_res.explanation
                        else:
                            sentiment_label, conf_val = clf.predict(rel.aspect_text, review_input, rel.opinion_text)
                            explanation = f"Evaluated by {model_choice}"

                        badge_cls = "badge-same" if pair_type_label == "same-language" else "badge-cross" if pair_type_label == "cross-lingual" else "badge-info"

                        results_data.append({
                            "Aspect Term": rel.aspect_text,
                            "Aspect Language": asp_lang_res.predicted_language,
                            "Opinion Expression": rel.opinion_text,
                            "Opinion Language": op_lang_res.predicted_language,
                            "Pair Type": pair_type_label,
                            "Predicted Sentiment": sentiment_label,
                            "Confidence": conf_val,
                            "Diagnostic Note": explanation,
                        })

                    res_df = pd.DataFrame(results_data)
                    st.dataframe(res_df, use_container_width=True)

                    # Summary cards for detected pairs
                    st.markdown("#### 📋 Aspect Breakdown Cards")
                    card_cols = st.columns(len(results_data))
                    for i, r in enumerate(results_data):
                        with card_cols[i % len(card_cols)]:
                            sent_icon = "🟢" if r["Predicted Sentiment"] == "Positive" else "🔴" if r["Predicted Sentiment"] == "Negative" else "🟡"
                            st.markdown(
                                f"""
                                <div class="metric-box">
                                    <strong>{r['Aspect Term']}</strong><br/>
                                    <span style="font-size:13px;color:#475569;">Opinion: <em>"{r['Opinion Expression']}"</em></span><br/>
                                    <span style="font-size:14px;font-weight:600;">{sent_icon} {r['Predicted Sentiment']}</span> 
                                    <small>(Conf: {r['Confidence']})</small><br/>
                                    <small>Aspect Lang: {r['Aspect Language']} | Opinion Lang: {r['Opinion Language']}</small><br/>
                                    <small>Type: <code>{r['Pair Type']}</code></small>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


# ==============================================================================
# PAGE 3: DATASET EXPLORER
# ==============================================================================
elif selected_page == "Page 3: Dataset Explorer":
    st.markdown("""
    <div class="academic-header">
        <h1>Dataset Explorer & Validator</h1>
        <p>Inspect, validate, and explore Aspect-Based Sentiment Analysis datasets</p>
    </div>
    """, unsafe_allow_html=True)

    upload_file = st.file_uploader("Upload an ABSA Dataset CSV:", type=["csv"])

    if upload_file is not None:
        try:
            raw_df = pd.read_csv(upload_file)
            val_res = validate_absa_dataframe(raw_df)

            if val_res.is_valid:
                st.session_state["dataset"] = val_res.cleaned_df
                st.session_state["validation_result"] = val_res
                st.session_state["is_demo_dataset"] = False
                st.session_state["dataset_name"] = upload_file.name
                st.success(f"Successfully uploaded and validated `{upload_file.name}`!")
            else:
                st.error("Validation failed. Please correct the schema and data errors detailed below.")
                st.session_state["validation_result"] = val_res
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    df = st.session_state.get("dataset")
    val_res = st.session_state.get("validation_result")

    if df is not None and val_res is not None:
        # Validation summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", val_res.total_rows)
        c2.metric("Valid ABSA Pairs", val_res.valid_rows)
        c3.metric("Unique Reviews", val_res.stats.get("unique_reviews", 0))
        c4.metric("Categories", len(val_res.stats.get("product_categories", {})))

        if val_res.warnings:
            with st.expander(f"⚠️ Validation Warnings ({len(val_res.warnings)})", expanded=False):
                for w in val_res.warnings:
                    st.write(f"- {w}")

        if val_res.errors:
            with st.expander(f"❌ Validation Errors ({len(val_res.errors)})", expanded=True):
                for err in val_res.errors:
                    st.write(f"- {err}")

        st.markdown("---")
        st.markdown("### 🔍 Dataset Filtering & Interactive Table")

        filt_col1, filt_col2 = st.columns(2)
        categories = ["All"] + sorted(list(df["product_category"].dropna().unique()))
        sentiments = ["All"] + sorted(list(df["sentiment"].dropna().unique()))

        with filt_col1:
            sel_cat = st.selectbox("Filter by Product Category:", categories)
        with filt_col2:
            sel_sent = st.selectbox("Filter by Sentiment:", sentiments)

        filtered_df = df.copy()
        if sel_cat != "All":
            filtered_df = filtered_df[filtered_df["product_category"] == sel_cat]
        if sel_sent != "All":
            filtered_df = filtered_df[filtered_df["sentiment"] == sel_sent]

        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("### 📊 Distribution Analytics")
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.plotly_chart(plot_sentiment_distribution(filtered_df), use_container_width=True)
        with ch_col2:
            st.plotly_chart(plot_language_pair_distribution(filtered_df), use_container_width=True)

        # Download sanitized data
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Cleaned / Filtered CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_absa_dataset.csv",
            mime="text/csv",
        )
    else:
        st.info("No dataset currently loaded. Upload a CSV file or load the sample dataset.")


# ==============================================================================
# PAGE 4: MODEL COMPARISON
# ==============================================================================
elif selected_page == "Page 4: Model Comparison":
    st.markdown("""
    <div class="academic-header">
        <h1>Model Comparison Benchmark</h1>
        <p>Comparative performance analysis across Classical ML and Multilingual Transformer architectures</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("dataset")
    if df is None:
        st.warning("Please load a dataset first.")
    else:
        st.write(
            "This table compares validated models evaluated on the held-out test split. "
            "To uphold scientific integrity, metrics are shown only when computed; uncomputed models display **Not Available**."
        )

        # Check for executed experiment results in results/
        baseline_file = "results/baseline_metrics.csv"
        gap_file = "results/alignment_gap_results.csv"

        if os.path.exists(baseline_file) and os.path.exists(gap_file):
            df_base = pd.read_csv(baseline_file)
            df_gap = pd.read_csv(gap_file)
            merged_results = pd.merge(df_base, df_gap, on="Model", how="outer")

            # Check full model roster
            full_roster = [
                "Lexicon Baseline",
                "TF-IDF + Logistic Regression",
                "TF-IDF + Linear SVM",
                "mBERT",
                "XLM-RoBERTa",
                "MuRIL",
            ]

            display_rows = []
            for m in full_roster:
                match = merged_results[merged_results["Model"] == m]
                if not match.empty:
                    row = match.iloc[0]
                    cat = "Rule/Lexicon" if "Lexicon" in m else "Classical ML" if "TF-IDF" in m else "Multilingual Transformer"
                    display_rows.append({
                        "Model": m,
                        "Architecture": cat,
                        "Accuracy": round(float(row["Accuracy"]), 4) if pd.notna(row["Accuracy"]) else None,
                        "Macro-F1": round(float(row["Macro_F1"]), 4) if pd.notna(row["Macro_F1"]) else None,
                        "Same-Language F1": round(float(row["Same_Language_F1"]), 4) if pd.notna(row["Same_Language_F1"]) else None,
                        "Cross-Lingual F1": round(float(row["Cross_Lingual_F1"]), 4) if pd.notna(row["Cross_Lingual_F1"]) else None,
                        "Alignment Gap (pp)": round(float(row["Alignment_Gap_pp"]), 2) if pd.notna(row["Alignment_Gap_pp"]) else None,
                        "Status": "Evaluated on Held-Out Split",
                    })
                else:
                    cat = "Multilingual Transformer" if m in ["mBERT", "XLM-RoBERTa", "MuRIL"] else "Classical ML"
                    display_rows.append({
                        "Model": m,
                        "Architecture": cat,
                        "Accuracy": None,
                        "Macro-F1": None,
                        "Same-Language F1": None,
                        "Cross-Lingual F1": None,
                        "Alignment Gap (pp)": None,
                        "Status": "Not evaluated",
                    })

            comp_df = pd.DataFrame(display_rows)
            st.dataframe(comp_df, use_container_width=True)

            evaluated_df = comp_df[comp_df["Status"] == "Evaluated on Held-Out Split"]
            if not evaluated_df.empty:
                st.plotly_chart(plot_model_comparison(evaluated_df), use_container_width=True)
                st.plotly_chart(plot_same_vs_cross_f1(evaluated_df), use_container_width=True)
        else:
            st.info("No saved experiment results found in `results/`. Run experiments on Page 7 or execute `python3 -m src.run_experiments`.")


# ==============================================================================
# PAGE 5: CROSS-LINGUAL GAP ANALYSIS
# ==============================================================================
elif selected_page == "Page 5: Cross-Lingual Gap Analysis":
    st.markdown("""
    <div class="academic-header">
        <h1>Cross-Lingual Gap Analysis</h1>
        <p>Quantifying the Cross-Lingual Aspect–Opinion Alignment Gap in Code-Mixed Reviews</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("dataset")
    if df is None:
        st.warning("Please load a dataset first.")
    else:
        st.markdown(
            """
            > **Definition:**
            > $$\text{Alignment Gap} = \text{Macro-F1}_{\text{same-language}} - \text{Macro-F1}_{\text{cross-lingual}}$$
            > A **positive** alignment gap indicates degraded model performance on cross-lingual pairs 
            > relative to same-language pairs.
            """
        )

        model_to_inspect = st.selectbox(
            "Select Evaluated Model for Gap Diagnostics:",
            options=["Lexicon Baseline"] + [m for m in st.session_state["models"].keys() if m != "Lexicon Baseline"],
        )

        m_obj = st.session_state["models"].get(model_to_inspect)
        if isinstance(m_obj, LexiconSentimentClassifier):
            preds = [m_obj.predict(row["review_text"], row["opinion_term"], row["aspect_term"]).predicted_sentiment for _, row in df.iterrows()]
        else:
            preds, _ = m_obj.predict_batch(df)

        eval_df = df.copy()
        eval_df["predicted_sentiment"] = preds

        gap_detail = analyze_alignment_gap(eval_df, model_name=model_to_inspect)

        g_col1, g_col2, g_col3 = st.columns(3)
        g_col1.metric(
            "Same-Language Macro-F1",
            f"{gap_detail.f1_same_lang:.3f}" if gap_detail.f1_same_lang is not None else "N/A",
            f"{gap_detail.n_same} pairs ({gap_detail.unique_reviews_same} reviews)",
        )
        g_col2.metric(
            "Cross-Lingual Macro-F1",
            f"{gap_detail.f1_cross_lang:.3f}" if gap_detail.f1_cross_lang is not None else "N/A",
            f"{gap_detail.n_cross} pairs ({gap_detail.unique_reviews_cross} reviews)",
        )
        gap_display = f"{gap_detail.alignment_gap_pp:+.2f} pp" if gap_detail.alignment_gap_pp is not None else "N/A"
        g_col3.metric("Alignment Gap (Δ F1)", gap_display, "Difference in % points")

        if gap_detail.is_conclusive:
            st.success(f"🔬 **Statistical Status:** Conclusive Evidence ({gap_detail.evidence_level})")
        else:
            st.warning(f"⚠️ **Statistical Status:** Inconclusive (Exploratory Only) — Sample size is below the scientific adequacy threshold (minimum 15 pairs per subset).")

        st.info(f"**Research Finding:** {gap_detail.interpretation}")

        if gap_detail.gap_ci_95:
            st.write(f"**95% Cluster Bootstrap Confidence Interval on Gap:** `[{gap_detail.gap_ci_95[0]:.2f} pp, {gap_detail.gap_ci_95[1]:.2f} pp]`")
        else:
            st.caption("ℹ️ *95% Cluster Bootstrap Confidence Interval not computed: requires at least 4 unique reviews per subset to avoid degenerate sampling variance.*")

        # Linguistic breakdown table
        st.markdown("### 🧩 Subgroup Performance by Language Combination")
        if gap_detail.pair_breakdown:
            breakdown_records = []
            for combo, stats in gap_detail.pair_breakdown.items():
                breakdown_records.append({
                    "Language Direction (Aspect → Opinion)": combo,
                    "Pair Type": stats["pair_type"],
                    "Sample Count": stats["count"],
                    "Macro-F1": stats["f1_macro"],
                    "Accuracy": stats["accuracy"],
                })
            b_df = pd.DataFrame(breakdown_records)
            st.dataframe(b_df, use_container_width=True)

            # Export gap analysis
            b_csv = io.StringIO()
            b_df.to_csv(b_csv, index=False)
            st.download_button(
                "📥 Export Gap Analysis CSV",
                data=b_csv.getvalue(),
                file_name=f"alignment_gap_{model_to_inspect.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )


# ==============================================================================
# PAGE 6: TRANSLITERATION NORMALIZATION
# ==============================================================================
elif selected_page == "Page 6: Transliteration Normalization":
    st.markdown("""
    <div class="academic-header">
        <h1>Transliteration Normalization Pipeline</h1>
        <p>Conservative Normalization: Repeated Character Reduction and Phonetic Canonicalization</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("dataset")

    st.markdown("### ⚙️ Pipeline Configuration")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        reduce_flooding = st.checkbox("Reduce Character Flooding (e.g., 'sooo' → 'soo')", value=True)
    with p_col2:
        canonical_spelling = st.checkbox("Phonetic Spelling Normalization (e.g., 'bohot' → 'bahut')", value=True)
    with p_col3:
        clean_ws = st.checkbox("Collapse Irregular Whitespace", value=True)

    config = NormalizationConfig(
        reduce_repeated_chars=reduce_flooding,
        normalize_phonetic_spelling=canonical_spelling,
        normalize_whitespace=clean_ws,
    )
    normalizer = TransliterationNormalizer(config)

    st.markdown("---")
    st.markdown("### 🧪 Interactive Text Tester")
    test_text = st.text_input("Enter text to test normalization rules:", value="Camera quality bohot achhi hai aur display ekdum vibranttt hai!")
    norm_test_res = normalizer.normalize(test_text)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("**Original Text:**")
        st.code(norm_test_res.original_text, language="text")
    with t_col2:
        st.markdown("**Normalized Text:**")
        st.code(norm_test_res.normalized_text, language="text")

    if norm_test_res.modifications:
        st.write("**Modifications Detected:**", norm_test_res.modifications)

    if df is not None:
        st.markdown("---")
        st.markdown("### 🔬 Corpus Impact Analysis (Condition A vs. Condition B)")
        if st.button("Run Corpus Normalization Benchmark", type="primary"):
            norm_df, stats = normalizer.normalize_dataframe(df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Corpus Reviews", stats["total_records"])
            c2.metric("Modified Reviews", stats["changed_records"])
            c3.metric("Change Ratio", f"{stats['change_percentage']}%")

            # Comparative performance evaluation
            lex = LexiconSentimentClassifier()
            # Cond A: Original
            preds_orig = [lex.predict(row["review_text"], row["opinion_term"], row["aspect_term"]).predicted_sentiment for _, row in df.iterrows()]
            m_orig = compute_sentiment_metrics(df["sentiment"].tolist(), preds_orig, compute_ci=False)

            # Cond B: Normalized
            preds_norm = [lex.predict(row["normalized_review_text"], row["normalized_opinion_term"], row["normalized_aspect_term"]).predicted_sentiment for _, row in norm_df.iterrows()]
            m_norm = compute_sentiment_metrics(norm_df["sentiment"].tolist(), preds_norm, compute_ci=False)

            diff_f1 = round((m_norm.macro_f1 - m_orig.macro_f1) * 100.0, 2)

            comp_stats = pd.DataFrame([{
                "Model": "Lexicon Baseline",
                "Original F1": m_orig.macro_f1,
                "Normalized F1": m_norm.macro_f1,
                "Delta (pp)": f"{diff_f1:+.2f} pp",
            }])
            st.dataframe(comp_stats, use_container_width=True)
            st.plotly_chart(plot_normalization_impact(comp_stats), use_container_width=True)

    norm_comp_file = "results/normalization_comparison.csv"
    if os.path.exists(norm_comp_file):
        st.markdown("---")
        st.markdown("### 📊 Held-Out Test Normalization Benchmark (Condition A vs. Condition B)")
        st.write("Controlled evaluation across all models on held-out test split:")
        norm_csv_df = pd.read_csv(norm_comp_file)
        st.dataframe(norm_csv_df, use_container_width=True)
        st.plotly_chart(plot_normalization_impact(norm_csv_df.rename(columns={"Original_Macro_F1": "Original F1", "Normalized_Macro_F1": "Normalized F1"})), use_container_width=True)


# ==============================================================================
# PAGE 7: EXPERIMENTS AND RESULTS
# ==============================================================================
elif selected_page == "Page 7: Experiments and Results":
    st.markdown("""
    <div class="academic-header">
        <h1>Experiments & Model Training</h1>
        <p>Train and evaluate Classical ML and Multilingual Transformer models on held-out test splits</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("dataset")
    if df is None:
        st.warning("Please load a dataset first.")
    else:
        st.markdown("### 🧪 Experiment Setup")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            model_type = st.selectbox(
                "Select Model to Train & Evaluate:",
                ["TF-IDF + Logistic Regression", "TF-IDF + Linear SVM", "mBERT", "XLM-RoBERTa", "MuRIL"],
            )

        with exp_col2:
            random_seed = st.number_input("Random Seed (for strict reproducibility):", value=42, step=1)

        st.markdown("#### Data Splitting Strategy (Strict `review_id` Grouping)")
        st.write(
            "To prevent data leakage, aspect-opinion pairs originating from the same customer review "
            "are grouped together and placed strictly in either Train, Validation, or Test."
        )

        splits = split_by_review_id(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_seed)
        s1, s2, s3 = st.columns(3)
        s1.metric("Train Set", f"{len(splits.train_df)} pairs", f"{splits.train_reviews} reviews")
        s2.metric("Validation Set", f"{len(splits.val_df)} pairs", f"{splits.val_reviews} reviews")
        s3.metric("Test Set (Held-Out)", f"{len(splits.test_df)} pairs", f"{splits.test_reviews} reviews")

        if st.button("🚀 Run Training and Evaluation", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            if "TF-IDF" in model_type:
                status_text.text(f"Training {model_type}...")
                m_type = "logistic_regression" if "Logistic" in model_type else "linear_svm"
                model = ClassicalABSAModel(model_type=m_type, random_state=random_seed)
                model.fit(splits.train_df)
                progress_bar.progress(60)

                status_text.text("Evaluating on held-out test set...")
                test_preds, test_confs = model.predict_batch(splits.test_df)
                progress_bar.progress(100)
                status_text.text("Evaluation complete!")

                # Store model
                st.session_state["models"][model_type] = model

            else:
                # Multilingual Transformer
                status_text.text(f"Initializing {model_type} ({SUPPORTED_TRANSFORMERS[model_type]})...")
                trans_model = MultilingualABSAModel(model_name=model_type)

                def update_prog(p):
                    pct = int(min(90, (p.epoch / p.total_epochs) * 80))
                    progress_bar.progress(pct)
                    status_text.text(f"Epoch {p.epoch}/{p.total_epochs} - Train Loss: {p.train_loss:.4f} | Val F1: {p.val_macro_f1:.4f}")

                train_res = trans_model.fit(
                    splits.train_df,
                    val_df=splits.val_df,
                    epochs=2,
                    batch_size=4,
                    progress_callback=update_prog,
                )
                progress_bar.progress(90)
                status_text.text("Evaluating transformer on held-out test set...")
                test_preds, test_confs = trans_model.predict_batch(splits.test_df)
                progress_bar.progress(100)
                status_text.text("Transformer fine-tuning and evaluation complete!")

                st.session_state["models"][model_type] = trans_model

            # Calculate and display metrics
            eval_metrics = compute_sentiment_metrics(
                splits.test_df["sentiment"].tolist(),
                test_preds,
                compute_ci=True,
            )

            st.markdown("### 🏆 Held-Out Test Evaluation Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Accuracy", f"{eval_metrics.accuracy:.4f}")
            r2.metric("Macro-F1", f"{eval_metrics.macro_f1:.4f}")
            r3.metric("Macro-Precision", f"{eval_metrics.macro_precision:.4f}")
            r4.metric("Macro-Recall", f"{eval_metrics.macro_recall:.4f}")

            if eval_metrics.bootstrap_ci_f1:
                st.info(f"**95% Bootstrap Confidence Interval on Macro-F1:** `[{eval_metrics.bootstrap_ci_f1[0]:.4f}, {eval_metrics.bootstrap_ci_f1[1]:.4f}]`")

            # Confusion Matrix
            cm_fig = plot_confusion_matrix_figure(eval_metrics.confusion_mat, eval_metrics.labels)
            st.plotly_chart(cm_fig, use_container_width=True)

            # Per-class metrics
            per_class_df = pd.DataFrame(eval_metrics.per_class).T
            st.dataframe(per_class_df, use_container_width=True)

            # Error Analysis for this test run
            test_df_eval = splits.test_df.copy()
            test_df_eval["predicted_sentiment"] = test_preds
            err_records, err_df, err_stats = run_error_analysis(test_df_eval, model_name=model_type)

            st.markdown("### ⚠️ Diagnosed Prediction Errors on Test Partition")
            st.write(f"Total Errors: **{err_stats['total_errors']}** (Error Rate: **{err_stats['overall_error_rate_pct']}%**)")
            if not err_df.empty:
                st.dataframe(err_df, use_container_width=True)
                st.plotly_chart(plot_error_categories(err_stats["category_counts"]), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📁 Saved Research Experiment Runs & Configurations")
        from src.run_experiments import run_experiment_suite

        if st.button("⚡ Run Full Automated Experiment Suite (All Baselines & Save Artifacts)", use_container_width=True):
            with st.spinner("Executing full baseline suite, gap analysis, and normalization benchmarks..."):
                suite_res = run_experiment_suite(dataset_path=st.session_state["dataset_name"] if os.path.exists(st.session_state["dataset_name"]) else "data/research_reviews_annotated.csv", random_seed=int(random_seed))
                st.success("Full experiment suite executed successfully! Artifacts written to `results/`.")
                st.rerun()

        cfg_dir = "results/experiment_configurations"
        if os.path.exists(cfg_dir):
            cfg_files = sorted([f for f in os.listdir(cfg_dir) if f.endswith(".json")], reverse=True)
            if cfg_files:
                selected_cfg = st.selectbox("Inspect Saved Experiment Configuration:", cfg_files)
                with open(os.path.join(cfg_dir, selected_cfg), "r") as f:
                    cfg_data = json.load(f)
                st.json(cfg_data)



# ==============================================================================
# PAGE 8: ABOUT / METHODOLOGY
# ==============================================================================
elif selected_page == "Page 8: About / Methodology":
    st.markdown("""
    <div class="academic-header">
        <h1>Research Methodology & System Documentation</h1>
        <p>Scientific Foundations, Experimental Design, and Reproducibility Specifications</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        r"""
        ### 1. Research Problem
        Code-mixed languages—specifically **Hinglish** (Hindi written in Roman or Devanagari script intermingled 
        with English)—pose significant challenges for conventional NLP models.
        In product reviews, customers frequently refer to product features (**aspect terms**) in English (e.g. *Camera quality*, *battery life*) 
        while expressing sentiment (**opinion expressions**) in Hindi (e.g. *achhi hai*, *bahut kharab nikla*).

        ### 2. The Alignment Gap Hypothesis
        Standard multilingual encoders (mBERT, XLM-RoBERTa, MuRIL) are pre-trained predominantly on monolingual texts or 
        parallel translation corpora. When aspect and opinion occur across language boundaries:
        $$\\text{Alignment Gap} = \\text{F1}_{\\text{same-language}} - \\text{F1}_{\\text{cross-lingual}}$$
        We hypothesize that **cross-lingual aspect-opinion associations suffer from an attention misalignment**, 
        yielding lower F1 scores compared to monolingual associations.

        ### 3. Pipeline Decoupling & Modular Architecture
        To prevent confounding extraction errors with sentiment classification:
        - **Aspect Term Extraction (ATE):** Sequence span extraction of entity features.
        - **Opinion Term Extraction (OTE):** Subjective evaluative span detection.
        - **Aspect-Opinion Relation Extraction (AORE):** Clause-aware link prediction.
        - **Aspect Sentiment Classification (ASC):** Polarity prediction conditioned on the aspect and opinion spans.
        - **Span Language Identification:** Disambiguates English, Romanized Hindi, and Devanagari Hindi at the token/span level.

        ### 4. Transliteration Normalization
        Romanized Hindi exhibits high orthographic variance (e.g., *bohot* vs. *bahut*, *accha* vs. *achha*). 
        Our conservative normalizer applies:
        1. Character flooding reduction ($k \\ge 3$ repeated characters reduced to 2).
        2. Conservative canonical dictionary mapping (no translation; semantic preservation).
        3. Offset preservation to maintain alignment with gold annotations.

        ### 5. Documented Failure Modes (Error Analysis Taxonomy)
        The system classifies observed model prediction errors into 10 structured diagnostic categories:
        1. **Aspect extraction failure** (Missing or boundary-shifted target entity span)
        2. **Opinion extraction failure** (Missing evaluative phrase or intensifier)
        3. **Aspect-opinion binding failure** (Multi-aspect review cross-clause context bleed)
        4. **Transliteration recognition failure** (Non-standard phonetic Romanized spelling variance)
        5. **Sentiment classification failure** (Standard polarity classification misclassification)
        6. **Negation handling failure** (Failure to invert polarity when negated by 'nahi', 'not')
        7. **Language-identification failure** (Disagreement between predicted and true span language)
        8. **Annotation disagreement** (Conflict between human label and deterministic derivation)
        9. **Implicit aspect or opinion** (Latent sentiment lacking explicit anchor token)
        10. **Other / unresolved** (Complex discourse or idiomatic exceptions)

        ### 6. Research Integrity & Literature Foundations
        - **Literature Survey:** For detailed analysis of 10 seminal peer-reviewed research papers, see `docs/LITERATURE_SURVEY.md`.
        - **No Synthetic Numbers:** Evaluation metrics and charts are strictly computed from actual model inferences.
        - **Grouped Splitting:** Splitting is performed strictly by `review_id` to prevent identical customer reviews from leaking into test partitions.
        - **Uncertainty Reporting:** Review-level clustered bootstrap confidence intervals are computed only when sample sizes are statistically sufficient ($N \ge 10$).
        """
    )
