"""
Reproducible Experiment Runner for Hinglish ABSA and Cross-Lingual Gap Analysis.
Executes baseline models, normalization comparisons, alignment gap quantification,
error analysis, and saves traceable machine-readable artifacts into results/.
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from src.data_loader import load_dataset, split_by_review_id, verify_no_leakage, get_split_distributions
from src.preprocessing import TransliterationNormalizer, NormalizationConfig
from src.sentiment_classifier import LexiconSentimentClassifier
from src.classical_models import ClassicalABSAModel
from src.evaluation import compute_sentiment_metrics, compute_cross_lingual_gap
from src.gap_analysis import analyze_alignment_gap
from src.error_analysis import run_error_analysis


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file for experiment reproducibility tracking."""
    if not os.path.exists(filepath):
        return "NOT_FOUND"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def run_experiment_suite(
    dataset_path: str = "data/research_reviews_annotated.csv",
    output_dir: str = "results",
    random_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    include_transformers: bool = False,
    transformer_name: str = "mBERT",
) -> Dict[str, Any]:
    """
    Executes the entire research evaluation pipeline and outputs all artifacts into results/.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "experiment_configurations"), exist_ok=True)
    os.makedirs("models/saved_checkpoints", exist_ok=True)

    # 1. Load and validate dataset
    if not os.path.exists(dataset_path):
        # Fallback to sample_reviews.csv if research dataset is not yet created
        dataset_path = "data/sample_reviews.csv"

    df, val_res = load_dataset(dataset_path)
    if not val_res.is_valid:
        raise ValueError(f"Dataset validation failed: {val_res.errors}")

    dataset_hash = compute_file_sha256(dataset_path)

    # 2. Group-aware split on review_id (Strict leakage prevention)
    splits = split_by_review_id(
        df=val_res.cleaned_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_seed,
    )
    verify_no_leakage(splits)
    split_info = get_split_distributions(splits)

    # 3. Transliteration Normalization Condition
    normalizer = TransliterationNormalizer(NormalizationConfig())
    train_df_norm, _ = normalizer.normalize_dataframe(splits.train_df)
    test_df_norm, norm_stats = normalizer.normalize_dataframe(splits.test_df)

    # Tables to collect
    baseline_records = []
    alignment_gap_records = []
    per_class_records = []
    normalization_records = []

    models_to_evaluate = [
        ("Lexicon Baseline", "lexicon", None),
        ("TF-IDF + Logistic Regression", "logistic_regression", None),
        ("TF-IDF + Linear SVM", "linear_svm", None),
    ]

    for model_name, model_type, _ in models_to_evaluate:
        exp_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}"

        # Train model if ML baseline
        if model_type == "lexicon":
            clf = LexiconSentimentClassifier()
            # Predictions on Condition A (Original Text)
            preds = [clf.predict(row["review_text"], row["opinion_term"], row["aspect_term"]).predicted_sentiment for _, row in splits.test_df.iterrows()]
            # Predictions on Condition B (Normalized Text)
            preds_norm = [clf.predict(row["normalized_review_text"], row["normalized_opinion_term"], row["normalized_aspect_term"]).predicted_sentiment for _, row in test_df_norm.iterrows()]
        else:
            clf = ClassicalABSAModel(model_type=model_type, random_state=random_seed)
            clf.fit(splits.train_df)
            preds, _ = clf.predict_batch(splits.test_df)

            # Fit on normalized train for Condition B
            clf_norm = ClassicalABSAModel(model_type=model_type, random_state=random_seed)
            clf_norm.fit(train_df_norm)
            preds_norm, _ = clf_norm.predict_batch(test_df_norm)

            # Save checkpoint
            model_save_path = f"models/saved_checkpoints/{model_type}_seed{random_seed}.joblib"
            clf.save(model_save_path)

        # Condition A evaluation
        eval_df = splits.test_df.copy()
        eval_df["predicted_sentiment"] = preds

        # Metrics
        overall_m = compute_sentiment_metrics(eval_df["sentiment"].tolist(), preds, compute_ci=True)
        gap_detail = analyze_alignment_gap(eval_df, model_name=model_name, min_samples=2)

        # Condition B metrics
        eval_df_norm = test_df_norm.copy()
        eval_df_norm["predicted_sentiment"] = preds_norm
        overall_m_norm = compute_sentiment_metrics(eval_df_norm["sentiment"].tolist(), preds_norm, compute_ci=False)
        gap_detail_norm = analyze_alignment_gap(eval_df_norm, model_name=model_name, min_samples=2)

        # Save predictions log
        pred_csv = os.path.join(output_dir, "predictions", f"pred_{model_type}.csv")
        eval_df.to_csv(pred_csv, index=False)

        # Save confusion matrix
        cm_json = os.path.join(output_dir, "confusion_matrices", f"cm_{model_type}.json")
        with open(cm_json, "w") as f:
            json.dump({
                "labels": overall_m.labels,
                "confusion_matrix": overall_m.confusion_mat,
            }, f, indent=2)

        # Run error analysis
        _, _, err_stats = run_error_analysis(
            df=eval_df,
            model_name=model_name,
            output_dir=output_dir,
        )

        # Record baseline metrics
        baseline_records.append({
            "Experiment_ID": exp_id,
            "Model": model_name,
            "Accuracy": overall_m.accuracy,
            "Macro_Precision": overall_m.macro_precision,
            "Macro_Recall": overall_m.macro_recall,
            "Macro_F1": overall_m.macro_f1,
            "Macro_F1_CI_95": f"[{overall_m.bootstrap_ci_f1[0]}, {overall_m.bootstrap_ci_f1[1]}]" if overall_m.bootstrap_ci_f1 else "N/A",
            "Test_Samples": len(splits.test_df),
        })

        # Record alignment gap
        ci_str = f"[{gap_detail.gap_ci_95[0]}, {gap_detail.gap_ci_95[1]}]" if gap_detail.gap_ci_95 else "N/A"
        alignment_gap_records.append({
            "Model": model_name,
            "Same_Language_F1": gap_detail.f1_same_lang,
            "Cross_Lingual_F1": gap_detail.f1_cross_lang,
            "Alignment_Gap_pp": gap_detail.alignment_gap_pp,
            "Gap_95_CI": ci_str,
            "Same_Lang_Count": gap_detail.n_same,
            "Cross_Ling_Count": gap_detail.n_cross,
            "Interpretation": gap_detail.interpretation,
        })

        # Record per-class metrics
        for cls_name, cls_m in overall_m.per_class.items():
            per_class_records.append({
                "Model": model_name,
                "Class": cls_name,
                "Precision": cls_m["precision"],
                "Recall": cls_m["recall"],
                "F1": cls_m["f1"],
                "Support": cls_m["support"],
            })

        # Record normalization comparison
        delta_f1_pp = round((overall_m_norm.macro_f1 - overall_m.macro_f1) * 100.0, 2)
        delta_gap_pp = (
            round((gap_detail_norm.alignment_gap_pp - gap_detail.alignment_gap_pp), 2)
            if (gap_detail_norm.alignment_gap_pp is not None and gap_detail.alignment_gap_pp is not None)
            else None
        )

        normalization_records.append({
            "Model": model_name,
            "Original_Macro_F1": overall_m.macro_f1,
            "Normalized_Macro_F1": overall_m_norm.macro_f1,
            "Delta_F1_pp": delta_f1_pp,
            "Original_Alignment_Gap_pp": gap_detail.alignment_gap_pp,
            "Normalized_Alignment_Gap_pp": gap_detail_norm.alignment_gap_pp,
            "Delta_Gap_pp": delta_gap_pp,
            "Modified_Test_Reviews": norm_stats["changed_records"],
        })

        # Save experiment config JSON
        cfg_path = os.path.join(output_dir, "experiment_configurations", f"{exp_id}.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "experiment_id": exp_id,
                "timestamp": datetime.now().isoformat(),
                "dataset_path": dataset_path,
                "dataset_sha256": dataset_hash,
                "model_name": model_name,
                "random_seed": random_seed,
                "train_test_split": {"train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio},
                "split_distributions": split_info,
                "metrics_condition_a": {
                    "accuracy": overall_m.accuracy,
                    "macro_f1": overall_m.macro_f1,
                    "same_lang_f1": gap_detail.f1_same_lang,
                    "cross_lang_f1": gap_detail.f1_cross_lang,
                    "alignment_gap_pp": gap_detail.alignment_gap_pp,
                },
                "metrics_condition_b_normalized": {
                    "accuracy": overall_m_norm.accuracy,
                    "macro_f1": overall_m_norm.macro_f1,
                    "same_lang_f1": gap_detail_norm.f1_same_lang,
                    "cross_lang_f1": gap_detail_norm.f1_cross_lang,
                    "alignment_gap_pp": gap_detail_norm.alignment_gap_pp,
                },
                "error_analysis_summary": err_stats,
            }, f, indent=2)

    # 4. Optional Transformer fine-tuning
    if include_transformers:
        from src.transformer_models import MultilingualABSAModel
        exp_id_t = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{transformer_name}"
        t_model = MultilingualABSAModel(model_name=transformer_name)
        t_model.fit(
            splits.train_df,
            val_df=splits.val_df,
            epochs=2,
            batch_size=4,
            save_dir=f"models/saved_checkpoints/{transformer_name.lower()}",
        )
        t_preds, _ = t_model.predict_batch(splits.test_df)
        t_df = splits.test_df.copy()
        t_df["predicted_sentiment"] = t_preds
        t_m = compute_sentiment_metrics(t_df["sentiment"].tolist(), t_preds, compute_ci=True)
        t_gap = analyze_alignment_gap(t_df, model_name=transformer_name, min_samples=2)

        baseline_records.append({
            "Experiment_ID": exp_id_t,
            "Model": transformer_name,
            "Accuracy": t_m.accuracy,
            "Macro_Precision": t_m.macro_precision,
            "Macro_Recall": t_m.macro_recall,
            "Macro_F1": t_m.macro_f1,
            "Macro_F1_CI_95": f"[{t_m.bootstrap_ci_f1[0]}, {t_m.bootstrap_ci_f1[1]}]" if t_m.bootstrap_ci_f1 else "N/A",
            "Test_Samples": len(splits.test_df),
        })
        alignment_gap_records.append({
            "Model": transformer_name,
            "Same_Language_F1": t_gap.f1_same_lang,
            "Cross_Lingual_F1": t_gap.f1_cross_lang,
            "Alignment_Gap_pp": t_gap.alignment_gap_pp,
            "Gap_95_CI": f"[{t_gap.gap_ci_95[0]}, {t_gap.gap_ci_95[1]}]" if t_gap.gap_ci_95 else "N/A",
            "Same_Lang_Count": t_gap.n_same,
            "Cross_Ling_Count": t_gap.n_cross,
            "Interpretation": t_gap.interpretation,
        })

    # Save master CSV tables
    df_baselines = pd.DataFrame(baseline_records)
    df_baselines.to_csv(os.path.join(output_dir, "baseline_metrics.csv"), index=False)

    df_gap = pd.DataFrame(alignment_gap_records)
    df_gap.to_csv(os.path.join(output_dir, "alignment_gap_results.csv"), index=False)

    df_per_class = pd.DataFrame(per_class_records)
    df_per_class.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)

    df_norm_res = pd.DataFrame(normalization_records)
    df_norm_res.to_csv(os.path.join(output_dir, "normalization_comparison.csv"), index=False)

    return {
        "baseline_metrics": df_baselines.to_dict(orient="records"),
        "alignment_gaps": df_gap.to_dict(orient="records"),
        "normalization_comparison": df_norm_res.to_dict(orient="records"),
        "split_summary": split_info,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hinglish ABSA Experiments")
    parser.add_argument("--dataset", default="data/research_reviews_annotated.csv", help="Dataset path")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include_transformers", action="store_true", help="Include transformer fine-tuning")
    args = parser.parse_args()

    print(f"Executing experiment suite on '{args.dataset}' with seed {args.seed}...")
    res = run_experiment_suite(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        random_seed=args.seed,
        include_transformers=args.include_transformers,
    )
    print("Experiments completed successfully! Results written to:", args.output_dir)
