"""
Classical machine learning baselines for Aspect Sentiment Classification:
- TF-IDF + Logistic Regression
- TF-IDF + Linear Support Vector Machine (LinearSVC)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib


def create_absa_feature_text(aspect: str, review: str, opinion: Optional[str] = None) -> str:
    """
    Creates an aspect-conditioned text representation for TF-IDF featurization.
    Example: 'Camera quality [SEP] achhi hai [SEP] Camera quality achhi hai, but battery...'
    """
    aspect_str = str(aspect or "").strip()
    review_str = str(review or "").strip()
    op_str = str(opinion or "").strip()
    if op_str:
        return f"{aspect_str} [SEP] {op_str} [SEP] {review_str}"
    return f"{aspect_str} [SEP] {review_str}"


class ClassicalABSAModel:
    """
    Wrapper for Scikit-learn based TF-IDF ABSA classifiers.
    """

    def __init__(self, model_type: str = "logistic_regression", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.is_trained = False
        self.classes_ = ["Negative", "Neutral", "Positive"]

        # Configurable TF-IDF Pipeline
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000,
            token_pattern=r"(?u)\b\w+\b",
        )

        if model_type == "logistic_regression":
            classifier = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
        elif model_type == "linear_svm":
            classifier = LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=random_state,
                dual="auto",
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", classifier),
        ])

    def fit(self, train_df: pd.DataFrame) -> "ClassicalABSAModel":
        """
        Trains model on labelled DataFrame containing 'aspect_term', 'review_text', 'sentiment'.
        """
        X = [
            create_absa_feature_text(row["aspect_term"], row["review_text"], row.get("opinion_term"))
            for _, row in train_df.iterrows()
        ]
        y = train_df["sentiment"].values

        self.pipeline.fit(X, y)
        self.classes_ = list(self.pipeline.classes_)
        self.is_trained = True
        return self

    def predict(self, aspect: str, review: str, opinion: Optional[str] = None) -> Tuple[str, float]:
        """
        Predicts sentiment for a single aspect-review instance.
        Returns (predicted_label, confidence).
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        feature_text = create_absa_feature_text(aspect, review, opinion)
        pred_label = self.pipeline.predict([feature_text])[0]

        # Calculate confidence score
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = self.pipeline.predict_proba([feature_text])[0]
            confidence = float(np.max(proba))
        else:
            # LinearSVC decision function
            decision = self.pipeline.decision_function([feature_text])[0]
            if decision.ndim == 1 and len(decision) > 1:
                # Softmax approximation of decision values
                exp_d = np.exp(decision - np.max(decision))
                probs = exp_d / np.sum(exp_d)
                confidence = float(np.max(probs))
            else:
                confidence = 0.75

        return pred_label, round(confidence, 2)

    def predict_batch(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        Generates predictions for a dataframe.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        X = [
            create_absa_feature_text(row["aspect_term"], row["review_text"], row.get("opinion_term"))
            for _, row in df.iterrows()
        ]
        preds = self.pipeline.predict(X).tolist()

        clf = self.pipeline.named_steps["clf"]
        confs = []
        if hasattr(clf, "predict_proba"):
            probs = self.pipeline.predict_proba(X)
            confs = [round(float(np.max(p)), 2) for p in probs]
        else:
            # LinearSVC decision function
            decisions = self.pipeline.decision_function(X)
            for d in decisions:
                if d.ndim == 1 and len(d) > 1:
                    exp_d = np.exp(d - np.max(d))
                    probs = exp_d / np.sum(exp_d)
                    confs.append(round(float(np.max(probs)), 2))
                else:
                    confs.append(0.75)

        return preds, confs

    def save(self, filepath: str) -> None:
        """Saves model pipeline to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClassicalABSAModel":
        """Loads model pipeline from disk."""
        return joblib.load(filepath)
