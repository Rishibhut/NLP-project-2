"""
Unit tests for baseline models:
- Lexicon Sentiment Classifier
- Classical TF-IDF Logistic Regression & LinearSVC
- Aspect Extractor
- Opinion Extractor
- Relation Extractor
"""

import pytest
import pandas as pd
from src.sentiment_classifier import LexiconSentimentClassifier
from src.classical_models import ClassicalABSAModel
from src.aspect_extractor import AspectExtractor
from src.opinion_extractor import OpinionExtractor
from src.relation_extractor import RelationExtractor


def test_lexicon_classifier_positive():
    clf = LexiconSentimentClassifier()
    res = clf.predict("Camera quality achhi hai", opinion_span="achhi hai")
    assert res.predicted_sentiment == "Positive"
    assert res.score > 0
    assert res.lexicon_coverage > 0


def test_lexicon_classifier_negative_with_negation():
    clf = LexiconSentimentClassifier()
    res = clf.predict("Display soft nahi hai", opinion_span="soft nahi hai")
    assert res.predicted_sentiment == "Negative"
    assert len(res.negations_found) > 0


def test_aspect_extractor():
    extractor = AspectExtractor()
    text = "Camera quality achhi hai, but battery backup bahut kharab hai."
    spans = extractor.extract_aspects(text)
    span_texts = [s.text.lower() for s in spans]
    assert "camera quality" in span_texts
    assert "battery backup" in span_texts


def test_opinion_extractor():
    extractor = OpinionExtractor()
    text = "Camera quality achhi hai, but battery backup bahut kharab hai."
    spans = extractor.extract_opinions(text)
    span_texts = [s.text.lower() for s in spans]
    assert any("achhi hai" in st for st in span_texts)
    assert any("bahut kharab hai" in st for st in span_texts)


def test_relation_extractor():
    asp_ext = AspectExtractor()
    op_ext = OpinionExtractor()
    rel_ext = RelationExtractor()

    text = "Camera quality achhi hai, but battery backup bahut kharab hai."
    asps = asp_ext.extract_aspects(text)
    ops = op_ext.extract_opinions(text)
    rels = rel_ext.extract_relations(text, asps, ops)

    assert len(rels) >= 2
    # Ensure camera quality links to positive/achhi, battery backup links to kharab
    pairs = [(r.aspect_text.lower(), r.opinion_text.lower()) for r in rels]
    assert any("camera quality" in a and "achhi" in o for a, o in pairs)
    assert any("battery backup" in a and "kharab" in o for a, o in pairs)


def test_classical_model_fit_predict():
    data = {
        "review_id": [1, 2, 3, 4, 5, 6],
        "review_text": [
            "Good camera quality",
            "Very bad battery backup",
            "Average screen display",
            "Superb camera quality",
            "Terrible sound",
            "Decent fit",
        ],
        "aspect_term": ["camera quality", "battery backup", "screen display", "camera quality", "sound", "fit"],
        "opinion_term": ["Good", "Very bad", "Average", "Superb", "Terrible", "Decent"],
        "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"],
    }
    df = pd.DataFrame(data)

    model = ClassicalABSAModel(model_type="logistic_regression", random_state=42)
    model.fit(df)

    pred, conf = model.predict(aspect="camera quality", review="Awesome camera quality", opinion="Awesome")
    assert pred in ["Positive", "Negative", "Neutral"]
    assert 0.0 <= conf <= 1.0
