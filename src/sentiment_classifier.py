"""
Lexicon-based sentiment baseline for Aspect-Based Sentiment Analysis.
Features transparent polarity scoring, conservative negation handling,
and explicit lexicon coverage reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import re

# Curated polar sentiment lexicon with polarity weights
HINGLISH_ENGLISH_POLAR_LEXICON: Dict[str, float] = {
    # Strong Positives (+1.0)
    "good": 1.0, "great": 1.0, "excellent": 1.0, "awesome": 1.0, "amazing": 1.0,
    "perfect": 1.0, "love": 1.0, "best": 1.0, "fast": 0.8, "smooth": 0.8,
    "vibrant": 0.8, "comfortable": 0.9, "durable": 0.9, "clear": 0.8, "loud": 0.6,
    "responsive": 0.8, "crisp": 0.8, "neat": 0.7, "timely": 0.7, "safe": 0.8,
    "simple": 0.7, "attractive": 0.8, "top": 0.7, "super": 0.9,
    "accha": 1.0, "acchi": 1.0, "acche": 1.0, "achha": 1.0, "achhi": 1.0, "achhe": 1.0,
    "badhiya": 1.0, "badiya": 1.0, "shaandar": 1.0, "shandar": 1.0, "laajawab": 1.0,
    "lajawab": 1.0, "mast": 0.9, "zabardast": 1.0, "sahi": 0.8, "badiyaa": 1.0,
    # Weak Positives / Neutral Context (0.2)
    "theek": 0.2, "thik": 0.2, "okay": 0.1, "ok": 0.1, "average": 0.0, "ordinary": 0.0,
    # Negatives (-1.0)
    "bad": -1.0, "poor": -1.0, "terrible": -1.0, "horrible": -1.0, "worst": -1.0,
    "slow": -0.8, "useless": -1.0, "noisy": -0.8, "loose": -0.7, "stiff": -0.6,
    "cheap": -0.7, "weak": -0.8, "annoying": -0.9, "blur": -0.8, "damage": -0.9,
    "kharab": -1.0, "khraab": -1.0, "bekaar": -1.0, "bekar": -1.0, "ghatiya": -1.0,
    "ganda": -0.9, "bakwaas": -1.0, "bakwas": -1.0, "phat": -0.8, "garam": -0.5,
    "static": -0.6, "disconnect": -0.8, "chhota": -0.3, "tight": -0.5, "lag": -0.7
}

# Negation operators
NEGATION_WORDS: Set[str] = {
    "nahi", "nhi", "nahin", "na", "mat", "not", "never", "no", "without", "kabhi"
}

# Intensifiers
INTENSIFIERS: Dict[str, float] = {
    "bahut": 1.3, "bohot": 1.3, "bhot": 1.3, "kaafi": 1.2, "bilkul": 1.4,
    "ekdum": 1.4, "very": 1.3, "really": 1.3, "extremely": 1.5, "super": 1.4,
    "thoda": 0.7, "slightly": 0.7
}


@dataclass
class SentimentResult:
    predicted_sentiment: str  # 'Positive', 'Negative', 'Neutral'
    score: float
    confidence: float
    lexicon_coverage: float
    polar_terms_found: List[str] = field(default_factory=list)
    negations_found: List[str] = field(default_factory=list)
    explanation: str = ""


class LexiconSentimentClassifier:
    """
    Transparent lexicon-based baseline sentiment classifier.
    Handles negation flipping and reports lexical coverage.
    """

    def __init__(self, lexicon: Optional[Dict[str, float]] = None):
        self.lexicon = lexicon or HINGLISH_ENGLISH_POLAR_LEXICON
        self.negations = NEGATION_WORDS
        self.intensifiers = INTENSIFIERS

    def predict(self, text: str, opinion_span: Optional[str] = None, aspect_span: Optional[str] = None) -> SentimentResult:
        """
        Predicts aspect-level sentiment.
        If opinion_span is provided, focuses heavily on the opinion expression.
        """
        # Determine analysis target: opinion_span has highest relevance, otherwise sentence/aspect window
        target_text = opinion_span if (opinion_span and opinion_span.strip()) else text
        if not target_text or not target_text.strip():
            return SentimentResult(
                predicted_sentiment="Neutral",
                score=0.0,
                confidence=0.33,
                lexicon_coverage=0.0,
                explanation="Target text was empty; default to Neutral.",
            )

        tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", target_text.lower())
        if not tokens:
            return SentimentResult(
                predicted_sentiment="Neutral",
                score=0.0,
                confidence=0.33,
                lexicon_coverage=0.0,
                explanation="No tokens found in target.",
            )

        polar_found = []
        negations_found = []
        scores: List[float] = []
        matched_tokens_count = 0

        for i, tok in enumerate(tokens):
            if tok in self.negations:
                negations_found.append(tok)

            if tok in self.lexicon:
                base_weight = self.lexicon[tok]
                matched_tokens_count += 1
                polar_found.append(f"{tok}({base_weight})")

                # Check preceding window for intensifiers and negations (up to 3 tokens back)
                multiplier = 1.0
                is_negated = False

                start_win = max(0, i - 3)
                window_tokens = tokens[start_win:i]

                for win_tok in window_tokens:
                    if win_tok in self.intensifiers:
                        multiplier *= self.intensifiers[win_tok]
                    if win_tok in self.negations:
                        is_negated = True

                effective_score = base_weight * multiplier
                if is_negated:
                    # Negation flips positive to negative, or neutralizes negative
                    effective_score = -0.8 if effective_score > 0 else 0.3
                    polar_found.append(f"[NEGATED by '{window_tokens}']")

                scores.append(effective_score)

        lexicon_coverage = round(matched_tokens_count / max(1, len(tokens)), 2)

        if not scores:
            # Check if review has general negation (e.g., "nahi hai")
            has_general_neg = any(tok in self.negations for tok in tokens)
            if has_general_neg:
                return SentimentResult(
                    predicted_sentiment="Negative",
                    score=-0.5,
                    confidence=0.60,
                    lexicon_coverage=lexicon_coverage,
                    polar_terms_found=polar_found,
                    negations_found=negations_found,
                    explanation="No explicit polar term matched, but negation found.",
                )
            return SentimentResult(
                predicted_sentiment="Neutral",
                score=0.0,
                confidence=0.50,
                lexicon_coverage=0.0,
                explanation="No polar terms matched in lexicon; classified as Neutral baseline.",
            )

        total_score = sum(scores)
        mean_score = total_score / len(scores)

        if mean_score > 0.15:
            sentiment = "Positive"
            confidence = min(0.95, 0.60 + 0.15 * len(scores))
        elif mean_score < -0.15:
            sentiment = "Negative"
            confidence = min(0.95, 0.60 + 0.15 * len(scores))
        else:
            sentiment = "Neutral"
            confidence = 0.65

        explanation = f"Mean score: {mean_score:.2f} based on: {', '.join(polar_found)}"

        return SentimentResult(
            predicted_sentiment=sentiment,
            score=round(mean_score, 3),
            confidence=round(confidence, 2),
            lexicon_coverage=lexicon_coverage,
            polar_terms_found=polar_found,
            negations_found=negations_found,
            explanation=explanation,
        )
