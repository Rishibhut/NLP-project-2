"""
Language identification module for code-mixed Hinglish at the span and token level.
Distinguishes English, Romanized Hindi, Devanagari Hindi, Mixed, and Uncertain spans.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
import unicodedata

# High-frequency Romanized Hindi markers, copulas, pronouns, and postpositions
HINDI_ROMANIZED_LEXICON = {
    # Postpositions and pronouns
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "pe", "tak", "liye",
    "yeh", "ye", "voh", "woh", "wo", "is", "us", "in", "un", "mera", "meri",
    "mere", "tera", "teri", "tere", "apna", "apni", "apne", "hum", "aap", "tum",
    # Copulas and auxiliaries
    "hai", "hain", "he", "hn", "tha", "thi", "the", "thay", "hoga", "hogi",
    "hoge", "raha", "rahi", "rahe", "karta", "karti", "karte", "gaya", "gayi", "gaye",
    # Conjunctions and discourse markers
    "aur", "lekin", "par", "magar", "kyunki", "isliye", "bhi", "to", "toh",
    "ya", "agar", "warna", "matlab", "yani",
    # Common Hindi adjectives and adverbs
    "bahut", "bohot", "bhot", "kam", "zyada", "jyada", "accha", "acchi", "acche",
    "achha", "achhi", "achhe", "kharab", "khraab", "bekaar", "bekar", "ghatiya",
    "badhiya", "badiya", "sasta", "mehnga", "thoda", "bilkul", "ekdum", "ekdam",
    "sahi", "galat", "asli", "nakli", "saaf", "seedha", "tez", "dheema", "naram",
    # Negation
    "nahi", "nhi", "nahin", "na", "mat", "kabhi",
    # Common Hindi nouns/verbs in reviews
    "cheez", "dikhne", "dekhne", "aawaz", "keemat", "dhoop", "phat", "lagta",
    "lagti", "lagte", "chhota", "bada", "mili", "mila", "mile", "karo", "lo", "liya"
}

# High-frequency English function words and review vocabulary
ENGLISH_LEXICON = {
    "the", "a", "an", "is", "are", "was", "were", "and", "but", "or", "because",
    "very", "too", "much", "more", "good", "bad", "great", "excellent", "poor",
    "awesome", "terrible", "horrible", "best", "worst", "quality", "product",
    "phone", "display", "battery", "camera", "sound", "speaker", "charger",
    "screen", "delivery", "fast", "slow", "smooth", "clear", "loud", "soft",
    "light", "heavy", "comfortable", "perfect", "amazing", "durable", "cheap",
    "expensive", "support", "service", "backup", "issue", "problem", "works",
    "working", "features", "look", "design", "fit", "size", "color", "buttons",
    "keys", "switches", "material", "finish", "zoom", "connectivity", "touch",
    "response", "cushioning", "bass", "audio", "mic", "clarity", "value", "money"
}

# Ambiguous tokens that could be English or Romanized Hindi depending on context
AMBIGUOUS_TOKENS = {"is", "to", "no", "so", "in", "me", "us", "he", "par"}


@dataclass
class LanguagePrediction:
    span_text: str
    predicted_language: str  # 'English', 'Hindi', 'Mixed', 'Uncertain'
    confidence: float
    hindi_token_count: int
    english_token_count: int
    devanagari_token_count: int
    uncertain_token_count: int
    notes: str


class SpanLanguageIdentifier:
    """
    Identifies language of text spans with sensitivity to Romanized Hindi,
    English, Devanagari script, and code-mixed spans.
    """

    def __init__(self):
        self.hindi_lex = HINDI_ROMANIZED_LEXICON
        self.eng_lex = ENGLISH_LEXICON
        self.ambiguous = AMBIGUOUS_TOKENS

    def has_devanagari(self, text: str) -> bool:
        """Checks for Devanagari Unicode range U+0900 to U+097F."""
        return any("\u0900" <= ch <= "\u097F" for ch in text)

    def identify_token(self, token: str) -> str:
        """Identifies language of a single token."""
        token_clean = re.sub(r"[^\w]", "", token).lower()
        if not token_clean:
            return "Uncertain"

        if self.has_devanagari(token):
            return "Hindi"

        if token_clean in self.ambiguous:
            return "Ambiguous"

        is_hindi = token_clean in self.hindi_lex
        is_english = token_clean in self.eng_lex

        if is_hindi and not is_english:
            return "Hindi"
        if is_english and not is_hindi:
            return "English"

        # Heuristic morphological analysis
        # Hindi suffixes: -wala, -wali, -wale, -pan, -iyo
        if any(token_clean.endswith(sfx) for sfx in ["wala", "wali", "wale", "kar", "karne"]):
            return "Hindi"

        # English suffixes: -ing, -tion, -ed, -ment, -able, -ive, -ly
        if any(token_clean.endswith(sfx) for sfx in ["ing", "tion", "ment", "able", "ive", "ly"]):
            return "English"

        return "Uncertain"

    def identify_span(self, span: str) -> LanguagePrediction:
        """
        Classifies language of a span (aspect term or opinion expression).
        """
        if not span or not span.strip():
            return LanguagePrediction(
                span_text=span,
                predicted_language="Uncertain",
                confidence=0.0,
                hindi_token_count=0,
                english_token_count=0,
                devanagari_token_count=0,
                uncertain_token_count=0,
                notes="Empty span",
            )

        # Devanagari script presence check
        if self.has_devanagari(span):
            return LanguagePrediction(
                span_text=span,
                predicted_language="Hindi",
                confidence=0.99,
                hindi_token_count=1,
                english_token_count=0,
                devanagari_token_count=1,
                uncertain_token_count=0,
                notes="Devanagari script detected",
            )

        tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", span)
        if not tokens:
            return LanguagePrediction(
                span_text=span,
                predicted_language="Uncertain",
                confidence=0.0,
                hindi_token_count=0,
                english_token_count=0,
                devanagari_token_count=0,
                uncertain_token_count=0,
                notes="No alphanumeric tokens found",
            )

        h_count = 0
        e_count = 0
        u_count = 0

        for tok in tokens:
            t_lang = self.identify_token(tok)
            if t_lang == "Hindi":
                h_count += 1
            elif t_lang == "English":
                e_count += 1
            else:
                u_count += 1

        total_informative = h_count + e_count

        if total_informative == 0:
            # All tokens are ambiguous or unknown
            return LanguagePrediction(
                span_text=span,
                predicted_language="Uncertain",
                confidence=0.30,
                hindi_token_count=0,
                english_token_count=0,
                devanagari_token_count=0,
                uncertain_token_count=u_count,
                notes="Tokens ambiguous or not in lexical inventory",
            )

        h_ratio = h_count / total_informative
        e_ratio = e_count / total_informative

        if h_count > 0 and e_count > 0:
            # Intra-span code-mixing
            return LanguagePrediction(
                span_text=span,
                predicted_language="Mixed",
                confidence=round(max(h_ratio, e_ratio), 2),
                hindi_token_count=h_count,
                english_token_count=e_count,
                devanagari_token_count=0,
                uncertain_token_count=u_count,
                notes="Both Hindi and English tokens present in span",
            )

        if h_count > e_count:
            conf = min(0.95, 0.65 + 0.1 * h_count)
            return LanguagePrediction(
                span_text=span,
                predicted_language="Hindi",
                confidence=round(conf, 2),
                hindi_token_count=h_count,
                english_token_count=e_count,
                devanagari_token_count=0,
                uncertain_token_count=u_count,
                notes="Romanized Hindi tokens identified",
            )
        else:
            conf = min(0.95, 0.65 + 0.1 * e_count)
            return LanguagePrediction(
                span_text=span,
                predicted_language="English",
                confidence=round(conf, 2),
                hindi_token_count=h_count,
                english_token_count=e_count,
                devanagari_token_count=0,
                uncertain_token_count=u_count,
                notes="English tokens identified",
            )

    def classify_pair_type(self, aspect_lang: str, opinion_lang: str) -> str:
        """
        Classifies whether an aspect-opinion association is same-language or cross-lingual.
        """
        if aspect_lang in {"Uncertain", "Mixed", "Other"} or opinion_lang in {"Uncertain", "Mixed", "Other"}:
            return "Uncertain"
        if aspect_lang == opinion_lang:
            return "same-language"
        return "cross-lingual"
