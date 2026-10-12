"""
Preprocessing and Transliteration Normalization module.
Implements a conservative, configurable normalization pipeline for Hinglish text.
Preserves original strings, maintains character alignment, and provides detailed audit statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import re
import pandas as pd

# Canonical Hinglish phonetic/spelling mapping dictionary
# Strictly preserves word semantics and never translates Hindi into English.
CANONICAL_TRANSLITERATION_MAP: Dict[str, str] = {
    # Positive descriptors
    "achha": "accha",
    "achhi": "acchi",
    "achhe": "acche",
    "bohot": "bahut",
    "bohoot": "bahut",
    "bhot": "bahut",
    "badiya": "badhiya",
    "bdhiya": "badhiya",
    "shandar": "shaandar",
    "shandaar": "shaandar",
    "lajawab": "laajawab",
    "mast": "zabardast",
    # Negative descriptors
    "khraab": "kharab",
    "kharb": "kharab",
    "bekar": "bekaar",
    "bkaar": "bekaar",
    "ghatya": "ghatiya",
    "ghatia": "ghatiya",
    "bakwas": "bakwaas",
    # Modifiers and auxiliaries
    "zyada": "zyada",
    "jyada": "zyada",
    "thoda": "thoda",
    "thoda-sa": "thoda",
    "bilkul": "bilkul",
    "blkl": "bilkul",
    "ekdum": "ekdam",
    "ekdm": "ekdam",
    "h": "hai",
    "hn": "hain",
    "thaa": "tha",
    "thii": "thi",
    "thee": "the",
    "nahi": "nahi",
    "nhi": "nahi",
    "nahin": "nahi",
    "ni": "nahi",
}


@dataclass
class NormalizationConfig:
    reduce_repeated_chars: bool = True
    normalize_phonetic_spelling: bool = True
    normalize_whitespace: bool = True
    clean_punctuation: bool = False  # False by default to preserve negation and sentence bounds
    max_repeated_chars: int = 2


@dataclass
class NormalizationResult:
    original_text: str
    normalized_text: str
    is_modified: bool
    modifications: List[Tuple[str, str]] = field(default_factory=list)


class TransliterationNormalizer:
    """
    Configurable transliteration normalization engine for Hindi-English code-mixed reviews.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        self.canonical_map = CANONICAL_TRANSLITERATION_MAP

    def reduce_character_flooding(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Reduces elongated repeated characters (e.g., 'acccchhaaa' -> 'accha', 'soooo' -> 'soo').
        Preserves double letters that are valid in Romanized Hindi or English.
        """
        mods = []

        def repl(match: re.Match) -> str:
            original = match.group(0)
            char = original[0]
            # Replace runs of 3 or more with up to max_repeated_chars (usually 2)
            replaced = char * min(len(original), self.config.max_repeated_chars)
            if replaced != original:
                mods.append((original, replaced))
            return replaced

        # Regex for any character repeated 3 or more times
        pattern = re.compile(r"([a-zA-Z])\1{2,}", re.IGNORECASE)
        reduced_text = pattern.sub(repl, text)
        return reduced_text, mods

    def normalize_tokens(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Normalizes known phonetic spelling variants using the conservative dictionary.
        """
        mods = []

        def token_replacer(match: re.Match) -> str:
            token = match.group(0)
            token_lower = token.lower()
            if token_lower in self.canonical_map:
                canonical = self.canonical_map[token_lower]
                if canonical != token_lower:
                    # Preserve titlecase if original was titlecase
                    if token.istitle():
                        out = canonical.capitalize()
                    elif token.isupper():
                        out = canonical.upper()
                    else:
                        out = canonical
                    mods.append((token, out))
                    return out
            return token

        # Word boundary regex
        word_pattern = re.compile(r"\b[a-zA-Z\-]+\b")
        normalized_text = word_pattern.sub(token_replacer, text)
        return normalized_text, mods

    def normalize(self, text: str) -> NormalizationResult:
        """
        Normalizes a single text string according to the active configuration.
        """
        if not text or not isinstance(text, str):
            return NormalizationResult(original_text=str(text or ""), normalized_text=str(text or ""), is_modified=False)

        current = text
        all_mods = []

        # 1. Whitespace normalization
        if self.config.normalize_whitespace:
            cleaned_ws = re.sub(r"[ \t]+", " ", current).strip()
            if cleaned_ws != current:
                all_mods.append(("excess whitespace", "single space"))
                current = cleaned_ws

        # 2. Repeated character reduction
        if self.config.reduce_repeated_chars:
            current, flood_mods = self.reduce_character_flooding(current)
            all_mods.extend(flood_mods)

        # 3. Phonetic spelling canonicalization
        if self.config.normalize_phonetic_spelling:
            current, spell_mods = self.normalize_tokens(current)
            all_mods.extend(spell_mods)

        # 4. Optional conservative punctuation normalization
        if self.config.clean_punctuation:
            # Replace multiple punctuation like ?? or !! with single ? or !
            punc_clean = re.sub(r"([?!.,])\1+", r"\1", current)
            if punc_clean != current:
                all_mods.append(("repeated punctuation", "single punctuation"))
                current = punc_clean

        return NormalizationResult(
            original_text=text,
            normalized_text=current,
            is_modified=(current != text),
            modifications=all_mods,
        )

    def normalize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalizes review_text and associated terms in an ABSA dataframe.
        Stores normalized_review_text, normalized_aspect_term, normalized_opinion_term
        while rigorously preserving original columns for experimental contrast.
        """
        res_df = df.copy()

        norm_reviews = []
        norm_aspects = []
        norm_opinions = []
        changed_reviews_count = 0
        total_token_mods = 0

        for _, row in res_df.iterrows():
            rev_res = self.normalize(str(row["review_text"]))
            asp_res = self.normalize(str(row["aspect_term"]))
            op_res = self.normalize(str(row["opinion_term"]))

            if rev_res.is_modified:
                changed_reviews_count += 1
            total_token_mods += len(rev_res.modifications)

            norm_reviews.append(rev_res.normalized_text)
            norm_aspects.append(asp_res.normalized_text)
            norm_opinions.append(op_res.normalized_text)

        res_df["normalized_review_text"] = norm_reviews
        res_df["normalized_aspect_term"] = norm_aspects
        res_df["normalized_opinion_term"] = norm_opinions

        stats = {
            "total_records": len(df),
            "changed_records": changed_reviews_count,
            "change_percentage": round((changed_reviews_count / max(1, len(df))) * 100, 2),
            "total_token_modifications": total_token_mods,
        }

        return res_df, stats
