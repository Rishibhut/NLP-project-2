"""
Opinion Term Extraction (OTE) module for Hinglish product reviews.
Extracts subjective and evaluative spans, intensifier-adjective combinations,
negated opinion spans, and idiom patterns.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import re

# Curated polar opinion expressions in Hinglish & English
DEFAULT_OPINION_PATTERNS: List[str] = [
    # Multi-word Hinglish expressions
    r"\b(?:bahut|bohot|kaafi|bilkul|ekdum|thoda)\s+(?:accha|acchi|acche|achha|achhi|achhe|badhiya|badiya|shaandar|laajawab|lajawab)\s*(?:hai|hain|tha|thi|lagta|lagti)?\b",
    r"\b(?:bahut|bohot|kaafi|bilkul|ekdum|thoda)\s+(?:kharab|khraab|bekaar|bekar|ghatiya|slow|loose|chhota|noisy)\s*(?:hai|hain|tha|thi|lagta|nikla)?\b",
    r"\b(?:bilkul\s+)?(?:kaam\s+nahi\s+karta|blur\s+karta\s+hai|phat\s+rahi\s+hai|heating\s+problem\s+de\s+raha\s+hai|disconnect\s+hoti\s+rehti\s+hai)\b",
    r"\b(?:soft|accurate|clear)\s+nahi\s+hai\b",
    r"\b(?:theek\s+thaak\s+hain|theek\s+thaak\s+thi|average\s+hai|ordinary\s+hai)\b",
    r"\b(?:accha|acchi|acche|achha|achhi|achhe)\s+(?:hai|hain|tha|thi)\b",
    r"\b(?:kharab|khraab|bekaar|bekar|ghatiya)\s+(?:hai|hain|nikli|nikla)\b",
    r"\b(?:crystal\s+clear|really\s+powerful|works\s+perfectly|value\s+for\s+money|is\s+amazing|top\s+notch\s+hai|noticeable\s+and\s+annoying)\b",
    r"\b(?:are\s+responsive|feel\s+great|is\s+neat|are\s+timely|is\s+fast|is\s+soft)\b",
    r"\b(?:durable\s+lagta\s+hai|weak\s+hai|attractive\s+hai|safe\s+thi|on\s+schedule\s+hui)\b",
    r"\b(?:are\s+stiff\s+and\s+feel\s+cheap|test\s+pass\s+ho\s+gaya|was\s+super\s+simple|clear\s+tha)\b",
    r"\b(?:time\s+par\s+nahi\s+hui|time\s+par\s+nahi\s+tha|delivery\s+late\s+hui|late\s+delivery)\b",
    r"\b(?:kaafi\s+achhi\s+hai|kaafi\s+acchi\s+hai|kaafi\s+accha\s+hai|kaafi\s+badiya\s+hai)\b",
]

# Single-word polar anchors
POLAR_ANCHORS: Set[str] = {
    "accha", "acchi", "acche", "achha", "achhi", "achhe", "badhiya", "badiya",
    "shaandar", "laajawab", "lajawab", "mast", "kharab", "khraab", "bekaar",
    "bekar", "ghatiya", "ganda", "bakwaas", "slow", "vibrant", "smooth",
    "crystal", "powerful", "comfortable", "perfect", "useless", "noisy",
    "light", "ordinary", "average", "responsive", "stiff", "cheap", "durable",
    "weak", "attractive", "clear", "annoying"
}


@dataclass
class OpinionSpan:
    text: str
    start_char: int
    end_char: int
    confidence: float
    source: str


class OpinionExtractor:
    """
    Extracts opinion expressions from Hinglish reviews using syntactic-lexical pattern matching.
    """

    def __init__(self, custom_patterns: Optional[List[str]] = None):
        self.patterns = custom_patterns or DEFAULT_OPINION_PATTERNS
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def extract_opinions(self, text: str) -> List[OpinionSpan]:
        """
        Extracts opinion expression spans from review text.
        """
        if not text or not text.strip():
            return []

        spans: List[OpinionSpan] = []
        matched_intervals: List[Tuple[int, int]] = []

        def overlaps(s: int, e: int) -> bool:
            return any(not (e <= existing_s or s >= existing_e) for existing_s, existing_e in matched_intervals)

        # 1. Match multi-word regex patterns
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if not overlaps(start, end):
                    matched_intervals.append((start, end))
                    spans.append(OpinionSpan(
                        text=text[start:end],
                        start_char=start,
                        end_char=end,
                        confidence=0.85,
                        source="pattern_matcher",
                    ))

        # 2. Fallback window search around remaining polar anchors
        tokens_with_offsets = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\b\w+\b", text)]
        for i, (tok, s, e) in enumerate(tokens_with_offsets):
            if tok.lower() in POLAR_ANCHORS and not overlaps(s, e):
                # Expand window 1 word left (intensifier/negation) and 1 word right (auxiliary)
                win_start = s
                win_end = e
                if i > 0:
                    prev_tok, prev_s, _ = tokens_with_offsets[i - 1]
                    if prev_tok.lower() in {"bahut", "bohot", "kaafi", "bilkul", "ekdum", "nahi", "not", "thoda", "really", "very"}:
                        win_start = prev_s
                if i < len(tokens_with_offsets) - 1:
                    next_tok, _, next_e = tokens_with_offsets[i + 1]
                    if next_tok.lower() in {"hai", "hain", "tha", "thi", "lagta", "lagti", "mila", "nikla", "thaak"}:
                        win_end = next_e

                if not overlaps(win_start, win_end):
                    matched_intervals.append((win_start, win_end))
                    spans.append(OpinionSpan(
                        text=text[win_start:win_end],
                        start_char=win_start,
                        end_char=win_end,
                        confidence=0.75,
                        source="anchor_window",
                    ))

        spans.sort(key=lambda s: s.start_char)
        return spans
