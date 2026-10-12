"""
Aspect Term Extraction (ATE) module for Hinglish product reviews.
Provides rule-based noun-phrase and domain-lexicon span extraction,
annotation-assisted extraction, and evaluation utilities.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import re

# Comprehensive product aspect lexicon across major e-commerce categories
DEFAULT_ASPECT_LEXICON: Set[str] = {
    # Electronics / Mobile / Laptops
    "camera quality", "camera", "battery backup", "battery life", "battery",
    "display", "screen brightness", "screen", "screen resolution", "touch response",
    "touch", "build quality", "build", "sound quality", "sound", "speaker",
    "processor", "performance", "design", "look", "weight", "portability",
    "strap quality", "strap", "keyboard switches", "key travel", "keyboard",
    "mic clarity", "mic", "microphone", "charger", "cooling system", "cooling",
    "fingerprint sensor", "face unlock", "app connectivity", "refresh rate",
    "noise cancellation feature", "noise cancellation", "fans", "switches",
    "buttons", "hardware", "software updates", "software", "interface", "ui",
    # Audio
    "bass", "audio latency", "latency", "volume", "earbuds", "aawaz",
    # Footwear & Clothing
    "sole", "fit", "fabric", "stitching", "color", "size", "material",
    "finish", "cushioning", "arch support", "grip",
    # Service / General
    "delivery time", "delivery", "packaging", "customer support", "customer care",
    "price", "keemat", "deal", "value", "cheez", "product"
}


@dataclass
class AspectSpan:
    text: str
    start_char: int
    end_char: int
    confidence: float
    source: str  # 'rule_lexicon', 'annotation_assisted', 'model'


class AspectExtractor:
    """
    Extracts aspect terms from Hinglish reviews using domain-informed span matching,
    regex patterns, and optional annotation assistance.
    """

    def __init__(self, custom_lexicon: Optional[Set[str]] = None):
        self.lexicon = sorted(
            list(custom_lexicon or DEFAULT_ASPECT_LEXICON),
            key=lambda x: len(x),
            reverse=True  # Match longest noun phrases first
        )

    def extract_aspects(self, text: str) -> List[AspectSpan]:
        """
        Extracts candidate aspect spans from the input review text.
        """
        if not text or not text.strip():
            return []

        spans: List[AspectSpan] = []
        text_lower = text.lower()
        matched_intervals: List[Tuple[int, int]] = []

        def overlaps(s: int, e: int) -> bool:
            return any(not (e <= existing_s or s >= existing_e) for existing_s, existing_e in matched_intervals)

        for term in self.lexicon:
            term_lower = term.lower()
            # Match term with word boundaries where applicable
            pattern = r"\b" + re.escape(term_lower) + r"\b"
            for match in re.finditer(pattern, text_lower):
                start, end = match.start(), match.end()
                if not overlaps(start, end):
                    matched_intervals.append((start, end))
                    # Get exact case from original text
                    original_span = text[start:end]
                    spans.append(AspectSpan(
                        text=original_span,
                        start_char=start,
                        end_char=end,
                        confidence=0.85,
                        source="rule_lexicon",
                    ))

        # Sort spans by appearance in text
        spans.sort(key=lambda s: s.start_char)
        return spans


def evaluate_span_extraction(
    predicted_spans: List[List[str]],
    gold_spans: List[List[str]],
    match_mode: str = "exact"  # 'exact' or 'overlap'
) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F1 for extracted spans against gold spans.
    """
    total_pred = 0
    total_gold = 0
    true_positives = 0

    for preds, golds in zip(predicted_spans, gold_spans):
        p_clean = [p.strip().lower() for p in preds if p.strip()]
        g_clean = [g.strip().lower() for g in golds if g.strip()]

        total_pred += len(p_clean)
        total_gold += len(g_clean)

        matched_gold_indices = set()
        for p in p_clean:
            found = False
            for idx, g in enumerate(g_clean):
                if idx in matched_gold_indices:
                    continue
                if match_mode == "exact":
                    if p == g:
                        found = True
                        matched_gold_indices.add(idx)
                        break
                elif match_mode == "overlap":
                    if p in g or g in p:
                        found = True
                        matched_gold_indices.add(idx)
                        break
            if found:
                true_positives += 1

    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)
