"""
Aspect-Opinion Relation Extraction (AORE) module.
Predicts semantic association links between extracted aspect spans and opinion spans
using clause segmentation, token distance, and syntactic proximity.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import re
from src.aspect_extractor import AspectSpan
from src.opinion_extractor import OpinionSpan


@dataclass
class AspectOpinionRelation:
    aspect_text: str
    opinion_text: str
    aspect_span: Tuple[int, int]
    opinion_span: Tuple[int, int]
    confidence: float
    distance_chars: int
    same_clause: bool


class RelationExtractor:
    """
    Links candidate aspects to corresponding candidate opinions.
    """

    def __init__(self):
        # Conjunctions and punctuation that signal clause boundaries
        self.clause_delimiters = re.compile(r"[,;]|\b(?:but|aur|lekin|par|magar|and|while|whereas)\b", re.IGNORECASE)

    def get_clause_indices(self, text: str) -> List[Tuple[int, int]]:
        """Splits review into character ranges representing clauses."""
        matches = list(self.clause_delimiters.finditer(text))
        if not matches:
            return [(0, len(text))]

        clauses = []
        start = 0
        for m in matches:
            end = m.start()
            if end > start:
                clauses.append((start, end))
            start = m.end()
        if start < len(text):
            clauses.append((start, len(text)))
        return clauses

    def extract_relations(
        self,
        text: str,
        aspects: List[AspectSpan],
        opinions: List[OpinionSpan],
    ) -> List[AspectOpinionRelation]:
        """
        Predicts valid aspect-opinion links from extracted spans.
        """
        if not aspects or not opinions:
            return []

        clauses = self.get_clause_indices(text)

        def get_clause_id(char_idx: int) -> int:
            for cid, (cs, ce) in enumerate(clauses):
                if cs <= char_idx <= ce:
                    return cid
            return -1

        relations: List[AspectOpinionRelation] = []
        assigned_opinions: Set[int] = set()

        for asp in aspects:
            asp_mid = (asp.start_char + asp.end_char) / 2
            asp_clause = get_clause_id(int(asp_mid))

            best_op_idx = -1
            best_score = float("-inf")
            min_dist = float("inf")
            same_clause_flag = False

            for op_idx, op in enumerate(opinions):
                op_mid = (op.start_char + op.end_char) / 2
                op_clause = get_clause_id(int(op_mid))

                # Distance in characters
                dist = abs(asp_mid - op_mid)
                is_same_clause = (asp_clause != -1 and asp_clause == op_clause)

                # Score metric: reward same clause, penalize distance, penalize already assigned opinions
                score = 100.0 - (dist * 0.5)
                if is_same_clause:
                    score += 50.0
                if op_idx in assigned_opinions:
                    score -= 30.0

                if score > best_score:
                    best_score = score
                    best_op_idx = op_idx
                    min_dist = int(dist)
                    same_clause_flag = is_same_clause

            if best_op_idx != -1:
                chosen_op = opinions[best_op_idx]
                assigned_opinions.add(best_op_idx)
                conf = min(0.95, max(0.50, round(1.0 - (min_dist / max(1, len(text))), 2)))
                relations.append(AspectOpinionRelation(
                    aspect_text=asp.text,
                    opinion_text=chosen_op.text,
                    aspect_span=(asp.start_char, asp.end_char),
                    opinion_span=(chosen_op.start_char, chosen_op.end_char),
                    confidence=conf,
                    distance_chars=min_dist,
                    same_clause=same_clause_flag,
                ))

        return relations


def evaluate_relation_extraction(
    predicted_pairs: List[List[Tuple[str, str]]],
    gold_pairs: List[List[Tuple[str, str]]],
) -> Tuple[float, float, float]:
    """
    Evaluates exact aspect-opinion relation links (pairs).
    Returns (Precision, Recall, F1).
    """
    total_pred = 0
    total_gold = 0
    true_positives = 0

    for preds, golds in zip(predicted_pairs, gold_pairs):
        p_clean = [(a.strip().lower(), o.strip().lower()) for a, o in preds if a.strip() and o.strip()]
        g_clean = [(a.strip().lower(), o.strip().lower()) for a, o in golds if a.strip() and o.strip()]

        total_pred += len(p_clean)
        total_gold += len(g_clean)

        matched_gold_indices = set()
        for p_asp, p_op in p_clean:
            found = False
            for idx, (g_asp, g_op) in enumerate(g_clean):
                if idx in matched_gold_indices:
                    continue
                # Match when aspect and opinion match (exact or substring overlap)
                asp_match = (p_asp == g_asp) or (p_asp in g_asp) or (g_asp in p_asp)
                op_match = (p_op == g_op) or (p_op in g_op) or (g_op in p_op)
                if asp_match and op_match:
                    found = True
                    matched_gold_indices.add(idx)
                    break
            if found:
                true_positives += 1

    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)
