"""
Automated tests for data leakage prevention and group-aware splitting.
Guarantees that aspect-opinion records originating from the same customer review
never cross train/val/test partitions.
"""

import pytest
import pandas as pd
from src.data_loader import split_by_review_id, verify_no_leakage, get_split_distributions


def test_no_review_id_data_leakage():
    # Synthetic test fixture with multiple aspects per review
    records = []
    for r_id in range(1, 21):
        # 2 aspects per review
        records.append({
            "review_id": r_id,
            "review_text": f"Review {r_id} display is good and battery is bad",
            "aspect_term": "display",
            "opinion_term": "good",
            "sentiment": "Positive",
            "aspect_language": "English",
            "opinion_language": "English",
            "pair_type": "same-language",
            "product_category": "Smartphones",
        })
        records.append({
            "review_id": r_id,
            "review_text": f"Review {r_id} display is good and battery is bad",
            "aspect_term": "battery",
            "opinion_term": "bad",
            "sentiment": "Negative",
            "aspect_language": "English",
            "opinion_language": "English",
            "pair_type": "same-language",
            "product_category": "Smartphones",
        })

    df = pd.DataFrame(records)
    splits = split_by_review_id(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

    # 1. Verify verify_no_leakage passes without exception
    assert verify_no_leakage(splits) is True

    # 2. Assert disjoint sets of review_ids
    train_ids = set(splits.train_df["review_id"])
    val_ids = set(splits.val_df["review_id"])
    test_ids = set(splits.test_df["review_id"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # 3. All original reviews accounted for
    all_split_ids = train_ids.union(val_ids).union(test_ids)
    assert all_split_ids == set(range(1, 21))


def test_split_distributions_structure():
    data = {
        "review_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "review_text": ["Sample text"] * 11,
        "aspect_term": ["camera"] * 11,
        "opinion_term": ["achha"] * 11,
        "sentiment": ["Positive", "Negative"] * 5 + ["Neutral"],
        "aspect_language": ["English"] * 11,
        "opinion_language": ["Hindi"] * 11,
        "pair_type": ["cross-lingual"] * 11,
        "product_category": ["Electronics"] * 11,
    }
    df = pd.DataFrame(data)
    splits = split_by_review_id(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42)
    dist = get_split_distributions(splits)

    assert dist["leakage_free"] is True
    assert "train" in dist
    assert "validation" in dist
    assert "test" in dist
    assert dist["train"]["records"] > 0
    assert dist["test"]["records"] > 0


def test_small_dataset_edge_case():
    # Only 2 reviews
    df = pd.DataFrame({
        "review_id": [1, 2],
        "review_text": ["text 1", "text 2"],
        "aspect_term": ["a1", "a2"],
        "opinion_term": ["o1", "o2"],
        "sentiment": ["Positive", "Negative"],
        "aspect_language": ["English", "Hindi"],
        "opinion_language": ["English", "Hindi"],
        "pair_type": ["same-language", "same-language"],
        "product_category": ["General", "General"],
    })
    splits = split_by_review_id(df)
    assert splits.train_reviews == 2
    assert splits.test_reviews == 2
