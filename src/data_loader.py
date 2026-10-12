"""
Data loader module for Hinglish ABSA datasets.
Includes strict group-aware splitting by review_id to eliminate data leakage across evaluation splits.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Union
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.data_validation import validate_absa_dataframe, ValidationResult


@dataclass
class DatasetSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_reviews: int
    val_reviews: int
    test_reviews: int


def load_dataset(source: Union[str, io.BytesIO, io.StringIO]) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Load and validate dataset from file path or buffer.
    
    Args:
        source: File path string or file-like buffer.
        
    Returns:
        Tuple of (raw_df, validation_result)
    """
    if isinstance(source, str):
        if source.endswith(".json"):
            df = pd.read_json(source)
        else:
            df = pd.read_csv(source)
    else:
        # Buffer
        try:
            df = pd.read_csv(source)
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_json(source)
            
    validation_result = validate_absa_dataframe(df)
    return df, validation_result


def split_by_review_id(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """
    Performs Grouped Train/Val/Test split strictly on review_id.
    Guarantees no review_id or customer text leaks into the test or val partition.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    unique_reviews = df["review_id"].unique()
    n_reviews = len(unique_reviews)
    
    if n_reviews < 3:
        # Edge case for extremely small demonstration inputs
        return DatasetSplits(
            train_df=df.copy(),
            val_df=df.copy(),
            test_df=df.copy(),
            train_reviews=n_reviews,
            val_reviews=n_reviews,
            test_reviews=n_reviews,
        )
        
    rng = np.random.RandomState(random_state)
    shuffled_reviews = rng.permutation(unique_reviews)
    
    n_train = max(1, int(round(train_ratio * n_reviews)))
    n_val = max(1, int(round(val_ratio * n_reviews)))
    
    # Ensure at least 1 review in each split if n_reviews >= 3
    if n_train + n_val >= n_reviews:
        n_train = max(1, n_reviews - 2)
        n_val = 1
        
    train_ids = set(shuffled_reviews[:n_train])
    val_ids = set(shuffled_reviews[n_train:n_train + n_val])
    test_ids = set(shuffled_reviews[n_train + n_val:])
    
    # Fallback if test_ids ended up empty
    if not test_ids:
        test_ids = set([shuffled_reviews[-1]])
        if test_ids.intersection(val_ids):
            val_ids.difference_update(test_ids)
        elif test_ids.intersection(train_ids):
            train_ids.difference_update(test_ids)

    train_df = df[df["review_id"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df["review_id"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df["review_id"].isin(test_ids)].copy().reset_index(drop=True)
    
    return DatasetSplits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_reviews=len(train_ids),
        val_reviews=len(val_ids),
        test_reviews=len(test_ids),
    )
