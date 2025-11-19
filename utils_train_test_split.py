"""
Utility functions for train/test splits
"""
import numpy as np
import pandas as pd
from pathlib import Path

def create_train_test_split(csv_path: str, train_ratio: float = 0.8, seed: int = 42):
    """
    Split WTP data into train and test sets

    Args:
        csv_path: Path to full dataset
        train_ratio: Fraction for training (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        train_path, test_path: Paths to train and test CSV files
    """
    df = pd.read_csv(csv_path)
    M = len(df)

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(M)
    n_train = int(M * train_ratio)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_test = df.iloc[test_indices].reset_index(drop=True)

    # Save to temporary files
    base_path = Path(csv_path).stem
    train_path = f"{base_path}_train.csv"
    test_path = f"{base_path}_test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    return train_path, test_path, n_train, M - n_train
