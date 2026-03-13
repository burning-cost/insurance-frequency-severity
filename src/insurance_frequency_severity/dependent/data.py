"""
data.py
-------
Dataset utilities and feature preprocessing for frequency-severity models.

The dataset format expected throughout the library is:
  - X: feature matrix (n_policies, n_features) — numeric
  - n_claims: claim count per policy (int, ≥ 0)
  - avg_severity: average claim amount (float; 0 for policies with no claims)
  - exposure: time at risk, typically in years (float, > 0)

``prepare_features`` converts a Pandas DataFrame (which may contain categoricals)
to a numeric numpy array suitable for PyTorch, using sklearn's
``ColumnTransformer`` under the hood.  It returns the transformer so you can
apply the same encoding to held-out data.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    transformer: Optional[ColumnTransformer] = None,
) -> Tuple[np.ndarray, ColumnTransformer]:
    """Encode a DataFrame into a numeric matrix.

    Numeric columns are standardised (zero mean, unit variance).  Categorical
    columns are one-hot encoded (unknown categories at inference are set to
    all-zero rows, not an error).

    Parameters
    ----------
    df:
        Input data.  Must contain all columns in ``numeric_cols`` and
        ``categorical_cols``.
    numeric_cols:
        Names of numeric/continuous columns.
    categorical_cols:
        Names of categorical columns.  Pass ``None`` or ``[]`` if there are no
        categoricals.
    transformer:
        A fitted ``ColumnTransformer`` from a previous call to this function.
        Pass this when encoding held-out or test data to ensure the same
        encoding is applied.  When ``None``, a new transformer is fitted on
        ``df``.

    Returns
    -------
    X: np.ndarray of shape (n, n_features_out)
        Encoded feature matrix.
    transformer: ColumnTransformer
        Fitted transformer (reuse for test data).

    Examples
    --------
    >>> X_train, ct = prepare_features(df_train, numeric_cols=["age", "value"],
    ...                                categorical_cols=["vehicle_class"])
    >>> X_test, _  = prepare_features(df_test, numeric_cols=["age", "value"],
    ...                                categorical_cols=["vehicle_class"],
    ...                                transformer=ct)
    """
    categorical_cols = categorical_cols or []

    if transformer is None:
        transformers = []
        if numeric_cols:
            transformers.append(
                ("num", StandardScaler(), numeric_cols)
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            )
        transformer = ColumnTransformer(transformers, remainder="drop")
        transformer.fit(df)

    X = transformer.transform(df).astype(np.float32)
    return X, transformer


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class FreqSevDataset(Dataset):
    """PyTorch Dataset for frequency-severity data.

    Each item is a dict with keys ``x``, ``log_exposure``, ``n_claims``,
    ``avg_severity``.

    Parameters
    ----------
    X:
        Feature matrix, shape (n, p).  Should be float32.
    n_claims:
        Claim counts, shape (n,).  Integer or float.
    avg_severity:
        Average severity (total loss / n_claims), shape (n,).  Zero for
        policies with no claims.
    exposure:
        Exposure (years at risk), shape (n,).  Must be > 0.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_claims: np.ndarray,
        avg_severity: np.ndarray,
        exposure: np.ndarray,
    ) -> None:
        n = len(X)
        if not (len(n_claims) == len(avg_severity) == len(exposure) == n):
            raise ValueError("All arrays must have the same length.")
        if np.any(exposure <= 0):
            raise ValueError("All exposure values must be strictly positive.")

        self.x = torch.tensor(X, dtype=torch.float32)
        self.n_claims = torch.tensor(n_claims, dtype=torch.float32)
        self.avg_severity = torch.tensor(avg_severity, dtype=torch.float32)
        self.log_exposure = torch.tensor(
            np.log(exposure.astype(np.float32)), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.x[idx],
            "log_exposure": self.log_exposure[idx],
            "n_claims": self.n_claims[idx],
            "avg_severity": self.avg_severity[idx],
        }

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_claims_col: str = "n_claims",
        avg_severity_col: str = "avg_severity",
        exposure_col: str = "exposure",
    ) -> "FreqSevDataset":
        """Construct from a Pandas DataFrame (features already numeric).

        Parameters
        ----------
        df:
            DataFrame with at least the feature columns plus target columns.
        feature_cols:
            Ordered list of feature column names.  These are used as-is;
            call ``prepare_features`` first if you need encoding.
        n_claims_col, avg_severity_col, exposure_col:
            Column names for the target variables.
        """
        X = df[feature_cols].to_numpy(dtype=np.float32)
        n_claims = df[n_claims_col].to_numpy(dtype=np.float32)
        avg_sev = df[avg_severity_col].to_numpy(dtype=np.float32)
        exposure = df[exposure_col].to_numpy(dtype=np.float32)
        return cls(X, n_claims, avg_sev, exposure)


def make_train_val_loaders(
    dataset: FreqSevDataset,
    val_fraction: float = 0.1,
    batch_size: int = 512,
    seed: int = 42,
) -> Tuple:
    """Split a dataset into train/val DataLoaders.

    Parameters
    ----------
    dataset:
        The full dataset.
    val_fraction:
        Fraction of data to use for validation.
    batch_size:
        Mini-batch size.
    seed:
        Random seed for reproducible splits.

    Returns
    -------
    train_loader, val_loader : DataLoader, DataLoader
    """
    from torch.utils.data import random_split, DataLoader

    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
