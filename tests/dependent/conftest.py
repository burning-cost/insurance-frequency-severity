"""
Shared fixtures for the insurance_frequency_severity.dependent test suite.

All fixtures use small datasets (n <= 500) so that tests run on CPU in <30s
total.  We never run torch model training on the Raspberry Pi directly -- but
these unit tests are intentionally small enough that a single forward pass or a
5-epoch training run is feasible.
"""
import numpy as np
import pytest
import torch

from insurance_frequency_severity.dependent.benchmarks import make_dependent_claims, feature_cols


@pytest.fixture(scope="session")
def small_dataset():
    """200-policy synthetic dataset with γ=-0.15."""
    df_train, df_test = make_dependent_claims(
        n_policies=500, gamma=-0.15, n_features=4, seed=0
    )
    fc = feature_cols(df_train)
    return {
        "df_train": df_train,
        "df_test": df_test,
        "feature_cols": fc,
        "X_train": df_train[fc].values.astype(np.float32),
        "X_test": df_test[fc].values.astype(np.float32),
        "n_claims_train": df_train["n_claims"].values,
        "n_claims_test": df_test["n_claims"].values,
        "avg_sev_train": df_train["avg_severity"].values,
        "avg_sev_test": df_test["avg_severity"].values,
        "exposure_train": df_train["exposure"].values,
        "exposure_test": df_test["exposure"].values,
    }


@pytest.fixture(scope="session")
def independent_dataset():
    """200-policy synthetic dataset with γ=0 (null case)."""
    df_train, df_test = make_dependent_claims(
        n_policies=400, gamma=0.0, n_features=4, seed=1
    )
    fc = feature_cols(df_train)
    return {
        "df_train": df_train,
        "df_test": df_test,
        "feature_cols": fc,
        "X_train": df_train[fc].values.astype(np.float32),
        "n_claims_train": df_train["n_claims"].values,
        "avg_sev_train": df_train["avg_severity"].values,
        "exposure_train": df_train["exposure"].values,
    }
