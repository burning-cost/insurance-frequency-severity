"""
benchmarks.py
-------------
Synthetic data generators for validating the dependent frequency-severity model.

``make_dependent_claims`` generates a portfolio where the average severity is
explicitly shifted by γ·N — the Garrido-Genest-Schulz conditional covariate
model.  This is the canonical ground truth for testing whether the model
recovers the correct γ.

``make_independent_claims`` generates the null case (γ=0) for comparison.

Both generators produce a Pandas DataFrame in the format expected by
``DependentFSModel.fit``.

Typical usage::

    df_train, df_test = make_dependent_claims(n_policies=20_000, gamma=-0.15)
    model = DependentFSModel(use_explicit_gamma=True)
    model.fit(
        df_train[feature_cols].values,
        df_train['n_claims'].values,
        df_train['avg_severity'].values,
        df_train['exposure'].values,
    )
    print(f"Recovered γ={model.gamma_:.4f}, true γ=-0.15")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def make_dependent_claims(
    n_policies: int = 10_000,
    gamma: float = -0.15,
    base_freq: float = 0.08,
    base_sev: float = 3_000.0,
    phi: float = 1.5,
    n_features: int = 5,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic motor insurance claims with known frequency-severity dependence.

    Data generating process
    -----------------------
    For each policy i with covariates xᵢ and exposure tᵢ:

    1. log λᵢ = log(tᵢ) + β₀ + Xᵢ · β_freq       (Poisson frequency)
    2. Nᵢ ~ Poisson(λᵢ · tᵢ)
    3. log μᵢ = α₀ + Xᵢ · β_sev + γ · Nᵢ         (GGS conditional severity)
    4. If Nᵢ > 0: Ȳᵢ ~ Gamma(shape=Nᵢ/φ, mean=μᵢ); else Ȳᵢ = 0

    The parameter γ is the true dependence parameter the model should recover.
    Negative γ (default −0.15) means higher-claim-count policies have lower
    average severity — the pattern typically found in UK motor.

    Parameters
    ----------
    n_policies:
        Number of policies in the full dataset (train + test combined).
    gamma:
        True dependence parameter.  γ=0 gives independence; γ<0 gives negative
        frequency-severity correlation (typical for motor).
    base_freq:
        Baseline claim frequency (λ at x=0, t=1).
    base_sev:
        Baseline average severity (μ at x=0, N=0), in pounds.
    phi:
        Gamma dispersion parameter φ.  Higher values give more severity
        variability.
    n_features:
        Number of synthetic covariates.  First half affect frequency, second
        half affect severity; they all share the same feature matrix so the
        trunk has genuine heterogeneity to exploit.
    seed:
        Random seed.
    test_fraction:
        Fraction of data to hold out as test set.

    Returns
    -------
    df_train, df_test : pd.DataFrame
        DataFrames with columns:
        ``feature_0``, …, ``feature_{n_features-1}``,
        ``exposure``, ``n_claims``, ``avg_severity``, ``total_loss``,
        ``true_lambda``, ``true_mu``.

    Examples
    --------
    >>> df_train, df_test = make_dependent_claims(n_policies=20_000, gamma=-0.15)
    >>> df_train.head()
    """
    rng = np.random.default_rng(seed)

    # --- Covariates ---
    X = rng.standard_normal((n_policies, n_features)).astype(np.float32)

    # --- Exposure: mix of short- and full-year policies ---
    exposure = rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0], size=n_policies).astype(np.float32)

    # --- Regression coefficients ---
    n_freq_feats = max(1, n_features // 2)
    n_sev_feats = n_features - n_freq_feats
    beta_freq = rng.uniform(-0.3, 0.3, size=n_freq_feats).astype(np.float32)
    beta_sev = rng.uniform(-0.2, 0.2, size=n_sev_feats).astype(np.float32)

    # --- Frequency ---
    log_lambda_base = np.log(base_freq) + X[:, :n_freq_feats] @ beta_freq
    lambda_ = np.exp(log_lambda_base) * exposure
    n_claims = rng.poisson(lambda_).astype(np.float32)

    # --- Conditional severity ---
    log_mu = np.log(base_sev) + X[:, n_freq_feats:] @ beta_sev + gamma * n_claims
    mu = np.exp(log_mu)

    avg_severity = np.zeros(n_policies, dtype=np.float32)
    pos = n_claims > 0
    if pos.sum() > 0:
        alpha = n_claims[pos] / phi
        rate = n_claims[pos] / (phi * mu[pos] + 1e-10)
        avg_severity[pos] = rng.gamma(shape=alpha, scale=1.0 / rate).astype(np.float32)

    total_loss = n_claims * avg_severity

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["exposure"] = exposure
    df["n_claims"] = n_claims
    df["avg_severity"] = avg_severity
    df["total_loss"] = total_loss
    df["true_lambda"] = lambda_ / exposure  # per unit exposure
    df["true_mu"] = mu

    # --- Split ---
    n_test = max(1, int(n_policies * test_fraction))
    df_test = df.iloc[-n_test:].copy().reset_index(drop=True)
    df_train = df.iloc[:-n_test].copy().reset_index(drop=True)

    return df_train, df_test


def make_independent_claims(
    n_policies: int = 10_000,
    base_freq: float = 0.08,
    base_sev: float = 3_000.0,
    phi: float = 1.5,
    n_features: int = 5,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic claims with γ=0 (frequency-severity independence).

    Identical to ``make_dependent_claims`` with ``gamma=0``.  Use this as the
    null comparison to verify the model does not overfit spurious dependence.

    Parameters
    ----------
    n_policies, base_freq, base_sev, phi, n_features, seed, test_fraction:
        Same as ``make_dependent_claims``.

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    return make_dependent_claims(
        n_policies=n_policies,
        gamma=0.0,
        base_freq=base_freq,
        base_sev=base_sev,
        phi=phi,
        n_features=n_features,
        seed=seed,
        test_fraction=test_fraction,
    )


def feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of feature column names from a benchmark dataset."""
    return [c for c in df.columns if c.startswith("feature_")]
