"""Tests for dependent.benchmarks — synthetic data generators."""
import numpy as np
import pandas as pd
import pytest

from insurance_frequency_severity.dependent.benchmarks import (
    feature_cols,
    make_dependent_claims,
    make_independent_claims,
)


class TestMakeDependentClaims:
    def test_returns_two_dataframes(self):
        df_train, df_test = make_dependent_claims(n_policies=200)
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)

    def test_train_test_sizes(self):
        df_train, df_test = make_dependent_claims(n_policies=200, test_fraction=0.2)
        assert len(df_train) == 160
        assert len(df_test) == 40

    def test_required_columns(self):
        df_train, _ = make_dependent_claims(n_policies=100)
        required = {"n_claims", "avg_severity", "exposure", "total_loss"}
        assert required.issubset(df_train.columns)

    def test_feature_columns_exist(self):
        df_train, _ = make_dependent_claims(n_policies=100, n_features=5)
        fc = [c for c in df_train.columns if c.startswith("feature_")]
        assert len(fc) == 5

    def test_exposure_positive(self):
        df_train, _ = make_dependent_claims(n_policies=100)
        assert (df_train["exposure"] > 0).all()

    def test_n_claims_nonnegative(self):
        df_train, _ = make_dependent_claims(n_policies=100)
        assert (df_train["n_claims"] >= 0).all()

    def test_avg_severity_zero_when_no_claims(self):
        df_train, _ = make_dependent_claims(n_policies=500)
        zero_mask = df_train["n_claims"] == 0
        assert (df_train.loc[zero_mask, "avg_severity"] == 0.0).all()

    def test_avg_severity_positive_when_claims(self):
        df_train, _ = make_dependent_claims(n_policies=500)
        pos_mask = df_train["n_claims"] > 0
        if pos_mask.sum() > 0:
            assert (df_train.loc[pos_mask, "avg_severity"] > 0).all()

    def test_total_loss_consistency(self):
        df_train, _ = make_dependent_claims(n_policies=200)
        expected = df_train["n_claims"] * df_train["avg_severity"]
        np.testing.assert_allclose(df_train["total_loss"].values, expected.values, rtol=1e-4)

    def test_reproducibility(self):
        df1, _ = make_dependent_claims(n_policies=100, seed=7)
        df2, _ = make_dependent_claims(n_policies=100, seed=7)
        np.testing.assert_array_equal(df1["n_claims"].values, df2["n_claims"].values)

    def test_different_seeds_differ(self):
        df1, _ = make_dependent_claims(n_policies=100, seed=1)
        df2, _ = make_dependent_claims(n_policies=100, seed=2)
        assert not np.array_equal(df1["n_claims"].values, df2["n_claims"].values)

    def test_gamma_negative_induces_negative_correlation(self):
        """With γ<0, policyholders with more claims should have lower avg severity."""
        df, _ = make_dependent_claims(n_policies=5000, gamma=-0.3, seed=0)
        pos = df["n_claims"] > 0
        if pos.sum() > 50:
            corr = np.corrcoef(df.loc[pos, "n_claims"], df.loc[pos, "avg_severity"])[0, 1]
            assert corr < 0, f"Expected negative correlation with γ<0, got {corr:.3f}"

    def test_gamma_zero_no_strong_correlation(self):
        """With γ=0, correlation should be near zero."""
        df, _ = make_dependent_claims(n_policies=5000, gamma=0.0, seed=0)
        pos = df["n_claims"] > 0
        if pos.sum() > 50:
            corr = np.corrcoef(df.loc[pos, "n_claims"], df.loc[pos, "avg_severity"])[0, 1]
            assert abs(corr) < 0.3, f"Expected ~zero correlation with γ=0, got {corr:.3f}"

    def test_true_columns_present(self):
        df, _ = make_dependent_claims(n_policies=100)
        assert "true_lambda" in df.columns
        assert "true_mu" in df.columns

    def test_true_lambda_positive(self):
        df, _ = make_dependent_claims(n_policies=100)
        assert (df["true_lambda"] > 0).all()

    def test_claim_count_int_compatible(self):
        df, _ = make_dependent_claims(n_policies=100)
        assert np.all(df["n_claims"] == df["n_claims"].astype(int))


class TestMakeIndependentClaims:
    def test_returns_two_dataframes(self):
        df_train, df_test = make_independent_claims(n_policies=100)
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)

    def test_same_structure_as_dependent(self):
        df_dep, _ = make_dependent_claims(n_policies=100, gamma=0.0, seed=3)
        df_ind, _ = make_independent_claims(n_policies=100, seed=3)
        # Identical because gamma=0 in make_dependent_claims
        np.testing.assert_array_equal(df_dep["n_claims"].values, df_ind["n_claims"].values)

    def test_exposure_positive(self):
        df, _ = make_independent_claims(n_policies=100)
        assert (df["exposure"] > 0).all()


class TestFeatureCols:
    def test_returns_feature_columns(self):
        df, _ = make_dependent_claims(n_policies=50, n_features=3)
        fc = feature_cols(df)
        assert len(fc) == 3
        assert all(c.startswith("feature_") for c in fc)

    def test_no_non_feature_cols(self):
        df, _ = make_dependent_claims(n_policies=50, n_features=4)
        fc = feature_cols(df)
        for c in ("n_claims", "avg_severity", "exposure", "total_loss"):
            assert c not in fc
