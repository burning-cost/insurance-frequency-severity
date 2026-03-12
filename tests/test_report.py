"""
Tests for report.py — JointModelReport.

Key tests:
- to_html() returns a non-empty HTML string
- to_dict() returns expected keys
- Report renders without error when optional arguments are missing
- Report includes correction histogram when correction_df provided
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_frequency_severity.report import JointModelReport
from insurance_frequency_severity.joint import JointFreqSev
from insurance_frequency_severity.diagnostics import DependenceTest


@pytest.fixture(scope="module")
def fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
    import numpy as np
    dgp = nb_gamma_dgp
    n, s = dgp["n"], dgp["s"]
    data = pd.DataFrame({"claim_count": n, "avg_severity": s})
    rng = np.random.default_rng(99)

    model = JointFreqSev(
        freq_glm=mock_freq_glm,
        sev_glm=mock_sev_glm,
        copula="sarmanov",
        kernel_theta=dgp["kernel_theta"],
        kernel_alpha=dgp["kernel_alpha"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data, rng=rng)
    return model


@pytest.fixture(scope="module")
def fitted_test(nb_gamma_dgp):
    dgp = nb_gamma_dgp
    n, s = dgp["n"], dgp["s"]
    mask = (n > 0) & (s > 0)
    rng = np.random.default_rng(99)
    test = DependenceTest(n_permutations=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test.fit(n[mask], s[mask], rng=rng)
    return test


class TestJointModelReport:
    def test_to_html_returns_string(self, fitted_model, fitted_test):
        report = JointModelReport(fitted_model, fitted_test)
        html = report.to_html()
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_contains_key_sections(self, fitted_model, fitted_test):
        report = JointModelReport(fitted_model, fitted_test)
        html = report.to_html()
        assert "Dependence" in html
        assert "omega" in html or "rho" in html

    def test_to_dict_has_keys(self, fitted_model, fitted_test):
        report = JointModelReport(fitted_model, fitted_test)
        d = report.to_dict()
        assert "omega" in d
        assert "copula_family" in d
        assert "kendall_tau" in d
        assert "spearman_rho_test" in d

    def test_report_without_dependence_test(self, fitted_model):
        """Report should render without dependence test."""
        report = JointModelReport(fitted_model)
        html = report.to_html()
        assert isinstance(html, str)
        assert len(html) > 50

    def test_report_with_correction_df(self, fitted_model, fitted_test):
        """Report with correction histogram."""
        corrections = fitted_model.premium_correction()
        report = JointModelReport(fitted_model, fitted_test)
        html = report.to_html(correction_df=corrections)
        assert "correction" in html.lower()

    def test_report_with_scatter(self, fitted_model, fitted_test, nb_gamma_dgp):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        report = JointModelReport(fitted_model, fitted_test)
        html = report.to_html(n=n, s=s)
        assert "img" in html

    def test_to_html_writes_file(self, fitted_model, tmp_path):
        report = JointModelReport(fitted_model)
        out_path = str(tmp_path / "report.html")
        report.to_html(output_path=out_path)
        with open(out_path) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content

    def test_copula_comparison_in_dict(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        from insurance_frequency_severity.diagnostics import compare_copulas
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        rng = np.random.default_rng(77)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comp = compare_copulas(n, s, mock_freq_glm, mock_sev_glm, rng=rng)

        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pd.DataFrame({"claim_count": n, "avg_severity": s})
            model.fit(data, rng=rng)

        report = JointModelReport(model, copula_comparison=comp)
        d = report.to_dict()
        assert "copula_comparison" in d
        assert isinstance(d["copula_comparison"], list)
