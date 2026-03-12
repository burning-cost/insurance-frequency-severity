"""
report.py — HTML report generation for joint frequency-severity models.

JointModelReport produces a self-contained HTML report suitable for sharing
with a pricing team. The report includes:

1. Dependence test results (Kendall tau, Spearman rho)
2. Copula parameter estimates with confidence intervals
3. Premium correction distribution (histogram)
4. Scatter plot of N vs S with dependence annotation
5. AIC/BIC comparison across copula families (if compare_copulas was run)

The report is designed to be interpretable by a non-technical pricing
manager: correction factors are shown as percentages, not mathematical
formulas.
"""

from __future__ import annotations

import base64
import io
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class JointModelReport:
    """
    Generates an HTML report for a fitted JointFreqSev model.

    Parameters
    ----------
    joint_model : JointFreqSev
        A fitted joint model.
    dependence_test : DependenceTest, optional
        Pre-fitted dependence test results.
    copula_comparison : DataFrame, optional
        Output from compare_copulas().

    Examples
    --------
    >>> report = JointModelReport(model, test)
    >>> report.to_html("model_report.html")
    >>> d = report.to_dict()
    """

    def __init__(
        self,
        joint_model: Any,
        dependence_test: Optional[Any] = None,
        copula_comparison: Optional[pd.DataFrame] = None,
    ):
        self.joint_model = joint_model
        self.dependence_test = dependence_test
        self.copula_comparison = copula_comparison

    def to_dict(self) -> Dict:
        """
        Return report data as a plain dictionary.

        Useful for programmatic consumption without generating HTML.
        """
        result = {}

        model = self.joint_model
        if hasattr(model, "omega_") and model.omega_ is not None:
            result["copula_family"] = model.copula_family
            result["omega"] = model.omega_
            result["spearman_rho"] = model.rho_
            result["omega_ci"] = model.omega_ci_
            result["aic"] = model.aic_
            result["bic"] = model.bic_
            result["n_policies"] = model._n_obs
            result["n_claims"] = model._n_claims

        if self.dependence_test is not None and hasattr(self.dependence_test, "tau_"):
            dt = self.dependence_test
            result["kendall_tau"] = dt.tau_
            result["kendall_pval"] = dt.tau_pval_
            result["spearman_rho_test"] = dt.rho_s_
            result["spearman_pval"] = dt.rho_s_pval_

        if self.copula_comparison is not None:
            result["copula_comparison"] = self.copula_comparison.to_dict(orient="records")

        return result

    def _make_fig_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _plot_correction_distribution(
        self, correction_df: pd.DataFrame
    ) -> Optional[str]:
        """Histogram of premium correction factors."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            factors = correction_df["correction_factor"].values
            fig, ax = plt.subplots(figsize=(8, 4))

            ax.hist(factors, bins=40, edgecolor="white", color="#2271b5", alpha=0.85)
            ax.axvline(1.0, color="#cc0000", linewidth=1.5, linestyle="--", label="Independence (1.0)")
            ax.axvline(float(np.mean(factors)), color="#f28c28", linewidth=1.5,
                       linestyle="-", label=f"Mean = {np.mean(factors):.4f}")

            ax.set_xlabel("Premium correction factor")
            ax.set_ylabel("Number of policies")
            ax.set_title("Distribution of premium correction factors")
            ax.legend()
            ax.grid(True, alpha=0.3)

            result = self._make_fig_base64(fig)
            plt.close(fig)
            return result
        except Exception:
            return None

    def _plot_dependence_scatter(
        self, n: np.ndarray, s: np.ndarray
    ) -> Optional[str]:
        """Scatter of N vs S for positive-claim observations."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mask = (n > 0) & (s > 0) & np.isfinite(s)
            n_pos = n[mask]
            s_pos = s[mask]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(n_pos, s_pos, alpha=0.3, s=10, color="#2271b5")
            ax.set_xlabel("Claim count (N)")
            ax.set_ylabel("Average claim severity (S)")
            ax.set_title("Claim count vs. average severity")

            if self.dependence_test and hasattr(self.dependence_test, "tau_"):
                tau = self.dependence_test.tau_
                rho = self.dependence_test.rho_s_
                ax.text(
                    0.05, 0.95,
                    f"Kendall tau = {tau:.3f}\nSpearman rho = {rho:.3f}",
                    transform=ax.transAxes, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                )

            ax.grid(True, alpha=0.3)
            result = self._make_fig_base64(fig)
            plt.close(fig)
            return result
        except Exception:
            return None

    def to_html(
        self,
        output_path: Optional[str] = None,
        n: Optional[np.ndarray] = None,
        s: Optional[np.ndarray] = None,
        correction_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate HTML report.

        Parameters
        ----------
        output_path : str, optional
            If provided, write HTML to this file.
        n : array-like, optional
            Claim counts for dependence scatter plot.
        s : array-like, optional
            Severities for scatter plot.
        correction_df : DataFrame, optional
            Output from model.premium_correction(). If None, no histogram.

        Returns
        -------
        HTML string.
        """
        data = self.to_dict()
        model = self.joint_model

        sections = []

        # --- Header ---
        sections.append("""
        <h1>Joint Frequency-Severity Model Report</h1>
        <p style="color: #666;">Burning Cost | insurance-frequency-severity</p>
        <hr>
        """)

        # --- Dependence test ---
        if self.dependence_test is not None and hasattr(self.dependence_test, "tau_"):
            dt = self.dependence_test
            summary_df = dt.summary()
            sections.append("<h2>Dependence Tests</h2>")
            sections.append("<p>Tests H<sub>0</sub>: independence between N and S "
                           "(positive-claim observations only).</p>")
            sections.append(summary_df.to_html(index=False, classes="table"))

        # --- Copula fit results ---
        if hasattr(model, "omega_") and model.omega_ is not None:
            dep_df = model.dependence_summary()
            sections.append("<h2>Fitted Dependence Parameter</h2>")
            sections.append(dep_df.to_html(index=False, classes="table"))

            ci = model.omega_ci_
            if ci:
                direction = "negative" if model.omega_ < 0 else "positive"
                sections.append(
                    f"<p><strong>Interpretation:</strong> The estimated dependence parameter "
                    f"({model.copula_family} omega = {model.omega_:.4f}) indicates "
                    f"<strong>{direction} frequency-severity dependence</strong>. "
                    f"95% CI: ({ci[0]:.4f}, {ci[1]:.4f}). "
                )
                if model.rho_ is not None:
                    sections.append(
                        f"Spearman's rank correlation (Monte Carlo) = {model.rho_:.4f}.</p>"
                    )

        # --- Copula comparison ---
        if self.copula_comparison is not None:
            sections.append("<h2>Copula Family Comparison</h2>")
            sections.append(self.copula_comparison.to_html(index=False, classes="table"))
            best = self.copula_comparison.iloc[0]["copula"]
            sections.append(f"<p>Best fitting copula by AIC: <strong>{best}</strong>.</p>")

        # --- Plots ---
        if n is not None and s is not None:
            n = np.asarray(n)
            s = np.asarray(s)
            scatter_b64 = self._plot_dependence_scatter(n, s)
            if scatter_b64:
                sections.append("<h2>Frequency vs Severity</h2>")
                sections.append(
                    f'<img src="data:image/png;base64,{scatter_b64}" '
                    f'style="max-width:700px;"><br>'
                )

        if correction_df is not None:
            hist_b64 = self._plot_correction_distribution(correction_df)
            if hist_b64:
                sections.append("<h2>Premium Correction Factor Distribution</h2>")
                sections.append(
                    f'<img src="data:image/png;base64,{hist_b64}" '
                    f'style="max-width:800px;"><br>'
                )
                mean_corr = float(correction_df["correction_factor"].mean())
                pct_gt1 = float((correction_df["correction_factor"] > 1.0).mean() * 100)
                sections.append(
                    f"<p>Mean correction factor: {mean_corr:.4f}. "
                    f"{pct_gt1:.1f}% of policies have correction > 1.0 "
                    f"(i.e., dependence increases expected cost).</p>"
                )

        # --- CSS ---
        css = """
        <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }
        h1 { color: #2271b5; }
        h2 { color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
        table.table { border-collapse: collapse; width: 100%; margin: 12px 0; }
        table.table th, table.table td {
            border: 1px solid #ddd; padding: 8px 12px; text-align: left;
        }
        table.table th { background: #f5f5f5; font-weight: bold; }
        table.table tr:nth-child(even) { background: #fafafa; }
        </style>
        """

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Joint Frequency-Severity Model Report</title>
{css}
</head>
<body>
{"".join(sections)}
</body>
</html>"""

        if output_path:
            with open(output_path, "w") as f:
                f.write(html)

        return html
