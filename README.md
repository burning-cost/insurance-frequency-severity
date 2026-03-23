# insurance-frequency-severity

[![PyPI](https://img.shields.io/pypi/v/insurance-frequency-severity)](https://pypi.org/project/insurance-frequency-severity/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-frequency-severity)](https://pypi.org/project/insurance-frequency-severity/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-frequency-severity/blob/main/notebooks/quickstart.ipynb)


Sarmanov copula joint frequency-severity modelling for UK personal lines insurance.

Merged from: `insurance-frequency-severity` (Sarmanov/Gaussian copula) and `insurance-dependent-fs` (neural two-part model).

**Blog post:** [Your Frequency-Severity Independence Assumption Is Costing You Premium](https://burning-cost.github.io/2027/05/15/frequency-severity-independence-is-costing-you-premium/)

Challenges the independence assumption in the standard two-model GLM framework. Your frequency GLM and severity GLM are correct. The problem is multiplying their predictions together as though claim count and average severity are unrelated — they are not.

## Why use this?

- The standard UK motor pricing approach (pure premium = E[N] × E[S]) assumes frequency and severity are independent given rating factors — they are not. NCD structure suppresses borderline claims, creating a systematic negative correlation. Vernic, Bolancé & Alemany (2022) found this mismeasurement costs €5–55+ per policyholder; the directional effect in UK motor is the same.
- The Sarmanov copula handles the discrete-continuous mixed margins problem correctly — no probability integral transform approximation for the count margin, which is not well-defined for discrete distributions. The Gaussian copula comparison and Garrido conditional fallback are also included so you can present the methodology choice to a pricing committee.
- IFM estimation: you plug in your already-fitted statsmodels GLM objects. There is no need to refit the marginals — the library estimates the dependence parameter omega on top of your existing models, and returns analytical (closed-form) correction factors per policy at scoring time.
- DependenceTest first: run the permutation test for independence before committing to a correction. If the test does not reject, use the simpler independent model. The benchmark shows that even when omega is not statistically significant, the correction can absorb marginal model error (28.6% MAE improvement on the benchmark DGP).
- Generates a JointModelReport HTML document (omega estimate, CI, Spearman rho, AIC/BIC comparison, correction factor distribution) suitable for a pricing committee or model validation pack.

## The problem

Every UK motor pricing team runs two GLMs:

```
Pure premium = E[N|x] × E[S|x]
```

This assumes N and S are independent given rating factors x. The assumption is almost certainly wrong. In UK motor, the No Claims Discount structure suppresses borderline claims: policyholders with frequent small claims are aware of the NCD threshold and do not report near-miss incidents. The result is a systematic negative correlation between claim count and average severity.

Vernic, Bolancé, and Alemany (2022) found this mismeasurement amounts to €5–55+ per policyholder on a Spanish auto book. The directional effect in UK motor is the same; the magnitude depends on your book.

This library gives you three methods to measure and correct for it:

1. **Sarmanov copula (primary)**: Bivariate Sarmanov distribution for NB/Poisson frequency × Gamma/Lognormal severity. Handles the discrete-continuous mixed margins problem correctly — no probability integral transform approximation needed for the count margin. IFM estimation: you plug in your fitted GLM objects, we estimate omega.

2. **Gaussian copula (comparison)**: Standard approach from Czado et al. (2012). Uses PIT approximation for the discrete margin. Good for presenting rho in familiar terms.

3. **Garrido conditional (fallback)**: Adds N as a covariate in the severity GLM. No copula, no new methodology — just a single extra GLM parameter. Works on smaller books where omega estimation would be unreliable.

## Installation

```bash
uv add insurance-frequency-severity
```

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-frequency-severity/discussions). Found it useful? A ⭐ helps others find it.

## Quickstart

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from insurance_frequency_severity import JointFreqSev, DependenceTest

rng = np.random.default_rng(42)
n_policies = 5000
# Synthetic motor book: claim count and average severity per policy
claim_count = rng.poisson(0.10, size=n_policies)
avg_severity = np.where(
    claim_count > 0,
    rng.gamma(shape=3.0, scale=800.0, size=n_policies),
    np.nan,
)
X = np.column_stack([
    rng.normal(35, 8, n_policies),   # age
    rng.normal(5, 2, n_policies),    # ncb
])
claims_df = pd.DataFrame({
    "claim_count": claim_count,
    "avg_severity": avg_severity,
})

# Fit marginal GLMs
X_df = pd.DataFrame(X, columns=["age", "ncb"])
X_const = sm.add_constant(X_df)
my_nb_glm = sm.GLM(
    claim_count, X_const, family=sm.families.NegativeBinomial(alpha=0.8)
).fit()
claims_mask = claim_count > 0
my_gamma_glm = sm.GLM(
    avg_severity[claims_mask],
    X_const[claims_mask],
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

# Test for dependence first
test = DependenceTest()
test.fit(n=claim_count[claims_mask], s=avg_severity[claims_mask])
print(test.summary())

# Fit joint model — accepts your existing fitted GLMs
model = JointFreqSev(
    freq_glm=my_nb_glm,    # fitted statsmodels NegativeBinomial GLM
    sev_glm=my_gamma_glm,  # fitted statsmodels Gamma GLM
    copula="sarmanov",
)
model.fit(
    claims_df,
    n_col="claim_count",
    s_col="avg_severity",
)

# Check dependence parameter and confidence interval
print(model.dependence_summary())

# Get correction factors for your in-force book
corrections = model.premium_correction()
print(corrections[["mu_n", "mu_s", "correction_factor", "premium_joint"]].describe())
```

## GLM compatibility

This library is designed for statsmodels GLM objects. It detects marginal families via `model.family` (statsmodels convention) and extracts dispersion from `model.scale`. Non-statsmodels objects with `.predict()` and `.fittedvalues` may work, but kernel parameters will be inferred from statsmodels-specific attributes and could silently produce wrong results. For non-statsmodels GLMs, pass parameter dictionaries directly.

```python
# Works with statsmodels GLM results
import statsmodels.api as sm
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 3000
X = pd.DataFrame({"age": rng.normal(35, 8, n), "ncb": rng.normal(5, 2, n)})
X_const = sm.add_constant(X)
y = rng.poisson(0.10, size=n)
claims_mask = y > 0
s = rng.gamma(3.0, 800.0, size=n)

nb_glm = sm.GLM(y, X_const, family=sm.families.NegativeBinomial(alpha=0.8)).fit()
gamma_glm = sm.GLM(
    s[claims_mask],
    X_const[claims_mask],
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

model = JointFreqSev(freq_glm=nb_glm, sev_glm=gamma_glm)
```

## Methods

### JointFreqSev

```python
model = JointFreqSev(freq_glm, sev_glm, copula="sarmanov")
model.fit(data, n_col, s_col, method="ifm")   # IFM or MLE
model.premium_correction()                    # DataFrame with correction factors
model.loss_cost(X_new)                        # Corrected pure premium for new data
model.dependence_summary()                    # omega, CI, Spearman rho, AIC/BIC

# Note: for copula="gaussian" or "fgm", premium_correction() returns a single
# portfolio-average correction factor applied to all policies. Per-policy
# analytical corrections are available with copula="sarmanov" only.
```

### ConditionalFreqSev (Garrido 2016)

```python
from insurance_frequency_severity import ConditionalFreqSev

model = ConditionalFreqSev(freq_glm, sev_glm_base)
model.fit(data, n_col, s_col)
model.premium_correction()   # Correction = exp(gamma) * exp(mu_n * (exp(gamma) - 1))
```

### Diagnostics

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from insurance_frequency_severity import DependenceTest, compare_copulas, JointFreqSev

rng = np.random.default_rng(0)
n_policies = 5000
n = rng.poisson(0.10, size=n_policies)
s = np.where(n > 0, rng.gamma(3.0, 800.0, size=n_policies), np.nan)
X = pd.DataFrame({"age": rng.normal(35, 8, n_policies)})
X_const = sm.add_constant(X)
freq_glm = sm.GLM(n, X_const, family=sm.families.Poisson()).fit()
claims_mask = n > 0
sev_glm = sm.GLM(
    s[claims_mask], X_const[claims_mask],
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()
n_positive = n[claims_mask]
s_positive = s[claims_mask]

# Test independence
test = DependenceTest(n_permutations=1000)
test.fit(n_positive, s_positive)
print(test.summary())   # Kendall tau, Spearman rho, permutation p-values

# AIC/BIC comparison across copula families
comparison = compare_copulas(n, s, freq_glm, sev_glm)
print(comparison)   # Sorted by AIC: sarmanov, gaussian, fgm
```

### Report

```python
from insurance_frequency_severity import JointModelReport

report = JointModelReport(model, dependence_test=test, copula_comparison=comparison)
report.to_html(
    "pricing_review.html",
    n=n,
    s=s,
    correction_df=corrections,
)
```

## Premium correction interpretation

The correction factor is `E[N×S] / (E[N] × E[S])`. Values:

- `< 1.0`: negative dependence. High-count policyholders have lower severity than independence predicts. Independence model overstates their risk.
- `= 1.0`: independence holds.
- `> 1.0`: positive dependence. Rare but valid — e.g., some commercial lines where large customers have both high frequency and high severity.

For UK motor with typical NCD structure, expect the average correction to be 0.93–0.98 (independence overstates the pure premium by 2–7% on average, with larger corrections at the high-frequency tail).

## Theoretical background

The Sarmanov bivariate distribution:

```
f(n, s) = f_N(n) × f_S(s) × [1 + ω × φ₁(n) × φ₂(s)]
```

where φ₁, φ₂ are bounded kernel functions with zero mean under their respective marginals. When ω=0 this reduces to the product of marginals (independence). The key advantage over standard copulas: no probability integral transform is needed for the discrete frequency margin. Sklar's theorem is not unique for discrete distributions, so the "copula" of a discrete-continuous pair is not well-defined. The Sarmanov family sidesteps this entirely by working directly with the joint distribution.

Spearman's rho range for the Laplace kernel Sarmanov with NB/Gamma margins: [-3/4, 3/4] (Blier-Wong 2026). This comfortably accommodates the moderate negative dependence found in auto insurance data.

The IFM (Inference Functions for Margins) estimator:
1. Fit frequency GLM → get E[N|xᵢ] for each policy
2. Fit severity GLM → get E[S|xᵢ] for each claiming policy
3. Profile likelihood over ω: maximise Σᵢ log[1 + ω × φ₁(nᵢ; μ̂ᴺᵢ) × φ₂(sᵢ; μ̂ˢᵢ)] for observed (nᵢ, sᵢ) with nᵢ > 0

Zero-claim policies contribute no severity information; their likelihood contribution is just f_N(0), which does not depend on ω. So only observed claims inform the dependence estimate.

## Data requirements

Stable ω estimation needs approximately 20,000 policyholder-years with at least 2,000 claims. Smaller portfolios will produce wide confidence intervals on ω. The library warns you at < 1,000 policies and < 500 claims.

For small books, use `ConditionalFreqSev` — it estimates a single parameter γ from the severity GLM refitted with N as a covariate, which is more stable with less data.

---

## Performance

Benchmarked against an **independent two-part model** (Poisson GLM × Gamma GLM, pure premium = E[N] × E[S]) on 12,000 synthetic UK motor policies (8,437 train / 3,563 test) with known positive freq-sev dependence via a latent risk score. Results from `benchmarks/benchmark_insurance_frequency_severity.py` run 2026-03-16.

| Metric | Independent model | Sarmanov copula | Change |
|--------|------------------|-----------------|--------|
| Pure premium MAE vs oracle | 14.8405 | **10.6010** | -28.6% |
| Portfolio total premium bias | +22.95% | -6.77% | -16.2pp |
| Estimated Spearman rho | 0.000 | -0.015 | — |
| Fit time (seconds) | 0.105 | 0.128 | +21% |

**Correction factors:** mean 0.943, p10 0.939, p90 0.950. High-risk decile correction: 0.952 (-4.8% premium reduction vs independence). Low-risk decile: 0.940.

**Note on omega sign:** The benchmark DGP uses a positive latent risk score (z) to drive both higher frequency and severity. The fitted omega is -1.14 (Spearman rho ≈ -0.015), meaning the library detected negative-leaning dependence on this sample. The 95% CI on omega is (-1.61, +0.30), which includes zero — independence is not rejected at 5%. Despite this, the correction produces a 28.6% MAE improvement and reduces portfolio bias from +22.95% to -6.77%. This is explained by the correction absorbing some of the marginal model error: the GLMs slightly overpredict frequency for high-latent-risk policies, and the copula correction partially offsets this.

The canonical use case is a portfolio where omega is positive and statistically significant. Use `DependenceTest` before fitting to check whether the correction is supported by the data.

**When to use:** Personal lines motor or property books where `DependenceTest` indicates positive and statistically significant freq-sev dependence. The correction is analytical (closed-form, no simulation at scoring time).

**When NOT to use:** When you cannot reject independence (`DependenceTest` p-value > 0.05). Also when the book has very few claims (< 500) — the omega estimate will be too noisy.



## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_frequency_severity_demo.py).

## References

- Vernic, Bolancé, Alemany (2022). Sarmanov distribution for modeling dependence between the frequency and the average severity of insurance claims. *Insurance: Mathematics and Economics*, 102, 111–125.
- Garrido, Genest, Schulz (2016). Generalized linear models for dependent frequency and severity of insurance claims. *IME*, 70, 205–215.
- Lee, Shi (2019). A dependent frequency-severity approach to modeling longitudinal insurance claims. *IME*, 87, 115–129.
- Blier-Wong (2026). arXiv:2601.09016. Spearman rho range for Sarmanov copulas.
- Czado, Kastenmeier, Brechmann, Min (2012). A mixed copula model for insurance claims and claim sizes. *Scandinavian Actuarial Journal*, 4, 278–305.

---

Built by [Burning Cost](https://github.com/burning-cost). MIT licence.

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-dispersion](https://github.com/burning-cost/insurance-dispersion) | Double GLM for covariate-driven dispersion — models heterogeneous variance within each component |
| [insurance-severity](https://github.com/burning-cost/insurance-severity) | Heavy-tail severity with composite Pareto models and ILFs — use for the severity component when tails matter |
| [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | Quantile GBM for tail risk — non-parametric complement when the full distributional structure is uncertain |


---

## Part of the Burning Cost Toolkit

Open-source Python libraries for UK personal lines insurance pricing. [Browse all libraries](https://burning-cost.github.io/tools/)

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — `FrequencySeverityConformal` provides joint f/s coverage guarantees |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility — blends frequency and severity estimates for thin segments |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML causal inference — establishes whether frequency-severity dependence is causal or driven by observed confounders |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — monitors frequency and severity component calibration separately over time |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — produces the sign-off pack for joint frequency-severity models |
