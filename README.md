# insurance-frequency-severity

**Sarmanov copula joint frequency-severity modelling — analytical premium correction without refitting your GLMs.**

[![PyPI](https://img.shields.io/pypi/v/insurance-frequency-severity)](https://pypi.org/project/insurance-frequency-severity/) [![Downloads](https://img.shields.io/pypi/dm/insurance-frequency-severity)](https://pypi.org/project/insurance-frequency-severity/) [![Python](https://img.shields.io/pypi/pyversions/insurance-frequency-severity)](https://pypi.org/project/insurance-frequency-severity/) [![License](https://img.shields.io/pypi/l/insurance-frequency-severity)](https://pypi.org/project/insurance-frequency-severity/)

---

## The problem

Every UK motor pricing team multiplies a Poisson frequency GLM by a Gamma severity GLM and calls it pure premium. This assumes claim count and average severity are independent given the rating factors — they are not.

In UK motor, the NCD structure suppresses borderline claims: policyholders aware of the NCD threshold do not report near-miss incidents. The result is a systematic negative correlation between claim count and average severity. Ignoring this biases the pure premium, and the bias concentrates in your highest-risk accounts. Vernic, Bolancé and Alemany (2022) measured this at €5–55+ per policyholder on a Spanish auto book. The directional effect in UK motor is the same.

**Blog post:** [Your Frequency-Severity Independence Assumption Is Costing You Premium](https://burning-cost.github.io/2026/05/15/frequency-severity-independence-is-costing-you-premium/)

---

## Why this library?

Standard copulas (Gaussian, Clayton) require a probability integral transform for the discrete frequency margin — and Sklar's theorem is not unique for discrete distributions. The Sarmanov bivariate distribution sidesteps this entirely by working directly with the joint density, giving you analytically closed-form per-policy correction factors without simulation.

IFM estimation means you plug in your already-fitted statsmodels GLM objects. The library estimates the dependence parameter omega on top of your existing models. You do not refit the marginals.

---

## Compared to alternatives

| | Independent GLM multiplication | Gaussian copula | Tweedie single model | **insurance-frequency-severity** |
|---|---|---|---|---|
| Handles discrete-continuous margins correctly | No (assumption) | Partial (PIT approximation) | N/A | Yes (Sarmanov) |
| Per-policy correction factors | No | Portfolio average only | N/A | Yes |
| Uses existing GLM objects | Yes | Requires refitting | No | Yes (IFM) |
| Test for dependence first | No | No | No | Yes (`DependenceTest`) |
| AIC/BIC copula comparison | No | No | No | Yes |
| HTML model report | No | No | No | Yes (`JointModelReport`) |

---

## Quickstart

```bash
uv add insurance-frequency-severity
```

```python
import pandas as pd
from insurance_frequency_severity import JointFreqSev, DependenceTest

# Test for dependence before committing to a correction
test = DependenceTest()
test.fit(n=claim_count[claims_mask], s=avg_severity[claims_mask])
print(test.summary())  # Kendall tau, Spearman rho, permutation p-values

# Fit joint model on top of your existing fitted GLMs
policy_df = pd.DataFrame({"claim_count": claim_count, "avg_severity": avg_severity})
model = JointFreqSev(freq_glm=my_nb_glm, sev_glm=my_gamma_glm, copula="sarmanov")
model.fit(policy_df, n_col="claim_count", s_col="avg_severity")

corrections = model.premium_correction()
print(corrections[["mu_n", "mu_s", "correction_factor", "premium_joint"]].describe())
```

---

## The three methods

**Sarmanov copula (primary)** — the recommended approach for books with enough data (≥20,000 policyholder-years, ≥2,000 claims). Handles the discrete-continuous mixed margins problem correctly. Per-policy analytical correction factors, no simulation.

**Gaussian copula (comparison)** — the standard actuarial approach. Uses PIT approximation for the discrete frequency margin. Good for presenting results in familiar terms, or for comparing rho estimates. Returns a portfolio-average correction factor, not per-policy factors.

**Garrido conditional fallback** (`ConditionalFreqSev`) — adds claim count N as a covariate in the severity GLM. One extra GLM parameter. More stable on small books where omega estimation from the Sarmanov would be unreliable.

---

## Complete example

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from insurance_frequency_severity import (
    JointFreqSev,
    ConditionalFreqSev,
    DependenceTest,
    compare_copulas,
    JointModelReport,
)

rng = np.random.default_rng(42)
n_policies = 5000
claim_count = rng.poisson(0.10, size=n_policies)
avg_severity = np.where(
    claim_count > 0,
    rng.gamma(shape=3.0, scale=800.0, size=n_policies),
    np.nan,
)
X = pd.DataFrame({
    "age": rng.normal(35, 8, n_policies),
    "ncb": rng.normal(5, 2, n_policies),
})
X_const = sm.add_constant(X)
claims_mask = claim_count > 0

my_nb_glm = sm.GLM(
    claim_count, X_const,
    family=sm.families.NegativeBinomial(alpha=0.8),
).fit()
my_gamma_glm = sm.GLM(
    avg_severity[claims_mask], X_const[claims_mask],
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

# Step 1: test for dependence
test = DependenceTest(n_permutations=1000)
test.fit(claim_count[claims_mask], avg_severity[claims_mask])
print(test.summary())

# Step 2: compare copula families
comparison = compare_copulas(claim_count, avg_severity, my_nb_glm, my_gamma_glm)
print(comparison)  # sorted by AIC: sarmanov, gaussian, fgm

# Step 3: fit and correct
policy_df = pd.DataFrame({"claim_count": claim_count, "avg_severity": avg_severity})
model = JointFreqSev(freq_glm=my_nb_glm, sev_glm=my_gamma_glm, copula="sarmanov")
model.fit(policy_df, n_col="claim_count", s_col="avg_severity")
print(model.dependence_summary())  # omega, CI, Spearman rho, AIC/BIC
corrections = model.premium_correction()

# Step 4: generate model report
report = JointModelReport(model, dependence_test=test, copula_comparison=comparison)
report.to_html("pricing_review.html", n=claim_count, s=avg_severity, correction_df=corrections)
```

---

## Garrido conditional fallback

```python
from insurance_frequency_severity import ConditionalFreqSev

policy_df = pd.DataFrame({"claim_count": claim_count, "avg_severity": avg_severity})

model = ConditionalFreqSev(my_nb_glm, my_gamma_glm)
model.fit(policy_df, n_col="claim_count", s_col="avg_severity")
model.premium_correction()
```

Use this when you have fewer than 1,000 claims and cannot reliably estimate omega.

---

## Reading the correction factors

`premium_correction()` returns the factor `E[N×S] / (E[N] × E[S])` per policy:

- `< 1.0`: negative dependence. High-count policyholders have lower severity than independence predicts. Independence model overstates their risk.
- `= 1.0`: independence holds.
- `> 1.0`: positive dependence — valid in some commercial lines where large customers have both high frequency and high severity.

For UK motor with typical NCD structure, expect the average correction to be 0.93–0.98, with larger corrections at the high-frequency tail.

---

## Validated performance

On a 30,000-policy synthetic UK motor book with planted Sarmanov dependence (omega=3.5):

| Metric | Independence | Sarmanov copula |
|---|---|---|
| Portfolio premium bias | −3% to −8% | ~0% |
| High-risk decile correction factor | 1.00 | 1.05–1.15× |
| Omega recovery relative error | — | 10–20% |
| Fit time | < 1s | < 1s |

In a benchmark on 12,000 synthetic policies with latent freq-sev dependence, the Sarmanov correction reduced pure premium MAE vs oracle by 28.6% and portfolio bias from +22.95% to −6.77%.

Always run `DependenceTest` before fitting. If independence cannot be rejected (p > 0.05) and your book has fewer than 1,000 claims, use `ConditionalFreqSev` instead.

Full validation notebook: `notebooks/databricks_validation.py`.

---

## Data requirements

Stable omega estimation requires approximately 20,000 policyholder-years with at least 2,000 claims. The library warns at < 1,000 policies and < 500 claims. Zero-claim policies contribute no information about the dependence parameter — only observed (n > 0, s) pairs enter the likelihood.

---

## Theoretical background

The Sarmanov bivariate distribution:

```
f(n, s) = f_N(n) * f_S(s) * [1 + omega * phi_1(n) * phi_2(s)]
```

where phi_1 and phi_2 are bounded kernel functions with zero mean under their marginals. When omega=0 this reduces to the independence model. The key advantage: no probability integral transform is needed for the discrete frequency margin, which is required by Gaussian/Clayton copulas and is not well-defined for discrete distributions.

IFM estimation: fit frequency GLM → fit severity GLM → profile likelihood over omega using only observed (n > 0, s) pairs. Closed-form, no simulation.

Reference: Vernic, Bolancé, Alemany (2022), *Insurance: Mathematics and Economics*, 102, 111–125.

---

## Limitations

- Stable omega estimation requires ≥20,000 policyholder-years and ≥2,000 claims. Smaller books produce wide confidence intervals. Always check `DependenceTest` first.
- Per-policy analytical corrections are only available with `copula="sarmanov"`. Gaussian and FGM copulas return a portfolio-average factor only.
- The library wraps statsmodels GLM objects. Non-statsmodels models may work via `.predict()` but kernel parameters are inferred from statsmodels-specific attributes.
- The correction is not recalibrated as the portfolio evolves. If the NCD scale is restructured, re-estimate omega on recent data.

---

## Part of the Burning Cost stack

Takes claims data and your existing fitted GLMs. Feeds Sarmanov-corrected joint premium estimates into [insurance-optimise](https://github.com/burning-cost/insurance-optimise) and [insurance-conformal](https://github.com/burning-cost/insurance-conformal). [See the full stack](https://burning-cost.github.io/stack/)

| Library | Description |
|---|---|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — joint frequency-severity coverage guarantees |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility — blends frequency and severity estimates for thin segments |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — monitors frequency and severity calibration separately |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — sign-off pack for joint frequency-severity models |

---

## References

**Sarmanov copula foundations**

- Sarmanov, O.V. (1966). "Generalized normal correlation and two-dimensional Fréchet classes." *Soviet Mathematics Doklady*, 7, 596–599. (Original Sarmanov bivariate distribution construction.)
- Lee, M.T. & Cha, J.H. (2015). "On two general classes of discrete bivariate distributions." *The American Statistician*, 69(3), 221–230. [doi:10.1080/00031305.2015.1044710](https://doi.org/10.1080/00031305.2015.1044710) (Sarmanov family properties relevant to count-continuous joint models.)

**Insurance frequency-severity joint modelling**

- Vernic, R., Bolancé, C. & Alemany, R. (2022). "Sarmanov distribution for modeling dependence between the frequency and the average severity of insurance claims." *Insurance: Mathematics and Economics*, 102, 111–125. [doi:10.1016/j.insmatheco.2021.11.003](https://doi.org/10.1016/j.insmatheco.2021.11.003)
- Garrido, J., Genest, C. & Schulz, J. (2016). "Generalized linear models for dependent frequency and severity of insurance claims." *Insurance: Mathematics and Economics*, 70, 205–215. [doi:10.1016/j.insmatheco.2016.06.006](https://doi.org/10.1016/j.insmatheco.2016.06.006)
- Lee, G. & Shi, P. (2019). "A dependent frequency-severity approach to modeling longitudinal insurance claims." *Insurance: Mathematics and Economics*, 87, 115–129. [doi:10.1016/j.insmatheco.2019.04.004](https://doi.org/10.1016/j.insmatheco.2019.04.004)
- Czado, C., Kastenmeier, R., Brechmann, E.C. & Min, A. (2012). "A mixed copula model for insurance claims and claim sizes." *Scandinavian Actuarial Journal*, 4, 278–305. [doi:10.1080/03461238.2010.546009](https://doi.org/10.1080/03461238.2010.546009)
- Frees, E.W. & Valdez, E.A. (1998). "Understanding Relationships Using Copulas." *North American Actuarial Journal*, 2(1), 1–25. [doi:10.1080/10920277.1998.10595667](https://doi.org/10.1080/10920277.1998.10595667) (Foundational copula reference for actuarial dependence modelling.)

---

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-frequency-severity/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-frequency-severity/issues)
- **Blog and tutorials:** [burning-cost.github.io](https://burning-cost.github.io)
- **Training course:** [Insurance Pricing in Python](https://burning-cost.github.io/course) — Module 4 covers frequency-severity modelling. £97 one-time.

## Licence

MIT

## Related Libraries

| Library | Description |
|---------|-------------|
| [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility — apply to frequency-severity components when segments have thin data |
| [`insurance-severity`](https://github.com/burning-cost/insurance-severity) | Spliced severity distributions — parametric severity modelling to feed into the frequency-severity framework |
| [`insurance-distributional`](https://github.com/burning-cost/insurance-distributional) | Distributional GBMs — alternative when you want a single model for the full loss distribution |
