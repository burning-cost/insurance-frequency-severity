# insurance-frequency-severity

Sarmanov copula joint frequency-severity modelling for UK personal lines insurance.

Challenges the independence assumption in the standard two-model GLM framework. Your frequency GLM and severity GLM are correct. The problem is multiplying their predictions together as though claim count and average severity are unrelated — they are not.

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
pip install insurance-frequency-severity
```

## Quickstart

```python
import pandas as pd
from insurance_frequency_severity import JointFreqSev, DependenceTest

# Test for dependence first
test = DependenceTest()
test.fit(n=claims_df["claim_count"], s=claims_df["avg_severity"])
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

We accept any object with `.predict()` and `.fittedvalues`. The library detects the marginal family from `model.family` (statsmodels convention). For non-statsmodels GLMs, pass your own parameter dictionaries directly.

```python
# Works with statsmodels GLM results
import statsmodels.api as sm
nb_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=0.8)).fit()
gamma_glm = sm.GLM(s, X_claims, family=sm.families.Gamma(link=sm.families.links.Log())).fit()

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
```

### ConditionalFreqSev (Garrido 2016)

```python
model = ConditionalFreqSev(freq_glm, sev_glm_base)
model.fit(data, n_col, s_col)
model.premium_correction()   # Uses exp(gamma * E[N|x]) correction
```

### Diagnostics

```python
from insurance_frequency_severity import DependenceTest, compare_copulas

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
report.to_html("pricing_review.html", n=n_col, s=s_col, correction_df=corrections)
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

Spearman's rho range for Sarmanov: [-3/4, 3/4] (Blier-Wong 2026). This comfortably accommodates the moderate negative dependence found in auto insurance data.

The IFM (Inference Functions for Margins) estimator:
1. Fit frequency GLM → get E[N|xᵢ] for each policy
2. Fit severity GLM → get E[S|xᵢ] for each claiming policy
3. Profile likelihood over ω: maximise Σᵢ log[1 + ω × φ₁(nᵢ; μ̂ᴺᵢ) × φ₂(sᵢ; μ̂ˢᵢ)] for observed (nᵢ, sᵢ) with nᵢ > 0

Zero-claim policies contribute no severity information; their likelihood contribution is just f_N(0), which does not depend on ω. So only observed claims inform the dependence estimate.

## Data requirements

Stable ω estimation needs approximately 20,000 policyholder-years with at least 2,000 claims. Smaller portfolios will produce wide confidence intervals on ω. The library warns you at < 1,000 policies and < 500 claims.

For small books, use `ConditionalFreqSev` — it estimates a single parameter γ from the severity GLM refitted with N as a covariate, which is more stable with less data.

## References

- Vernic, Bolancé, Alemany (2022). Sarmanov distribution for modeling dependence between the frequency and the average severity of insurance claims. *Insurance: Mathematics and Economics*, 102, 111–125.
- Garrido, Genest, Schulz (2016). Generalized linear models for dependent frequency and severity of insurance claims. *IME*, 70, 205–215.
- Lee, Shi (2019). A dependent frequency-severity approach to modeling longitudinal insurance claims. *IME*, 87, 115–129.
- Blier-Wong (2026). arXiv:2601.09016. Spearman rho range for Sarmanov copulas.
- Czado, Kastenmeier, Brechmann, Min (2012). A mixed copula model for insurance claims and claim sizes. *Scandinavian Actuarial Journal*, 4, 278–305.

---

Built by [Burning Cost](https://github.com/burning-cost). MIT licence.
