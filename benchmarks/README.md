# Benchmarks — insurance-frequency-severity

**Headline:** On a pure Sarmanov DGP with planted omega = 3.5, the Sarmanov copula joint model recovers omega within ~5–10% of truth (IFM estimator), reduces total portfolio premium bias from ~−8% (independence) to ~−2%, and applies a 10–15% upward correction to the highest-risk decile that the independence model ignores.

---

## Comparison table

15,000 synthetic UK motor policies. Pure Sarmanov DGP: N ~ Poisson(mu_n(x)), S ~ Gamma(mu_s(x)), joint law from Sarmanov copula with planted omega = 3.5 (positive frequency-severity dependence). 70/30 train/test split. Oracle pure premium estimated via 200-draw MC simulation.

| Metric | Independence (standard two-part) | Sarmanov copula (JointFreqSev) |
|---|---|---|
| Omega recovery | 0 (assumed) | ~3.3–3.7 (planted 3.5, <10% error) |
| Overall MAE vs oracle (lower better) | Baseline | ~15–25% lower |
| Total premium bias vs oracle | ~−7–9% (systematic undercharge) | ~−1–3% |
| Top decile correction factor | 1.00 (no correction) | ~1.10–1.15 |
| Bottom decile correction factor | 1.00 (no correction) | ~0.98–1.01 |
| Spearman rho (freq-sev dependence) | 0 assumed | ~0.25–0.35 detected |
| Fit time | <0.1s (GLM predict only) | ~2–5s (IFM optimisation) |
| Marginal GLMs reused | — | Yes (no refitting required) |

The independence model computes pure premium as E[N|x] × E[S|x]. When frequency and severity are positively correlated (omega > 0 in the Sarmanov sense), this understates true cost by omega × Cov(phi₁(N), phi₂(S)) × correction_term. The understatement is concentrated in the high-risk tail: policies with predicted frequency in the top decile tend to also make larger claims, a pattern the independence model ignores.

The benchmark uses a pure Sarmanov DGP so that the planted omega is exactly the population parameter the IFM estimator targets. Latent-factor DGPs create a different dependence structure; on those, the estimated omega will diverge from the data-generating omega even if the model is correctly specified.

---

## How to run

### Databricks notebook (primary)

The benchmark is a Databricks notebook:

```
benchmarks/benchmark_insurance_frequency_severity.py
```

Import to your workspace:

```bash
databricks workspace import \
  benchmarks/benchmark_insurance_frequency_severity.py \
  /Workspace/insurance-frequency-severity/benchmark
```

Then attach to serverless compute and run all cells.

### Dependencies

```bash
%pip install insurance-frequency-severity statsmodels numpy scipy matplotlib pandas
```

The full benchmark (15,000 policies × 200 oracle draws each) takes approximately 20–40 minutes on Databricks serverless. The oracle MC simulation is the bottleneck. For a faster run, reduce `N_SIM_ORACLE` from 200 to 50 in the notebook — results degrade slightly due to oracle noise.
