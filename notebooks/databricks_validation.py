# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # insurance-frequency-severity: Validation on a Synthetic UK Motor Portfolio
# MAGIC
# MAGIC This notebook validates insurance-frequency-severity on a 30,000-policy synthetic
# MAGIC UK motor book with a known positive frequency-severity dependence structure.
# MAGIC
# MAGIC The central claim of this library is that the standard actuarial two-part model
# MAGIC (pure premium = E[N] × E[S]) understates aggregate risk when frequency and severity
# MAGIC are positively correlated. In commercial and fleet motor, high-frequency policyholders
# MAGIC also tend to have higher severity — the same risk profile that generates more claims
# MAGIC also generates larger ones. Multiplying independent marginal predictions misses this.
# MAGIC
# MAGIC What this notebook shows:
# MAGIC
# MAGIC 1. A 30,000-policy synthetic book with planted positive Sarmanov omega
# MAGIC 2. Independence assumption (freq × sev) — standard industry practice
# MAGIC 3. Empirical correction — portfolio-level A/E adjustment, no parametric model
# MAGIC 4. Sarmanov copula from the library — recovers omega and produces per-policy corrections
# MAGIC 5. Portfolio premium comparison: how much the independence assumption understates
# MAGIC 6. Segment-level breakdown: where the bias concentrates
# MAGIC
# MAGIC **Expected result:** The independence model understates aggregate expected loss cost
# MAGIC by 3-8% when omega is moderate-positive. The Sarmanov model recovers the planted omega
# MAGIC within 20% and corrects the premium shortfall to within 2%. The bias concentrates in
# MAGIC the top risk decile (high-frequency, high-severity commercial risks).
# MAGIC
# MAGIC ---
# MAGIC *Part of the [Burning Cost](https://burning-cost.github.io) insurance pricing toolkit.*

# COMMAND ----------

# MAGIC %pip install insurance-frequency-severity statsmodels numpy scipy pandas -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from insurance_frequency_severity import JointFreqSev, SarmanovCopula, DependenceTest

warnings.filterwarnings("ignore")

print(f"Validation run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data-Generating Process
# MAGIC
# MAGIC The DGP is a 30,000-policy UK motor book (70/30 train/test split) with:
# MAGIC - Poisson claim frequency, mean 0.06-0.25 claims/year depending on risk profile
# MAGIC - Gamma claim severity, mean £800-£4,500 depending on vehicle group and age
# MAGIC - Positive freq-sev dependence via a planted Sarmanov omega = 3.5
# MAGIC
# MAGIC **Why positive dependence is realistic for some UK motor lines:**
# MAGIC Fleet and commercial motor policies: high-frequency drivers (young, urban, heavy
# MAGIC usage) also tend to have more serious incidents. The same factors that drive claim
# MAGIC count (exposure, risk appetite, vehicle power) also drive claim cost. This is
# MAGIC different from personal motor where NCD suppression creates negative dependence.
# MAGIC
# MAGIC **The validation target:** plant omega = 3.5, recover omega = ~3.5.
# MAGIC Independence model premium / true premium should be < 1.0, with the gap
# MAGIC proportional to the strength of dependence.
# MAGIC
# MAGIC We use the library's own `SarmanovCopula.sample()` to generate the data — this is
# MAGIC a pure Sarmanov DGP, so the IFM estimator in `JointFreqSev` targets exactly
# MAGIC the planted parameter with no mismatch.

# COMMAND ----------

RNG = np.random.default_rng(42)
N   = 30_000

# Rating covariates — UK motor
driver_age  = np.clip(RNG.normal(38, 11, N), 17, 80)
ncd         = np.clip(RNG.normal(3, 1.5, N), 0, 9).round().astype(float)
urban       = RNG.binomial(1, 0.58, N).astype(float)
veh_grp     = RNG.choice(["A", "B", "C", "D", "E"], N, p=[0.25, 0.25, 0.20, 0.18, 0.12])
exposure    = RNG.uniform(0.5, 1.0, N)

# True frequency mean — Poisson
log_mu_n = (
    -2.9
    - 0.015 * driver_age
    - 0.10  * ncd
    + 0.20  * urban
    + np.where(veh_grp == "B", 0.12,
      np.where(veh_grp == "C", 0.28,
      np.where(veh_grp == "D", 0.48,
      np.where(veh_grp == "E", 0.72, 0.0))))
)
mu_n_true = np.exp(log_mu_n) * exposure

# True severity mean — Gamma
log_mu_s = (
    6.4
    + 0.006 * driver_age
    - 0.03  * ncd
    + 0.09  * urban
    + np.where(veh_grp == "B", 0.18,
      np.where(veh_grp == "C", 0.42,
      np.where(veh_grp == "D", 0.78,
      np.where(veh_grp == "E", 1.08, 0.0))))
)
mu_s_true   = np.exp(log_mu_s)
GAMMA_SHAPE = 3.0

# Planted Sarmanov omega — moderate positive dependence
OMEGA_PLANTED = 3.5
KERNEL_THETA  = 0.5
KERNEL_ALPHA  = 0.001

print(f"Portfolio: {N:,} policies")
print(f"Planted Sarmanov omega: {OMEGA_PLANTED}")
print(f"Mean frequency (unweighted): {mu_n_true.mean():.4f} claims/PY")
print(f"Mean severity (unweighted):  £{mu_s_true.mean():,.0f}")
print(f"Independence pure premium:   £{(mu_n_true * mu_s_true).mean():,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Sample (N, S) from pure Sarmanov DGP
# MAGIC
# MAGIC We call `SarmanovCopula.sample()` per policy. The sampler uses acceptance-rejection
# MAGIC on the Sarmanov density kernel. This is slow for 30k policies individually — we batch
# MAGIC in groups of 1,000 and sample each batch with matching marginal parameters.
# MAGIC
# MAGIC Allow 3-5 minutes. This is the DGP cost, not the fitting cost.

# COMMAND ----------

sarm_gen = SarmanovCopula(
    freq_family="poisson",
    sev_family="gamma",
    omega=OMEGA_PLANTED,
    kernel_theta=KERNEL_THETA,
    kernel_alpha=KERNEL_ALPHA,
)

# Verify omega is within feasible range at representative marginal params
rep_fp = {"mu": float(np.median(mu_n_true))}
rep_sp = {"mu": float(np.median(mu_s_true)), "shape": GAMMA_SHAPE}
omega_lo, omega_hi = sarm_gen.omega_bounds(rep_fp, rep_sp)
print(f"Omega feasible range at median params: [{omega_lo:.2f}, {omega_hi:.2f}]")
print(f"Planted omega {OMEGA_PLANTED} is {'VALID' if omega_lo <= OMEGA_PLANTED <= omega_hi else 'OUT OF BOUNDS'}")
print()

t_dgp = time.perf_counter()

claim_counts   = np.zeros(N, dtype=int)
avg_severities = np.zeros(N, dtype=float)

BATCH = 1_000
for i in range(0, N, BATCH):
    end = min(i + BATCH, N)
    for j in range(i, end):
        fp = {"mu": float(mu_n_true[j])}
        sp = {"mu": float(mu_s_true[j]), "shape": GAMMA_SHAPE}
        ns, ss = sarm_gen.sample(1, fp, sp, rng=RNG)
        claim_counts[j]   = int(ns[0])
        avg_severities[j] = float(ss[0])

dgp_time = time.perf_counter() - t_dgp

claim_rate = (claim_counts > 0).mean()
print(f"DGP sampling complete in {dgp_time:.0f}s")
print(f"Claim rate:                {claim_rate:.2%}")
print(f"Mean claim count:          {claim_counts.mean():.4f}")
print(f"Mean severity (claimants): £{avg_severities[claim_counts > 0].mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Oracle expected loss cost
# MAGIC
# MAGIC The oracle E[N×S|x] per policy is computed by simulating 200 draws from the
# MAGIC planted Sarmanov distribution for each policy and averaging. This is the target
# MAGIC that any correct joint model should approximate.

# COMMAND ----------

print("Computing oracle E[N*S|x] per policy (200 simulations per policy)...")
t_oracle = time.perf_counter()

N_SIM = 200
oracle_pp = np.zeros(N)
for i in range(N):
    fp = {"mu": float(mu_n_true[i])}
    sp = {"mu": float(mu_s_true[i]), "shape": GAMMA_SHAPE}
    ns, ss = sarm_gen.sample(N_SIM, fp, sp, rng=RNG)
    oracle_pp[i] = float(np.mean(ns * ss))

oracle_time = time.perf_counter() - t_oracle

indep_pp     = mu_n_true * mu_s_true
sarmanov_lift = (oracle_pp.mean() / indep_pp.mean() - 1.0)

print(f"Oracle computed in {oracle_time:.0f}s")
print(f"Oracle mean pure premium:      £{oracle_pp.mean():.4f}")
print(f"Independence mean pure premium: £{indep_pp.mean():.4f}")
print(f"Sarmanov lift over independence: {sarmanov_lift:+.2%}")
print(f"  (this is the premium the independence model will miss)")

# COMMAND ----------

# Train / test split
df = pd.DataFrame({
    "driver_age":   driver_age,
    "ncd":          ncd,
    "urban":        urban,
    "veh_grp":      veh_grp,
    "exposure":     exposure,
    "claim_count":  claim_counts.astype(float),
    "avg_severity": avg_severities,
    "oracle_pp":    oracle_pp,
    "indep_pp":     indep_pp,
})

split_rng  = np.random.default_rng(99)
train_mask = split_rng.random(N) < 0.70
train_df   = df[train_mask].reset_index(drop=True)
test_df    = df[~train_mask].reset_index(drop=True)

print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print(f"Train claim rate: {(train_df.claim_count > 0).mean():.2%}")
print(f"Train claims: {int((train_df.claim_count > 0).sum()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test for Dependence First
# MAGIC
# MAGIC `DependenceTest` runs a permutation test for independence between N and S on
# MAGIC the observed (N > 0) pairs. If this does not reject, the independence model
# MAGIC is statistically acceptable. Always run this before fitting the joint model.

# COMMAND ----------

t0 = time.perf_counter()

pos_mask = train_df.claim_count > 0
test_dep = DependenceTest(n_permutations=1000)
test_dep.fit(
    n=train_df.loc[pos_mask, "claim_count"].values,
    s=train_df.loc[pos_mask, "avg_severity"].values,
)
dep_time = time.perf_counter() - t0

print(test_dep.summary())
print(f"\nTest completed in {dep_time:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Marginal GLMs
# MAGIC
# MAGIC Standard industry two-part model: Poisson frequency GLM + Gamma severity GLM.
# MAGIC These are the inputs to `JointFreqSev` — we do not refit the marginals, we
# MAGIC plug them in and estimate omega on top.

# COMMAND ----------

FEAT = ["driver_age", "ncd", "urban", "C(veh_grp)"]

t0 = time.perf_counter()
freq_glm = smf.glm(
    f"claim_count ~ {' + '.join(FEAT)}",
    data=train_df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(train_df["exposure"].clip(1e-6)),
).fit()

sev_train   = train_df[train_df.claim_count > 0].copy()
sev_glm = smf.glm(
    f"avg_severity ~ {' + '.join(FEAT)}",
    data=sev_train,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()
glm_time = time.perf_counter() - t0

print(f"Marginal GLMs fitted in {glm_time:.2f}s")
print(f"Freq GLM deviance explained: {(1 - freq_glm.deviance/freq_glm.null_deviance):.1%}")
print(f"Sev GLM dispersion (phi):    {sev_glm.scale:.4f}  => shape ~ {1/sev_glm.scale:.2f}  (true: {GAMMA_SHAPE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Independence Model (Industry Standard)
# MAGIC
# MAGIC Pure premium = E[N|x] × E[S|x]. No correction for dependence.
# MAGIC This is what most UK pricing teams produce.

# COMMAND ----------

t0_base = time.perf_counter()
mu_n_test    = freq_glm.predict(test_df)
mu_s_test    = sev_glm.predict(test_df)
pp_indep     = mu_n_test * mu_s_test
base_time    = time.perf_counter() - t0_base

oracle_test  = test_df["oracle_pp"].values
mae_indep    = float(np.abs(pp_indep - oracle_test).mean())
bias_indep   = float((pp_indep.sum() - oracle_test.sum()) / oracle_test.sum())

print(f"Independence baseline:")
print(f"  MAE vs oracle:     £{mae_indep:.4f}")
print(f"  Portfolio bias:    {bias_indep:+.2%}  (negative = understates risk)")
print(f"  Total portfolio:   £{pp_indep.sum():,.0f}  vs oracle £{oracle_test.sum():,.0f}")
print(f"  Shortfall:         £{oracle_test.sum() - pp_indep.sum():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Empirical Correction
# MAGIC
# MAGIC The simplest correction: compute the ratio of observed total loss to predicted
# MAGIC total loss on the training set and apply it as a flat scalar to the test premiums.
# MAGIC This is calibration, not modelling — it corrects the aggregate but not the
# MAGIC segment-level mis-ranking.

# COMMAND ----------

mu_n_train   = freq_glm.predict(train_df)
mu_s_train   = sev_glm.predict(train_df)
pp_indep_tr  = mu_n_train * mu_s_train

# Empirical correction factor: observed total loss / predicted total loss
# "Observed total loss" = sum over claimants of (count * severity)
obs_total_loss_train = float(
    (train_df["claim_count"] * train_df["avg_severity"]).sum()
)
pred_total_loss_train = float(pp_indep_tr.sum())
empirical_cf = obs_total_loss_train / pred_total_loss_train

pp_empirical = pp_indep * empirical_cf
mae_empirical = float(np.abs(pp_empirical - oracle_test).mean())
bias_empirical = float((pp_empirical.sum() - oracle_test.sum()) / oracle_test.sum())

print(f"Empirical correction factor: {empirical_cf:.4f}")
print(f"  MAE vs oracle:     £{mae_empirical:.4f}")
print(f"  Portfolio bias:    {bias_empirical:+.2%}")
print()
print("Note: empirical correction fixes the aggregate bias but applies the same")
print("factor to all policies — it does not correct the segment-level ranking.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sarmanov Copula — omega Recovery
# MAGIC
# MAGIC `JointFreqSev` accepts the fitted GLM objects and estimates omega via IFM
# MAGIC (Inference Functions for Margins) on the training claims data.
# MAGIC
# MAGIC The key validation: does the estimated omega match the planted omega?
# MAGIC IFM is asymptotically unbiased for pure Sarmanov data.

# COMMAND ----------

t0_lib = time.perf_counter()

joint = JointFreqSev(
    freq_glm=freq_glm,
    sev_glm=sev_glm,
    copula="sarmanov",
    kernel_theta=KERNEL_THETA,
    kernel_alpha=KERNEL_ALPHA,
)
joint.fit(
    train_df,
    n_col="claim_count",
    s_col="avg_severity",
    exposure_col="exposure",
)
lib_time = time.perf_counter() - t0_lib

dep_summary  = joint.dependence_summary()
omega_hat    = float(dep_summary["omega"].iloc[0])
spearman_rho = float(dep_summary["spearman_rho"].iloc[0])

print(f"JointFreqSev fit time: {lib_time:.3f}s")
print()
print(f"Planted omega:   {OMEGA_PLANTED:.4f}")
print(f"Estimated omega: {omega_hat:.4f}")
print(f"Relative error:  {abs(omega_hat - OMEGA_PLANTED)/OMEGA_PLANTED:.1%}")
print(f"Sign correct:    {'YES' if (omega_hat > 0) == (OMEGA_PLANTED > 0) else 'NO'}")
print(f"Spearman rho:    {spearman_rho:.4f}")
print()
print(dep_summary.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Sarmanov Premium Correction

# COMMAND ----------

corrections = joint.premium_correction(
    X=test_df[["driver_age", "ncd", "urban", "veh_grp"]],
    exposure=test_df["exposure"].values,
)
pp_joint = corrections["premium_joint"].values
cf_vals  = corrections["correction_factor"].values

mae_joint  = float(np.abs(pp_joint - oracle_test).mean())
bias_joint = float((pp_joint.sum() - oracle_test.sum()) / oracle_test.sum())

# Decile analysis: where does the correction matter most?
pred_decile = pd.qcut(pp_indep, q=10, labels=False, duplicates="drop")

print(f"Sarmanov correction:")
print(f"  MAE vs oracle:     £{mae_joint:.4f}")
print(f"  Portfolio bias:    {bias_joint:+.2%}")
print(f"  Total portfolio:   £{pp_joint.sum():,.0f}  vs oracle £{oracle_test.sum():,.0f}")
print()
print(f"Correction factors (>1.0 = positive dependence, more premium needed):")
print(f"  Mean: {cf_vals.mean():.4f}  p10: {np.percentile(cf_vals,10):.4f}  "
      f"p90: {np.percentile(cf_vals,90):.4f}")
print(f"  High-risk decile (D10): {cf_vals[pred_decile==9].mean():.4f}")
print(f"  Low-risk decile  (D1):  {cf_vals[pred_decile==0].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Segment-Level Breakdown
# MAGIC
# MAGIC The premium shortfall from independence concentrates in high-risk segments.
# MAGIC Low-frequency policyholders have near-zero claim count variance, so the
# MAGIC independence vs joint premium difference is negligible. High-frequency segments
# MAGIC are where positive freq-sev correlation bites hardest.

# COMMAND ----------

print("Premium comparison by risk decile:")
print(f"\n{'Decile':>8} {'n':>7} {'Oracle':>10} {'Indep':>9} {'Sarmanov':>10} "
      f"{'Indep bias':>12} {'Sarm bias':>11}")
print("-" * 72)

for d in range(10):
    m = pred_decile == d
    if m.sum() < 5:
        continue
    n_d = m.sum()
    or_d  = oracle_test[m].mean()
    in_d  = pp_indep[m].mean()
    sa_d  = pp_joint[m].mean()
    bi_in = (in_d - or_d) / or_d
    bi_sa = (sa_d - or_d) / or_d
    print(f"  {d+1:>6}  {n_d:>7}  £{or_d:>8.2f}  £{in_d:>7.2f}  £{sa_d:>8.2f}  "
          f"{bi_in:>11.1%}  {bi_sa:>10.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Results Summary

# COMMAND ----------

mae_impr_sarm = (mae_indep - mae_joint) / mae_indep
mae_impr_emp  = (mae_indep - mae_empirical) / mae_indep

print("=" * 72)
print("VALIDATION SUMMARY")
print("=" * 72)
print(f"\n{'Metric':<40} {'Independence':>14} {'Empirical CF':>13} {'Sarmanov':>10}")
print("-" * 72)
print(f"{'MAE vs oracle (lower better)':<40} £{mae_indep:>12.4f} £{mae_empirical:>11.4f} £{mae_joint:>8.4f}")
print(f"{'Portfolio bias':<40} {bias_indep:>14.2%} {bias_empirical:>13.2%} {bias_joint:>10.2%}")
print(f"{'MAE improvement vs independence':<40} {'—':>14} {mae_impr_emp:>13.1%} {mae_impr_sarm:>10.1%}")
print()
print(f"Omega recovery:")
print(f"  Planted:         {OMEGA_PLANTED:.4f}")
print(f"  Estimated:       {omega_hat:.4f}")
print(f"  Relative error:  {abs(omega_hat - OMEGA_PLANTED)/OMEGA_PLANTED:.1%}")
print(f"  Sign correct:    {'YES' if (omega_hat > 0) == (OMEGA_PLANTED > 0) else 'NO'}")
print()
print(f"Sarmanov lift (true E[NS] / E[N]*E[S]): {sarmanov_lift:+.2%}")
print()
print("EXPECTED PERFORMANCE (30k-policy motor book, Sarmanov DGP, omega=3.5):")
print("  Independence understates aggregate premium by 3-8% with moderate omega")
print("  Empirical CF fixes aggregate bias but not segment ranking")
print("  Sarmanov copula corrects both: MAE improves 20-35%, segment rank preserved")
print("  Omega recovered within 20% on 21k training policies with 2k+ claims")
print(f"  Fit time: {lib_time:.3f}s — IFM profile likelihood, closed-form correction")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. When to Use This — Practical Guidance
# MAGIC
# MAGIC **Use Sarmanov copula correction when:**
# MAGIC
# MAGIC - `DependenceTest` rejects independence (p < 0.05). Run the test first — the
# MAGIC   correction is not free, and if the data does not support dependence, you are adding
# MAGIC   a free parameter for noise.
# MAGIC - The portfolio is commercial motor, fleet, or property where high-frequency
# MAGIC   policyholders also have high severity. UK personal motor with NCD suppression
# MAGIC   typically shows negative omega (independence model overstates risk) — check the
# MAGIC   sign of omega before applying.
# MAGIC - You have ≥ 20,000 policyholder-years and ≥ 2,000 claims. Omega estimation is
# MAGIC   noisy below these thresholds — the library warns you.
# MAGIC - You want per-policy correction factors, not just a portfolio-level scalar. The
# MAGIC   Sarmanov correction is analytical (no simulation at scoring time), so it is
# MAGIC   production-safe at any portfolio size.
# MAGIC
# MAGIC **Use empirical correction (flat scalar) when:**
# MAGIC
# MAGIC - The portfolio is too small for stable omega estimation (< 1,000 claims).
# MAGIC - You just need aggregate calibration, not per-policy adjustments.
# MAGIC - The pricing committee is not ready for copula methodology — the empirical CF
# MAGIC   is a defensible stepping stone.
# MAGIC
# MAGIC **Use `ConditionalFreqSev` (Garrido fallback) when:**
# MAGIC
# MAGIC - Data is small: 500-2,000 claims. The Garrido model adds claim count N as a
# MAGIC   covariate in the severity GLM — one extra parameter that is more stable than omega.
# MAGIC
# MAGIC **When NOT to use any correction:**
# MAGIC
# MAGIC - `DependenceTest` does not reject independence. A correction on independent data
# MAGIC   adds variance without reducing bias.
# MAGIC - The marginal GLMs themselves are misspecified. The copula corrects dependence,
# MAGIC   not marginal model error. Fix the GLMs first.
# MAGIC
# MAGIC **Data requirements:**
# MAGIC
# MAGIC - At minimum: observed (N_i, S_i) pairs for each claiming policyholder (N_i > 0).
# MAGIC   Zero-claim policies contribute no severity information and do not constrain omega.
# MAGIC - The IFM estimator accepts your existing fitted statsmodels GLM objects — no
# MAGIC   need to refit. Pass `freq_glm` and `sev_glm` directly to `JointFreqSev`.
# MAGIC - Covariates must match between the GLMs and the data passed to `premium_correction()`.
# MAGIC
# MAGIC **On the omega sign in personal motor:**
# MAGIC
# MAGIC UK personal motor with NCD suppression typically has negative omega — frequent
# MAGIC claimants suppress borderline claims, so conditional severity given high N is
# MAGIC lower than independence would predict. A negative correction reduces the premium.
# MAGIC Always check: correction factor > 1.0 means you have been under-charging;
# MAGIC < 1.0 means over-charging.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *insurance-frequency-severity v0.2+ | [GitHub](https://github.com/burning-cost/insurance-frequency-severity) | [Burning Cost](https://burning-cost.github.io)*
