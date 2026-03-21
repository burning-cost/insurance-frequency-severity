# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line
#
# Run end-to-end on Databricks. Do not run locally.

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: Sarmanov Copula vs Independent Frequency x Severity
# MAGIC
# MAGIC **Library:** `insurance-frequency-severity` — Sarmanov copula joint frequency-severity
# MAGIC modelling with analytical premium correction for UK personal lines pricing
# MAGIC
# MAGIC **Baseline:** Independent two-part model — Poisson GLM for frequency, Gamma GLM for
# MAGIC severity, pure premium = E[N] x E[S]. This is the standard industry approach.
# MAGIC
# MAGIC **Dataset:** 15,000 synthetic UK motor policies generated from a **pure Sarmanov DGP**
# MAGIC with a planted omega. The model should recover this planted omega directly.
# MAGIC Previously the benchmark used a latent-factor DGP, which creates a different
# MAGIC dependence structure — the Sarmanov omega the model estimates does not correspond
# MAGIC to any single planted parameter in a latent-factor model. That mismatch is now fixed.
# MAGIC
# MAGIC **Date:** 2026-03-21
# MAGIC **Library version:** 0.2.5
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Every pricing team assumes frequency and severity are independent. The standard
# MAGIC actuarial two-part model produces:
# MAGIC
# MAGIC     Pure premium = E[N|x] x E[S|x]
# MAGIC
# MAGIC This is only correct when Cov(N, S|x) = 0. When N and S are positively correlated,
# MAGIC the independence model understates the true expected loss cost. The bias concentrates
# MAGIC in high-risk segments where high-frequency policyholders also make large claims.
# MAGIC
# MAGIC The Sarmanov copula captures this dependence explicitly via omega. The correction
# MAGIC is analytical for Poisson frequency and Gamma severity. At portfolio level, the
# MAGIC correction is typically 3-8%. For the top risk decile, it can exceed 15%.
# MAGIC
# MAGIC **Key validation in this benchmark:** planted omega should equal recovered omega.
# MAGIC If the DGP is pure Sarmanov and the marginals are correctly specified, the IFM
# MAGIC estimator recovers omega accurately.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-frequency-severity statsmodels numpy scipy matplotlib pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from insurance_frequency_severity import JointFreqSev, SarmanovCopula

warnings.filterwarnings("ignore")
print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded. No torch dependency.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pure Sarmanov DGP — planted omega
# MAGIC
# MAGIC We generate (N, S) pairs directly from the Sarmanov distribution using the
# MAGIC library's own `SarmanovCopula.sample()` acceptance-rejection sampler.
# MAGIC This means:
# MAGIC - The planted omega IS the population Sarmanov omega
# MAGIC - The IFM estimator in JointFreqSev targets exactly this parameter
# MAGIC - No latent-factor mismatch; no indirect correspondence
# MAGIC
# MAGIC Covariates drive per-policy (mu_n, mu_s). We draw one (N, S) per policy from
# MAGIC the Sarmanov joint with that policy's marginal parameters.

# COMMAND ----------

rng = np.random.default_rng(42)
N_POL = 15_000

# UK motor rating covariates
age      = np.clip(rng.normal(38, 10, N_POL), 17, 80)
ncd      = np.clip(rng.normal(3, 1.5, N_POL), 0, 9).round().astype(int).astype(float)
urban    = rng.binomial(1, 0.55, N_POL).astype(float)
veh_grp  = rng.choice(["A","B","C","D","E"], N_POL, p=[0.25,0.25,0.20,0.18,0.12])
exposure = rng.uniform(0.6, 1.0, N_POL)

# True Poisson frequency mean (linear predictor + offset)
log_mu_n = (
    -2.8
    - 0.018 * age
    - 0.12  * ncd
    + 0.22  * urban
    + np.where(veh_grp=="B", 0.10,
      np.where(veh_grp=="C", 0.25,
      np.where(veh_grp=="D", 0.45,
      np.where(veh_grp=="E", 0.70, 0.0))))
)
mu_n_true = np.exp(log_mu_n) * exposure

# True Gamma severity mean
log_mu_s = (
    6.5
    + 0.005 * age
    - 0.03  * ncd
    + 0.08  * urban
    + np.where(veh_grp=="B", 0.20,
      np.where(veh_grp=="C", 0.45,
      np.where(veh_grp=="D", 0.80,
      np.where(veh_grp=="E", 1.10, 0.0))))
)
mu_s_true = np.exp(log_mu_s)
GAMMA_SHAPE = 3.0

# Planted Sarmanov omega (moderate positive — positive freq-sev dependence)
OMEGA_TRUE   = 3.5
KERNEL_THETA = 0.5    # Laplace exponent for frequency kernel (Poisson)
KERNEL_ALPHA = 0.001  # Laplace exponent for severity kernel (Gamma)

print(f"Planted omega:      {OMEGA_TRUE}")
print(f"Kernel theta:       {KERNEL_THETA}")
print(f"Kernel alpha:       {KERNEL_ALPHA}")
print(f"N policies:         {N_POL:,}")
print(f"Gamma shape:        {GAMMA_SHAPE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Sample (N, S) from pure Sarmanov DGP

# COMMAND ----------

sarm = SarmanovCopula(
    freq_family="poisson",
    sev_family="gamma",
    omega=OMEGA_TRUE,
    kernel_theta=KERNEL_THETA,
    kernel_alpha=KERNEL_ALPHA,
)

# Check omega is within feasible bounds at representative parameters
rep_freq = {"mu": float(np.median(mu_n_true))}
rep_sev  = {"mu": float(np.median(mu_s_true)), "shape": GAMMA_SHAPE}
omega_min, omega_max = sarm.omega_bounds(rep_freq, rep_sev)
print(f"Omega feasible range at median params: [{omega_min:.2f}, {omega_max:.2f}]")
print(f"Planted omega {OMEGA_TRUE} is {'VALID' if omega_min <= OMEGA_TRUE <= omega_max else 'OUT OF BOUNDS - reduce omega'}")

print("\nSampling from pure Sarmanov DGP (per-policy exact sampling)...")
t_dgp = time.perf_counter()

claim_counts  = np.zeros(N_POL, dtype=int)
avg_severities = np.zeros(N_POL, dtype=float)

BATCH = 500
for i in range(0, N_POL, BATCH):
    batch_end = min(i + BATCH, N_POL)
    for j in range(i, batch_end):
        fp = {"mu": float(mu_n_true[j])}
        sp = {"mu": float(mu_s_true[j]), "shape": GAMMA_SHAPE}
        n_s, s_s = sarm.sample(1, fp, sp, rng=rng)
        claim_counts[j]   = int(n_s[0])
        avg_severities[j] = float(s_s[0])

dgp_time = time.perf_counter() - t_dgp
print(f"DGP sampling time: {dgp_time:.1f}s")
print(f"Claim rate: {(claim_counts>0).mean():.2%}")
print(f"Mean claim count: {claim_counts.mean():.4f}")
print(f"Mean severity (positive policies): "
      f"{avg_severities[claim_counts>0].mean():.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Oracle: E[N*S|x] via Sarmanov simulation

# COMMAND ----------

print("Computing oracle E[N*S|x] per policy (N_SIM=200)...")
t_oracle = time.perf_counter()

N_SIM_ORACLE = 200
oracle_pp = np.zeros(N_POL)
for i in range(N_POL):
    fp = {"mu": float(mu_n_true[i])}
    sp = {"mu": float(mu_s_true[i]), "shape": GAMMA_SHAPE}
    n_sim, s_sim = sarm.sample(N_SIM_ORACLE, fp, sp, rng=rng)
    oracle_pp[i] = float(np.mean(n_sim * s_sim))

oracle_time = time.perf_counter() - t_oracle
print(f"Oracle time: {oracle_time:.1f}s")
print(f"Oracle mean pp:       {oracle_pp.mean():.4f}")
print(f"Independence mean pp: {(mu_n_true * mu_s_true).mean():.4f}")
print(f"Sarmanov lift:        {(oracle_pp.mean()/(mu_n_true*mu_s_true).mean()-1):.2%}")

# COMMAND ----------

df = pd.DataFrame({
    "age": age, "ncd": ncd, "urban": urban, "veh_grp": veh_grp,
    "exposure": exposure,
    "claim_count": claim_counts.astype(float),
    "avg_severity": avg_severities,
    "oracle_pp": oracle_pp,
})

rng_split = np.random.default_rng(99)
train_mask = rng_split.random(N_POL) < 0.70
train_df = df[train_mask].reset_index(drop=True)
test_df  = df[~train_mask].reset_index(drop=True)

print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print(f"Train claim rate: {(train_df.claim_count > 0).mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Marginal GLMs

# COMMAND ----------

FEAT = ["age", "ncd", "urban", "C(veh_grp)"]

freq_glm = smf.glm(
    f"claim_count ~ {' + '.join(FEAT)}", data=train_df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(train_df["exposure"].clip(1e-6)),
).fit()

claims_train = train_df[train_df.claim_count > 0].copy()
sev_glm = smf.glm(
    f"avg_severity ~ {' + '.join(FEAT)}", data=claims_train,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

print(f"Freq GLM deviance explained: {(1 - freq_glm.deviance/freq_glm.null_deviance):.1%}")
print(f"Sev GLM scale (phi):         {sev_glm.scale:.4f}  => shape ~ {1/sev_glm.scale:.2f}  (true: {GAMMA_SHAPE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Independence

# COMMAND ----------

t0_base = time.perf_counter()
mu_n_test     = freq_glm.predict(test_df)
mu_s_test     = sev_glm.predict(test_df)
pp_indep_test = mu_n_test * mu_s_test
base_time     = time.perf_counter() - t0_base

oracle_test   = test_df["oracle_pp"].values
mae_indep     = float(np.abs(pp_indep_test - oracle_test).mean())
rel_bias_indep = float((pp_indep_test.sum() - oracle_test.sum()) / oracle_test.sum())

print(f"Independence MAE vs oracle: {mae_indep:.4f}")
print(f"Independence relative bias: {rel_bias_indep:+.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sarmanov Joint Model — omega recovery

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

print(f"Fit time: {lib_time:.3f}s")
print()
print(f"Planted omega:   {OMEGA_TRUE:.4f}")
print(f"Estimated omega: {omega_hat:.4f}")
print(f"Relative error:  {abs(omega_hat - OMEGA_TRUE)/OMEGA_TRUE:.1%}")
print(f"Spearman rho:    {spearman_rho:.4f}")
print()
print(dep_summary.to_string(index=False))

# COMMAND ----------

corrections   = joint.premium_correction(
    X=test_df[["age","ncd","urban","veh_grp"]],
    exposure=test_df["exposure"].values,
)
pp_joint_test = corrections["premium_joint"].values

mae_joint     = float(np.abs(pp_joint_test - oracle_test).mean())
rel_bias_joint = float((pp_joint_test.sum() - oracle_test.sum()) / oracle_test.sum())
mae_impr      = (mae_indep - mae_joint) / mae_indep

pred_decile   = pd.qcut(pp_indep_test, 10, labels=False, duplicates="drop")
cf            = corrections["correction_factor"].values

print(f"Sarmanov MAE:       {mae_joint:.4f}  (independence: {mae_indep:.4f})")
print(f"MAE improvement:    {mae_impr:.1%}")
print(f"Sarmanov bias:      {rel_bias_joint:+.2%}  (independence: {rel_bias_indep:+.2%})")
print(f"High-risk decile correction: {cf[pred_decile==9].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results Table

# COMMAND ----------

print("=" * 68)
print(f"{'Metric':<40} {'Independence':>13} {'Sarmanov':>13}")
print("=" * 68)

rows = [
    ("Overall MAE (lower better)",            mae_indep,           mae_joint),
    ("Total premium bias abs (lower better)", abs(rel_bias_indep), abs(rel_bias_joint)),
    ("Estimated Spearman rho",                0.0,                 spearman_rho),
    ("Fit time (s)",                          base_time,           lib_time),
]
for name, b, l in rows:
    delta = (l - b) / abs(b) * 100 if b != 0 else 0.0
    print(f"{name:<40} {b:>13.4f} {l:>13.4f}   {delta:+.1f}%")

print("=" * 68)
print()
print(f"Omega recovery:")
print(f"  Planted:          {OMEGA_TRUE:.4f}")
print(f"  Estimated:        {omega_hat:.4f}")
print(f"  Relative error:   {abs(omega_hat - OMEGA_TRUE)/OMEGA_TRUE:.1%}")
sign_ok = (omega_hat > 0) == (OMEGA_TRUE > 0)
print(f"  Sign correct:     {'YES' if sign_ok else 'NO'}")
print()
print("Correction factors:")
print(f"  Mean: {cf.mean():.4f}  p10: {np.percentile(cf,10):.4f}  p90: {np.percentile(cf,90):.4f}")
print(f"  High-risk decile: {cf[pred_decile==9].mean():.4f}")
print(f"  Low-risk decile:  {cf[pred_decile==0].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Plots

# COMMAND ----------

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Plot 1: premium by decile
dec_ind, dec_jt, dec_or, dlabels = [], [], [], []
for d in range(10):
    m = pred_decile == d
    if m.sum() < 5:
        continue
    dec_ind.append(pp_indep_test[m].mean())
    dec_jt.append(pp_joint_test[m].mean())
    dec_or.append(oracle_test[m].mean())
    dlabels.append(d + 1)

ax1.plot(dlabels, dec_or,  "k-o",  lw=2, label="Oracle (DGP)", ms=7)
ax1.plot(dlabels, dec_ind, "b--s", lw=2, label=f"Independence (bias={rel_bias_indep:+.1%})", ms=7)
ax1.plot(dlabels, dec_jt,  "r-^",  lw=2, label=f"Sarmanov (bias={rel_bias_joint:+.1%})", ms=7)
ax1.set_xlabel("Predicted premium decile")
ax1.set_ylabel("Mean pure premium")
ax1.set_title("Pure premium by decile")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: omega recovery bar
ax2.bar(["Planted", "Estimated"], [OMEGA_TRUE, omega_hat],
        color=["steelblue", "tomato"], alpha=0.85, width=0.4)
ax2.axhline(OMEGA_TRUE, color="steelblue", ls="--", lw=1.5)
ax2.set_ylabel("Omega")
ax2.set_title(f"Omega recovery\nPlanted={OMEGA_TRUE:.2f}, Estimated={omega_hat:.4f}, "
              f"Error={abs(omega_hat-OMEGA_TRUE)/OMEGA_TRUE:.1%}")
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: MAE by decile
mae_ind_d, mae_jt_d = [], []
for d in range(10):
    m = pred_decile == d
    if m.sum() < 5:
        continue
    mae_ind_d.append(float(np.abs(pp_indep_test[m] - oracle_test[m]).mean()))
    mae_jt_d.append(float(np.abs(pp_joint_test[m] - oracle_test[m]).mean()))

ax3.plot(dlabels, mae_ind_d, "b--s", lw=2, label="Independence", ms=7)
ax3.plot(dlabels, mae_jt_d,  "r-^",  lw=2, label="Sarmanov", ms=7)
ax3.set_xlabel("Premium decile")
ax3.set_ylabel("MAE vs oracle")
ax3.set_title("MAE by decile")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: correction factor by decile
cf_by_d = [cf[pred_decile==d-1].mean() for d in dlabels]
ax4.bar(dlabels, cf_by_d, color="tomato", alpha=0.8)
ax4.axhline(1.0, color="black", ls="--", lw=1.5, label="Independence=1")
ax4.set_xlabel("Premium decile")
ax4.set_ylabel("Correction factor")
ax4.set_title(f"Premium correction (omega={omega_hat:.4f})")
ax4.legend()
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    f"insurance-frequency-severity: Pure Sarmanov DGP omega={OMEGA_TRUE} recovery benchmark",
    fontsize=11, fontweight="bold",
)
plt.savefig("/tmp/benchmark_frequency_severity.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved /tmp/benchmark_frequency_severity.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verdict

# COMMAND ----------

print("=" * 64)
print("VERDICT: Pure Sarmanov DGP — Omega Recovery Benchmark")
print("=" * 64)
print()
print(f"Planted omega:    {OMEGA_TRUE:.4f}")
print(f"Estimated omega:  {omega_hat:.4f}")
print(f"Relative error:   {abs(omega_hat - OMEGA_TRUE)/OMEGA_TRUE:.1%}")
print(f"Sign correct:     {'YES' if (omega_hat>0)==(OMEGA_TRUE>0) else 'NO'}")
print()
print(f"MAE improvement (Sarmanov vs independence): {mae_impr:.1%}")
print(f"Portfolio bias: independence={rel_bias_indep:+.2%}, Sarmanov={rel_bias_joint:+.2%}")
print()
print("-" * 64)
print("DGP is pure Sarmanov: planted omega = population parameter.")
print("IFM estimator targets this directly. Previously used latent-factor")
print("DGP produced a different dependence structure — the Sarmanov omega")
print("estimated by IFM did not correspond to any planted value, causing")
print("apparent sign/magnitude mismatch. Fixed in v0.2.5.")
print("-" * 64)

if abs(omega_hat - OMEGA_TRUE) / OMEGA_TRUE < 0.20:
    print("\nPASS: omega recovered within 20% of planted value.")
else:
    print(f"\nNOTE: omega error > 20%. Check kernel params / sample size.")

if mae_joint < mae_indep:
    print("PASS: Sarmanov reduces MAE vs independence.")
else:
    print("NOTE: MAE not reduced (dependence correction may be small at this omega).")

if __name__ == "__main__":
    pass
