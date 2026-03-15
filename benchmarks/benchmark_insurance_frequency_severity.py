# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-frequency-severity Sarmanov copula vs independence assumption
# MAGIC
# MAGIC **Library:** `insurance-frequency-severity` — Sarmanov copula joint frequency-severity
# MAGIC modelling with analytical premium correction for mixed discrete-continuous margins.
# MAGIC
# MAGIC **Baseline:** independent two-part model — Poisson GLM for frequency, Gamma GLM
# MAGIC for severity. Pure premium = E[N|x] * E[S|x]. This is the standard UK actuarial
# MAGIC approach and what every team builds first.
# MAGIC
# MAGIC **Dataset:** 12,000 synthetic UK motor policies with known positive freq-sev
# MAGIC dependence. A latent risk score drives both higher claim frequency and higher
# MAGIC severity. The true expected loss cost is computed from the DGP analytically.
# MAGIC
# MAGIC **Date:** 2026-03-15
# MAGIC **Library version:** 0.2.0
# MAGIC
# MAGIC ---
# MAGIC The independence model is a special case of the Sarmanov copula (omega=0).
# MAGIC When omega > 0, policies with high claim frequency also tend to make large
# MAGIC claims. The independence premium = E[N] * E[S] understates true cost by
# MAGIC omega * E[N*phi_1(N)] * E[S*phi_2(S)]. For the top risk decile, this bias
# MAGIC can exceed 10–15%, which is the segment where mispricing costs most.

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

from insurance_frequency_severity import JointFreqSev

warnings.filterwarnings("ignore")
print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Correlated Frequency-Severity Data
# MAGIC
# MAGIC DGP: a latent risk score z drives both frequency and severity.
# MAGIC - N | x ~ Poisson(mu_n(x))  where mu_n depends on covariates + z
# MAGIC - S | x, N>0 ~ Gamma(shape, scale=mu_s(x)/shape)  where mu_s depends on covariates + z
# MAGIC - z ~ Gamma(2, 0.5), mean=1. Policies with high z have both more and bigger claims.
# MAGIC
# MAGIC True E[N*S|x] = E[N|x]*E[S|x] + omega * corr_term.
# MAGIC We compute the true aggregate premium analytically for the oracle comparison.

# COMMAND ----------

rng = np.random.default_rng(42)
N_POL = 12_000

age      = np.clip(rng.normal(38, 10, N_POL), 17, 80)
ncd      = np.clip(rng.normal(3, 1.5, N_POL), 0, 9).round().astype(int).astype(float)
urban    = rng.binomial(1, 0.55, N_POL).astype(float)
veh_grp  = rng.choice(["A","B","C","D","E"], N_POL, p=[0.25,0.25,0.20,0.18,0.12])
exposure = rng.uniform(0.6, 1.0, N_POL)

# Latent risk score — creates positive freq-sev dependence
z = rng.gamma(2, 0.5, N_POL)  # mean = 1, CV = 1/sqrt(2)

# True frequency: depends on covariates + latent z
log_mu_n = (
    -2.8
    - 0.018 * age
    - 0.12  * ncd
    + 0.22  * urban
    + np.where(veh_grp=="B",0.10,np.where(veh_grp=="C",0.25,np.where(veh_grp=="D",0.45,np.where(veh_grp=="E",0.70,0.0))))
    + 0.35  * (z - 1)   # latent risk drives frequency
)
mu_n_true = np.exp(log_mu_n) * exposure
claim_counts = rng.poisson(mu_n_true)

# True severity: depends on covariates + latent z (positive dependence!)
log_mu_s = (
    6.5
    + 0.005 * age
    - 0.03  * ncd
    + 0.08  * urban
    + np.where(veh_grp=="B",0.20,np.where(veh_grp=="C",0.45,np.where(veh_grp=="D",0.80,np.where(veh_grp=="E",1.10,0.0))))
    + 0.25  * (z - 1)   # same latent risk drives severity
)
mu_s_true = np.exp(log_mu_s)
GAMMA_SHAPE = 3.0

# Generate observed severities for policies with claims
avg_sev = np.where(
    claim_counts > 0,
    rng.gamma(GAMMA_SHAPE, mu_s_true / GAMMA_SHAPE),
    0.0,
)

df = pd.DataFrame({
    "age": age, "ncd": ncd, "urban": urban, "veh_grp": veh_grp,
    "exposure": exposure, "z": z,
    "claim_count": claim_counts.astype(float),
    "avg_severity": avg_sev,
    "mu_n_true": mu_n_true,
    "mu_s_true": mu_s_true,
    "true_pure_premium": mu_n_true * mu_s_true * (1 + 0.30 * (z - 1)),  # approx joint
})

# True oracle: simulate 100k samples per policy average
print("Computing oracle E[N*S|x] via simulation (this defines ground truth)...")
oracle_pp = np.zeros(N_POL)
N_SIM = 500
for i in range(N_POL):
    z_sim = rng.gamma(2, 0.5, N_SIM)
    log_n = (-2.8 - 0.018*age[i] - 0.12*ncd[i] + 0.22*urban[i]
             + (0.10 if veh_grp[i]=="B" else 0.25 if veh_grp[i]=="C" else 0.45 if veh_grp[i]=="D" else 0.70 if veh_grp[i]=="E" else 0.0)
             + 0.35*(z_sim - 1))
    log_s = (6.5 + 0.005*age[i] - 0.03*ncd[i] + 0.08*urban[i]
             + (0.20 if veh_grp[i]=="B" else 0.45 if veh_grp[i]=="C" else 0.80 if veh_grp[i]=="D" else 1.10 if veh_grp[i]=="E" else 0.0)
             + 0.25*(z_sim - 1))
    n_sim = rng.poisson(np.exp(log_n) * exposure[i])
    s_sim = rng.gamma(GAMMA_SHAPE, np.exp(log_s) / GAMMA_SHAPE)
    oracle_pp[i] = float(np.mean(n_sim * s_sim))

df["oracle_pp"] = oracle_pp

train_mask = rng.random(N_POL) < 0.70
train_df = df[train_mask].reset_index(drop=True)
test_df  = df[~train_mask].reset_index(drop=True)

print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print(f"Observed frequency: {train_df.claim_count.sum()/train_df.exposure.sum():.4f}")
print(f"Claim rate (policies with claims): {(train_df.claim_count > 0).mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit Marginal GLMs (used by both baseline and library)

# COMMAND ----------

FEAT = ["age", "ncd", "urban", "C(veh_grp)"]
formula_freq = f"claim_count ~ {' + '.join(FEAT)}"
formula_sev  = f"avg_severity ~ {' + '.join(FEAT)}"

freq_glm = smf.glm(
    formula_freq, data=train_df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(train_df["exposure"].clip(1e-6)),
).fit()

claims_df = train_df[train_df.claim_count > 0].copy()
sev_glm = smf.glm(
    formula_sev, data=claims_df,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

print(f"Freq GLM deviance explained: {(1-freq_glm.deviance/freq_glm.null_deviance):.1%}")
print(f"Sev GLM scale (phi):         {sev_glm.scale:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Independence Pure Premium

# COMMAND ----------

t0_base = time.perf_counter()

mu_n_train = freq_glm.predict(train_df)   # predicted claims
mu_s_train = sev_glm.predict(train_df)    # predicted severity per claim
pp_indep_train = mu_n_train * mu_s_train

mu_n_test = freq_glm.predict(test_df)
mu_s_test = sev_glm.predict(test_df)
pp_indep_test = mu_n_test * mu_s_test

base_time = time.perf_counter() - t0_base
oracle_test = test_df["oracle_pp"].values

mae_indep = float(np.abs(pp_indep_test - oracle_test).mean())
rel_bias_indep = float((pp_indep_test.sum() - oracle_test.sum()) / oracle_test.sum())

print(f"Baseline time: {base_time:.3f}s")
print(f"Independence MAE vs oracle: {mae_indep:.4f}")
print(f"Independence total premium: {pp_indep_test.sum():.0f}")
print(f"Oracle total premium:        {oracle_test.sum():.0f}")
print(f"Relative bias: {rel_bias_indep:+.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Library: Sarmanov Copula Joint Model

# COMMAND ----------

t0_lib = time.perf_counter()

joint = JointFreqSev(
    freq_glm=freq_glm,
    sev_glm=sev_glm,
    copula="sarmanov",
)
joint.fit(
    train_df,
    n_col="claim_count",
    s_col="avg_severity",
    exposure_col="exposure",
)

lib_time = time.perf_counter() - t0_lib

dep_summary = joint.dependence_summary()
omega_hat = float(dep_summary["omega"].iloc[0])
spearman_rho = float(dep_summary["spearman_rho"].iloc[0])

print(f"Library fit time: {lib_time:.3f}s")
print(f"Estimated omega: {omega_hat:.4f}")
print(f"Spearman rho:    {spearman_rho:.4f}")
print()
print(dep_summary.to_string(index=False))

# COMMAND ----------

# Compute corrected pure premiums on test set
corrections = joint.premium_correction(X=test_df[["age","ncd","urban","veh_grp"]],
                                        exposure=test_df["exposure"].values)
pp_joint_test = corrections["premium_joint"].values

mae_joint = float(np.abs(pp_joint_test - oracle_test).mean())
rel_bias_joint = float((pp_joint_test.sum() - oracle_test.sum()) / oracle_test.sum())

print(f"Joint MAE vs oracle:    {mae_joint:.4f}")
print(f"Independence MAE:        {mae_indep:.4f}")
print(f"MAE improvement:         {(mae_indep - mae_joint)/mae_indep:.1%}")
print()
print(f"Joint total premium:     {pp_joint_test.sum():.0f}")
print(f"Independence total:      {pp_indep_test.sum():.0f}")
print(f"Oracle total:            {oracle_test.sum():.0f}")
print(f"Joint relative bias:     {rel_bias_joint:+.2%}")
print(f"Independence bias:       {rel_bias_indep:+.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results Table

# COMMAND ----------

print("=" * 68)
print(f"{'Metric':<40} {'Independence':>13} {'Sarmanov':>13}")
print("=" * 68)

# Per-decile analysis
decile_labels = []
mae_ind_dec, mae_jt_dec = [], []
pred_decile = pd.qcut(pp_indep_test, 10, labels=False, duplicates="drop")
for d in range(10):
    m = pred_decile == d
    if m.sum() < 5: continue
    mae_ind_dec.append(float(np.abs(pp_indep_test[m] - oracle_test[m]).mean()))
    mae_jt_dec.append(float(np.abs(pp_joint_test[m] - oracle_test[m]).mean()))
    decile_labels.append(d+1)

rows = [
    ("Overall MAE vs oracle (lower better)",  mae_indep,        mae_joint),
    ("Total premium bias (abs, lower better)", abs(rel_bias_indep), abs(rel_bias_joint)),
    ("Estimated Spearman rho",                0.0,              spearman_rho),
    ("Fit time (s)",                           base_time,        lib_time),
]
for name, b, l in rows:
    delta = (l - b)/abs(b)*100 if b != 0 else 0.0
    print(f"{name:<40} {b:>13.4f} {l:>13.4f}   {delta:+.1f}%")

print("=" * 68)
print()
print("Correction factors (premium_joint / premium_independent):")
cf = corrections["correction_factor"].values
print(f"  Mean: {cf.mean():.4f}  p10: {np.percentile(cf,10):.4f}  p90: {np.percentile(cf,90):.4f}")
print(f"  High-risk decile correction: {cf[pred_decile==9].mean():.4f}")
print(f"  Low-risk decile correction:  {cf[pred_decile==0].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Plots

# COMMAND ----------

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Plot 1: Predicted vs oracle by decile
dec_ind, dec_jt, dec_or = [], [], []
for d in range(10):
    m = pred_decile == d
    if m.sum() < 5: continue
    dec_ind.append(pp_indep_test[m].mean())
    dec_jt.append(pp_joint_test[m].mean())
    dec_or.append(oracle_test[m].mean())

ax1.plot(range(1,len(dec_or)+1), dec_or,  "k-o", linewidth=2, label="Oracle (true DGP)", markersize=7)
ax1.plot(range(1,len(dec_ind)+1), dec_ind, "b--s", linewidth=2, label=f"Independence (bias={rel_bias_indep:+.1%})", markersize=7)
ax1.plot(range(1,len(dec_jt)+1),  dec_jt,  "r-^", linewidth=2, label=f"Sarmanov (bias={rel_bias_joint:+.1%})", markersize=7)
ax1.set_xlabel("Predicted premium decile"); ax1.set_ylabel("Mean pure premium")
ax1.set_title("Pure premium by decile: oracle vs models"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Plot 2: Correction factor by risk decile
ax2.bar(np.array(decile_labels), [corrections["correction_factor"].values[pred_decile==d-1].mean() for d in decile_labels],
        color="tomato", alpha=0.8)
ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Independence (factor=1)")
ax2.set_xlabel("Premium decile"); ax2.set_ylabel("Sarmanov correction factor")
ax2.set_title(f"Premium correction by decile\nomega={omega_hat:.4f}, Spearman rho={spearman_rho:.4f}")
ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: MAE by decile
ax3.plot(decile_labels, mae_ind_dec, "b--s", linewidth=2, label="Independence", markersize=7)
ax3.plot(decile_labels, mae_jt_dec,  "r-^", linewidth=2, label="Sarmanov", markersize=7)
ax3.set_xlabel("Premium decile"); ax3.set_ylabel("MAE vs oracle")
ax3.set_title("Pure premium MAE by decile"); ax3.legend(); ax3.grid(True, alpha=0.3)

# Plot 4: Dependence: claim count vs severity by decile
n_decile = pd.qcut(df.claim_count.clip(upper=5), 5, labels=False, duplicates="drop")
sev_by_n = []
for d in range(5):
    m = (n_decile == d) & (df.claim_count > 0)
    if m.sum() < 10: sev_by_n.append(np.nan); continue
    sev_by_n.append(float(df.loc[m, "avg_severity"].mean()))

ax4.bar(np.arange(1,6), sev_by_n, color="steelblue", alpha=0.8)
ax4.set_xlabel("Claim count quintile (1=lowest)"); ax4.set_ylabel("Mean severity")
ax4.set_title("Positive freq-sev dependence: higher claim count\ncorrelates with higher severity (omega > 0)")
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle("insurance-frequency-severity: Sarmanov Copula vs Independence",
             fontsize=13, fontweight="bold")
plt.savefig("/tmp/benchmark_frequency_severity.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved /tmp/benchmark_frequency_severity.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verdict

# COMMAND ----------

print("=" * 62)
print("VERDICT: Sarmanov Copula vs Independence Assumption")
print("=" * 62)
print()
print(f"Estimated dependence: omega={omega_hat:.4f}, Spearman rho={spearman_rho:.4f}")
print()
print(f"Overall MAE: independence={mae_indep:.4f}, Sarmanov={mae_joint:.4f}")
mae_improvement = (mae_indep - mae_joint) / mae_indep
print(f"MAE improvement: {mae_improvement:.1%}")
print()
print(f"Total premium bias:")
print(f"  Independence: {rel_bias_indep:+.2%}")
print(f"  Sarmanov:     {rel_bias_joint:+.2%}")
print()
high_risk_cf = corrections["correction_factor"].values[pred_decile==9].mean()
print(f"High-risk decile correction factor: {high_risk_cf:.4f}")
print(f"  ({(high_risk_cf-1)*100:.1f}% underprice if independence assumed)")
print()
if mae_joint < mae_indep:
    print("  Sarmanov copula reduces pure premium estimation error")
    print("  by capturing the positive freq-sev dependence driven by")
    print("  latent risk heterogeneity. The independence model")
    print("  systematically underprices the highest-risk segment.")
else:
    print("  Both models achieve similar performance on this dataset.")
    print("  Dependence is detectable (omega != 0) but correction is small.")

if __name__ == "__main__":
    pass
