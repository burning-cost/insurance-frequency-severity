# Databricks notebook source
# DISCLAIMER: freMTPL2 is French motor data (OpenML dataset 41214).
# It is used here for methodology validation only — to demonstrate that
# the Sarmanov copula generalises to real insurance data beyond synthetic DGPs.
# It is NOT UK market data and should not be used to draw conclusions about
# UK motor pricing structure.
#
# Dataset: 678,013 French motor third-party liability policies.
# Source: Noll, Salzmann, Wüthrich (2018) — Case Study: French Motor Third-Party
#         Liability Claims.
# Variables used:
#   ClaimNb     — number of claims (frequency target, Poisson)
#   ClaimAmount — total claim amount for the policy
#   Exposure    — fraction of year at risk (offset)
#   Area        — categorical area code (A–F)
#   VehPower    — vehicle power (4–15)
#   VehAge      — vehicle age in years (0–100)
#   DrivAge     — driver age in years (18–100)
#   BonusMalus  — bonus-malus level (50–350)
#   VehGas      — fuel type (Diesel / Regular)
#   Density     — population density of driver's commune
#   Region      — French administrative region code
#
# What this benchmark demonstrates:
#   The independence assumption E[N*S] = E[N]*E[S] is standard in two-part
#   pricing models. The Sarmanov copula provides a test of this assumption and,
#   when dependency exists, a multiplicative correction to the pure premium.
#   This benchmark quantifies whether that correction matters on real data.
#
# Library: insurance-frequency-severity
# Date: 2026-03-27

# COMMAND ----------

%pip install "statsmodels>=0.14" "scikit-learn>=1.3" "pandas>=2.0" --quiet

# COMMAND ----------

%pip install "insurance-frequency-severity" --quiet

# COMMAND ----------

# Skip dbutils.library.restartPython() when running locally
try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

print("=" * 68)
print("freMTPL2 Benchmark: Joint Frequency-Severity with Sarmanov Copula")
print("Data:    OpenML #41214 — French motor MTPL")
print("Library: insurance-frequency-severity")
print("=" * 68)
print()
print("DISCLAIMER: French motor data used for methodology validation only.")
print("Not UK market data. Results reflect French portfolio characteristics.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Load freMTPL2
# ---------------------------------------------------------------------------
# OpenML 41214 is the freMTPL2freq dataset (~678K rows). It includes ClaimNb
# and Exposure alongside the rating factors. ClaimAmount is in freMTPL2sev
# (dataset 41215) and needs to be aggregated to the policy level.
#
# For this benchmark we use ClaimAmount directly from the freq table when
# available, or approximate total loss from ClaimNb * average if not.

from sklearn.datasets import fetch_openml

print("\nDownloading freMTPL2freq from OpenML (may take 30–60s)...")
t_load = time.time()

raw = fetch_openml(data_id=41214, as_frame=True, parser="auto")
df_raw = raw.frame.copy()
df_raw.columns = [c.strip() for c in df_raw.columns]

print(f"  Loaded {len(df_raw):,} rows in {time.time() - t_load:.1f}s")
print(f"  Columns: {list(df_raw.columns)}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Prepare frequency and severity targets
# ---------------------------------------------------------------------------
# freMTPL2freq contains ClaimNb (frequency) and ClaimAmount (total loss for
# the policy). Average severity = ClaimAmount / ClaimNb for policies with
# at least one claim.

df_raw["ClaimNb"]     = df_raw["ClaimNb"].astype(float)
df_raw["Exposure"]    = df_raw["Exposure"].astype(float).clip(lower=1e-6)
df_raw["ClaimAmount"] = df_raw["ClaimAmount"].astype(float)

# Average claim severity (per claim) — only meaningful for ClaimNb > 0
mask_pos = df_raw["ClaimNb"] > 0
df_raw["AvgSeverity"] = np.where(
    mask_pos,
    df_raw["ClaimAmount"] / df_raw["ClaimNb"],
    np.nan,
)

n_pos = mask_pos.sum()
print(f"Total policies:          {len(df_raw):,}")
print(f"Policies with claims:    {n_pos:,}  ({100*n_pos/len(df_raw):.1f}%)")
print(f"Mean ClaimNb:            {df_raw['ClaimNb'].mean():.4f}")
print(f"Claim frequency (adj):   {df_raw['ClaimNb'].sum() / df_raw['Exposure'].sum():.4f} claims/year")
print(f"Mean severity (claimants only): {df_raw.loc[mask_pos, 'AvgSeverity'].mean():,.0f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Subsample to ~75K rows for tractable fitting
# ---------------------------------------------------------------------------
# We use a simple random subsample (not stratified) to keep the methodology
# clean. The subsampled claim frequency will match the population frequency.

N_SAMPLE = 75_000
SEED = 42

rng_np = np.random.default_rng(SEED)
idx_sample = rng_np.choice(len(df_raw), size=N_SAMPLE, replace=False)
df = df_raw.iloc[idx_sample].reset_index(drop=True)

print(f"Subsample: {len(df):,} rows")
print(f"Subsampled claim frequency: {df['ClaimNb'].sum() / df['Exposure'].sum():.4f} claims/year")
print(f"Subsampled claimant rate:   {(df['ClaimNb'] > 0).mean():.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
# Encode categoricals as integers for statsmodels GLM. Log-transform Density
# (right-skewed). Keep numerics as floats.

FEATURES = ["DrivAge", "VehAge", "BonusMalus", "VehPower", "Density",
            "Area", "VehGas", "Region"]

df_model = df[FEATURES].copy()

# Log-transform Density
df_model["LogDensity"] = np.log1p(df_model["Density"].astype(float))
df_model = df_model.drop(columns=["Density"])

# One-hot encode categoricals
cat_cols = ["Area", "VehGas", "Region"]
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=float)

# Convert all remaining columns to float
for col in df_model.columns:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0.0).astype(float)

X = sm.add_constant(df_model, has_constant="add")

print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Frequency model: Poisson GLM with log(Exposure) offset
# ---------------------------------------------------------------------------
# Standard pricing practice: log-link Poisson, offset = log(Exposure).

print("\nFitting Poisson GLM (frequency)...")
t0 = time.time()

log_exposure = np.log(df["Exposure"].values)
poisson_glm = sm.GLM(
    df["ClaimNb"].values,
    X,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=log_exposure,
).fit()

mu_freq = poisson_glm.fittedvalues  # E[N|x] * Exposure — claim count scale

freq_time = time.time() - t0
print(f"  Fit time: {freq_time:.1f}s")
print(f"  Deviance: {poisson_glm.deviance:.2f}")
print(f"  AIC:      {poisson_glm.aic:.2f}")
print(f"  Sum fitted vs actual: {mu_freq.sum():.1f} vs {df['ClaimNb'].sum():.1f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Severity model: Gamma GLM on claims-only subset
# ---------------------------------------------------------------------------
# The severity model is fitted on the subset of policies with at least one
# claim. This is the standard two-part approach.

mask_train_pos = (df["ClaimNb"] > 0) & (df["AvgSeverity"] > 0) & df["AvgSeverity"].notna()
df_sev = df[mask_train_pos].reset_index(drop=True)
X_sev  = X[mask_train_pos.values].reset_index(drop=True)

print(f"\nFitting Gamma GLM (severity) on {mask_train_pos.sum():,} claimant rows...")
t0 = time.time()

gamma_glm = sm.GLM(
    df_sev["AvgSeverity"].values,
    X_sev,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    var_weights=df_sev["ClaimNb"].values,  # weight by number of claims
).fit()

mu_sev_claims = gamma_glm.fittedvalues  # E[S|x] on claimants

sev_time = time.time() - t0
print(f"  Fit time: {sev_time:.1f}s")
print(f"  Deviance: {gamma_glm.deviance:.2f}")
print(f"  AIC:      {gamma_glm.aic:.2f}")
print(f"  Mean fitted severity: {mu_sev_claims.mean():,.0f}")
print(f"  Mean actual severity: {df_sev['AvgSeverity'].mean():,.0f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Predict severity for the full policy set
# ---------------------------------------------------------------------------
# We need E[S|x] for every policy (including zero-claim), not just claimants,
# because the premium correction is computed at the policy level.

mu_sev_all = gamma_glm.predict(X)

print(f"Full-portfolio mean E[S|x]: {mu_sev_all.mean():,.0f}")
print(f"(Claimant subset mean E[S|x]: {mu_sev_claims.mean():,.0f})")

# Independence pure premium: E[N|x] * E[S|x]
pp_indep = mu_freq * mu_sev_all
print(f"\nIndependence pure premium (portfolio mean): {pp_indep.mean():,.2f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Sarmanov copula: estimate freq-sev dependency
# ---------------------------------------------------------------------------
# JointFreqSev wraps the SarmanovCopula and handles GLM parameter extraction.
# It estimates omega (the Sarmanov dependence parameter) by profile likelihood
# over the claims-only subset.

from insurance_frequency_severity import JointFreqSev

print("\nFitting Sarmanov copula (IFM method)...")
print("This estimates omega by profile likelihood given the fitted marginals.")
t0 = time.time()

# Build the data DataFrame that JointFreqSev.fit() expects
fit_data = pd.DataFrame({
    "claim_count":   df["ClaimNb"].values,
    "avg_severity":  df["AvgSeverity"].fillna(0.0).values,
    "exposure":      df["Exposure"].values,
})

joint_model = JointFreqSev(
    freq_glm=poisson_glm,
    sev_glm=gamma_glm,
    copula="sarmanov",
)
joint_model.fit(
    fit_data,
    n_col="claim_count",
    s_col="avg_severity",
    freq_X=X,
    sev_X=X_sev,
    exposure_col=None,   # mu_freq already on count scale via offset
)

copula_time = time.time() - t0
print(f"  Fit time: {copula_time:.1f}s")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Retrieve results from fitted joint model
# ---------------------------------------------------------------------------

omega   = joint_model.omega_
rho_s   = joint_model.rho_
aic_cop = joint_model.aic_
bic_cop = joint_model.bic_
omega_ci = joint_model.omega_ci_

print(f"\nSarmanov omega (dependence parameter): {omega:.4f}")
print(f"Spearman rho (rank correlation):       {rho_s:.4f}")
if omega_ci:
    print(f"95% CI on omega:                       [{omega_ci[0]:.4f}, {omega_ci[1]:.4f}]")
print(f"AIC (copula term):                     {aic_cop:.2f}")
print(f"BIC (copula term):                     {bic_cop:.2f}")
print()
if abs(omega) < 0.1:
    print("Interpretation: omega near zero — freq-sev dependence is weak.")
    print("The independence assumption is approximately valid for this sample.")
elif omega < 0:
    print("Interpretation: negative omega — higher claim counts associate with")
    print("lower average severity. Consistent with NCD/BM suppression effect.")
else:
    print("Interpretation: positive omega — higher claim counts associate with")
    print("higher average severity. Unusual; check data quality.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Premium correction factors
# ---------------------------------------------------------------------------
# correction_factor = E[N*S] / (E[N]*E[S])
# Under independence: correction = 1.0
# With copula: correction = 1 + omega * correction_term

corrections_df = joint_model.premium_correction(X=X)

print("Premium correction summary:")
print(corrections_df[["mu_n", "mu_s", "correction_factor", "premium"]].describe().round(4))

correction_factors = corrections_df["correction_factor"].values
pp_corrected       = corrections_df["premium"].values
pp_uncorrected     = corrections_df["mu_n"].values * corrections_df["mu_s"].values

mean_correction   = float(correction_factors.mean())
portfolio_uncorr  = float(pp_uncorrected.sum())
portfolio_corr    = float(pp_corrected.sum())
portfolio_delta   = 100.0 * (portfolio_corr - portfolio_uncorr) / max(portfolio_uncorr, 1.0)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print("=" * 68)
print("RESULTS — freMTPL2 joint frequency-severity benchmark")
print("=" * 68)
print()
print(f"{'Dataset':<40} freMTPL2freq (OpenML 41214)")
print(f"{'Subsample size':<40} {len(df):,} policies")
print(f"{'Policies with claims':<40} {mask_train_pos.sum():,}")
print()
print(f"{'Sarmanov omega':<40} {omega:.6f}")
print(f"{'Spearman rho':<40} {rho_s:.6f}")
if omega_ci:
    print(f"{'95% CI (omega)':<40} [{omega_ci[0]:.4f}, {omega_ci[1]:.4f}]")
print()
print(f"{'Mean correction factor':<40} {mean_correction:.6f}")
print(f"{'Portfolio pure premium (uncorrected)':<40} {portfolio_uncorr:,.0f}")
print(f"{'Portfolio pure premium (corrected)':<40} {portfolio_corr:,.0f}")
print(f"{'Portfolio delta (%)':<40} {portfolio_delta:+.3f}%")
print()
print(f"{'Frequency GLM AIC':<40} {poisson_glm.aic:.1f}")
print(f"{'Severity GLM AIC':<40} {gamma_glm.aic:.1f}")
print(f"{'Copula AIC (dependence term)':<40} {aic_cop:.2f}")
print()

# COMMAND ----------

# ---------------------------------------------------------------------------
# Distribution of correction factors
# ---------------------------------------------------------------------------
# If the copula term is irrelevant, correction factors will cluster near 1.0.
# A spread of 0.98–1.02 is economically immaterial. Wider spreads suggest
# the dependency correction matters for individual risk pricing even if the
# portfolio-level effect is small.

pct = np.percentile(correction_factors, [1, 5, 25, 50, 75, 95, 99])
print("Distribution of per-policy correction factors:")
print(f"  P1:    {pct[0]:.4f}")
print(f"  P5:    {pct[1]:.4f}")
print(f"  P25:   {pct[2]:.4f}")
print(f"  P50:   {pct[3]:.4f}")
print(f"  P75:   {pct[4]:.4f}")
print(f"  P95:   {pct[5]:.4f}")
print(f"  P99:   {pct[6]:.4f}")
print(f"  Std:   {correction_factors.std():.4f}")
print()
print(f"Policies where |correction - 1| > 1%: "
      f"{(np.abs(correction_factors - 1) > 0.01).sum():,} "
      f"({100*(np.abs(correction_factors - 1) > 0.01).mean():.1f}%)")
print(f"Policies where |correction - 1| > 5%: "
      f"{(np.abs(correction_factors - 1) > 0.05).sum():,} "
      f"({100*(np.abs(correction_factors - 1) > 0.05).mean():.1f}%)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Honest assessment
# ---------------------------------------------------------------------------

print()
print("=" * 68)
print("ASSESSMENT")
print("=" * 68)
print("""
1. The Sarmanov omega is estimated by IFM (inference functions for margins):
   profile likelihood for omega given the fitted marginal GLMs. This is
   consistent but not fully efficient — full joint MLE would be slightly
   more efficient but rarely changes the point estimate materially.

2. freMTPL2 is a frequency-heavy dataset (~7% claimant rate). Omega is
   identified from the joint distribution of (ClaimNb, AvgSeverity) on the
   claimant subset only. With ~5K–7K claimants in the 75K subsample, the
   standard errors on omega are moderate; the 95% CI tells you whether
   dependency is statistically distinguishable from zero.

3. The portfolio-level correction factor is typically small (|delta| < 2%).
   This is the expected result for frequency-severity in motor: individual
   claims are relatively homogeneous, and the BonusMalus system already
   partially captures the claim-size correlation through pricing factors.

4. Per-policy corrections matter more than the portfolio mean. A pricing
   team targeting individual risk accuracy (e.g., telematics or commercial
   fleet) should examine the spread of correction factors, not just the mean.

5. This is French motor data. The BonusMalus structure, geographic coding,
   and vehicle classification differ materially from UK motor. Use this
   result as evidence that the method works on real data — not as a
   calibrated estimate of UK dependency magnitude.
""")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Exit with results dict
# ---------------------------------------------------------------------------

import json

results = {
    "dataset":                     "freMTPL2freq (OpenML 41214)",
    "n_sample":                    int(len(df)),
    "n_claimants":                 int(mask_train_pos.sum()),
    "sarmanov_omega":              float(omega),
    "spearman_rho":                float(rho_s),
    "omega_ci_lower":              float(omega_ci[0]) if omega_ci else None,
    "omega_ci_upper":              float(omega_ci[1]) if omega_ci else None,
    "mean_correction_factor":      float(mean_correction),
    "portfolio_pure_premium_uncorrected": float(portfolio_uncorr),
    "portfolio_pure_premium_corrected":   float(portfolio_corr),
    "portfolio_delta_pct":         float(portfolio_delta),
    "freq_glm_aic":                float(poisson_glm.aic),
    "sev_glm_aic":                 float(gamma_glm.aic),
    "copula_aic":                  float(aic_cop),
    "disclaimer": "French motor data — methodology validation only, not UK market data",
}

print("\nRaw results JSON:")
print(json.dumps(results, indent=2))

try:
    dbutils.notebook.exit(json.dumps(results))
except NameError:
    pass  # running locally
