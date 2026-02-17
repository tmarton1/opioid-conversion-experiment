"""Explore data to find examples for the consistency illustration figure."""
import pandas as pd
import numpy as np

TIGHT = "tight(0.85,1.15)"
BROAD = "broad(0.6,1.7)"

# ── RANDOM DOSE ────────────────────────────────────────────────────────
pma = pd.read_csv("results/pair_model_analysis.csv")
naive = pma[(pma["include_nielsen"] == False) & (pma["include_examples"] == False) & (pma["ome"] == True)]
nielsen = pma[(pma["include_nielsen"] == True) & (pma["include_examples"] == False) & (pma["ome"] == True)]

# Load raw trial data for scatter plots later
raw = pd.read_csv("results/experiment_results.csv")

print("=" * 70)
print("RANDOM DOSE: Consistent-but-wrong (low CV, low accuracy)")
print("=" * 70)
# Relax R² requirement, focus on CV and accuracy
good = naive[
    (naive["observed_coefficient_cv"] < 0.3) &
    (naive[TIGHT] < 0.2) &
    (naive[BROAD] < 0.8) &
    (naive["n_observations"] >= 15)
].copy()
good["coeff_ratio"] = good["observed_coefficient_mean"] / good["expected_coefficient"]
good = good.sort_values("observed_coefficient_cv")

# Check Nielsen rescue for each
rescue_rows = []
for _, row in good.iterrows():
    n_row = nielsen[(nielsen["opioid_pair"] == row["opioid_pair"]) & (nielsen["model"] == row["model"])]
    if n_row.empty:
        continue
    n = n_row.iloc[0]
    rescue_rows.append({
        "opioid_pair": row["opioid_pair"],
        "model": row["model"],
        "naive_cv": row["observed_coefficient_cv"],
        "naive_tight": row[TIGHT],
        "naive_broad": row[BROAD],
        "naive_coeff_ratio": row["coeff_ratio"],
        "naive_r2": row["r_squared"],
        "naive_slope_p": row["regression_slope_pvalue"],
        "nielsen_tight": n[TIGHT],
        "nielsen_broad": n[BROAD],
        "nielsen_cv": n["observed_coefficient_cv"],
        "nielsen_coeff_ratio": n["observed_coefficient_mean"] / n["expected_coefficient"],
        "rescue": n[BROAD] - row[BROAD],
    })

rescue_df = pd.DataFrame(rescue_rows).sort_values("naive_cv")
print(f"\nConsistent-but-wrong with Nielsen rescue ({len(rescue_df)} candidates):")
print(rescue_df[rescue_df["rescue"] > 0.2][["opioid_pair", "model", "naive_cv", "naive_tight", "naive_broad",
        "naive_coeff_ratio", "naive_r2", "nielsen_tight", "nielsen_broad", "nielsen_cv", "rescue"]].head(20).to_string())

print("\n" + "=" * 70)
print("RANDOM DOSE: Noisy example (high CV, no neg slope)")
print("=" * 70)
noisy = naive[
    (naive["observed_coefficient_cv"] > 0.5) &
    (naive["regression_slope_pvalue"] > 0.1) &
    (naive["n_observations"] >= 15)
].sort_values("observed_coefficient_cv", ascending=False)
print(noisy[["opioid_pair", "model", "observed_coefficient_cv", TIGHT, BROAD,
             "r_squared", "regression_slope_pvalue"]].head(10).to_string())

print("\n" + "=" * 70)
print("RANDOM DOSE: Negative slope example")
print("=" * 70)
neg = naive[
    (naive["regression_slope_pvalue"] < 0.05) &
    (naive["regression_slope"] < 0) &
    (naive["n_observations"] >= 15)
].sort_values("regression_slope_pvalue")
print(neg[["opioid_pair", "model", "observed_coefficient_cv", TIGHT, BROAD,
           "regression_slope", "regression_slope_pvalue"]].head(10).to_string())

# ── VIGNETTES ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VIGNETTES: Low within-vignette variability, across-vignette disagreement")
print("=" * 70)

vr = pd.read_csv("results/reworded_vignettes_results.csv")
v_naive = vr[(vr["include_nielsen"] == False) & (vr["explain_units"] == False) &
             (vr["has_vignette_text"] == True) & (vr["multiple_from_opioids"] == False)]

rows = []
for (model, scenario), grp in v_naive.groupby(["model", "scenario_key"]):
    per_vig = grp.groupby("vignette_id").agg(
        mean_resp=("observed_to_dose", "mean"),
        std_resp=("observed_to_dose", "std"),
        n=("observed_to_dose", "count"),
    )
    if len(per_vig) < 3 or per_vig["n"].min() < 3:
        continue
    
    # Within-vignette: average CV across vignettes
    within_cvs = per_vig["std_resp"] / per_vig["mean_resp"].abs().clip(lower=0.01)
    avg_within_cv = within_cvs.mean()
    
    # Across-vignette: CV of the vignette means
    across_std = per_vig["mean_resp"].std()
    mean_of_means = per_vig["mean_resp"].mean()
    across_cv = across_std / abs(mean_of_means) if mean_of_means != 0 else np.nan
    
    expected = grp["expected_to_dose"].iloc[0]
    ratio = mean_of_means / expected if expected != 0 else np.nan
    
    rows.append({
        "model": model,
        "scenario_key": scenario,
        "n_vignettes": len(per_vig),
        "avg_within_cv": avg_within_cv,
        "across_vignette_cv": across_cv,
        "mean_response": mean_of_means,
        "expected_dose": expected,
        "ratio": ratio,
        "vignette_means": per_vig["mean_resp"].tolist(),
        "vignette_stds": per_vig["std_resp"].tolist(),
    })

vdf = pd.DataFrame(rows)

# Good: low within-CV, meaningful across-CV, wrong answer
good_vig = vdf[
    (vdf["avg_within_cv"] < 0.15) &
    (vdf["across_vignette_cv"] > 0.10) &
    (abs(vdf["ratio"] - 1) > 0.2)
].sort_values("avg_within_cv")

print(f"\nGood vignette candidates ({len(good_vig)}):")
if len(good_vig) > 0:
    print(good_vig[["scenario_key", "model", "avg_within_cv", "across_vignette_cv",
                    "ratio", "expected_dose", "n_vignettes"]].head(20).to_string())
else:
    # Relax criteria
    print("Relaxing criteria...")
    good_vig2 = vdf[
        (vdf["avg_within_cv"] < 0.25) &
        (vdf["across_vignette_cv"] > 0.08)
    ].sort_values("avg_within_cv")
    print(f"Relaxed candidates ({len(good_vig2)}):")
    print(good_vig2[["scenario_key", "model", "avg_within_cv", "across_vignette_cv",
                     "ratio", "expected_dose", "n_vignettes"]].head(20).to_string())
