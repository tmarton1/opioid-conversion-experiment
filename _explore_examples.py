"""Explore data to find good examples for the consistency figure."""
import pandas as pd
import numpy as np

# Load pair-level analysis (random doses)
pma = pd.read_csv("results/pair_model_analysis.csv")
# Filter to naive, no examples, OME pairs
naive = pma[(pma["include_nielsen"] == False) & (pma["include_examples"] == False) & (pma["ome"] == True)]
nielsen = pma[(pma["include_nielsen"] == True) & (pma["include_examples"] == False) & (pma["ome"] == True)]

print("=== RANDOM DOSE: Looking for consistent-but-wrong examples ===")
print(f"Naive rows: {len(naive)}, Nielsen rows: {len(nielsen)}")
print()

# Columns of interest
TIGHT = "tight(0.85,1.15)"
BROAD = "broad(0.6,1.7)"

# Good examples: low CV (consistent), low tight accuracy (wrong)
# Pair with model
good_candidates = naive[
    (naive["observed_coefficient_cv"] < 0.25) &  # consistent
    (naive[TIGHT] < 0.20) &  # clearly wrong
    (naive["r_squared"] > 0.8) &  # decent linear fit
    (naive["n_observations"] >= 15)
].copy()
good_candidates["abs_coeff_error"] = abs(good_candidates["observed_coefficient_mean"] / good_candidates["expected_coefficient"] - 1)
good_candidates = good_candidates.sort_values("observed_coefficient_cv")

print(f"Good candidates (CV<0.15, tight<0.15, R²>0.9): {len(good_candidates)}")
print(good_candidates[["opioid_pair", "model", "observed_coefficient_cv", TIGHT, BROAD, 
                        "r_squared", "expected_coefficient", "observed_coefficient_mean",
                        "abs_coeff_error", "regression_slope_pvalue"]].head(20).to_string())
print()

# Check Nielsen rescue for these candidates
for _, row in good_candidates.head(10).iterrows():
    pair = row["opioid_pair"]
    model = row["model"]
    n_row = nielsen[(nielsen["opioid_pair"] == pair) & (nielsen["model"] == model)]
    if not n_row.empty:
        n = n_row.iloc[0]
        print(f"  {pair} / {model[:30]}: naive tight={row[TIGHT]:.2f} broad={row[BROAD]:.2f} CV={row['observed_coefficient_cv']:.3f} -> "
              f"nielsen tight={n[TIGHT]:.2f} broad={n[BROAD]:.2f} CV={n['observed_coefficient_cv']:.3f}")

print()
print("=== Looking for noisy example (high CV, no neg slope) ===")
noisy = naive[
    (naive["observed_coefficient_cv"] > 0.5) &
    (naive["regression_slope_pvalue"] > 0.1) &  # no significant slope
    (naive["n_observations"] >= 15)
].sort_values("observed_coefficient_cv", ascending=False)
print(f"Noisy candidates: {len(noisy)}")
print(noisy[["opioid_pair", "model", "observed_coefficient_cv", TIGHT, BROAD,
             "r_squared", "regression_slope", "regression_slope_pvalue"]].head(10).to_string())

print()
print("=== Looking for negative slope example ===")
neg_slope = naive[
    (naive["regression_slope_pvalue"] < 0.05) &
    (naive["regression_slope"] < 0) &
    (naive["n_observations"] >= 15)
].sort_values("regression_slope_pvalue")
print(f"Negative slope candidates: {len(neg_slope)}")
print(neg_slope[["opioid_pair", "model", "observed_coefficient_cv", TIGHT, BROAD,
                 "r_squared", "regression_slope", "regression_slope_pvalue"]].head(10).to_string())

print()
print("=== VIGNETTES: Looking for low within-vignette variability but across-vignette disagreement ===")
# Load vignette results
vr = pd.read_csv("results/reworded_vignettes_results.csv")
# Single-from, naive, has text
import json
with open("resources/reworded_vignettes.json") as f:
    vig_data = json.load(f)
single_scenarios = [k for k, v in vig_data.items() if not v.get("multiple_from_opioids", False)]

vr_naive = vr[(vr["include_nielsen"] == False) & (vr["explain_units"] == False) & 
              (vr["has_vignette_text"] == True) &
              (vr["scenario_key"].isin(single_scenarios))]

print(f"Vignette naive text rows: {len(vr_naive)}")

# For each model x scenario: compute within-vignette CV and across-vignette spread
rows = []
for (model, scenario), grp in vr_naive.groupby(["model", "scenario_key"]):
    # Each vignette_id is a different wording
    per_vig = grp.groupby("vignette_id")["observed_to_dose"].agg(["mean", "std", "count"])
    if len(per_vig) < 3:
        continue
    # Within-vignette CV: average of per-vignette CVs
    within_cvs = (per_vig["std"] / per_vig["mean"].abs().clip(lower=0.01))
    avg_within_cv = within_cvs.mean()
    # Across-vignette spread: std of vignette means
    across_std = per_vig["mean"].std()
    across_cv = across_std / abs(per_vig["mean"].mean()) if per_vig["mean"].mean() != 0 else np.nan
    # Expected dose
    expected = grp["expected_to_dose"].iloc[0]
    avg_response = per_vig["mean"].mean()
    ratio = avg_response / expected if expected != 0 else np.nan
    
    rows.append({
        "model": model,
        "scenario_key": scenario,
        "n_vignettes": len(per_vig),
        "avg_within_cv": avg_within_cv,
        "across_vignette_cv": across_cv,
        "across_vignette_std": across_std,
        "avg_response": avg_response,
        "expected_dose": expected,
        "ratio": ratio,
        "vignette_means": per_vig["mean"].tolist(),
    })

vdf = pd.DataFrame(rows)
# Good vignette examples: low within-vignette CV, but high across-vignette CV
# AND wrong (ratio far from 1)
good_vig = vdf[
    (vdf["avg_within_cv"] < 0.1) &  # very consistent within vignette
    (vdf["across_vignette_cv"] > 0.15) &  # different across vignettes
    (abs(vdf["ratio"] - 1) > 0.3)  # wrong
].sort_values("avg_within_cv")

print(f"\nGood vignette candidates: {len(good_vig)}")
print(good_vig[["scenario_key", "model", "avg_within_cv", "across_vignette_cv", 
                "ratio", "expected_dose", "n_vignettes"]].head(15).to_string())
