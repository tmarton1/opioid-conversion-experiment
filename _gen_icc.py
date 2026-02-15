#!/usr/bin/env python3
"""Generate ICC analysis CSV for the naive condition."""
from analysis.icc_analysis import ICCAnalysis

icc = ICCAnalysis(
    "results/reworded_vignettes_results.csv",
    filters={"include_nielsen": False, "explain_units": False},
)
icc.save("results/icc_analysis.csv")

# Print summary
df = icc.results
print(f"\nOverall mean ICC: {df['icc'].mean():.3f}")
print(f"Median ICC: {df['icc'].median():.3f}")
print(f"Range: [{df['icc'].min():.3f}, {df['icc'].max():.3f}]")
print()

# Per-model mean ICC
print("=== Mean ICC by model ===")
model_means = df.groupby("model")["icc"].mean().sort_values(ascending=False)
for m, v in model_means.items():
    print(f"  {m}: {v:.3f}")
print()

# Per-scenario mean ICC
print("=== Mean ICC by scenario ===")
scenario_means = df.groupby("scenario_key")["icc"].mean().sort_values(ascending=False)
for s, v in scenario_means.items():
    print(f"  {s}: {v:.3f}")
print()

# Significance summary
sig = (df["f_pvalue"] < 0.05).sum()
print(f"Significant (p<0.05): {sig}/{len(df)} ({sig/len(df)*100:.1f}%)")
