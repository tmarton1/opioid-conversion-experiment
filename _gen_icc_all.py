#!/usr/bin/env python3
"""Generate ICC analysis CSV and both ICC figures."""
import asyncio
from analysis.icc_analysis import ICCAnalysis
from visualize.icc_visualizer import save_icc_figures

# 1. Generate CSV
icc = ICCAnalysis(
    "results/reworded_vignettes_results.csv",
    filters={"include_nielsen": False, "explain_units": False},
)
icc.save("results/icc_analysis.csv")

# Summary
df = icc.results
print(f"\nOverall mean ICC: {df['icc'].mean():.3f}")
print(f"Median ICC: {df['icc'].median():.3f}")
sig = (df["f_pvalue"] < 0.05).sum()
print(f"Significant (p<0.05): {sig}/{len(df)} ({sig/len(df)*100:.1f}%)")
print()

# 2. Generate figures
asyncio.run(save_icc_figures())
print("\nDone!")
