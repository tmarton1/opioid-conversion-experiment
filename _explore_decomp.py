"""Explore variance decomposition data for visualization design."""
import pandas as pd
import numpy as np
from common.models import CONSISTENT_MODELS

vma = pd.read_csv("results/reworded_vignettes_model_analysis.csv")
vma = vma[(vma["include_nielsen"]==False) & (vma["explain_units"]==False) & (vma["has_vignette_text"]==True)]
vma = vma[vma["model"].isin(set(CONSISTENT_MODELS))]

icc = pd.read_csv("results/icc_analysis.csv")
icc = icc[icc["model"].isin(set(CONSISTENT_MODELS))]

tight_col = "tight(0.85,1.15)"

scen = vma.groupby(["model","scenario_key"]).agg(
    mean_cv=("observed_dose_cv","mean"),
    tight_prop=(tight_col,"mean"),
).reset_index()

merged = scen.merge(
    icc[["model","scenario_key","var_between","var_within","var_total","icc"]],
    on=["model","scenario_key"], how="inner"
)

n_models = merged["model"].nunique()
n_scen = merged["scenario_key"].nunique()
print(f"Merged: {len(merged)} rows, {n_models} models, {n_scen} scenarios")
print()

# Summary per model sorted by accuracy
models_sorted = sorted(
    merged["model"].unique(),
    key=lambda x: merged[merged["model"]==x]["tight_prop"].mean()
)
for m in models_sorted:
    mdf = merged[merged["model"]==m]
    acc = mdf["tight_prop"].mean()
    cv = mdf["mean_cv"].median()
    vw = mdf["var_within"].mean()
    vb = mdf["var_between"].mean()
    mi = mdf["icc"].mean()
    print(f"{m:40s}  acc={acc:.2f}  med_cv={cv:.3f}  var_within={vw:.4f}  var_between={vb:.4f}  icc={mi:.3f}")

print()
print("=== Distribution of within-vignette CV ===")
cv_vals = merged["mean_cv"]
print(f"Median: {cv_vals.median():.4f}")
print(f"Mean:   {cv_vals.mean():.4f}")
pct_zero = (cv_vals == 0).mean()
pct_005 = (cv_vals < 0.05).mean()
pct_010 = (cv_vals < 0.10).mean()
pct_020 = (cv_vals < 0.20).mean()
print(f"% with CV=0:    {pct_zero:.1%}")
print(f"% with CV<0.05: {pct_005:.1%}")
print(f"% with CV<0.10: {pct_010:.1%}")
print(f"% with CV<0.20: {pct_020:.1%}")

print()
print("=== Variance decomposition ===")
vw_mean = merged["var_within"].mean()
vb_mean = merged["var_between"].mean()
vt_mean = merged["var_total"].mean()
ratio = vb_mean / vt_mean
print(f"Mean var_within:  {vw_mean:.4f}")
print(f"Mean var_between: {vb_mean:.4f}")
print(f"Mean var_total:   {vt_mean:.4f}")
print(f"Ratio between/total: {ratio:.1%}")

print()
merged["_acc_bin"] = pd.cut(
    merged["tight_prop"],
    bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
    labels=["0-20%","20-50%","50-80%","80-100%"]
)
for ab in ["0-20%","20-50%","50-80%","80-100%"]:
    sub = merged[merged["_acc_bin"]==ab]
    if len(sub)==0:
        continue
    n = len(sub)
    cv = sub["mean_cv"].median()
    vw = sub["var_within"].mean()
    vb = sub["var_between"].mean()
    mi = sub["icc"].mean()
    print(f"Accuracy {ab:8s} (n={n:3d}): med_cv={cv:.3f}  var_within={vw:.4f}  var_between={vb:.4f}  icc={mi:.3f}")

print()
print("=== Precise-but-inaccurate: CV<0.1 AND tight<0.5 ===")
precise_inacc = merged[(merged["mean_cv"] < 0.1) & (merged["tight_prop"] < 0.5)]
n_pi = len(precise_inacc)
pct_pi = n_pi / len(merged)
print(f"Count: {n_pi}/{len(merged)} ({pct_pi:.1%})")
if n_pi > 0:
    print(f"Mean ICC: {precise_inacc['icc'].mean():.3f}")
    print(f"Mean var_between: {precise_inacc['var_between'].mean():.4f}")
    print(f"Mean var_within:  {precise_inacc['var_within'].mean():.4f}")

print()
print("=== Precise-and-accurate: CV<0.1 AND tight>=0.5 ===")
precise_acc = merged[(merged["mean_cv"] < 0.1) & (merged["tight_prop"] >= 0.5)]
n_pa = len(precise_acc)
pct_pa = n_pa / len(merged)
print(f"Count: {n_pa}/{len(merged)} ({pct_pa:.1%})")
if n_pa > 0:
    print(f"Mean ICC: {precise_acc['icc'].mean():.3f}")
    print(f"Mean var_between: {precise_acc['var_between'].mean():.4f}")
    print(f"Mean var_within:  {precise_acc['var_within'].mean():.4f}")
