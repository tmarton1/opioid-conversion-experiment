"""Re-run Spearman analysis with all 20 scenarios (drop multiple_from_opioids filter)."""
import pandas as pd
import numpy as np
from math import comb
from scipy.stats import spearmanr
from common.models import CONSISTENT_MODELS

vma = pd.read_csv("results/reworded_vignettes_model_analysis.csv")
icc = pd.read_csv("results/icc_analysis.csv")

# Filter to naive condition BUT keep multiple_from_opioids
vma = vma[
    (vma["include_nielsen"] == False)
    & (vma["explain_units"] == False)
    & (vma["has_vignette_text"] == True)
]
vma = vma[vma["model"].isin(set(CONSISTENT_MODELS))]

print(f"VMA: {len(vma)} rows, {vma['model'].nunique()} models, {vma['scenario_key'].nunique()} scenarios")
print()

print(f"{'Model':<38} | {'ρ(var,acc)':>10} {'p':>8} | {'ρ(cv,acc)':>10} {'p':>8} | {'ρ(var,icc)':>10} {'p':>8} | n")
print("-" * 115)

all_results = []
for m in CONSISTENT_MODELS:
    mdf = vma[vma["model"] == m]
    mi = icc[icc["model"] == m]
    if mdf.empty or mi.empty:
        continue

    sg = mdf.groupby("scenario_key").agg(
        tight_prop=("tight(0.85,1.15)", "mean"),
        mean_cv=("observed_dose_cv", "mean"),
    ).reset_index()
    merged = sg.merge(mi[["scenario_key", "var_total", "icc"]], on="scenario_key")

    short = m.split("/")[-1][:35]
    n = len(merged)

    rho1, p1 = spearmanr(merged["var_total"], merged["tight_prop"])
    rho2, p2 = spearmanr(merged["mean_cv"], merged["tight_prop"])
    rho3, p3 = spearmanr(merged["var_total"], merged["icc"])

    s1 = "*" if p1 < 0.05 else ""
    s2 = "*" if p2 < 0.05 else ""
    s3 = "*" if p3 < 0.05 else ""

    print(f"  {short:<36} | {rho1:>9.3f}{s1} {p1:>8.4f} | {rho2:>9.3f}{s2} {p2:>8.4f} | {rho3:>9.3f}{s3} {p3:>8.4f} | {n}")
    all_results.append({"model": m, "rho1": rho1, "p1": p1, "rho2": rho2, "p2": p2, "rho3": rho3, "p3": p3, "n": n})

rdf = pd.DataFrame(all_results)
print()
print("=== SUMMARY (all 20 scenarios) ===")
for label, rhocol, pcol, expected_dir in [
    ("Claim 1: var→acc", "rho1", "p1", "negative"),
    ("Claim 2: cv→acc", "rho2", "p2", "null"),
    ("Claim 3: var→icc", "rho3", "p3", "positive"),
]:
    n_sig = (rdf[pcol] < 0.05).sum()
    n_tot = len(rdf)
    if expected_dir == "negative":
        n_dir = (rdf[rhocol] < 0).sum()
        binom_p = sum(
            comb(n_tot, k) * 0.5**n_tot
            for k in range(n_dir, n_tot + 1)
        )
        print(f"  {label}: {n_sig}/{n_tot} sig, {n_dir}/{n_tot} negative (sign test p={binom_p:.4f}), median ρ={rdf[rhocol].median():.3f}")
    elif expected_dir == "positive":
        n_dir = (rdf[rhocol] > 0).sum()
        binom_p = sum(
            comb(n_tot, k) * 0.5**n_tot
            for k in range(n_dir, n_tot + 1)
        )
        print(f"  {label}: {n_sig}/{n_tot} sig, {n_dir}/{n_tot} positive (sign test p={binom_p:.4f}), median ρ={rdf[rhocol].median():.3f}")
    else:
        n_neg = (rdf[rhocol] < 0).sum()
        n_pos = (rdf[rhocol] > 0).sum()
        print(f"  {label}: {n_sig}/{n_tot} sig, {n_neg} neg / {n_pos} pos, median ρ={rdf[rhocol].median():.3f}")
