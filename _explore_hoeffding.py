"""Compare Spearman ρ vs Hoeffding's D on all three claims."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from common.models import CONSISTENT_MODELS

# ---------- Hoeffding's D implementation ----------
def _hoeffding_d(x, y):
    """Hoeffding's D statistic with tie correction.
    
    Returns (D, p_approx) where p_approx is from the asymptotic distribution.
    For n < 30, permutation p-values would be more reliable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    
    # Ranks (average rank for ties)
    from scipy.stats import rankdata
    R = rankdata(x)
    S = rankdata(y)
    
    # Bivariate ranks: Q_i = number of points (x_j, y_j) where x_j < x_i AND y_j < y_i
    Q = np.zeros(n)
    for i in range(n):
        Q[i] = np.sum((x < x[i]) & (y < y[i]))
    
    # Hoeffding's D
    D1 = np.sum((Q * (Q - 1)))
    D2 = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    D3 = np.sum((R - 2) * (S - 2) * Q)
    
    D = 30 * ((n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    
    return D


def _hoeffding_perm_p(x, y, n_perm=9999):
    """Permutation p-value for Hoeffding's D."""
    d_obs = _hoeffding_d(x, y)
    rng = np.random.default_rng(42)
    count = 0
    y_arr = np.asarray(y)
    for _ in range(n_perm):
        y_perm = rng.permutation(y_arr)
        if _hoeffding_d(x, y_perm) >= d_obs:
            count += 1
    return d_obs, (count + 1) / (n_perm + 1)


# ---------- Load data ----------
vma = pd.read_csv("results/reworded_vignettes_model_analysis.csv")
icc = pd.read_csv("results/icc_analysis.csv")

vma = vma[
    (vma["include_nielsen"] == False)
    & (vma["explain_units"] == False)
    & (vma["has_vignette_text"] == True)
    & (vma["multiple_from_opioids"] == False)
]
vma = vma[vma["model"].isin(set(CONSISTENT_MODELS))]

print(f"{'Model':<32} | {'Claim':<15} | {'Spearman ρ':>10} {'p':>8} | {'Hoeff D':>8} {'perm-p':>8}")
print("-" * 105)

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
    if len(merged) < 5:
        continue

    short = m.split("/")[-1][:28]

    for label, xcol, ycol in [
        ("var→acc", "var_total", "tight_prop"),
        ("cv→acc", "mean_cv", "tight_prop"),
        ("var→icc", "var_total", "icc"),
    ]:
        x, y = merged[xcol].values, merged[ycol].values
        # Skip if constant
        if np.std(x) == 0 or np.std(y) == 0:
            print(f"  {short:<30} | {label:<15} | {'const':>10} {'':>8} | {'const':>8} {'':>8}")
            continue

        rho, sp = spearmanr(x, y)
        hd, hp = _hoeffding_perm_p(x, y, n_perm=4999)
        
        sig_sp = "*" if sp < 0.05 else ""
        sig_hd = "*" if hp < 0.05 else ""
        
        print(f"  {short:<30} | {label:<15} | {rho:>9.3f}{sig_sp} {sp:>8.4f} | {hd:>7.4f}{sig_hd} {hp:>8.4f}")

print()
print("* = p < 0.05")
print()
print("KEY COMPARISON: Count significant results per test per claim")
print()

# Summary counts
results = []
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
    if len(merged) < 5:
        continue

    for label, xcol, ycol in [
        ("var→acc", "var_total", "tight_prop"),
        ("cv→acc", "mean_cv", "tight_prop"),
        ("var→icc", "var_total", "icc"),
    ]:
        x, y = merged[xcol].values, merged[ycol].values
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rho, sp = spearmanr(x, y)
        hd, hp = _hoeffding_perm_p(x, y, n_perm=4999)
        results.append({"model": m, "claim": label, "spearman_p": sp, "hoeffding_p": hp, "rho": rho, "D": hd})

rdf = pd.DataFrame(results)
for claim in ["var→acc", "cv→acc", "var→icc"]:
    sub = rdf[rdf["claim"] == claim]
    n_sp = (sub["spearman_p"] < 0.05).sum()
    n_hd = (sub["hoeffding_p"] < 0.05).sum()
    n_tot = len(sub)
    print(f"  {claim:<15}: Spearman sig {n_sp}/{n_tot}, Hoeffding sig {n_hd}/{n_tot}")
    
    # Direction check for Spearman
    if claim == "var→acc":
        n_neg = ((sub["rho"] < 0)).sum()
        print(f"    → {n_neg}/{n_tot} models have negative ρ (expected: all)")
    elif claim == "cv→acc":
        n_null = (sub["spearman_p"] >= 0.05).sum()
        print(f"    → {n_null}/{n_tot} models non-significant (expected: all)")
    elif claim == "var→icc":
        n_pos = ((sub["rho"] > 0)).sum()
        print(f"    → {n_pos}/{n_tot} models have positive ρ (expected: all)")
