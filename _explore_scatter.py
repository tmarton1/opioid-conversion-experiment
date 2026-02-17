"""Explore data for scenario-level scatter panels."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

CONSISTENT_MODELS = [
    "google/gemini-2.5-pro",
    "anthropic/claude-3.5-sonnet:20241022",
    "openai/gpt-4.1",
    "deepseek/deepseek-chat",
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.7-sonnet",
    "meta-llama/llama-4-maverick",
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-nano",
    "anthropic/claude-3.5-haiku",
    "meta-llama/llama-4-scout",
    "google/gemma-3-27b-it",
    "deepseek/deepseek-r1",
]

vma = pd.read_csv("results/reworded_vignettes_model_analysis.csv")
icc = pd.read_csv("results/icc_analysis.csv")

# Filter to naive, text, single-from
vma = vma[
    (vma["include_nielsen"] == False)
    & (vma["explain_units"] == False)
    & (vma["has_vignette_text"] == True)
    & (vma["multiple_from_opioids"] == False)
]
vma = vma[vma["model"].isin(set(CONSISTENT_MODELS))]

print(f"VMA: {len(vma)} rows, {vma['model'].nunique()} models, {vma['scenario_key'].nunique()} scenarios")

for m in CONSISTENT_MODELS[:3]:
    mdf = vma[vma["model"] == m]
    mi = icc[icc["model"] == m]

    # Scenario-level aggregation
    sg = mdf.groupby("scenario_key").agg(
        tight_prop=("tight(0.85,1.15)", "mean"),
        broad_prop=("broad(0.6,1.7)", "mean"),
        mean_cv=("observed_dose_cv", "mean"),
        n_vignettes=("tight(0.85,1.15)", "count"),
    ).reset_index()

    merged = sg.merge(mi[["scenario_key", "var_total", "icc"]], on="scenario_key")

    print(f"\n=== {m} ({len(merged)} scenarios) ===")

    # Claim 1: variance vs accuracy (expect negative ρ)
    rho1, p1 = spearmanr(merged["var_total"], merged["tight_prop"])
    print(f"  Claim 1 (var vs tight_acc): ρ={rho1:.3f}, p={p1:.4f}")

    # Claim 2: CV vs accuracy (expect null)
    rho2, p2 = spearmanr(merged["mean_cv"], merged["tight_prop"])
    print(f"  Claim 2 (cv vs tight_acc):  ρ={rho2:.3f}, p={p2:.4f}")

    # Claim 3: variance vs ICC (expect positive ρ)
    rho3, p3 = spearmanr(merged["var_total"], merged["icc"])
    print(f"  Claim 3 (var vs ICC):       ρ={rho3:.3f}, p={p3:.4f}")

    print(f"  var_total range: {merged['var_total'].min():.4f} – {merged['var_total'].max():.4f}")
    print(f"  tight_prop range: {merged['tight_prop'].min():.3f} – {merged['tight_prop'].max():.3f}")
    print(f"  mean_cv range: {merged['mean_cv'].min():.4f} – {merged['mean_cv'].max():.4f}")
    print(f"  icc range: {merged['icc'].min():.3f} – {merged['icc'].max():.3f}")

# All models summary
print("\n=== ALL MODELS SUMMARY ===")
print(f"{'Model':<40} {'ρ(var,acc)':>10} {'p':>8} {'ρ(cv,acc)':>10} {'p':>8} {'ρ(var,icc)':>10} {'p':>8}")
for m in CONSISTENT_MODELS:
    mdf = vma[vma["model"] == m]
    mi = icc[icc["model"] == m]
    sg = mdf.groupby("scenario_key").agg(
        tight_prop=("tight(0.85,1.15)", "mean"),
        mean_cv=("observed_dose_cv", "mean"),
    ).reset_index()
    merged = sg.merge(mi[["scenario_key", "var_total", "icc"]], on="scenario_key")

    rho1, p1 = spearmanr(merged["var_total"], merged["tight_prop"])
    rho2, p2 = spearmanr(merged["mean_cv"], merged["tight_prop"])
    rho3, p3 = spearmanr(merged["var_total"], merged["icc"])
    print(f"  {m:<38} {rho1:>10.3f} {p1:>8.4f} {rho2:>10.3f} {p2:>8.4f} {rho3:>10.3f} {p3:>8.4f}")
