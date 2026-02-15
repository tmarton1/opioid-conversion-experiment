"""Generate comparison CSV for mg/day vs mg/4h dosing (both with Nielsen).

Pairs the same opioid conversions across the two dosing formats and computes
per-pair accuracy differences.  Output format matches existing comparison CSVs
so _per_model_wilcoxon can read it directly.
"""
import pandas as pd
import numpy as np

TIGHT = "tight(0.85,1.15)"
BROAD = "broad(0.6,1.7)"

# mg/day + Nielsen: OME pairs from pair_model_analysis
df1 = pd.read_csv("results/pair_model_analysis.csv")
df1 = df1[(df1["ome"] == True) & (df1["include_nielsen"] == True) & (df1["include_examples"] == False)]
df1["_pair"] = list(zip(df1["from_opioid"], df1["to_opioid"]))

# mg/4h + Nielsen (no explain): from explain_units
df2 = pd.read_csv("results/explain_units_pair_model_analysis.csv")
df2 = df2[df2["explain_units"] == False]
df2["_pair"] = list(zip(df2["from_opioid"], df2["to_opioid"]))

# Find overlapping pairs
overlap = set(df1["_pair"]) & set(df2["_pair"])
# Exclude oral_morphine -> oral_morphine identity pair
overlap -= {("oral_morphine", "oral_morphine")}
print(f"Overlapping pairs: {len(overlap)}")

df1 = df1[df1["_pair"].isin(overlap)]
df2 = df2[df2["_pair"].isin(overlap)]

rows = []
for model in sorted(set(df1["model"]) & set(df2["model"])):
    m1 = df1[df1["model"] == model]
    m2 = df2[df2["model"] == model]
    for pair in sorted(overlap):
        r1 = m1[m1["_pair"] == pair]
        r2 = m2[m2["_pair"] == pair]
        if r1.empty or r2.empty:
            continue
        r1 = r1.iloc[0]
        r2 = r2.iloc[0]
        rows.append({
            "model": model,
            "opioid_pair": f"{pair[0]} -> {pair[1]}",
            "daily_tight": r1[TIGHT],
            "daily_broad": r1[BROAD],
            "fourh_tight": r2[TIGHT],
            "fourh_broad": r2[BROAD],
            "tight_diff": r1[TIGHT] - r2[TIGHT],  # daily - 4h
            "broad_diff": r1[BROAD] - r2[BROAD],
        })

result = pd.DataFrame(rows)
out = "results/comparison_dosing_unit_effect.csv"
result.to_csv(out, index=False)
print(f"Wrote {out}: {len(result)} rows")
print(f"Models: {result['model'].nunique()}, Pairs per model: {result.groupby('model').size().iloc[0]}")

# Quick summary
for model in sorted(result["model"].unique())[:3]:
    sub = result[result["model"] == model]
    print(f"  {model}: mean broad_diff = {sub['broad_diff'].mean():.3f}")
