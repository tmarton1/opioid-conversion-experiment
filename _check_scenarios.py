"""Check why only 18 scenarios after filtering."""
import pandas as pd
from common.models import CONSISTENT_MODELS

vma = pd.read_csv("results/reworded_vignettes_model_analysis.csv")
icc = pd.read_csv("results/icc_analysis.csv")

print("ICC scenarios:", sorted(icc["scenario_key"].unique()))
print(f"  count: {icc['scenario_key'].nunique()}")
print()

# Before filtering
print("VMA scenarios (all):", sorted(vma["scenario_key"].unique()))
print(f"  count: {vma['scenario_key'].nunique()}")
print()

# After filtering
vma_f = vma[
    (vma["include_nielsen"] == False)
    & (vma["explain_units"] == False)
    & (vma["has_vignette_text"] == True)
    & (vma["multiple_from_opioids"] == False)
]
vma_f = vma_f[vma_f["model"].isin(set(CONSISTENT_MODELS))]
print("VMA scenarios (filtered):", sorted(vma_f["scenario_key"].unique()))
print(f"  count: {vma_f['scenario_key'].nunique()}")
print()

# Which are missing?
icc_scen = set(icc["scenario_key"].unique())
vma_scen = set(vma_f["scenario_key"].unique())
print("In ICC but not in filtered VMA:", icc_scen - vma_scen)
print("In filtered VMA but not in ICC:", vma_scen - icc_scen)
print()

# Check why those scenarios are missing
missing = icc_scen - vma_scen
for s in missing:
    sub = vma[vma["scenario_key"] == s]
    print(f"\nScenario: {s}")
    print(f"  Total rows in VMA: {len(sub)}")
    if len(sub) > 0:
        print(f"  include_nielsen values: {sub['include_nielsen'].unique()}")
        print(f"  explain_units values: {sub['explain_units'].unique()}")
        print(f"  has_vignette_text values: {sub['has_vignette_text'].unique()}")
        print(f"  multiple_from_opioids values: {sub['multiple_from_opioids'].unique()}")
        print(f"  models: {sub['model'].unique()}")
        # Check how many pass each filter step
        f1 = sub[sub["include_nielsen"] == False]
        f2 = f1[f1["explain_units"] == False]
        f3 = f2[f2["has_vignette_text"] == True]
        f4 = f3[f3["multiple_from_opioids"] == False]
        f5 = f4[f4["model"].isin(set(CONSISTENT_MODELS))]
        print(f"  After nielsen filter: {len(f1)}")
        print(f"  After explain_units filter: {len(f2)}")
        print(f"  After vignette_text filter: {len(f3)}")
        print(f"  After multiple_from filter: {len(f4)}")
        print(f"  After model filter: {len(f5)}")
