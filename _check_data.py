import pandas as pd
raw = pd.read_csv("results/experiment_results.csv")
print("COLUMNS:", list(raw.columns))
print()
# Check sample for one chosen example
ex = raw[(raw["from_opioid"]=="oral_tramadol") & (raw["to_opioid"]=="oral_morphine") & 
         (raw["model"]=="anthropic/claude-3.5-sonnet:20241022") & 
         (raw["include_nielsen"]==False) & (raw["include_examples"]==False)]
print(f"oral_tramadol->oral_morphine / claude-3.5-sonnet naive: {len(ex)} rows")
print(ex[["from_dose","expected_to_dose","observed_to_dose"]].head(10))
print()
exn = raw[(raw["from_opioid"]=="oral_tramadol") & (raw["to_opioid"]=="oral_morphine") & 
          (raw["model"]=="anthropic/claude-3.5-sonnet:20241022") & 
          (raw["include_nielsen"]==True) & (raw["include_examples"]==False)]
print(f"oral_tramadol->oral_morphine / claude-3.5-sonnet Nielsen: {len(exn)} rows")
print(exn[["from_dose","expected_to_dose","observed_to_dose"]].head(10))
print()
# Also check iv_fentanyl
ex2 = raw[(raw["from_opioid"]=="iv_fentanyl") & (raw["to_opioid"]=="oral_morphine") & 
          (raw["model"]=="anthropic/claude-3.5-sonnet:20241022") & 
          (raw["include_nielsen"]==False) & (raw["include_examples"]==False)]
print(f"iv_fentanyl->oral_morphine / claude-3.5-sonnet naive: {len(ex2)} rows")
print(ex2[["from_dose","expected_to_dose","observed_to_dose"]].head(10))
print()
# Check vignette data 
vr = pd.read_csv("results/reworded_vignettes_results.csv")
print("VIGNETTE COLUMNS:", list(vr.columns))
print()
v1 = vr[(vr["scenario_key"]=="pediatric-burn-patient") & 
        (vr["model"]=="google/gemini-2.5-flash") &
        (vr["include_nielsen"]==False) & (vr["explain_units"]==False) &
        (vr["has_vignette_text"]==True)]
print(f"pediatric-burn-patient / gemini-2.5-flash naive: {len(v1)} rows")
for vid, g in v1.groupby("vignette_id"):
    print(f"  vignette {vid}: n={len(g)}, mean={g['observed_to_dose'].mean():.2f}, "
          f"std={g['observed_to_dose'].std():.2f}, expected={g['expected_to_dose'].iloc[0]}")
