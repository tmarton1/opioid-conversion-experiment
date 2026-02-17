import pandas as pd
import numpy as np

raw = pd.read_csv("results/experiment_results.csv")

# Check a few noisy candidates
candidates = [
    ("transdermal_fentanyl", "oral_morphine", "ai21/jamba-mini-1.7"),
    ("im_buprenorphine", "oral_morphine", "ai21/jamba-mini-1.7"),
    ("iv_fentanyl", "oral_morphine", "openai/gpt-4.1"),
    ("sc_fentanyl", "oral_morphine", "google/gemini-2.5-pro"),
    ("iv_fentanyl", "oral_morphine", "google/gemini-2.5-pro"),
]

for fr, to, model in candidates:
    mask = ((raw["from_opioid"] == fr) & (raw["to_opioid"] == to) &
            (raw["model"] == model) & (raw["include_nielsen"] == False) &
            (raw["include_examples"] == False))
    sub = raw[mask]
    if len(sub) == 0:
        continue
    coeff = sub["observed_to_dose"] / sub["from_dose"]
    exp_coeff = (sub["expected_to_dose"] / sub["from_dose"]).iloc[0]
    short = model.split("/")[-1]
    print(f"\n{fr}->{to} / {short}:")
    print(f"  exp_coeff={exp_coeff:.4f}, n={len(sub)}")
    print(f"  coefficients: {sorted(coeff.values)[:5]} ... {sorted(coeff.values)[-5:]}")
    print(f"  mean={coeff.mean():.4f}, std={coeff.std():.4f}, cv={coeff.std()/abs(coeff.mean()):.4f}")
    print(f"  observed doses: {sorted(sub['observed_to_dose'].values)[:5]} ... {sorted(sub['observed_to_dose'].values)[-5:]}")
