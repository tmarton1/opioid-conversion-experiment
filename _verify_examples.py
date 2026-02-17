import pandas as pd
import numpy as np

raw = pd.read_csv("results/experiment_results.csv")

examples = [
    ("oral_tramadol", "oral_morphine", "anthropic/claude-3.5-sonnet:20241022"),
    ("oral_tapentadol", "oral_morphine", "mistralai/mixtral-8x22b-instruct"),
    ("iv_fentanyl", "oral_morphine", "anthropic/claude-3.5-sonnet:20241022"),
    ("transdermal_fentanyl", "oral_morphine", "ai21/jamba-mini-1.7"),
    ("transdermal_buprenorphine", "oral_morphine", "meta-llama/llama-3.3-70b-instruct"),
]

for fr, to, model in examples:
    for nielsen in [False, True]:
        mask = ((raw["from_opioid"] == fr) & (raw["to_opioid"] == to) &
                (raw["model"] == model) & (raw["include_nielsen"] == nielsen) &
                (raw["include_examples"] == False))
        sub = raw[mask]
        if len(sub) == 0:
            continue
        coeff = sub["observed_to_dose"] / sub["from_dose"]
        exp_coeff = (sub["expected_to_dose"] / sub["from_dose"]).iloc[0]
        cond = "Nielsen" if nielsen else "Naive"
        short = model.split("/")[-1]
        print(f"{fr}->{to} / {short:>30s} [{cond:>7s}]: "
              f"n={len(sub):2d}, exp_coeff={exp_coeff:.4f}, "
              f"obs_mean={coeff.mean():.4f}, obs_med={coeff.median():.4f}, "
              f"cv={coeff.std()/coeff.mean():.4f}, range=[{coeff.min():.3f}, {coeff.max():.3f}]")
    print()
