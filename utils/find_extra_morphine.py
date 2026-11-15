"""Find and remove duplicate morphine→morphine pairs for gpt-3.5-turbo and gpt-4.1-mini."""
from common.config import RESULTS_DIR
import pandas as pd

# ── Check experiment_results.csv (raw data) ──
raw = pd.read_csv(f"{RESULTS_DIR}/experiment_results.csv")
print("=== Raw experiment_results.csv columns (first 10) ===")
print(list(raw.columns[:15]))

# Find morphine→morphine rows for the two models
for model_short in ["gpt-3.5-turbo", "gpt-4.1-mini"]:
    mdf = raw[raw["model"].str.contains(model_short)]
    morph = mdf[
        (mdf["from_opioid"].str.contains("morphine", na=False))
        & (mdf["to_opioid"].str.contains("morphine", na=False))
    ]
    print(f"\n=== {model_short}: morphine→morphine rows ({len(morph)}) ===")
    cols_to_show = ["from_opioid", "to_opioid", "model", "include_nielsen"]
    if "experiment_id" in morph.columns:
        cols_to_show.insert(0, "experiment_id")
    if "timestamp" in morph.columns:
        cols_to_show.append("timestamp")
    if "date" in morph.columns:
        cols_to_show.append("date")
    print(morph[cols_to_show].drop_duplicates().to_string(index=False))

# ── Check pair_model_analysis.csv ──
pair = pd.read_csv(f"{RESULTS_DIR}/pair_model_analysis.csv")
for model_short in ["gpt-3.5-turbo", "gpt-4.1-mini"]:
    mdf = pair[pair["model"].str.contains(model_short)]
    morph = mdf[mdf["opioid_pair"].str.contains("morphine.*morphine", na=False)]
    print(f"\n=== {model_short} pair_model_analysis morphine pairs ===")
    print(morph[["opioid_pair", "model", "include_nielsen", "include_examples",
                 "n_observations", "weight"]].to_string(index=False))

# Also check: what are the 50 vs 49 pairs?
ref_model = "openai/gpt-4.1"
ref_pairs = set(pair[pair["model"] == ref_model]["opioid_pair"].unique())
for model_short in ["gpt-3.5-turbo", "gpt-4.1-mini"]:
    full_model = f"openai/{model_short}"
    m_pairs = set(pair[pair["model"] == full_model]["opioid_pair"].unique())
    extra = m_pairs - ref_pairs
    print(f"\n{model_short} extra pairs vs gpt-4.1: {extra}")
