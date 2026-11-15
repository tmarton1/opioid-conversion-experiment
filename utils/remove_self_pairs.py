"""
Remove oral_morphine -> oral_morphine self-pair rows from experiment CSVs,
since they only exist for two models and are not a valid conversion.
"""
from common.config import RESULTS_DIR
import pandas as pd

files = [
    f"{RESULTS_DIR}/experiment_results.csv",
    f"{RESULTS_DIR}/experiment_results_by_ome.csv",
]

for f in files:
    df = pd.read_csv(f)
    before = len(df)
    mask = (df["from_opioid"] == "oral_morphine") & (df["to_opioid"] == "oral_morphine")
    n_remove = mask.sum()
    df = df[~mask]
    df.to_csv(f, index=False)
    print(f"{f}: removed {n_remove} rows ({before} -> {len(df)})")

# Also remove from pair_model_analysis.csv
pma = f"{RESULTS_DIR}/pair_model_analysis.csv"
df = pd.read_csv(pma)
before = len(df)
mask = df["opioid_pair"] == "oral_morphine -> oral_morphine"
n_remove = mask.sum()
df = df[~mask]
df.to_csv(pma, index=False)
print(f"{pma}: removed {n_remove} rows ({before} -> {len(df)})")

# And from consensus
cons = f"{RESULTS_DIR}/opioid_pair_consensus.csv"
df = pd.read_csv(cons)
before = len(df)
mask = (df["from_opioid"] == "oral_morphine") & (df["to_opioid"] == "oral_morphine")
n_remove = mask.sum()
df = df[~mask]
df.to_csv(cons, index=False)
print(f"{cons}: removed {n_remove} rows ({before} -> {len(df)})")

print("\nDone! Verify pair counts:")
pma_df = pd.read_csv(f"{RESULTS_DIR}/pair_model_analysis.csv")
for m in ["openai/gpt-3.5-turbo", "openai/gpt-4.1-mini", "openai/gpt-4.1"]:
    naive = pma_df[(pma_df["model"] == m) & (pma_df["include_nielsen"] == False) & (pma_df["include_examples"] == False)]
    print(f"  {m}: {naive['opioid_pair'].nunique()} pairs")
