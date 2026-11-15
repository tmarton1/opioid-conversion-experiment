from common.config import RESULTS_DIR
import pandas as pd

# Load both datasets
pairs_df = pd.read_csv(f'{RESULTS_DIR}/pair_model_analysis.csv')
vignettes_df = pd.read_csv(f'{RESULTS_DIR}/reworded_vignettes_model_analysis.csv')

# Get unique opioid pairs from each
pairs_opioids = pairs_df[['from_opioid', 'to_opioid']].drop_duplicates().sort_values(['from_opioid', 'to_opioid'])
vignette_opioids = vignettes_df[['from_opioid', 'to_opioid']].drop_duplicates().sort_values(['from_opioid', 'to_opioid'])

# Make tuples for set operations
pairs_set = set(zip(pairs_opioids['from_opioid'], pairs_opioids['to_opioid']))
vignette_set = set(zip(vignette_opioids['from_opioid'], vignette_opioids['to_opioid']))

shared = pairs_set & vignette_set
only_pairs = pairs_set - vignette_set
only_vignettes = vignette_set - pairs_set

print(f"OPIOID PAIR EXPERIMENT: {len(pairs_set)} unique pairs")
print(f"VIGNETTE EXPERIMENT:    {len(vignette_set)} unique pairs")
print(f"SHARED:                 {len(shared)} pairs")
print(f"ONLY IN PAIR EXP:       {len(only_pairs)} pairs")
print(f"ONLY IN VIGNETTES:      {len(only_vignettes)} pairs")

print("\n" + "="*70)
print("SHARED PAIRS:")
print("="*70)
for f, t in sorted(shared):
    print(f"  {f:40s} -> {t}")

print("\n" + "="*70)
print("ONLY IN OPIOID PAIR EXPERIMENT:")
print("="*70)
for f, t in sorted(only_pairs):
    print(f"  {f:40s} -> {t}")

print("\n" + "="*70)
print("ONLY IN VIGNETTE EXPERIMENT:")
print("="*70)
for f, t in sorted(only_vignettes):
    print(f"  {f:40s} -> {t}")

# Also check OME status
print("\n" + "="*70)
print("VIGNETTE PAIRS BY OME STATUS:")
print("="*70)
vig_ome = vignettes_df[['from_opioid', 'to_opioid', 'ome']].drop_duplicates().sort_values(['ome', 'from_opioid', 'to_opioid'])
for _, row in vig_ome.iterrows():
    in_pairs = "✓ also in pair exp" if (row['from_opioid'], row['to_opioid']) in pairs_set else "✗ NOT in pair exp"
    print(f"  ome={str(row['ome']):5s}  {row['from_opioid']:40s} -> {row['to_opioid']:30s} {in_pairs}")

# Count vignettes per pair
print("\n" + "="*70)
print("VIGNETTE COUNT PER OPIOID PAIR (from reworded_vignettes_model_analysis):")
print("="*70)
# Use a single prompt condition to count distinct vignettes per pair
sample = vignettes_df[
    (vignettes_df['include_nielsen'] == False)
]
pair_vignette_counts = sample.groupby(['from_opioid', 'to_opioid'])['vignette_id'].nunique().reset_index()
pair_vignette_counts.columns = ['from_opioid', 'to_opioid', 'n_vignettes']
pair_vignette_counts = pair_vignette_counts.sort_values('n_vignettes', ascending=False)
for _, row in pair_vignette_counts.iterrows():
    in_pairs = "✓" if (row['from_opioid'], row['to_opioid']) in pairs_set else "✗"
    print(f"  {in_pairs} {row['from_opioid']:40s} -> {row['to_opioid']:30s} ({row['n_vignettes']} vignettes)")
