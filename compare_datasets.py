import pandas as pd
import json

# Load both files
vig = pd.read_csv('results/reworded_vignettes_results.csv')
scen = pd.read_csv('results/scenario_experiment_results.csv')

print('REWORDED_VIGNETTES_RESULTS.CSV:')
print(f'Total rows: {len(vig):,}')
print(f'Unique vignettes: {vig["vignette_id"].nunique()}')
print(f'Unique scenarios: {vig["scenario_key"].nunique()}')
print(f'Experiment IDs: {sorted(vig["experiment_id"].unique())}')
print()

print('SCENARIO_EXPERIMENT_RESULTS.CSV:')
print(f'Total rows: {len(scen):,}')
print(f'Unique vignettes: {scen["vignette_id"].nunique()}')
print(f'Unique scenarios: {scen["scenario_key"].nunique()}')
print(f'Experiment IDs: {sorted(scen["experiment_id"].unique())}')
print()

# Check vignette distribution
print('Sample vignettes in reworded_vignettes (top 10):')
print(vig['vignette_id'].value_counts().head(10))
print()

print('Sample vignettes in scenario_experiment (top 10):')
print(scen['vignette_id'].value_counts().head(10))
print()

# Load the vignette JSON to understand structure
with open('resources/reworded_vignettes.json', 'r') as f:
    reworded_vignettes = json.load(f)

with open('resources/original_vignettes.json', 'r') as f:
    original_vignettes = json.load(f)

print(f'Number of vignettes in reworded_vignettes.json: {len(reworded_vignettes)}')
print(f'Number of vignettes in original_vignettes.json: {len(original_vignettes)}')
print()

# Compare vignettes in each dataset
vig_vignettes = set(vig['vignette_id'].unique())
scen_vignettes = set(scen['vignette_id'].unique())

print(f'Vignettes only in reworded_vignettes_results: {vig_vignettes - scen_vignettes}')
print(f'Vignettes only in scenario_experiment_results: {scen_vignettes - vig_vignettes}')
print(f'Vignettes in both: {len(vig_vignettes & scen_vignettes)}')
