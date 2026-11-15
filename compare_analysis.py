from common.config import RESULTS_DIR
import pandas as pd

# Load the model analysis files
reworded_scenarios = pd.read_csv(f'{RESULTS_DIR}/reworded_scenarios_model_analysis.csv')
reworded_vignettes = pd.read_csv(f'{RESULTS_DIR}/reworded_vignettes_model_analysis.csv')

print('REWORDED_SCENARIOS_MODEL_ANALYSIS.CSV:')
print(f'Total rows: {len(reworded_scenarios):,}')
print(f'Columns: {list(reworded_scenarios.columns)}')
print(f'Unique scenarios: {reworded_scenarios["scenario_key"].nunique()}')
if 'vignette_id' in reworded_scenarios.columns:
    print(f'Unique vignettes: {reworded_scenarios["vignette_id"].nunique() if "vignette_id" in reworded_scenarios.columns else "N/A"}')
print()
print('Sample rows:')
print(reworded_scenarios.head(2).to_string())
print()

print('-' * 80)
print()

print('REWORDED_VIGNETTES_MODEL_ANALYSIS.CSV:')
print(f'Total rows: {len(reworded_vignettes):,}')
print(f'Columns: {list(reworded_vignettes.columns)}')
print(f'Unique vignettes: {reworded_vignettes["vignette_id"].nunique()}')
print(f'Unique scenarios: {reworded_vignettes["scenario_key"].nunique()}')
print()
print('Sample rows:')
print(reworded_vignettes.head(2).to_string())
