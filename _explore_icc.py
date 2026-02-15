import json, pandas as pd
from collections import defaultdict

# Check vignette structure
with open('resources/reworded_vignettes.json') as f:
    vigs = json.load(f)

scenarios = defaultdict(list)
for vid, v in vigs.items():
    has_text = v.get('has_vignette_text', True)
    scenarios[v['scenario_key']].append((vid, has_text))

print('=== SCENARIOS AND VIGNETTES ===')
for s, vs in sorted(scenarios.items()):
    text_count = sum(1 for _, ht in vs if ht)
    empty_count = sum(1 for _, ht in vs if not ht)
    print(f'{s}: {text_count} text + {empty_count} empty = {len(vs)} total')
    for vid, ht in sorted(vs):
        label = 'TEXT' if ht else 'EMPTY'
        print(f'  {vid}: {label}')
print()

# Check results CSV structure
df = pd.read_csv('results/reworded_vignettes_results.csv')
print('=== RESULTS CSV COLUMNS ===')
print(df.columns.tolist())
print(f'Shape: {df.shape}')
print()
print('=== SAMPLE ROWS ===')
print(df[['model', 'vignette_id', 'scenario_key', 'observed_to_dose', 'expected_to_dose']].head(10))
print()
print('=== UNIQUE SCENARIOS ===')
print(sorted(df['scenario_key'].unique()))
print(f'Count: {df["scenario_key"].nunique()}')
print()
print('=== UNIQUE MODELS ===')
print(sorted(df['model'].unique()))
print(f'Count: {df["model"].nunique()}')
print()

# Check has_vignette_text column
if 'has_vignette_text' in df.columns:
    print('=== HAS_VIGNETTE_TEXT ===')
    print(df['has_vignette_text'].value_counts())
else:
    print('No has_vignette_text column - need to join from vignette JSON')

# Check trial counts per vignette per model
print()
print('=== TRIALS PER VIGNETTE PER MODEL (sample) ===')
trial_counts = df.groupby(['model', 'vignette_id']).size()
print(f'Min: {trial_counts.min()}, Max: {trial_counts.max()}, Mean: {trial_counts.mean():.1f}')
print(trial_counts.value_counts().sort_index())
