"""Explore vignette vs random-dose opioid pair overlap."""
import pandas as pd

# ── Vignette analysis CSV ──
vt = pd.read_csv('results/reworded_vignettes_model_analysis.csv')
vt_naive = vt[(vt['include_nielsen'] == False) & (vt['explain_units'] == False)]

# ── Random-dose pairs ──
rd = pd.read_csv('results/pair_model_analysis.csv')
rd_naive = rd[
    (rd['include_nielsen'] == False) &
    (rd['include_examples'] == False) &
    (rd['explain_units'] == False)
]

# ── Build overlap set ──
rd_pair_keys = set(zip(
    rd_naive['from_opioid'], rd_naive['to_opioid'],
    rd_naive['from_opioid_unit'], rd_naive['to_opioid_unit'],
))

vt_text_single = vt_naive[
    (vt_naive['has_vignette_text'] == True) &
    (vt_naive['multiple_from_opioids'] == False)
]
vt_scenario_pairs = vt_text_single[
    ['scenario_key', 'from_opioid', 'to_opioid', 'from_opioid_unit', 'to_opioid_unit']
].drop_duplicates()

overlap_scenarios = []
for _, r in vt_scenario_pairs.iterrows():
    key = (r.from_opioid, r.to_opioid, r.from_opioid_unit, r.to_opioid_unit)
    if key in rd_pair_keys:
        overlap_scenarios.append(r.scenario_key)

overlap_pair_keys = set()
for _, r in vt_scenario_pairs[vt_scenario_pairs['scenario_key'].isin(overlap_scenarios)].iterrows():
    overlap_pair_keys.add((r.from_opioid, r.to_opioid, r.from_opioid_unit, r.to_opioid_unit))

print(f'Overlapping scenarios: {len(overlap_scenarios)}')
print(f'Overlapping pair keys: {len(overlap_pair_keys)}')
for pk in sorted(overlap_pair_keys):
    print(f'  {pk}')

# ── Check granularity ──
# Text vignettes: how many rows per model for an overlapping scenario?
ex = vt_naive[
    (vt_naive['scenario_key'] == 'adult-male-dog-bite') &
    (vt_naive['has_vignette_text'] == True)
]
print(f'\nText vignettes for adult-male-dog-bite: {len(ex)} rows')
print(ex[['model', 'vignette_id']].to_string())

# Empty vignettes
ex2 = vt_naive[
    (vt_naive['scenario_key'] == 'adult-male-dog-bite') &
    (vt_naive['has_vignette_text'] == False)
]
print(f'\nEmpty vignettes for adult-male-dog-bite: {len(ex2)} rows')
print(ex2[['model', 'vignette_id']].to_string())

# Random dose: 1 row per model per pair
ex3 = rd_naive[
    (rd_naive['from_opioid'] == 'iv_morphine') &
    (rd_naive['to_opioid'] == 'oral_morphine')
]
print(f'\nRD rows for iv_morphine->oral_morphine: {len(ex3)} rows')
print(ex3[['model', 'from_opioid', 'to_opioid']].to_string())

# ── Per-model accuracy on overlapping pairs only ──
from common.models import CONSISTENT_MODELS

# Filter RD to overlapping pairs
rd_overlap = rd_naive[
    rd_naive.apply(lambda r: (r['from_opioid'], r['to_opioid'],
                              r['from_opioid_unit'], r['to_opioid_unit']) in overlap_pair_keys, axis=1)
]
rd_overlap = rd_overlap[rd_overlap['model'].isin(CONSISTENT_MODELS)]

# Filter vignette text to overlapping scenarios
vt_text_overlap = vt_naive[
    (vt_naive['scenario_key'].isin(overlap_scenarios)) &
    (vt_naive['has_vignette_text'] == True) &
    (vt_naive['multiple_from_opioids'] == False) &
    (vt_naive['model'].isin(CONSISTENT_MODELS))
]

# Filter vignette empty to overlapping scenarios
vt_empty_overlap = vt_naive[
    (vt_naive['scenario_key'].isin(overlap_scenarios)) &
    (vt_naive['has_vignette_text'] == False) &
    (vt_naive['multiple_from_opioids'] == False) &
    (vt_naive['model'].isin(CONSISTENT_MODELS))
]

print(f'\n=== Filtered data sizes ===')
print(f'RD overlap rows: {len(rd_overlap)} ({len(rd_overlap["model"].unique())} models × {rd_overlap.groupby(["from_opioid","to_opioid"]).ngroups} pairs)')
print(f'VT text overlap rows: {len(vt_text_overlap)} ({len(vt_text_overlap["model"].unique())} models)')
print(f'VT empty overlap rows: {len(vt_empty_overlap)} ({len(vt_empty_overlap["model"].unique())} models)')

# Per-model means
print('\n=== Per-model accuracy on overlapping pairs ===')
print(f'{"Model":<45} {"RD tight":>10} {"RD broad":>10} {"VT-T tight":>10} {"VT-T broad":>10} {"VT-E tight":>10} {"VT-E broad":>10}')
for m in sorted(CONSISTENT_MODELS):
    rd_m = rd_overlap[rd_overlap['model'] == m]
    vt_t_m = vt_text_overlap[vt_text_overlap['model'] == m]
    vt_e_m = vt_empty_overlap[vt_empty_overlap['model'] == m]
    print(f'{m:<45} {rd_m["tight(0.85,1.15)"].mean():>10.3f} {rd_m["broad(0.6,1.7)"].mean():>10.3f}'
          f' {vt_t_m["tight(0.85,1.15)"].mean():>10.3f} {vt_t_m["broad(0.6,1.7)"].mean():>10.3f}'
          f' {vt_e_m["tight(0.85,1.15)"].mean():>10.3f} {vt_e_m["broad(0.6,1.7)"].mean():>10.3f}')
