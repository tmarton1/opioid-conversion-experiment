#!/usr/bin/env python3
"""Assign vignette colors to the reworded_vignettes.json."""
import json
import re
from collections import defaultdict

# 4 muted vignette colors, distinct from the bright saturated model palette
VIGNETTE_COLORS = ['#2d5f8a', '#8a5c2d', '#5a8a2d', '#8a2d5a']
# steel blue, sienna, olive green, plum

with open('resources/reworded_vignettes.json') as f:
    vigs = json.load(f)

scenarios = defaultdict(list)
for vid, v in vigs.items():
    if v.get('has_vignette_text', True):
        scenarios[v['scenario_key']].append(vid)

# Assign colors by sorted vignette order within each scenario
for scenario_key, vids in scenarios.items():
    sorted_vids = sorted(vids, key=lambda x: int(re.search(r'\d+', x).group()))
    for i, vid in enumerate(sorted_vids):
        vigs[vid]['vignette_color'] = VIGNETTE_COLORS[i]

# Also assign a neutral color to empty vignettes
for vid, v in vigs.items():
    if not v.get('has_vignette_text', True):
        v['vignette_color'] = '#888888'

with open('resources/reworded_vignettes.json', 'w') as f:
    json.dump(vigs, f, indent=2)

print('Assigned vignette colors:')
for scenario_key in sorted(scenarios):
    vids = scenarios[scenario_key]
    sorted_vids = sorted(vids, key=lambda x: int(re.search(r'\d+', x).group()))
    colors = [vigs[vid]['vignette_color'] for vid in sorted_vids]
    pairs = [f"{vid}={c}" for vid, c in zip(sorted_vids, colors)]
    print(f"  {scenario_key}: {', '.join(pairs)}")

print(f"\nColor palette: {VIGNETTE_COLORS}")
print("Done.")
