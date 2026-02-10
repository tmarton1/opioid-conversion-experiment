import json
from collections import defaultdict

with open('resources/reworded_vignettes.json') as f:
    data = json.load(f)

scenarios = defaultdict(list)
for k, v in data.items():
    scenarios[v['scenario_key']].append(v)

for sk, vignettes in sorted(scenarios.items()):
    from_ops = set(str(v['from_opioid']) for v in vignettes)
    to_ops = set(str(v['to_opioid']) for v in vignettes)
    dose_dists = set(str(v.get('dose_distribution')) for v in vignettes)
    output_units = set(str(v.get('output_unit', 'default')) for v in vignettes)
    output_convs = set(str(v.get('output_conversion', 'default')) for v in vignettes)
    ids = [v['vignette_id'] for v in vignettes]
    identical = (len(from_ops) == 1 and len(to_ops) == 1 and len(dose_dists) == 1
                 and len(output_units) == 1 and len(output_convs) == 1)
    status = 'SAME' if identical else 'DIFF'
    print(f'{status} {sk}: {ids}')
    if not identical:
        print(f'  from_opioid variants: {len(from_ops)}, to_opioid: {len(to_ops)}, dose_dist: {len(dose_dists)}, output_unit: {len(output_units)}, output_conv: {len(output_convs)}')
