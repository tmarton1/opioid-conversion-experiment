import json

with open('resources/reworded_vignettes.json') as f:
    vigs = json.load(f)

# Look at structure
first_key = list(vigs.keys())[0]
print('Sample entry keys:', list(vigs[first_key].keys()))
print()

# Collect from_opioid / to_opioid
from_ops = set()
to_ops = set()
for vid, v in vigs.items():
    for key in ['from_opioid', 'to_opioid']:
        val = v.get(key)
        if isinstance(val, dict):
            name = f"{val.get('administration_method')}_{val.get('substance')}"
            if key == 'from_opioid':
                from_ops.add(name)
            else:
                to_ops.add(name)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    name = f"{item.get('administration_method')}_{item.get('substance')}"
                else:
                    name = str(item)
                if key == 'from_opioid':
                    from_ops.add(name)
                else:
                    to_ops.add(name)
        elif isinstance(val, str):
            if key == 'from_opioid':
                from_ops.add(val)
            else:
                to_ops.add(val)

all_ops = sorted(from_ops | to_ops)

with open('resources/nielsen_ome.json') as fj:
    nielsen = json.load(fj)

print(f'Unique opioids in vignettes: {len(all_ops)}')
for o in all_ops:
    p = nielsen.get(o, {}).get('priority', '???')
    print(f'  {o:40s} priority={p}')

print()
non12 = [o for o in all_ops if nielsen.get(o, {}).get('priority') not in (1, 2)]
print(f'Vignette opioids with priority != 1 or 2: {len(non12)}')
for o in non12:
    p = nielsen.get(o, {}).get('priority')
    print(f'  {o:40s} priority={p}')

# Also show the 5 opioids that need priority 3
print()
print("=== Opioids in random-dose 19 one-directional pairs with priority=None ===")
missing = ['im_buprenorphine', 'iv_buprenorphine', 'iv_methadone', 'parenteral_nalbuphine', 'transdermal_buprenorphine']
for m in missing:
    in_vignettes = m in all_ops
    print(f'  {m:40s} in vignettes: {in_vignettes}')
