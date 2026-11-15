from common.config import RESULTS_DIR
import pandas as pd
import json

with open('resources/nielsen_ome.json') as f:
    nielsen = json.load(f)

def get_priority(opioid):
    p = nielsen.get(opioid, {}).get('priority')
    return p if p is not None else 'None'

# ── 1. Random-dose experiment ──
print("=" * 70)
print("1. Random-dose experiment (experiment_results.csv)")
print("=" * 70)
rd = pd.read_csv(f'{RESULTS_DIR}/experiment_results.csv')
rd_pairs = rd[['from_opioid', 'to_opioid']].drop_duplicates().sort_values(['from_opioid', 'to_opioid'])
print(f"Total directed pairs: {len(rd_pairs)}")
all_ops = sorted(set(rd_pairs['from_opioid']) | set(rd_pairs['to_opioid']))
print(f"Unique opioids: {len(all_ops)}")

pairs_set = set(map(tuple, rd_pairs.values))
bidir = []
oneway = []
for a in all_ops:
    for b in all_ops:
        if a < b:
            ab = (a, b) in pairs_set
            ba = (b, a) in pairs_set
            if ab and ba:
                bidir.append((a, b))
            elif ab:
                oneway.append((a, b))
            elif ba:
                oneway.append((b, a))

print(f"\nBidirectional pairs: {len(bidir)} (= {len(bidir)*2} directed)")
for a, b in bidir:
    pa, pb = get_priority(a), get_priority(b)
    print(f"  {a} (P{pa}) <-> {b} (P{pb})")

print(f"\nOne-directional pairs: {len(oneway)}")
for a, b in oneway:
    pa, pb = get_priority(a), get_priority(b)
    print(f"  {a} (P{pa}) -> {b} (P{pb})")

# Summarize bidirectional pair priorities
bidir_ops = set()
for a, b in bidir:
    bidir_ops.add(a)
    bidir_ops.add(b)

print(f"\nOpioids in bidirectional pairs ({len(bidir_ops)}):")
for o in sorted(bidir_ops):
    print(f"  {o} (P{get_priority(o)})")

oneway_from = set(a for a, b in oneway)
oneway_only = oneway_from - bidir_ops
print(f"\nOpioids ONLY in one-directional pairs ({len(oneway_only)}):")
for o in sorted(oneway_only):
    print(f"  {o} (P{get_priority(o)})")

# ── 2. Explain-units experiment ──
print("\n" + "=" * 70)
print("2. Explain-units experiment (explain_units_experiment_results.csv)")
print("=" * 70)
eu = pd.read_csv(f'{RESULTS_DIR}/explain_units_experiment_results.csv')
eu_pairs = eu[['from_opioid', 'to_opioid']].drop_duplicates().sort_values(['from_opioid', 'to_opioid'])
print(f"Total directed pairs: {len(eu_pairs)}")
eu_ops = sorted(set(eu_pairs['from_opioid']) | set(eu_pairs['to_opioid']))
print(f"Unique opioids: {len(eu_ops)}")
for _, r in eu_pairs.iterrows():
    pf = get_priority(r.from_opioid)
    pt = get_priority(r.to_opioid)
    self_mark = " [SELF]" if r.from_opioid == r.to_opioid else ""
    print(f"  {r.from_opioid} (P{pf}) -> {r.to_opioid} (P{pt}){self_mark}")

# ── 3. Vignette experiment ──
print("\n" + "=" * 70)
print("3. Vignette experiment (reworded_vignettes_results.csv)")
print("=" * 70)
vig = pd.read_csv(f'{RESULTS_DIR}/reworded_vignettes_results.csv')
vig_pairs = vig[['from_opioid', 'to_opioid']].drop_duplicates().sort_values(['from_opioid', 'to_opioid'])
print(f"Total directed pairs: {len(vig_pairs)}")
vig_ops = sorted(set(vig_pairs['from_opioid']) | set(vig_pairs['to_opioid']))
print(f"Unique opioids: {len(vig_ops)}")
for _, r in vig_pairs.iterrows():
    pf = get_priority(r.from_opioid)
    pt = get_priority(r.to_opioid)
    print(f"  {r.from_opioid} (P{pf}) -> {r.to_opioid} (P{pt})")
