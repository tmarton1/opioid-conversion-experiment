import pandas as pd
import numpy as np

df = pd.read_csv('results/pair_model_analysis.csv')
from common.models import CONSISTENT_MODELS
cset = set(CONSISTENT_MODELS)
naive = df[(df['include_nielsen']==False) & (df['include_examples']==False)]
naive = naive[naive['model'].isin(cset)].copy()
print(f'Naive consistent: {len(naive)} rows, {naive.model.nunique()} models, {naive.opioid_pair.nunique()} pairs')

# Compute ratio = observed_coefficient_mean / expected_coefficient
naive['ratio'] = naive['observed_coefficient_mean'] / naive['expected_coefficient']
TIGHT_LO, TIGHT_HI = 0.85, 1.15

# Three categories
naive['neg_slope'] = (naive['regression_slope'] < 0) & (naive['regression_slope_pvalue'] < 0.05)
naive['precise'] = naive['observed_coefficient_cv'] <= 0.2
naive['tight_acc'] = (naive['ratio'] >= TIGHT_LO) & (naive['ratio'] <= TIGHT_HI)

def categorize(row):
    if row['neg_slope']:
        return 'dose_dependent'
    elif row['precise'] and row['tight_acc']:
        return 'precise_correct'
    elif row['precise'] and not row['tight_acc']:
        return 'precise_wrong'
    elif row['tight_acc']:
        return 'imprecise_correct'
    else:
        return 'imprecise_wrong'

naive['category'] = naive.apply(categorize, axis=1)

# Overall proportions
print()
print('=== OVERALL PROPORTIONS ===')
vc = naive['category'].value_counts()
for cat in ['precise_correct', 'precise_wrong', 'dose_dependent', 'imprecise_correct', 'imprecise_wrong']:
    n = vc.get(cat, 0)
    pct = 100 * n / len(naive)
    print(f'  {cat:25s}  {n:4d}  ({pct:5.1f}%)')

# Per-model proportions
print()
print('=== PER MODEL (%) ===')
ct = pd.crosstab(naive['model'].apply(lambda m: m.split('/')[-1]), naive['category'], normalize='index')
ct = ct.reindex(columns=['precise_correct', 'precise_wrong', 'dose_dependent', 'imprecise_correct', 'imprecise_wrong'], fill_value=0)
print((ct * 100).round(1).to_string())

# Also check: of the dose-dependent ones, are they also precise or imprecise?
print()
print('=== DOSE-DEPENDENT OVERLAP ===')
dd = naive[naive['neg_slope']]
print(f'  Total dose-dependent: {len(dd)}')
print(f'  Also precise (CV<=0.2): {(dd["precise"]).sum()} ({100*(dd["precise"]).mean():.1f}%)')
print(f'  Also tight-accurate: {(dd["tight_acc"]).sum()} ({100*(dd["tight_acc"]).mean():.1f}%)')

# Per-model count of dose-dependent
print()
print('=== DOSE-DEPENDENT COUNT PER MODEL ===')
dd_ct = dd.groupby(dd['model'].apply(lambda m: m.split('/')[-1])).size()
for m in sorted(dd_ct.index):
    print(f'  {m:35s}  {dd_ct[m]:3d}/{naive.opioid_pair.nunique()}')

print()
print(f'Unique pairs: {naive.opioid_pair.nunique()}')
print(f'Models: {naive.model.nunique()}')
print(f'Total cells: {len(naive)}')
