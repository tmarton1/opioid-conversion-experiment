import pandas as pd, numpy as np
from common.models import CONSISTENT_MODELS

vma = pd.read_csv('results/reworded_vignettes_model_analysis.csv')
mask = (
    (vma['include_nielsen'] == False) &
    (vma['explain_units'] == False) &
    (vma['has_vignette_text'] == True) &
    (vma['multiple_from_opioids'] == False) &
    (vma['model'].isin(set(CONSISTENT_MODELS)))
)
filt = vma[mask].copy()
cv = filt['observed_dose_cv'].dropna()

print('Distribution:')
print(f'  CV = 0: {(cv == 0).sum()} ({(cv == 0).mean():.1%})')
print(f'  0 < CV <= 0.1: {((cv > 0) & (cv <= 0.1)).sum()} ({((cv > 0) & (cv <= 0.1)).mean():.1%})')
print(f'  CV > 0.1: {(cv > 0.1).sum()} ({(cv > 0.1).mean():.1%})')
print()

nonzero = cv[cv > 0]
print(f'Nonzero CV count: {len(nonzero)} ({len(nonzero)/len(cv):.1%})')
if len(nonzero) > 0:
    print(f'Nonzero CV median: {nonzero.median():.4f}')
    med = nonzero.median()
    print(f'  nonzero < median({med:.3f}): {(nonzero < med).sum()}')
    print(f'  nonzero >= median: {(nonzero >= med).sum()}')

print()
print('Suggested 3-bin: CV=0, 0<CV<=0.2, CV>0.2')
bins = ['CV=0', '0<CV<=0.2', 'CV>0.2']
counts = [
    (cv == 0).sum(),
    ((cv > 0) & (cv <= 0.2)).sum(),
    (cv > 0.2).sum(),
]
for b, c in zip(bins, counts):
    print(f'  {b}: {c} ({c/len(cv):.1%})')

# Also check random dose CV for comparison
print()
pma = pd.read_csv('results/pair_model_analysis.csv')
pmask = (
    (pma['include_nielsen'] == False) &
    (pma['model'].isin(set(CONSISTENT_MODELS)))
)
pfilt = pma[pmask].copy()
pcv = pfilt['observed_coefficient_cv'].dropna()
print('Random dose CV:')
print(pcv.describe())
print(f'  CV = 0: {(pcv == 0).sum()} ({(pcv == 0).mean():.1%})')
print(f'  0 < CV < 0.1: {((pcv > 0) & (pcv < 0.1)).sum()} ({((pcv > 0) & (pcv < 0.1)).mean():.1%})')
print(f'  0.1 <= CV < 0.5: {((pcv >= 0.1) & (pcv < 0.5)).sum()}/{len(pcv)}')
print(f'  CV >= 0.5: {(pcv >= 0.5).sum()}/{len(pcv)}')
