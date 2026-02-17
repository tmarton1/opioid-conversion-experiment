import pandas as pd
from common.models import CONSISTENT_MODELS
rvm = pd.read_csv('results/reworded_vignettes_model_analysis.csv')
mask = (
    (rvm['include_nielsen'] == False) &
    (rvm['explain_units'] == False) &
    (rvm['has_vignette_text'] == True) &
    (rvm['multiple_from_opioids'] == False) &
    (rvm['model'].isin(set(CONSISTENT_MODELS)))
)
filt = rvm[mask]
print('Filtered shape:', filt.shape)
print('Models:', sorted(filt['model'].unique()))
print()

cv = filt['observed_dose_cv']
print('CV value counts:')
print(f'  0.0: {(cv == 0).sum()}')
print(f'  (0, 0.1): {((cv > 0) & (cv < 0.1)).sum()}')
print(f'  [0.1, 0.5): {((cv >= 0.1) & (cv < 0.5)).sum()}')
print(f'  >= 0.5: {(cv >= 0.5).sum()}')
print()

grp = filt.groupby(['model', 'scenario_key']).size()
print('Vignettes per model x scenario:')
print(grp.describe())
print()

# ICC data
icc = pd.read_csv('results/icc_analysis.csv')
icc = icc[icc['model'].isin(set(CONSISTENT_MODELS))]
print('ICC shape:', icc.shape)
print('var_total range:', icc['var_total'].min(), '-', icc['var_total'].max())
print('ICC range:', icc['icc'].min(), '-', icc['icc'].max())
print()

scenario_var = icc.groupby('scenario_key')['var_total'].mean()
import numpy as np
t1, t2 = np.percentile(scenario_var.values, [33.3, 66.7])
print(f'Variance tercile boundaries: {t1:.4f}, {t2:.4f}')
print('Scenarios per bin:')
print(f'  Low (<{t1:.4f}): {(scenario_var < t1).sum()}')
print(f'  Mid ({t1:.4f}-{t2:.4f}): {((scenario_var >= t1) & (scenario_var < t2)).sum()}')
print(f'  High (>={t2:.4f}): {(scenario_var >= t2).sum()}')
