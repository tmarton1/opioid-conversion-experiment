#!/usr/bin/env python3
"""Explore prompt conditions in vignette results."""
import pandas as pd

df = pd.read_csv('results/reworded_vignettes_results.csv')

# What prompt columns exist?
prompt_cols = ['include_nielsen', 'include_examples', 'ask_for_units', 'explain_units', 'has_vignette_text']
for c in prompt_cols:
    if c in df.columns:
        print(f"{c}: {sorted(df[c].unique())}")

print()

# Count trials per unique condition
conditions = df.groupby(prompt_cols).size().reset_index(name='count')
print("=== PROMPT CONDITIONS (row counts) ===")
print(conditions.to_string(index=False))
print()

# For text vignettes only
text_df = df[df['has_vignette_text'] == True]
text_conditions = text_df.groupby(['include_nielsen', 'include_examples', 'explain_units']).size().reset_index(name='count')
print("=== TEXT VIGNETTE CONDITIONS ===")
print(text_conditions.to_string(index=False))
print()

# Trials per vignette per model per condition
for _, cond in text_conditions.iterrows():
    nielsen = cond['include_nielsen']
    examples = cond['include_examples']
    explain = cond['explain_units']
    subset = text_df[
        (text_df['include_nielsen'] == nielsen) &
        (text_df['include_examples'] == examples) &
        (text_df['explain_units'] == explain)
    ]
    trials = subset.groupby(['model', 'vignette_id']).size()
    print(f"nielsen={nielsen}, examples={examples}, explain={explain}: "
          f"{trials.min()}-{trials.max()} trials/vignette/model (mean {trials.mean():.1f})")
