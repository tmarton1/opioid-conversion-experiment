import pandas as pd
import numpy as np

# Load both result files
vig = pd.read_csv('results/reworded_vignettes_results.csv')
scen = pd.read_csv('results/scenario_experiment_results.csv')

print("="*80)
print("PROMPT CONDITIONS ANALYSIS")
print("="*80)

# Define prompt condition columns
condition_cols = ['include_nielsen', 'include_examples', 'ask_for_units', 'explain_units']

def analyze_prompt_conditions(df, name):
    print(f"\n{name}")
    print("-"*80)
    
    # Get unique combinations of prompt conditions
    prompt_combos = df[condition_cols].drop_duplicates().sort_values(by=condition_cols)
    print(f"\nUnique prompt condition combinations: {len(prompt_combos)}")
    print(prompt_combos.to_string(index=False))
    
    # Count observations per combination
    print(f"\n\nObservations per prompt condition combination:")
    combo_counts = df.groupby(condition_cols).size().reset_index(name='count')
    print(combo_counts.to_string(index=False))
    
    # Analyze coverage per vignette
    print(f"\n\nPrompt conditions per vignette:")
    vignette_coverage = df.groupby('vignette_id')[condition_cols].apply(
        lambda x: len(x.drop_duplicates())
    ).reset_index(name='num_prompt_combos')
    print(f"Min prompt combos per vignette: {vignette_coverage['num_prompt_combos'].min()}")
    print(f"Max prompt combos per vignette: {vignette_coverage['num_prompt_combos'].max()}")
    print(f"Mean prompt combos per vignette: {vignette_coverage['num_prompt_combos'].mean():.2f}")
    
    # Analyze coverage per model
    print(f"\n\nPrompt conditions per model:")
    model_coverage = df.groupby('model')[condition_cols].apply(
        lambda x: len(x.drop_duplicates())
    ).reset_index(name='num_prompt_combos')
    print(model_coverage.to_string(index=False))
    
    # Unique models and vignettes
    print(f"\n\nUnique models: {df['model'].nunique()}")
    print(f"Unique vignettes: {df['vignette_id'].nunique()}")
    
    # Model list
    print(f"\n\nModels used:")
    for i, model in enumerate(sorted(df['model'].unique()), 1):
        count = len(df[df['model'] == model])
        print(f"{i:2d}. {model:50s} ({count:5d} observations)")
    
    # Repetitions analysis
    print(f"\n\nRepetitions analysis:")
    reps = df.groupby(['vignette_id', 'model'] + condition_cols).size()
    print(f"Min repetitions: {reps.min()}")
    print(f"Max repetitions: {reps.max()}")
    print(f"Mean repetitions: {reps.mean():.2f}")
    print(f"Median repetitions: {reps.median():.0f}")
    
    return prompt_combos

# Analyze both datasets
vig_combos = analyze_prompt_conditions(vig, "REWORDED_VIGNETTES_RESULTS")
scen_combos = analyze_prompt_conditions(scen, "SCENARIO_EXPERIMENT_RESULTS")

# Compare the experiments
print("\n\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"\nReworded vignettes has {len(vig)} rows across {vig['vignette_id'].nunique()} vignettes")
print(f"Scenario experiment has {len(scen)} rows across {scen['vignette_id'].nunique()} vignettes")
print(f"\nReworded vignettes uses {vig['model'].nunique()} models")
print(f"Scenario experiment uses {scen['model'].nunique()} models")
