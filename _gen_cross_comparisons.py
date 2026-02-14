from analysis.cross_experiment_comparison import CrossExperimentComparison
from common.models import CONSISTENT_MODELS

# 1. Random dose vs empty vignette (both naive)
CrossExperimentComparison(
    csv_a='results/pair_model_analysis.csv',
    csv_b='results/reworded_vignettes_model_analysis.csv',
    filter_a={'include_nielsen': False, 'include_examples': False, 'explain_units': False},
    filter_b={'include_nielsen': False, 'explain_units': False, 'has_vignette_text': False},
    label_a='random_dose', label_b='empty_vignette',
    models=CONSISTENT_MODELS,
).save('results/comparison_cross_rd_vs_empty.csv')

print()

# 2. Random dose vs text vignette (both naive)
CrossExperimentComparison(
    csv_a='results/pair_model_analysis.csv',
    csv_b='results/reworded_vignettes_model_analysis.csv',
    filter_a={'include_nielsen': False, 'include_examples': False, 'explain_units': False},
    filter_b={'include_nielsen': False, 'explain_units': False, 'has_vignette_text': True},
    label_a='random_dose', label_b='text_vignette',
    models=CONSISTENT_MODELS,
).save('results/comparison_cross_rd_vs_text.csv')

print()

# 3. Empty vignette vs text vignette (both naive)
CrossExperimentComparison(
    csv_a='results/reworded_vignettes_model_analysis.csv',
    csv_b='results/reworded_vignettes_model_analysis.csv',
    filter_a={'include_nielsen': False, 'explain_units': False, 'has_vignette_text': False},
    filter_b={'include_nielsen': False, 'explain_units': False, 'has_vignette_text': True},
    label_a='empty_vignette', label_b='text_vignette',
    models=CONSISTENT_MODELS,
).save('results/comparison_cross_empty_vs_text.csv')
