from analysis.cross_experiment_comparison import CrossExperimentComparison
from common.models import CONSISTENT_MODELS

# OME vs non-OME per-model comparison: naive condition
CrossExperimentComparison(
    csv_a='results/pair_model_analysis.csv',
    csv_b='results/pair_model_analysis.csv',
    filter_a={'include_nielsen': False, 'include_examples': False, 'explain_units': False, 'ome': True},
    filter_b={'include_nielsen': False, 'include_examples': False, 'explain_units': False, 'ome': False},
    label_a='ome', label_b='non_ome',
    models=CONSISTENT_MODELS,
).save('results/comparison_ome_vs_nonome_naive_permodel.csv')

print()

# OME vs non-OME per-model comparison: Nielsen condition
CrossExperimentComparison(
    csv_a='results/pair_model_analysis.csv',
    csv_b='results/pair_model_analysis.csv',
    filter_a={'include_nielsen': True, 'include_examples': False, 'explain_units': False, 'ome': True},
    filter_b={'include_nielsen': True, 'include_examples': False, 'explain_units': False, 'ome': False},
    label_a='ome', label_b='non_ome',
    models=CONSISTENT_MODELS,
).save('results/comparison_ome_vs_nonome_nielsen_permodel.csv')
