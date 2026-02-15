"""Generate cross-experiment matched CSVs and test the figure."""
from analysis.cross_experiment_matched import CrossExperimentMatched

# Step 1: Generate CSVs
cem = CrossExperimentMatched(
    pair_csv="results/pair_model_analysis.csv",
    vignette_csv="results/reworded_vignettes_model_analysis.csv",
)
cem.save("results")

# Step 2: Test figure
from visualize.comprehensive_figures import ComprehensiveFigureBuilder
fb = ComprehensiveFigureBuilder("results")
fig = fb.build_cross_experiment()

from bokeh.plotting import output_file, save
output_file("figures/cross_experiment_bars.html")
save(fig)
print("\nSaved figures/cross_experiment_bars.html")
print("DONE")
