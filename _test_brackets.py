"""Test build of the three modified figures."""
from visualize.comprehensive_figures import ComprehensiveFigureBuilder
from bokeh.io import save

fb = ComprehensiveFigureBuilder('results')

print("Building explain_units...")
fig1 = fb.build_explain_units()
save(fig1, 'figures/explain_units.html', title='Explain Units')
print("  OK")

print("Building full_treatment...")
fig2 = fb.build_full_treatment()
save(fig2, 'figures/full_treatment.html', title='Full Treatment')
print("  OK")

print("Building cross_experiment...")
fig3 = fb.build_cross_experiment()
save(fig3, 'figures/cross_experiment_bars.html', title='Cross Experiment')
print("  OK")

print("Building nielsen_effect_vignettes...")
fig4 = fb.build_nielsen_effect_vignettes()
save(fig4, 'figures/nielsen_effect_vignettes.html', title='Nielsen Effect Vignettes')
print("  OK")

print("Building ome_vs_nonome...")
fig5 = fb.build_ome_vs_nonome()
save(fig5, 'figures/ome_vs_nonome.html', title='OME vs Non-OME')
print("  OK")

print("=== ALL DONE ===")
