import sys
sys.path.insert(0, ".")

from visualize.comprehensive_figures import ComprehensiveFigureBuilder
cfb = ComprehensiveFigureBuilder()

# Test each builder
builders = [
    ("naive_vignette_bars", cfb.build_naive_vignette_bars),
    ("naive_random_dose_bars", cfb.build_naive_random_dose_bars),
    ("cv_strip_vignettes", cfb.build_cv_strip_vignettes),
    ("cv_strip_random_doses", cfb.build_cv_strip_random_doses),
    ("nielsen_effect_vignettes", cfb.build_nielsen_effect_vignettes),
    ("nielsen_effect_random_doses", cfb.build_nielsen_effect_random_doses),
    ("nielsen_by_ome_vignettes", cfb.build_nielsen_by_ome_vignettes),
    ("nielsen_by_ome_random_doses", cfb.build_nielsen_by_ome_random_doses),
    ("explain_units", cfb.build_explain_units),
    ("full_treatment", cfb.build_full_treatment),
    ("cross_experiment", cfb.build_cross_experiment),
]

for name, builder in builders:
    try:
        fig = builder()
        print(f"  OK:  {name}")
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
        import traceback; traceback.print_exc()

print("\nDone!")
