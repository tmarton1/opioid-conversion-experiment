"""
The reported set (author decision 36, confirmed in decision 37).

One configuration for what feeds the paper, the appendix and the private
differences report: registry entries, scores, conditions and subsets.
3A (distribution_metrics), 3B (circularity) and the safety tests
(extreme_breakdown.tail_tests) compute only this set.

Standard-anchored registry entries (above_range, below_range, overdose,
severe_overdose, underdose) do not depend on the score: they are computed once
and written with score == "any" (3B: subset "all" only). Under non_nielsen only
above_range and below_range are kept (fewer conversions are covered, so they
differ). Accuracy bands (tight, broad) and median fold error depend on the
score and are computed per score.
"""

ANY = "any"   # score label of rows for standard-anchored entries (computed once, not per score)
ANCHORED = ("above_range", "below_range", "overdose", "severe_overdose", "underdose")   # registry entries with anchor != "expected"
ACCURACY = ("tight", "broad")                      # anchor "expected": depend on the score; median_fold_error is the threshold-free outcome
MAIN_SCORES = ("nielsen", "all", "non_nielsen")    # the three the plan compares
EXTRA_SCORES = ("geomean",)                        # kept only where Section A/B numbers read it
ALL4 = ("all", "ome", "nonome", "disagree")        # conversion subsets used by 3B
MAIN7 = ("vignette_naive", "vignette_nielsen", "vignette_nielsen_units", "rd_naive", "rd_nielsen",
         "eu_4h_nielsen", "eu_4h_nielsen_units")   # conditions the report, numbers and figures read
MAIN5 = MAIN7[:5]
SUPP4 = ("rd_nielsen_examples", "vignette_empty_naive", "vignette_empty_nielsen", "vignette_empty_nielsen_units")

REPORTED_SET = {
    "distribution_summary": {   # 3A; extreme_errors.csv stays a view of the nielsen rows
        "conditions_by_score": {"nielsen": MAIN7 + SUPP4, "all": MAIN7, "non_nielsen": MAIN7, "geomean": MAIN5},
        # score-independent columns: written once, on rows with score == "any"; blank on every other row
        "anchored_columns": ("pct_above_range", "pct_below_range", "pct_overdose", "pct_severe_overdose",
                             "pct_underdose", "n_ge_10x_highest"),
        "anchored_conditions": MAIN7,
        "anchored_source_scores": ("nielsen", "all", "geomean"),   # same coverage as "any"; non_nielsen covers fewer items
    },
    "wilcoxon_threshold_free": {   # 3A: the 14 comparison x subset cells
        "scores": MAIN_SCORES + EXTRA_SCORES,
        "tight_scores": ("nielsen",),    # tight_* columns: published comparison only
        "grid": (("grounding_vignettes", "all"), ("grounding_vignettes", "ome"), ("grounding_vignettes", "nonome"),
                 ("grounding_random_doses", "all"), ("grounding_random_doses", "ome"), ("grounding_random_doses", "nonome"),
                 ("units_effect_vignettes", "all"), ("units_effect_vignettes", "ome"), ("units_effect_vignettes", "nonome"),
                 ("unit_explanation_random_doses", "all"), ("dosing_unit", "ome"),
                 ("grounding_empty_prompts", "all"), ("grounding_empty_prompts", "ome"), ("grounding_empty_prompts", "nonome")),
    },
    "safety_exceedance": {"conditions": MAIN7 + SUPP4[1:]},
    "circularity": {   # 3B
        "accuracy_outcomes": ("median_fold_error",) + ACCURACY,
        "accuracy_grid": {"nielsen": ALL4, "all": ALL4, "non_nielsen": ALL4, "geomean": ("all", "disagree")},
        "anchored": {
            "any_from_scores": ("nielsen", "all", "geomean"),                 # written once: score "any", subset "all"
            "coverage_restricted": ("non_nielsen",),                          # above / below range scored without Nielsen
            "coverage_restricted_outcomes": ("above_range", "below_range"),   # (fewer items covered)
        },
    },
    "safety_tests": {   # Nielsen score only: all 3 tests x all 5 entries
        "tests": (("vignette_naive", "vignette_nielsen"), ("vignette_naive", "vignette_nielsen_units"), ("rd_naive", "rd_nielsen")),
        "outcomes": ANCHORED,
    },
}

ANCHORED_SOURCE_SCORE = "nielsen"   # the score the "any" rows are computed under (full coverage)


def conditions_for(score):
    """Conditions 3A summarises under `score`."""
    return REPORTED_SET["distribution_summary"]["conditions_by_score"][score]


def accuracy_subsets(score):
    """Conversion subsets 3B uses for the accuracy outcomes under `score`."""
    return REPORTED_SET["circularity"]["accuracy_grid"][score]
