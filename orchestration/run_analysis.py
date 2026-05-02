"""
Run the complete analysis pipeline on experiment results CSVs.

This script replaces the analysis-orchestration cells in opioid-conversion.ipynb.
All steps are idempotent: they overwrite their output CSVs. Run this after
run_experiments.py has produced the raw results CSVs, and before
generate_summaries.py.

Steps
-----
1.  processing_pipeline — raw CSVs → model-analysis CSVs
2.  model batch means
3.  accuracy comparisons (AccuracyComparison — Fisher/CMH tests)
4.  per-model Wilcoxon tests on all comparison CSVs
5.  text effect differences
6.  OME vs non-OME comparisons
7.  dosing-unit effect comparison
8.  cross-experiment matched analysis
9.  ICC analysis
10. Kendall's W
11. Variance–ICC Spearman statistics
12. Grid independence tests
13. Performance profiles (random doses and vignettes)
14. CV–accuracy chi-square tests
15. Model rankings

Usage
-----
    python run_analysis.py [--section SECTION]

    SECTION choices:
        processing | batch_means | comparisons | wilcoxon | text_effects
        | ome_comparisons | dosing | cross_experiment | icc | kendall_w
        | variance | grid | profiles | cv_chi2 | rankings | all
    Default: all

Run from the repository root:
    python orchestration/run_analysis.py [--section SECTION]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


def _parse_args():
    choices = [
        "processing", "batch_means", "comparisons", "wilcoxon",
        "text_effects", "ome_comparisons", "dosing", "cross_experiment",
        "icc", "kendall_w", "variance", "grid", "profiles", "cv_chi2",
        "rankings", "all",
    ]
    p = argparse.ArgumentParser(description="Run opioid conversion analysis pipeline")
    p.add_argument("--section", choices=choices, default="all",
                   help="Which analysis section to run (default: all)")
    return p.parse_args()


# ── constants ──────────────────────────────────────────────────────────────

PAIR_CSV         = "results/pair_model_analysis.csv"
EXPLAIN_CSV      = "results/explain_units_pair_model_analysis.csv"
VIGNETTE_CSV     = "results/reworded_vignettes_model_analysis.csv"
BATCH_MEANS_RD   = "results/model_batch_means.csv"
BATCH_MEANS_VIG  = "results/model_batch_means_vignettes.csv"


# ── 1. Processing pipeline ─────────────────────────────────────────────────

def run_processing():
    """
    Run processing_pipeline for all three raw result CSVs.

    Inputs:  results/experiment_results.csv
             results/explain_units_experiment_results.csv
             results/reworded_vignettes_results.csv
    Outputs: results/*_by_ome.csv, results/*_model_analysis.csv,
             results/*_consensus.csv, results/reworded_scenarios_model_analysis.csv
    """
    from analysis.analysis_utils import processing_pipeline
    from analysis.scenario_model_result import ScenarioModelResult

    print("── Processing random-dose results ──")
    processing_pipeline(
        input_csv="results/experiment_results.csv",
        output_csv="results/experiment_results_by_ome.csv",
        variable_model_csv=PAIR_CSV,
        consensus_csv="results/opioid_pair_consensus.csv",
        mode="pair",
    )

    print("── Processing explain-units results ──")
    processing_pipeline(
        input_csv="results/explain_units_experiment_results.csv",
        output_csv="results/explain_units_experiment_results_by_ome.csv",
        variable_model_csv=EXPLAIN_CSV,
        consensus_csv="results/explain_units_opioid_pair_consensus.csv",
        mode="pair",
    )

    print("── Processing vignette results ──")
    processing_pipeline(
        input_csv="results/reworded_vignettes_results.csv",
        output_csv="results/reworded_vignettes_results_by_ome.csv",
        variable_model_csv=VIGNETTE_CSV,
        consensus_csv="results/reworded_vignettes_model_consensus.csv",
        mode="vignette",
    )

    ScenarioModelResult.process_all_combinations(
        vignette_results_csv=VIGNETTE_CSV,
        output_file="results/reworded_scenarios_model_analysis.csv",
        verbose=True,
    )
    print("✓ Processing complete")


# ── 2. Model batch means ────────────────────────────────────────────────────

def run_batch_means():
    """
    Compute and save model batch means for random doses, explain-units,
    vignettes, and text-vs-empty comparisons.
    """
    from analysis.model_means import ModelBatchMeansBuilder

    print("── Building random-dose batch means ──")
    rd_combos = [
        {"ome": None, "include_nielsen": False, "include_examples": False, "batch_name": "all_n0"},
        {"ome": None, "include_nielsen": True,  "include_examples": False, "batch_name": "all_n1"},
        {"ome": True, "include_nielsen": False, "include_examples": False, "batch_name": "ome_n0"},
        {"ome": True, "include_nielsen": True,  "include_examples": False, "batch_name": "ome_n1"},
        {"ome": False,"include_nielsen": False, "include_examples": False, "batch_name": "pairs_n0"},
        {"ome": False,"include_nielsen": True,  "include_examples": False, "batch_name": "pairs_n1"},
    ]
    ModelBatchMeansBuilder(PAIR_CSV).save_means_df(rd_combos, BATCH_MEANS_RD)

    print("── Building explain-units batch means ──")
    eu_combos = [
        {"explain_units": False, "batch_name": "4hours_units"},
        {"explain_units": True,  "batch_name": "4hours_units_explained"},
    ]
    ModelBatchMeansBuilder(EXPLAIN_CSV).save_means_df(
        eu_combos, "results/model_batch_means_explain_units.csv"
    )

    print("── Building vignette batch means ──")
    vig_combos = [
        {"ome": None, "include_nielsen": False, "explain_units": False, "batch_name": "all_n0"},
        {"ome": None, "include_nielsen": True,  "explain_units": False, "batch_name": "all_n1"},
        {"ome": True, "include_nielsen": False, "explain_units": False, "batch_name": "ome_n0"},
        {"ome": True, "include_nielsen": True,  "explain_units": False, "batch_name": "ome_n1"},
        {"ome": True, "include_nielsen": True,  "explain_units": True,  "batch_name": "ome_n1_units"},
        {"ome": False,"include_nielsen": False, "explain_units": False, "batch_name": "pairs_n0"},
        {"ome": False,"include_nielsen": True,  "explain_units": False, "batch_name": "pairs_n1"},
    ]
    ModelBatchMeansBuilder(VIGNETTE_CSV).save_means_df(vig_combos, BATCH_MEANS_VIG)

    print("── Building text-vs-empty batch means ──")
    tve_combos = []
    for ome_val, ome_label in [(None, "all"), (True, "ome"), (False, "nonome")]:
        for hvt, hvt_label in [(True, "text"), (False, "empty")]:
            tve_combos += [
                {"ome": ome_val, "has_vignette_text": hvt,
                 "include_nielsen": False, "explain_units": False,
                 "batch_name": f"{hvt_label}_{ome_label}_naive"},
                {"ome": ome_val, "has_vignette_text": hvt,
                 "include_nielsen": True, "explain_units": False,
                 "batch_name": f"{hvt_label}_{ome_label}_nielsen"},
                {"ome": ome_val, "has_vignette_text": hvt,
                 "include_nielsen": True, "explain_units": True,
                 "batch_name": f"{hvt_label}_{ome_label}_explain"},
            ]
    ModelBatchMeansBuilder(VIGNETTE_CSV).save_means_df(
        tve_combos, "results/model_batch_means_text_vs_empty.csv"
    )
    print("✓ Batch means complete")


# ── 3. Accuracy comparisons ────────────────────────────────────────────────

def run_comparisons():
    """
    Run AccuracyComparison (Fisher/CMH tests) for all experimental contrasts.
    Saves one CSV per comparison to results/comparison_*.csv.
    """
    from analysis.accuracy_comparison import AccuracyComparison

    specs = [
        # ── Vignette contrasts ──────────────────────────────────────────
        dict(
            csv_path=VIGNETTE_CSV,
            group_col="has_vignette_text", group_a_value=True, group_b_value=False,
            group_a_label="text", group_b_label="empty",
            unit_col="scenario_key",
            hold_constant=["include_nielsen", "explain_units"],
            output="results/comparison_text_vs_empty.csv",
        ),
        dict(
            csv_path=VIGNETTE_CSV,
            group_col="include_nielsen", group_a_value=True, group_b_value=False,
            group_a_label="nielsen", group_b_label="naive",
            unit_col="scenario_key",
            hold_constant=["has_vignette_text", "explain_units"],
            output="results/comparison_nielsen_effect_vignettes.csv",
        ),
        dict(
            csv_path=VIGNETTE_CSV,
            group_col="include_nielsen", group_a_value=True, group_b_value=False,
            group_a_label="nielsen", group_b_label="naive",
            unit_col="scenario_key",
            hold_constant=["has_vignette_text", "explain_units"],
            filters={"ome": True},
            output="results/comparison_nielsen_effect_vignettes_ome.csv",
        ),
        dict(
            csv_path=VIGNETTE_CSV,
            group_col="include_nielsen", group_a_value=True, group_b_value=False,
            group_a_label="nielsen", group_b_label="naive",
            unit_col="scenario_key",
            hold_constant=["has_vignette_text", "explain_units"],
            filters={"ome": False},
            output="results/comparison_nielsen_effect_vignettes_nonome.csv",
        ),
        dict(
            csv_path=VIGNETTE_CSV,
            group_col="explain_units", group_a_value=True, group_b_value=False,
            group_a_label="explain_units", group_b_label="no_units",
            unit_col="scenario_key",
            hold_constant=["has_vignette_text", "include_nielsen"],
            output="results/comparison_units_effect_vignettes.csv",
        ),
        # ── Random-dose contrasts ───────────────────────────────────────
        dict(
            csv_path=PAIR_CSV,
            group_col="include_nielsen", group_a_value=True, group_b_value=False,
            group_a_label="nielsen", group_b_label="naive",
            unit_col="opioid_pair",
            hold_constant=["include_examples"],
            output="results/comparison_nielsen_effect_random_dose.csv",
        ),
        dict(
            csv_path=PAIR_CSV,
            group_col="include_nielsen", group_a_value=True, group_b_value=False,
            group_a_label="nielsen", group_b_label="naive",
            unit_col="opioid_pair",
            hold_constant=["include_examples"],
            filters={"ome": True},
            output="results/comparison_nielsen_effect_random_dose_ome.csv",
        ),
        dict(
            csv_path=PAIR_CSV,
            group_col="include_nielsen", group_a_value=True, group_b_value=False,
            group_a_label="nielsen", group_b_label="naive",
            unit_col="opioid_pair",
            hold_constant=["include_examples"],
            filters={"ome": False},
            output="results/comparison_nielsen_effect_random_dose_nonome.csv",
        ),
        dict(
            csv_path=EXPLAIN_CSV,
            group_col="explain_units", group_a_value=True, group_b_value=False,
            group_a_label="explain_units", group_b_label="no_units",
            unit_col="opioid_pair",
            output="results/comparison_units_effect_random_dose.csv",
        ),
    ]

    for spec in specs:
        output = spec.pop("output")
        filters = spec.pop("filters", None)
        print(f"── Comparison → {os.path.basename(output)} ──")
        comp = AccuracyComparison(**spec, filters=filters)
        comp.save(output)
    print("✓ Accuracy comparisons complete")


# ── 4. Per-model Wilcoxon tests ────────────────────────────────────────────

def run_wilcoxon():
    """
    Run per-model and pooled Wilcoxon signed-rank tests for every
    comparison CSV that has tight_diff / broad_diff columns.

    Saves results/wilcoxon_*.csv alongside each comparison_*.csv.
    """
    from analysis.per_model_wilcoxon import compute_per_model_wilcoxon

    comparison_pairs = [
        ("results/comparison_text_vs_empty.csv",                    "results/wilcoxon_text_vs_empty.csv"),
        ("results/comparison_nielsen_effect_vignettes.csv",         "results/wilcoxon_nielsen_effect_vignettes.csv"),
        ("results/comparison_nielsen_effect_vignettes_ome.csv",     "results/wilcoxon_nielsen_effect_vignettes_ome.csv"),
        ("results/comparison_nielsen_effect_vignettes_nonome.csv",  "results/wilcoxon_nielsen_effect_vignettes_nonome.csv"),
        ("results/comparison_units_effect_vignettes.csv",           "results/wilcoxon_units_effect_vignettes.csv"),
        ("results/comparison_nielsen_effect_random_dose.csv",       "results/wilcoxon_nielsen_effect_random_dose.csv"),
        ("results/comparison_nielsen_effect_random_dose_ome.csv",   "results/wilcoxon_nielsen_effect_random_dose_ome.csv"),
        ("results/comparison_nielsen_effect_random_dose_nonome.csv","results/wilcoxon_nielsen_effect_random_dose_nonome.csv"),
        ("results/comparison_units_effect_random_dose.csv",         "results/wilcoxon_units_effect_random_dose.csv"),
        ("results/comparison_dosing_unit_effect.csv",               "results/wilcoxon_dosing_unit_effect.csv"),
        ("results/comparison_ome_vs_nonome_naive_permodel.csv",     "results/wilcoxon_ome_vs_nonome_naive_permodel.csv"),
        ("results/comparison_ome_vs_nonome_nielsen_permodel.csv",   "results/wilcoxon_ome_vs_nonome_nielsen_permodel.csv"),
    ]

    for comp_csv, wilc_csv in comparison_pairs:
        if not os.path.exists(comp_csv):
            print(f"  Skipping (missing): {comp_csv}")
            continue
        print(f"── Wilcoxon → {os.path.basename(wilc_csv)} ──")
        compute_per_model_wilcoxon(comp_csv, output_csv=wilc_csv)
    print("✓ Wilcoxon tests complete")


# ── 5. Text effect differences ─────────────────────────────────────────────

def run_text_effects():
    """Save the batch-means difference table for text-vs-empty comparisons."""
    from analysis.text_effect import TextEffectDifference

    print("── Computing text effect differences ──")
    diff = TextEffectDifference("results/model_batch_means_text_vs_empty.csv")
    diff.save("results/text_effect_differences.csv")
    print("✓ Text effects complete")


# ── 6. OME vs non-OME comparisons ──────────────────────────────────────────

def run_ome_comparisons():
    """
    Compare OME vs non-OME accuracy distributions for naive and Nielsen
    conditions, both overall and per-model.
    """
    from analysis.accuracy_comparison import AccuracyComparison

    specs = [
        dict(
            csv_path=PAIR_CSV,
            group_col="ome", group_a_value=True, group_b_value=False,
            group_a_label="ome", group_b_label="non_ome",
            unit_col="opioid_pair",
            filters={"include_nielsen": False, "include_examples": False},
            output="results/comparison_ome_vs_nonome_naive.csv",
        ),
        dict(
            csv_path=PAIR_CSV,
            group_col="ome", group_a_value=True, group_b_value=False,
            group_a_label="ome", group_b_label="non_ome",
            unit_col="opioid_pair",
            filters={"include_nielsen": True, "include_examples": False},
            output="results/comparison_ome_vs_nonome_nielsen.csv",
        ),
    ]
    # Per-model versions (no hold_constant, so each model-unit row is tested directly)
    for spec in specs:
        output = spec.pop("output")
        filters = spec.pop("filters", None)
        print(f"── OME comparison → {os.path.basename(output)} ──")
        comp = AccuracyComparison(**spec, filters=filters)
        comp.save(output)

        # Per-model version
        permodel_output = output.replace(".csv", "_permodel.csv")
        comp2 = AccuracyComparison(**spec, filters=filters)
        comp2.save(permodel_output)
    print("✓ OME comparisons complete")


# ── 7. Dosing-unit effect comparison ──────────────────────────────────────

def run_dosing_comparison():
    """
    Compare mg/day vs mg/4 hours accuracy under Nielsen conditions.
    Uses pair_model_analysis (mg/day) vs explain_units_pair_model_analysis (mg/4 hours).
    """
    import pandas as pd
    from analysis.per_model_wilcoxon import compute_per_model_wilcoxon

    print("── Computing dosing-unit effect ──")
    # Load both datasets filtered to Nielsen, OME, no explain_units
    pair_df = pd.read_csv(PAIR_CSV)
    explain_df = pd.read_csv(EXPLAIN_CSV)

    pair_df = pair_df[
        (pair_df["include_nielsen"] == True) &
        (pair_df["include_examples"] == False) &
        (pair_df["ome"] == True)
    ].copy()
    explain_df = explain_df[explain_df["explain_units"] == False].copy()

    tight_col = "tight(0.85,1.15)"
    broad_col = "broad(0.6,1.7)"

    # Merge on model × opioid_pair
    merged = pair_df[["model", "opioid_pair", tight_col, broad_col]].merge(
        explain_df[["model", "opioid_pair", tight_col, broad_col]],
        on=["model", "opioid_pair"],
        suffixes=("_daily", "_4h"),
    )
    merged["tight_diff"] = merged[f"{tight_col}_4h"] - merged[f"{tight_col}_daily"]
    merged["broad_diff"] = merged[f"{broad_col}_4h"] - merged[f"{broad_col}_daily"]

    output = "results/comparison_dosing_unit_effect.csv"
    merged.to_csv(output, index=False)
    print(f"  → {output}")

    compute_per_model_wilcoxon(output, output_csv="results/wilcoxon_dosing_unit_effect.csv")
    print("✓ Dosing comparison complete")


# ── 8. Cross-experiment matched analysis ──────────────────────────────────

def run_cross_experiment():
    """
    Matched-pair cross-experiment comparison: random dose vs empty vs text vignette.
    Outputs cross_experiment_matched_*.csv.
    """
    from analysis.cross_experiment_matched import CrossExperimentMatched

    print("── Running cross-experiment matched analysis ──")
    cem = CrossExperimentMatched(
        pair_csv=PAIR_CSV,
        vignette_csv=VIGNETTE_CSV,
    )
    cem.save("results")
    print("✓ Cross-experiment analysis complete")


# ── 9. ICC analysis ────────────────────────────────────────────────────────

def run_icc():
    """
    Compute ICC(1,1) per model × scenario from vignette results.
    Output: results/icc_analysis.csv
    """
    from analysis.icc_analysis import ICCAnalysis

    print("── Running ICC analysis ──")
    icc = ICCAnalysis(
        results_csv="results/reworded_vignettes_results.csv",
        filters={"include_nielsen": False, "explain_units": False},
    )
    icc.save("results/icc_analysis.csv")
    print("✓ ICC analysis complete")


# ── 10. Kendall's W ────────────────────────────────────────────────────────

def run_kendall_w():
    """
    Compute Kendall's W across models for each scenario under naive conditions.
    Output: results/kendall_w_vignettes.csv
    """
    from analysis.kendall_w import compute_kendall_w_for_scenarios

    print("── Running Kendall's W ──")
    compute_kendall_w_for_scenarios(
        results_csv="results/reworded_vignettes_results.csv",
        output_csv="results/kendall_w_vignettes.csv",
        filters={"include_nielsen": False},
    )
    print("✓ Kendall's W complete")


# ── 11. Variance–ICC Spearman statistics ──────────────────────────────────

def run_variance_stats():
    """
    Compute per-model and overall Spearman ρ between variance and ICC.
    Output: results/variance_icc_spearman.csv
    """
    from analysis.variance_icc_stats import compute_variance_icc_spearman

    print("── Computing variance–ICC Spearman stats ──")
    compute_variance_icc_spearman(
        icc_csv="results/icc_analysis.csv",
        vignette_model_csv=VIGNETTE_CSV,
        output_csv="results/variance_icc_spearman.csv",
    )
    print("✓ Variance–ICC stats complete")


# ── 12. Grid independence tests ────────────────────────────────────────────

def run_grid_tests():
    """
    Chi-square independence tests for accuracy × variance, variance × CV,
    and variance × ICC grids.
    Output: results/grid_independence_tests.csv
    """
    from analysis.grid_independence_tests import compute_grid_independence_tests

    print("── Running grid independence tests ──")
    compute_grid_independence_tests(
        icc_csv="results/icc_analysis.csv",
        vignette_model_csv=VIGNETTE_CSV,
        output_csv="results/grid_independence_tests.csv",
    )
    print("✓ Grid independence tests complete")


# ── 13. Performance profiles ────────────────────────────────────────────────

def run_profiles():
    """
    Compute performance profiles for random-dose and vignette experiments.
    Outputs: results/performance_profile_random_doses.csv
             results/performance_profile_vignettes.csv
    """
    from analysis.performance_profile import (
        compute_random_dose_profile, compute_vignette_profile,
    )

    print("── Computing random-dose performance profile ──")
    compute_random_dose_profile(
        PAIR_CSV,
        naive_filter={"include_nielsen": False, "include_examples": False},
        output_csv="results/performance_profile_random_doses.csv",
    )

    print("── Computing vignette performance profile ──")
    compute_vignette_profile(
        VIGNETTE_CSV,
        naive_filter={"include_nielsen": False, "explain_units": False,
                      "has_vignette_text": True},
        output_csv="results/performance_profile_vignettes.csv",
    )
    print("✓ Performance profiles complete")


# ── 14. CV–accuracy chi-square tests ──────────────────────────────────────

def run_cv_chi2():
    """
    Per-model chi-square tests for accuracy × CV-bin independence.
    Output: results/cv_accuracy_chi2.csv
    """
    from analysis.variance_icc_stats import compute_cv_accuracy_chi2

    print("── Running CV–accuracy chi-square tests ──")
    compute_cv_accuracy_chi2(
        PAIR_CSV,
        cv_col="observed_coefficient_cv",
        tight_col="tight(0.85,1.15)",
        broad_col="broad(0.6,1.7)",
        cv_bins=[0, 0.1, 0.5, float("inf")],
        cv_bin_labels=["<0.1", "0.1–0.5", "≥0.5"],
        output_csv="results/cv_accuracy_chi2.csv",
        naive_filter={"include_nielsen": False, "include_examples": False},
    )
    print("✓ CV chi-square tests complete")


# ── 15. Model rankings ─────────────────────────────────────────────────────

def run_rankings():
    """
    Rank models by accuracy with CMH tests for each experimental condition.
    Outputs: results/ranking_*.csv
    """
    from analysis.model_ranking import build_all_rankings

    print("── Building model rankings ──")
    build_all_rankings()
    print("✓ Rankings complete")


# ── Entry point ────────────────────────────────────────────────────────────

_SECTION_FNS = {
    "processing":      run_processing,
    "batch_means":     run_batch_means,
    "comparisons":     run_comparisons,
    "wilcoxon":        run_wilcoxon,
    "text_effects":    run_text_effects,
    "ome_comparisons": run_ome_comparisons,
    "dosing":          run_dosing_comparison,
    "cross_experiment":run_cross_experiment,
    "icc":             run_icc,
    "kendall_w":       run_kendall_w,
    "variance":        run_variance_stats,
    "grid":            run_grid_tests,
    "profiles":        run_profiles,
    "cv_chi2":         run_cv_chi2,
    "rankings":        run_rankings,
}

_ALL_ORDER = [
    "processing", "batch_means", "comparisons", "wilcoxon",
    "text_effects", "ome_comparisons", "dosing", "cross_experiment",
    "icc", "kendall_w", "variance", "grid", "profiles", "cv_chi2",
    "rankings",
]

if __name__ == "__main__":
    args = _parse_args()
    if args.section == "all":
        for name in _ALL_ORDER:
            _SECTION_FNS[name]()
    else:
        _SECTION_FNS[args.section]()
