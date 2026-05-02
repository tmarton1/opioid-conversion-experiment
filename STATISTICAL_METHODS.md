# Statistical Methods

All statistical tests performed in this repository, with the input CSVs consumed, output CSVs produced, and the figures or summary documents they contribute to.

For figure-level detail (what each figure renders, which CSVs it reads, and which `figure_stats.json` keys annotate it), see [FIGURES.md](FIGURES.md).

This document is organised two ways:
- [§ Method details](#method-details) — one section per statistical test, with code location, formula, inputs, and outputs.
- [§ By figure](#by-figure) — maps each figure slug to the statistical tests that feed it and the analysis files that produce its data.

---

## Summary table

| # | Method | Purpose | Input CSVs | Output CSVs | Figures / Summaries |
|---|--------|---------|-----------|------------|---------------------|
| 1 | [Cochran–Mantel–Haenszel (CMH)](#1-cochranmantelhænszel-cmh-test) | Rank models by accuracy with stratified pairwise tests | `pair_model_analysis.csv`, `reworded_vignettes_model_analysis.csv`, `explain_units_pair_model_analysis.csv` | `ranking_naive_vignettes.csv`, `ranking_naive_random_doses.csv`, `ranking_random_doses_nielsen.csv`, `ranking_random_doses_nielsen_units.csv`, `ranking_vignettes_nielsen_units.csv` | `naive_vignette_bars`, `naive_random_dose_bars`, `nielsen_ranked_random_doses`, `nielsen_units_ranked` |
| 2 | [Fisher's Exact Test (Bonferroni)](#2-fishers-exact-test) | Compare accuracy distributions between two prompt conditions, per model × unit | `pair_model_analysis.csv`, `reworded_vignettes_model_analysis.csv`, `explain_units_pair_model_analysis.csv` | `comparison_*.csv` (9 files) | `nielsen_effect_combined`, `nielsen_by_ome_combined`, `explain_units_full_treatment`, `cross_experiment_bars` |
| 3 | [Wilcoxon Signed-Rank](#3-wilcoxon-signed-rank-test) | Test per-model condition effect on unit-level accuracy; pooled test | `comparison_*.csv` | `wilcoxon_*.csv` (12 files) | `nielsen_effect_combined`, `nielsen_by_ome_combined`, `explain_units_full_treatment`, `cross_experiment_bars` |
| 4 | [Matched Wilcoxon (cross-experiment)](#4-matched-wilcoxon-cross-experiment) | Test whether accuracy differs between task formats on matched opioid pairs | `pair_model_analysis.csv`, `reworded_vignettes_model_analysis.csv` | `cross_experiment_matched_wilcoxon.csv` | `cross_experiment_bars` |
| 5 | [Spearman ρ (cross-experiment)](#5-spearman-ρ-cross-experiment) | Correlate model rank order across random-dose and vignette experiments | `cross_experiment_matched_means.csv` | `cross_experiment_matched_spearman.csv` | `cross_experiment_bars` |
| 6 | [ICC(1,1) + F-test](#6-icc11--f-test) | Quantify proportion of response variance from vignette wording vs noise | `reworded_vignettes_results.csv` | `icc_analysis.csv` | `variance_stacked`, `icc_bubble_grid`, `spearman_variance_icc`, `kendall_w_summary` |
| 7 | [Pairwise Mann–Whitney U (Bonferroni)](#7-pairwise-mannwhitney-u) | Test whether any two vignette wordings within a scenario produce different responses | `reworded_vignettes_results.csv` | `icc_analysis.csv` (`sig_pairs` column) | `icc_bubble_grid` |
| 8 | [Kendall's W](#8-kendalls-w) | Measure cross-model concordance in how wording rank-orders responses within a scenario | `reworded_vignettes_results.csv` | `kendall_w_vignettes.csv` | `kendall_w_summary` |
| 9 | [Spearman ρ (variance vs ICC)](#9-spearman-ρ-variance-vs-icc) | Correlate scenario total variance with ICC per model | `icc_analysis.csv`, `reworded_vignettes_model_analysis.csv` | `variance_icc_spearman.csv` | `spearman_variance_icc` |
| 10 | [Binomial sign test](#10-binomial-sign-test) | Test whether per-model Spearman ρ values are directionally consistent | `variance_icc_spearman.csv` | `variance_icc_spearman.csv` (`sign_test_p` column) | `spearman_variance_icc` |
| 11 | [Chi-square independence (grids)](#11-chi-square-independence-grids) | Test independence in 3×3 accuracy × variance, variance × CV, and variance × ICC grids | `icc_analysis.csv`, `reworded_vignettes_model_analysis.csv` | `grid_independence_tests.csv` | `icc_bubble_grid`, `cv_strip_vignettes` |
| 12 | [Chi-square (CV × accuracy)](#12-chi-square-cv--accuracy) | Test independence between CV bin and accuracy category per model | `pair_model_analysis.csv` | `cv_accuracy_chi2.csv` | `cv_strip_random_doses` |

---

## Method details

## Method details

### 1. Cochran–Mantel–Haenszel (CMH) test

**Code**: `analysis/model_ranking.py` → `ModelRanking.build_all_rankings()`

**Description**: For each pair of models, a one-sided CMH test is run treating each conversion unit (opioid pair or scenario key) as a stratum. The test evaluates whether model A's accuracy odds are uniformly higher than model B's across all strata. Bonferroni correction is applied across all `k(k−1)` pairwise comparisons. Models with no significant pairwise difference against their nearest neighbour share a rank.

**Applied to**:
- Naive vignette accuracy (`reworded_vignettes_model_analysis.csv`, `include_nielsen=False`)
- Naive random-dose accuracy (`pair_model_analysis.csv`, `include_nielsen=False`)
- Random-dose accuracy after Nielsen (`pair_model_analysis.csv`, `include_nielsen=True`)
- Random-dose accuracy after Nielsen + units (`explain_units_pair_model_analysis.csv`)
- Vignette accuracy after Nielsen + units (`reworded_vignettes_model_analysis.csv`, `include_nielsen=True`, `explain_units=True`)

**Output columns** (in `ranking_*.csv`): `model`, `rank`, `tight_rank`, `broad_rank`

**Pairwise detail** (in `ranking_*_pairs.csv`): `model_a`, `model_b`, `n_strata`, `tight_stat`, `tight_p`, `tight_p_bonf`, `broad_stat`, `broad_p`, `broad_p_bonf`

---

### 2. Fisher's Exact Test

**Code**: `analysis/accuracy_comparison.py` → `AccuracyComparison.run()`

**Description**: For each (model, unit) row in a condition-comparison CSV, a 2×2 Fisher exact test compares the count of trials in the tight/broad range between group A and group B (e.g. `include_nielsen=True` vs `False`). Bonferroni correction is applied within each comparison family (one family per unique unit × model combination). The direction of any significant difference is stored.

**Applied to** (9 comparisons → 9 `comparison_*.csv` files):

| Output file | Contrast | Unit col | Hold constant |
|---|---|---|---|
| `comparison_text_vs_empty.csv` | `has_vignette_text` True vs False | `scenario_key` | `include_nielsen`, `explain_units` |
| `comparison_nielsen_effect_vignettes.csv` | `include_nielsen` True vs False | `scenario_key` | `has_vignette_text`, `explain_units` |
| `comparison_nielsen_effect_vignettes_ome.csv` | Same, OME pairs only | `scenario_key` | same |
| `comparison_nielsen_effect_vignettes_nonome.csv` | Same, non-OME pairs only | `scenario_key` | same |
| `comparison_units_effect_vignettes.csv` | `explain_units` True vs False | `scenario_key` | `has_vignette_text`, `include_nielsen` |
| `comparison_nielsen_effect_random_dose.csv` | `include_nielsen` True vs False | `opioid_pair` | `include_examples` |
| `comparison_nielsen_effect_random_dose_ome.csv` | Same, OME pairs only | `opioid_pair` | same |
| `comparison_nielsen_effect_random_dose_nonome.csv` | Same, non-OME pairs only | `opioid_pair` | same |
| `comparison_units_effect_random_dose.csv` | `explain_units` True vs False | `opioid_pair` | — |

**Output columns**: `model`, `unit`, `group_a_tight`, `group_b_tight`, `tight_p`, `tight_p_bonf`, `tight_result`, `group_a_broad`, `group_b_broad`, `broad_p`, `broad_p_bonf`, `broad_result`

---

### 3. Wilcoxon Signed-Rank Test

**Code**: `analysis/per_model_wilcoxon.py` → `compute_per_model_wilcoxon()`

**Description**: For each model, treats units (opioid pairs or scenario keys) as paired observations and applies a Wilcoxon signed-rank test to the vector of per-unit accuracy differences (group A − group B) in `tight_diff` and `broad_diff` columns. A pooled test across all models × units is also computed. No additional correction is applied beyond what was done in the upstream `AccuracyComparison`.

**Input**: each `comparison_*.csv` file (which contains `tight_diff` and `broad_diff` columns)

**Output** (12 `wilcoxon_*.csv` files, one per comparison): `model`, `tight_stat`, `tight_p`, `tight_n`, `broad_stat`, `broad_p`, `broad_n`; row `POOLED` aggregates across all models.

---

### 4. Matched Wilcoxon (cross-experiment)

**Code**: `analysis/cross_experiment_matched.py` → `CrossExperimentMatched.run()`

**Description**: Identifies opioid pairs present in both the random-dose and vignette experiments. For each model, a Wilcoxon signed-rank test compares accuracy on these matched pairs across experiment types (random-dose vs empty vignette, empty vs text vignette, random-dose vs text vignette).

**Output**: `cross_experiment_matched_wilcoxon.csv` — columns: `model`, `contrast`, `tight_stat`, `tight_p`, `broad_stat`, `broad_p`

---

### 5. Spearman ρ (cross-experiment)

**Code**: `analysis/cross_experiment_matched.py` → `CrossExperimentMatched.run_spearman()`

**Description**: Spearman rank correlation between each model's mean accuracy in the random-dose experiment and its mean accuracy in the vignette experiment, computed on the matched-pair subset. Quantifies whether model ranking is consistent across task formats.

**Output**: `cross_experiment_matched_spearman.csv` — columns: `condition`, `rho`, `p_value`, `n_models`

---

### 6. ICC(1,1) + F-test

**Code**: `analysis/icc_analysis.py` → `ICCAnalysis.run()`

**Description**: For each (model × scenario) cell under naive prompting, a one-way random-effects ANOVA decomposes the total log(observed/expected) variance into:
- **Between-vignette** (wording effect): variance across the `k` wordings of a scenario
- **Within-vignette** (stochastic noise): variance across the `r` repetitions within each wording

ICC(1,1) = MS_between − MS_within / (MS_between + (k−1)·MS_within), where k = number of vignette wordings per scenario.

An F-test (H₀: σ²_between = 0) is included. Degrees of freedom: df_between = k−1, df_within = k(r−1).

**Output columns** (in `icc_analysis.csv`): `model`, `scenario_key`, `icc`, `f_stat`, `f_pvalue`, `k`, `r`, `ms_between`, `ms_within`, `var_between`, `var_within`, `sig_pairs`, `total_pairs`

---

### 7. Pairwise Mann–Whitney U

**Code**: `analysis/icc_analysis.py` → `ICCAnalysis.run()` (called inside the same loop as ICC)

**Description**: Within each (model × scenario) cell, all C(k, 2) pairs of vignette wordings are compared using a two-sided Mann–Whitney U test on the log-ratio distributions. Bonferroni correction is applied within each cell. The count of significant pairs (`sig_pairs`) and total pairs (`total_pairs`) are stored in `icc_analysis.csv` alongside the ICC value.

---

### 8. Kendall's W

**Code**: `analysis/kendall_w.py` → `compute_kendall_w_for_scenarios()`

**Description**: For each scenario, models act as raters and vignette wordings act as items. Kendall's W (coefficient of concordance) measures cross-model agreement on which wording produces higher vs lower log(observed/expected) ratios. W = 0 means models disagree completely; W = 1 means all models rank wordings identically. A χ² test is used for significance.

**Formula**: W = 12S / (m²(n³ − n)), where S is the sum of squared rank-sum deviations, m = number of models, n = number of vignette wordings.

**Output columns** (in `kendall_w_vignettes.csv`): `scenario_key`, `kendall_w`, `chi2`, `p_value`, `n_models`, `n_wordings`

---

### 9. Spearman ρ (variance vs ICC)

**Code**: `analysis/variance_icc_stats.py` → `compute_variance_icc_spearman()`

**Description**: For each model, Spearman rank correlation between scenario-level total log-ratio variance (from `reworded_vignettes_model_analysis.csv`) and ICC (from `icc_analysis.csv`). An overall Spearman ρ pooling all models × scenarios is also computed.

**Output** (in `variance_icc_spearman.csv`): `model`, `stat_type`, `rho`, `p_value`, `n`; row `OVERALL` contains the pooled estimate.

---

### 10. Binomial sign test

**Code**: `analysis/variance_icc_stats.py` → `compute_variance_icc_spearman()` (computed alongside §9)

**Description**: A binomial sign test on the direction of per-model Spearman ρ values (H₀: half positive, half negative). Tests whether most models show the same direction of variance–ICC association. Stored in `variance_icc_spearman.csv` alongside the Spearman results.

---

### 11. Chi-square independence (grids)

**Code**: `analysis/grid_independence_tests.py` → `compute_grid_independence_tests()`

**Description**: Chi-square tests of independence in three 3×3 contingency grids, constructed by binning across all model × scenario cells:
1. **Accuracy × total variance** — does scenario accuracy tier correlate with total variance tier?
2. **Variance × CV** — does total variance tier correlate with CV tier?
3. **Variance × ICC** — does total variance tier correlate with ICC tier?

Bins are tertiles. Cells with expected count < 5 are noted in the output.

**Output** (in `grid_independence_tests.csv`): `grid_name`, `chi2`, `p_value`, `df`, `n_cells_expected_lt5`

---

### 12. Chi-square (CV × accuracy)

**Code**: `analysis/variance_icc_stats.py` → `compute_cv_accuracy_chi2()`

**Description**: Per model, a chi-square test of independence between CV bin (<0.1, 0.1–0.5, ≥0.5) and accuracy category (tight / broad-only / incorrect) under naive conditions. CV is computed from the observed_coefficient_cv column in `pair_model_analysis.csv`.

**Output** (in `cv_accuracy_chi2.csv`): `model`, `chi2`, `p_value`, `df`, `n`

---

## Correction for multiple comparisons

| Context | Procedure |
|---|---|
| CMH pairwise model rankings | Bonferroni across all k(k−1) model pairs within a condition |
| Fisher exact per-unit comparisons | Bonferroni within each comparison family (one family per comparison CSV) |
| Mann–Whitney U within ICC cells | Bonferroni within each model × scenario cell |
| All others (Wilcoxon, Kendall's W, Spearman, chi-square) | No additional correction; tests are pre-specified |

---

---

## By figure

Each figure slug links to its full specification in [FIGURES.md](FIGURES.md).

### [`naive_vignette_bars`](FIGURES.md#1-naive_vignette_bars)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Model ranking | §1 CMH | `analysis/model_ranking.py` → `ModelRanking.build_all_rankings()` | `ranking_naive_vignettes.csv` |

### [`cv_strip_vignettes`](FIGURES.md#2-cv_strip_vignettes)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| CV × accuracy independence | §11 Chi-square | `analysis/variance_icc_stats.py` → `compute_cv_accuracy_chi2()` | `cv_accuracy_chi2.csv` |

> Source data: `reworded_vignettes_model_analysis.csv` (filtered naive)

### [`vignette_performance_profile`](FIGURES.md#3-vignette_performance_profile)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Performance profile aggregation | — | `analysis/performance_profile.py` → `compute_vignette_profile()` | `performance_profile_vignettes.csv` |
| CV × accuracy independence | §11 Chi-square | `analysis/variance_icc_stats.py` → `compute_cv_accuracy_chi2()` | `cv_accuracy_chi2.csv` |

### [`variance_stacked`](FIGURES.md#4-variance_stacked)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Variance decomposition | §6 ICC(1,1) | `analysis/icc_analysis.py` → `ICCAnalysis.run()` | `icc_analysis.csv` |
| Variance–ICC correlation | §9 Spearman ρ | `analysis/variance_icc_stats.py` → `compute_variance_icc_spearman()` | `variance_icc_spearman.csv` |

### [`spearman_variance_icc`](FIGURES.md#5-spearman_variance_icc)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Per-model Spearman ρ | §9 Spearman ρ | `analysis/variance_icc_stats.py` → `compute_variance_icc_spearman()` | `variance_icc_spearman.csv` |
| Directional consistency | §10 Binomial sign test | same call | `variance_icc_spearman.csv` (`sign_test_p` column) |

### [`icc_bubble_grid`](FIGURES.md#6-icc_bubble_grid)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Per-cell ICC | §6 ICC(1,1) + F-test | `analysis/icc_analysis.py` → `ICCAnalysis.run()` | `icc_analysis.csv` |
| Pairwise wording comparisons | §7 Mann–Whitney U | same call | `icc_analysis.csv` (`sig_pairs` column) |
| Grid independence | §11 Chi-square grids | `analysis/grid_independence_tests.py` → `compute_grid_independence_tests()` | `grid_independence_tests.csv` |

### [`kendall_w_summary`](FIGURES.md#7-kendall_w_summary)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Cross-model wording concordance | §8 Kendall's W | `analysis/kendall_w.py` → `compute_kendall_w_for_scenarios()` | `kendall_w_vignettes.csv` |
| Per-cell ICC | §6 ICC(1,1) | `analysis/icc_analysis.py` → `ICCAnalysis.run()` | `icc_analysis.csv` |

### [`naive_random_dose_bars`](FIGURES.md#8-naive_random_dose_bars)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Model ranking | §1 CMH | `analysis/model_ranking.py` → `ModelRanking.build_all_rankings()` | `ranking_naive_random_doses.csv` |

### [`cv_strip_random_doses`](FIGURES.md#9-cv_strip_random_doses)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Performance profile aggregation | — | `analysis/performance_profile.py` → `compute_random_dose_profile()` | `performance_profile_random_doses.csv` |
| CV × accuracy independence | §12 Chi-square CV | `analysis/variance_icc_stats.py` → `compute_cv_accuracy_chi2()` | `cv_accuracy_chi2.csv` |

> Slope significance for dose-dependent bias: computed inline in the visualizer from `pair_model_analysis.csv`.

### [`cross_experiment_bars`](FIGURES.md#10-cross_experiment_bars)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Matched-pair Wilcoxon | §4 Matched Wilcoxon | `analysis/cross_experiment_matched.py` → `CrossExperimentMatched.run()` | `cross_experiment_matched_wilcoxon.csv` |
| Cross-experiment rank correlation | §5 Spearman ρ | `analysis/cross_experiment_matched.py` → `CrossExperimentMatched.run_spearman()` | `cross_experiment_matched_spearman.csv` |

### [`nielsen_effect_combined`](FIGURES.md#11-nielsen_effect_combined)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Condition comparison | §2 Fisher's exact | `analysis/accuracy_comparison.py` → `AccuracyComparison.run()` | `comparison_nielsen_effect_*.csv` |
| Per-model improvement test | §3 Wilcoxon | `analysis/per_model_wilcoxon.py` → `compute_per_model_wilcoxon()` | `wilcoxon_nielsen_effect_vignettes.csv`, `wilcoxon_nielsen_effect_random_dose.csv` |

### [`nielsen_ranked_random_doses`](FIGURES.md#12-nielsen_ranked_random_doses)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Model ranking after Nielsen | §1 CMH | `analysis/model_ranking.py` → `ModelRanking.build_all_rankings()` | `ranking_random_doses_nielsen.csv` |

### [`nielsen_by_ome_combined`](FIGURES.md#13-nielsen_by_ome_combined)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| OME/non-OME Fisher exact | §2 Fisher's exact | `analysis/accuracy_comparison.py` → `AccuracyComparison.run()` | `comparison_nielsen_effect_*_ome.csv`, `*_nonome.csv` |
| OME/non-OME Wilcoxon | §3 Wilcoxon | `analysis/per_model_wilcoxon.py` → `compute_per_model_wilcoxon()` | `wilcoxon_nielsen_effect_*_ome.csv`, `*_nonome.csv` |

### [`explain_units_full_treatment`](FIGURES.md#14-explain_units_full_treatment)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Dosing-unit Fisher exact | §2 Fisher's exact | `analysis/accuracy_comparison.py` → `AccuracyComparison.run()` | `comparison_dosing_unit_effect.csv`, `comparison_units_effect_random_dose.csv` |
| Dosing-unit Wilcoxon | §3 Wilcoxon | `analysis/per_model_wilcoxon.py` → `compute_per_model_wilcoxon()` | `wilcoxon_dosing_unit_effect.csv`, `wilcoxon_units_effect_random_dose.csv` |

### [`nielsen_units_ranked`](FIGURES.md#15-nielsen_units_ranked)

| Step | Method (§) | Code file → method | Output CSV |
|------|-----------|-------------------|------------|
| Model ranking after Nielsen + units | §1 CMH | `analysis/model_ranking.py` → `ModelRanking.build_all_rankings()` | `ranking_random_doses_nielsen_units.csv`, `ranking_vignettes_nielsen_units.csv` |

---

## Output reference

| CSV | Produced by | Used by |
|---|---|---|
| `ranking_naive_vignettes.csv` | `model_ranking.py` | `naive_vignette_bars` |
| `ranking_naive_random_doses.csv` | `model_ranking.py` | `naive_random_dose_bars` |
| `ranking_random_doses_nielsen.csv` | `model_ranking.py` | `nielsen_ranked_random_doses` |
| `ranking_random_doses_nielsen_units.csv` | `model_ranking.py` | `nielsen_units_ranked` |
| `ranking_vignettes_nielsen_units.csv` | `model_ranking.py` | `nielsen_units_ranked` |
| `comparison_*.csv` | `accuracy_comparison.py` | figure significance brackets |
| `wilcoxon_*.csv` | `per_model_wilcoxon.py` | figure Wilcoxon stars |
| `cross_experiment_matched_wilcoxon.csv` | `cross_experiment_matched.py` | `cross_experiment_bars` |
| `cross_experiment_matched_spearman.csv` | `cross_experiment_matched.py` | `cross_experiment_bars` |
| `icc_analysis.csv` | `icc_analysis.py` | `variance_stacked`, `icc_bubble_grid`, `spearman_variance_icc`, `kendall_w_summary` |
| `kendall_w_vignettes.csv` | `kendall_w.py` | `kendall_w_summary` |
| `variance_icc_spearman.csv` | `variance_icc_stats.py` | `spearman_variance_icc` |
| `grid_independence_tests.csv` | `grid_independence_tests.py` | `icc_bubble_grid`, `cv_strip_vignettes` |
| `cv_accuracy_chi2.csv` | `variance_icc_stats.py` | `cv_strip_random_doses` |
| `performance_profile_random_doses.csv` | `performance_profile.py` | `cv_strip_random_doses`, `vignette_performance_profile` |
| `performance_profile_vignettes.csv` | `performance_profile.py` | `vignette_performance_profile` |
| `summary_w_figures/figure_stats.json` | `orchestration/extract_figure_stats.py` | `all_comprehensive_figures.pdf` annotations |
