# Figure Reference

All 15 figures produced by `visualize/comprehensive_figures.py` →
`ComprehensiveFigureBuilder`.  Each entry covers:
- **Slug** — the identifier used in `FIGURE_METADATA` and in filenames under `figures/`
- **Build method** — the `ComprehensiveFigureBuilder` method that constructs it
- **CSVs consumed** — every results CSV the method reads
- **Annotation fields** — keys from `summary_w_figures/figure_stats.json` interpolated
  into the PDF annotation for this figure
- **What the code does** — the rendering logic

See [STATISTICAL_METHODS.md](STATISTICAL_METHODS.md) for the statistical tests
that produce the underlying CSVs, and for a cross-reference by figure.

---

## Table of contents

| # | Slug | Title |
|---|------|-------|
| 1 | [`naive_vignette_bars`](#1-naive_vignette_bars) | LLMs are dangerously inaccurate at opioid dose conversion |
| 2 | [`cv_strip_vignettes`](#2-cv_strip_vignettes) | Models are precise: they give the same wrong answer consistently |
| 3 | [`vignette_performance_profile`](#3-vignette_performance_profile) | Most errors are precise: same wrong answer every time |
| 4 | [`variance_stacked`](#4-variance_stacked) | Most response variance comes from prompt wording, not stochastic noise |
| 5 | [`spearman_variance_icc`](#5-spearman_variance_icc) | Wording-driven variance tracks ICC across most models |
| 6 | [`icc_bubble_grid`](#6-icc_bubble_grid) | Scenario-level decomposition: wording vs stochastic variance |
| 7 | [`kendall_w_summary`](#7-kendall_w_summary) | Wording effect sizes and directions are model-specific |
| 8 | [`naive_random_dose_bars`](#8-naive_random_dose_bars) | The same accuracy pattern holds on controlled random-dose prompts |
| 9 | [`cv_strip_random_doses`](#9-cv_strip_random_doses) | Precision replicates; dose-dependent bias emerges |
| 10 | [`cross_experiment_bars`](#10-cross_experiment_bars) | Performance structure is consistent across prompt formats |
| 11 | [`nielsen_effect_combined`](#11-nielsen_effect_combined) | Adding a reference table improves accuracy on both task formats |
| 12 | [`nielsen_ranked_random_doses`](#12-nielsen_ranked_random_doses) | Model ranking after adding the Nielsen reference table |
| 13 | [`nielsen_by_ome_combined`](#13-nielsen_by_ome_combined) | Nielsen improvement concentrates on oral morphine conversions |
| 14 | [`explain_units_full_treatment`](#14-explain_units_full_treatment) | Non-daily dosing confuses models; unit explanation rescues accuracy |
| 15 | [`nielsen_units_ranked`](#15-nielsen_units_ranked) | Best achievable accuracy after all prompt augmentations |

---

## Figure details

### 1. `naive_vignette_bars`

**Title**: LLMs are dangerously inaccurate at opioid dose conversion

**Build method**: `ComprehensiveFigureBuilder.build_naive_vignette_bars()`
→ calls standalone `naive_performance_bars(ranking_csv=…)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/ranking_naive_vignettes.csv` | Pre-computed CMH model ranking; supplies model order, tight/broad accuracy per model, and pairwise significance brackets |

**What the code does**

Reads the CMH ranking CSV and draws a horizontal bar chart, one row per model.
Each row has two bars sharing the same left edge: a dark (opaque) bar for
tight-range accuracy and a translucent extension to broad-range accuracy.
Models are ordered from best (top) to worst (bottom) by CMH rank.  Significance
brackets between adjacent-ranked models are drawn from the pairwise `tight_p_bonf`
column; stars or "n.s." are placed at the right edge of the longer bar.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_vignette_scenarios` | Total unique vignette scenarios (denominator of accuracy) |
| `naive_vig_best_tight_pct` | Best model's tight-range accuracy (%) |
| `naive_vig_worst_tight_pct` | Worst model's tight-range accuracy (%) |
| `naive_vig_mean_tight_pct` | Mean tight accuracy across all models (%) |
| `naive_vig_best_broad_pct` | Best model's broad-range accuracy (%) |
| `tight_lower_x`, `tight_upper_x` | Tight accuracy range bounds |
| `broad_lower_x`, `broad_upper_x` | Broad accuracy range bounds |

---

### 2. `cv_strip_vignettes`

**Title**: Models are precise: they give the same wrong answer consistently

**Build method**: `ComprehensiveFigureBuilder.build_cv_strip_vignettes()`
→ calls `cv_strip_with_ratio(model_csv=…)` and, if absent, regenerates
`cv_accuracy_chi2.csv` via `analysis.variance_icc_stats.compute_cv_accuracy_chi2()`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/reworded_vignettes_model_analysis.csv` | Per-scenario-key aggregated vignette results; filtered to naive (no Nielsen, no explain_units, has_vignette_text=True) |
| `results/cv_accuracy_chi2.csv` | Chi-square p-values per model for CV × accuracy independence; displayed as annotations in the bottom panel |

**What the code does**

Three-panel figure.

*Top strip chart*: each dot is one (model, scenario_key) cell; x-axis is CV of
observed/expected log-ratios across the `r` repetitions within that scenario.
Dots are colour-coded by model.  Most cluster near zero — responses are
deterministic.

*Middle scatter*: x = CV, y = mean observed/expected ratio (log scale).  Four
dashed horizontal lines mark the tight and broad accuracy bounds.  Shows the
joint distribution of precision and calibration.

*Bottom circle grid (mosaic)*: rows are models, columns are CV bins (CV=0,
0<CV≤0.2, CV>0.2), columns within each model cell are accuracy categories
(tight, broad-only, incorrect).  Circle area is proportional to the fraction of
scenarios falling in that cell.  Chi-square p-values from `cv_accuracy_chi2.csv`
are annotated per model.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_vignette_scenarios` | Total scenarios per model |
| `n_sig_cv_chi2_vignettes` | Number of models with significant CV × accuracy chi-square |
| `n_models` | Total model count |

---

### 3. `vignette_performance_profile`

**Title**: Most errors are precise: same wrong answer every time

**Build method**: `ComprehensiveFigureBuilder.build_vignette_performance_profile()`
→ calls `vignette_performance_profile(model_csv=…, profile_csv=…)` for the top
panel and `_cv_accuracy_mosaic(…)` for the bottom panel; regenerates
`performance_profile_vignettes.csv` if absent.

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/reworded_vignettes_model_analysis.csv` | Source data (filtered to naive) |
| `results/performance_profile_vignettes.csv` | Pre-aggregated profile: per-model proportions of Correct, Precise-but-wrong, Imprecise |
| `results/cv_accuracy_chi2.csv` | Chi-square p-values for mosaic annotation |

**What the code does**

*Top panel (performance profile)*: horizontal stacked bar per model.  Each bar is
divided into three categories computed per (model, scenario) response set:
- **Correct** — all repetitions within tight range
- **Precise but incorrect** — CV = 0 (deterministic) but outside tight range
- **Imprecise** — CV > 0 and not tight

Bars are sorted by overall vignette rank.

*Bottom panel (mosaic)*: same circle-area grid as `cv_strip_vignettes` but
coloured to highlight the Precise-incorrect mass.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `vig_mean_correct_pct` | Mean correct proportion across models (%) |
| `vig_mean_precise_wrong_pct` | Mean precise-but-wrong proportion (%) |
| `vig_mean_imprecise_pct` | Mean imprecise proportion (%) |
| `n_vignette_scenarios` | Observations per model |

---

### 4. `variance_stacked`

**Title**: Most response variance comes from prompt wording, not stochastic noise

**Build method**: `ComprehensiveFigureBuilder.build_variance_stacked()`
→ calls `visualize.icc_visualizer.variance_stacked_figure(icc_csv=…, vignette_model_csv=…, spearman_csv=…)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/icc_analysis.csv` | Per-(model, scenario) ICC(1,1) decomposition: `var_between`, `var_within`, `var_total`, `icc` |
| `results/reworded_vignettes_model_analysis.csv` | Provides naive tight accuracy per model for sorting and annotation |
| `results/variance_icc_spearman.csv` | Per-model Spearman ρ shown in an inset summary |

**What the code does**

*Left stacked bars*: one horizontal bar per model (sorted best→worst vignette
accuracy).  Each bar is divided into `var_between / var_total` (between-wording
variance; light colour) and `var_within / var_total` (within-wording noise; dark
colour), averaged across the model's 20 scenarios.  The tight accuracy for each
model is annotated at the right end.

*Right scatter*: one dot per (model, scenario) cell (260 total); x = total
log-ratio variance, y = ICC.  Colour = model.  Illustrates the strong positive
correlation between total variance and ICC — scenarios where models vary a lot
do so primarily because of wording, not noise.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_model_scenario_cells` | Total model × scenario cells (= rows in icc_analysis.csv restricted to CONSISTENT_MODELS) |
| `mean_icc` | Mean ICC across all cells |
| `median_icc` | Median ICC |
| `mean_between_var_pct` | Mean fraction of variance attributable to wording (%) |
| `n_models_icc_gt_half` | Models with mean ICC > 0.5 |
| `n_models` | Total model count |

---

### 5. `spearman_variance_icc`

**Title**: Wording-driven variance tracks ICC across most models

**Build method**: `ComprehensiveFigureBuilder.build_spearman_variance_icc()`
→ calls `visualize.icc_visualizer.spearman_bars_figure(icc_csv=…, vignette_model_csv=…, spearman_csv=…)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/variance_icc_spearman.csv` | Per-model Spearman ρ between scenario variance and ICC; includes OVERALL row and `n_individually_sig` column |
| `results/icc_analysis.csv` | Used to recover scenario counts |
| `results/reworded_vignettes_model_analysis.csv` | Naive accuracy for model sort order |

**What the code does**

Horizontal bar chart, one row per model, sorted by vignette accuracy.  Bar
length = Spearman ρ (variance vs ICC) for that model's 20 scenarios.  Bars to
the right of zero indicate models where scenarios with more total variance also
have a larger wording-attributable fraction (positive ICC–variance coupling).
Individual bars are coloured by significance (p < 0.05).  A summary line below
shows the overall pooled ρ, overall p-value, and binomial sign-test result.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_scenarios_per_model` | Scenarios per model (= n in each Spearman test) |
| `spearman_rho_variance_icc` | Overall pooled Spearman ρ |
| `spearman_p_variance_icc` | Overall pooled p-value |
| `n_positive_rho_models` | Models with ρ > 0 |
| `n_sig_rho_models` | Models with individually significant ρ (p < 0.05) |
| `n_total_rho_models` | Total models |

---

### 6. `icc_bubble_grid`

**Title**: Scenario-level decomposition: wording vs stochastic variance

**Build method**: `ComprehensiveFigureBuilder.build_icc_bubble_grid()`
→ calls `visualize.icc_visualizer.bubble_grid_figure(icc_csv=…, vignette_model_csv=…)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/icc_analysis.csv` | `var_between`, `var_total`, `icc`, `sig_pairs`, `total_pairs` per cell |
| `results/reworded_vignettes_model_analysis.csv` | Naive accuracy for model sort order |

**What the code does**

Grid with models on the y-axis and scenarios on the x-axis (13 × 20 = 260
cells).  In each cell, two concentric circles are drawn:
- **Outer** (faint) — area ∝ total log-ratio variance (`var_total`)
- **Inner** (bold) — area ∝ between-wording variance (`var_between`)

When the inner circle is large relative to the outer, most variance is
wording-driven (high ICC).  When the circles are nearly equal, wording explains
almost everything; when the inner is tiny, responses are deterministically
uniform across all wordings.

Each cell is also annotated with `sig_pairs / total_pairs` — the count of
wording pairs with a significant Bonferroni-corrected Mann–Whitney U test.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `icc_max_pairs` | Max pairwise comparisons per cell (C(k,2) = 6 for k=4 wordings) |
| `icc_total_sig_pairs` | Total significant pairwise comparisons across all cells |
| `icc_total_pairs` | Total pairwise comparisons across all cells |
| `n_models_with_sig_icc` | Models with at least one significant F-test |
| `n_model_scenario_cells` | Total cells |

---

### 7. `kendall_w_summary`

**Title**: Wording effect sizes and directions are model-specific

**Build method**: `ComprehensiveFigureBuilder.build_kendall_w_summary()`
→ calls `visualize.icc_visualizer.kendall_w_summary(kendall_csv=…, icc_csv=…)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/kendall_w_vignettes.csv` | Per-scenario Kendall's W and p-value |
| `results/icc_analysis.csv` | Per-model mean ICC for sorting and bar comparison |

**What the code does**

Paired bar chart, one group per scenario (sorted by mean ICC across models).
- **Blue bars**: mean ICC across all models for that scenario
- **Red bars**: Kendall's W for that scenario

The juxtaposition illustrates the diagnostic: high ICC + low W means that
wording strongly affects individual model responses, but different models are
sensitive to different aspects of the phrasing — there is no consistent
cross-model direction of the wording effect.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `mean_kendall_w` | Mean Kendall's W across all scenarios |
| `median_kendall_w` | Median Kendall's W |
| `n_sig_kendall_w` | Scenarios with significant W (p < 0.05) |
| `mean_icc` | Mean ICC for comparison |
| `n_vignette_scenarios` | Total scenarios |

---

### 8. `naive_random_dose_bars`

**Title**: The same accuracy pattern holds on controlled random-dose prompts

**Build method**: `ComprehensiveFigureBuilder.build_naive_random_dose_bars()`
→ calls `naive_performance_bars(ranking_csv=…)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/ranking_naive_random_doses.csv` | CMH ranking for random-dose naive condition; tight/broad accuracy and pairwise significance |

**What the code does**

Identical layout to `naive_vignette_bars` (§1) but using random-dose prompts:
one randomly-sampled numeric dose per opioid pair × model × repetition.
Renders a horizontal bar chart sorted by CMH rank with tight + broad
bars and significance brackets.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_opioid_pairs` | Total unique opioid pairs in the naive random-dose condition |
| `naive_rd_best_tight_pct` | Best model's tight accuracy (%) |
| `naive_rd_worst_tight_pct` | Worst model's tight accuracy (%) |
| `naive_rd_mean_tight_pct` | Mean tight accuracy across models (%) |
| `naive_rd_best_broad_pct` | Best model's broad accuracy (%) |
| `tight_lower_x`, `tight_upper_x` | Tight accuracy range bounds |
| `broad_lower_x`, `broad_upper_x` | Broad accuracy range bounds |

---

### 9. `cv_strip_random_doses`

**Title**: Precision replicates; dose-dependent bias emerges

**Build method**: `ComprehensiveFigureBuilder.build_cv_strip_random_doses()`
→ custom three-panel assembly using `cv_strip_chart()`, a scatter panel
assembled inline, and `error_profile_bars()`.

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/pair_model_analysis.csv` | Per-(model, opioid_pair) aggregated results: `observed_coefficient_cv`, `observed_coefficient_mean`, `expected_coefficient`; filtered to naive |
| `results/performance_profile_random_doses.csv` | Pre-aggregated error profile: Correct, Precise-wrong, Other-wrong, Dose-dep-neg, Dose-dep-pos per model |

**What the code does**

*Top strip chart*: each dot = (model, opioid_pair).  x = CV of responses across
repetitions.  Dots with a statistically significant *negative* OLS slope of
response vs dose are outlined in black (dose-dependent underdosing); significant
*positive* slope dots are outlined in grey.

*Middle scatter*: x = CV, y = mean observed/expected coefficient ratio (log
scale) with accuracy threshold lines.  Shows joint distribution of determinism
and calibration, parallel to the vignette figure.

*Bottom stacked bars*: per-model performance profile with five categories:
Correct, Precise-wrong, Other-wrong, Dose-dep-neg, Dose-dep-pos.  The
Dose-dep categories are specific to random doses (vignettes collapse into three
categories).

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_opioid_pairs` | Total unique pairs |
| `rd_mean_correct_pct` | Mean correct proportion (%) |
| `rd_mean_precise_wrong_pct` | Mean precise-but-wrong proportion (%) |
| `rd_mean_dose_dep_neg_pct` | Mean dose-dependent-negative proportion (%) |
| `n_sig_cv_chi2_rd` | Models with significant CV × accuracy chi-square |
| `n_models` | Total model count |

---

### 10. `cross_experiment_bars`

**Title**: Performance structure is consistent across prompt formats

**Build method**: `ComprehensiveFigureBuilder.build_cross_experiment()`
→ calls `cross_experiment_bars(matched_means_csv=…, matched_wilcoxon_csv=…)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/cross_experiment_matched_means.csv` | Per-model mean tight/broad accuracy on the 11 matched pairs in each of the three conditions (random-dose, empty vignette, text vignette) |
| `results/cross_experiment_matched_wilcoxon.csv` | Per-model and POOLED Wilcoxon p-values for the three pairwise contrasts (RD vs text, RD vs empty, empty vs text) |

**What the code does**

Grouped bar chart.  Each model has three groups of bars (random-dose, empty
vignette, text vignette), restricted to the 11 opioid pairs that appear in
both experiments.  Accuracy bars show both tight (dark) and broad (light
extension).  Three significance bracket sets are drawn per model using
`cross_experiment_matched_wilcoxon.csv`.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_matched_pairs` | Opioid pairs present in both experiments |
| `cross_pooled_broad_p_rd_text` | POOLED Wilcoxon broad p: random-dose vs text vignette |
| `cross_pooled_broad_p_rd_empty` | POOLED Wilcoxon broad p: random-dose vs empty vignette |
| `cross_pooled_broad_p_empty_text` | POOLED Wilcoxon broad p: empty vs text vignette |
| `cross_spearman_rho_rd_text_broad` | Mean model-level Spearman ρ: RD vs text (broad range) |
| `cross_n_sig_spearman_rd_text` | Models with significant cross-experiment Spearman ρ |

---

### 11. `nielsen_effect_combined`

**Title**: Adding a reference table improves accuracy on both task formats

**Build method**: `ComprehensiveFigureBuilder.build_nielsen_effect_combined()`
→ calls `nielsen_effect_bars()` twice (vignettes then random doses) and stacks
with `column()`.

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/model_batch_means_vignettes.csv` | Naive (`all_n0`) and Nielsen (`all_n1`) mean tight/broad accuracy per model — vignettes |
| `results/comparison_nielsen_effect_vignettes.csv` | Per-(model, scenario) Fisher exact results for Nielsen contrast — used by `per_model_wilcoxon` lookup |
| `results/wilcoxon_nielsen_effect_vignettes.csv` | Pre-computed Wilcoxon per-model and POOLED results — vignettes |
| `results/model_batch_means.csv` | Naive and Nielsen mean accuracy — random doses |
| `results/comparison_nielsen_effect_random_dose.csv` | Per-(model, pair) Fisher exact results — random doses |
| `results/wilcoxon_nielsen_effect_random_dose.csv` | Pre-computed Wilcoxon per-model and POOLED — random doses |

**What the code does**

Two stacked panels, one per experiment type.  In each panel, each model has two
grouped bars: naive and Nielsen-augmented accuracy (tight + broad extension).
Wilcoxon significance stars (from `wilcoxon_*.csv`) are drawn per model between
the naive and Nielsen bars.  Models are sorted by overall rank.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `nielsen_n_sig_tight_vignettes` | Models with significant tight improvement — vignettes |
| `nielsen_n_sig_tight_rd` | Models with significant tight improvement — random doses |
| `nielsen_mean_tight_improvement_vignettes` | Mean tight accuracy gain — vignettes (pp) |
| `nielsen_mean_tight_improvement_rd` | Mean tight accuracy gain — random doses (pp) |
| `nielsen_pooled_tight_p_vignettes` | POOLED Wilcoxon p — vignettes tight |
| `nielsen_pooled_tight_p_rd` | POOLED Wilcoxon p — random doses tight |
| `n_models` | Total model count |

---

### 12. `nielsen_ranked_random_doses`

**Title**: Model ranking after adding the Nielsen reference table

**Build method**: `ComprehensiveFigureBuilder.build_nielsen_ranked_random_doses()`
→ calls `naive_performance_bars(ranking_csv=ranking_random_doses_nielsen.csv)`

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/ranking_random_doses_nielsen.csv` | CMH ranking after Nielsen augmentation; tight/broad accuracy and pairwise significance |

**What the code does**

Identical layout to `naive_random_dose_bars` (§8) but after Nielsen augmentation.
The CMH ranking may differ substantially from the naive ranking, revealing which
models benefit most from reference data.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `n_opioid_pairs` | Total unique pairs |
| `nielsen_rd_best_tight_pct` | Best model tight accuracy after Nielsen (%) |
| `nielsen_rd_mean_tight_pct` | Mean tight accuracy after Nielsen (%) |
| `best_rd_model_after_nielsen` | Short name of top model after Nielsen |

---

### 13. `nielsen_by_ome_combined`

**Title**: Nielsen improvement concentrates on oral morphine conversions

**Build method**: `ComprehensiveFigureBuilder.build_nielsen_by_ome_combined()`
→ four calls to `nielsen_effect_bars()` in a column: vignettes OME, vignettes
non-OME, RD OME, RD non-OME.

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/model_batch_means_vignettes.csv` | OME (`ome_n0`, `ome_n1`) and non-OME (`pairs_n0`, `pairs_n1`) batch means — vignettes |
| `results/comparison_nielsen_effect_vignettes_ome.csv` | Nielsen contrast, OME pairs only — vignettes |
| `results/comparison_nielsen_effect_vignettes_nonome.csv` | Nielsen contrast, non-OME pairs only — vignettes |
| `results/wilcoxon_nielsen_effect_vignettes_ome.csv` | Per-model / POOLED Wilcoxon — vignettes OME |
| `results/wilcoxon_nielsen_effect_vignettes_nonome.csv` | Per-model / POOLED Wilcoxon — vignettes non-OME |
| `results/model_batch_means.csv` | OME and non-OME batch means — random doses |
| `results/comparison_nielsen_effect_random_dose_ome.csv` | Nielsen contrast, OME only — random doses |
| `results/comparison_nielsen_effect_random_dose_nonome.csv` | Nielsen contrast, non-OME only — random doses |
| `results/wilcoxon_nielsen_effect_random_dose_ome.csv` | Per-model / POOLED Wilcoxon — RD OME |
| `results/wilcoxon_nielsen_effect_random_dose_nonome.csv` | Per-model / POOLED Wilcoxon — RD non-OME |

**What the code does**

Four panels stacked vertically, each a Nielsen-effect grouped bar chart for a
specific experiment × OME stratum.  The stacking makes it visually obvious that
improvement is much larger for OME (oral morphine equivalent) pairs, where the
Nielsen table provides direct conversion factors, than for non-OME pairs.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `vig_ome_naive_tight_pct` | Vignette OME naive mean tight (%) |
| `vig_ome_nielsen_tight_pct` | Vignette OME Nielsen mean tight (%) |
| `vig_ome_tight_improvement_pp` | Vignette OME tight improvement (pp) |
| `vig_nonome_tight_improvement_pp` | Vignette non-OME tight improvement (pp) |
| `rd_ome_tight_improvement_pp` | RD OME tight improvement (pp) |
| `rd_nonome_tight_improvement_pp` | RD non-OME tight improvement (pp) |
| `nielsen_vig_ome_pooled_p` | POOLED Wilcoxon p — vignettes OME |
| `nielsen_vig_nonome_pooled_p` | POOLED Wilcoxon p — vignettes non-OME |
| `nielsen_rd_ome_pooled_p` | POOLED Wilcoxon p — RD OME |
| `nielsen_rd_nonome_pooled_p` | POOLED Wilcoxon p — RD non-OME |

---

### 14. `explain_units_full_treatment`

**Title**: Non-daily dosing confuses models; unit explanation rescues accuracy

**Build method**: `ComprehensiveFigureBuilder.build_explain_units_full_treatment()`
→ stacks `explain_units_bars()` (top) and `full_treatment_bars()` (bottom).

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/pair_model_analysis.csv` | Naive + Nielsen OME per-(model, pair) results |
| `results/explain_units_pair_model_analysis.csv` | Results with ask_for_units + explain_units flags |
| `results/comparison_units_effect_random_dose.csv` | Fisher exact: explain_units=True vs False |
| `results/wilcoxon_units_effect_random_dose.csv` | Per-model / POOLED Wilcoxon for units-explanation effect |
| `results/comparison_dosing_unit_effect.csv` | Per-(model, pair) accuracy in daily vs 4h format |
| `results/wilcoxon_dosing_unit_effect.csv` | Per-model / POOLED Wilcoxon for daily→4h drop |
| `results/comparison_nielsen_effect_random_dose_ome.csv` | Nielsen contrast OME — for full pipeline |
| `results/model_batch_means.csv` | Naive batch means for pipeline baseline |

**What the code does**

*Top panel (`explain_units_bars`)*: three-condition grouped bars per model:
(1) Nielsen + daily (mg/day), (2) Nielsen + 4h (mg/4h, no explanation),
(3) Nielsen + 4h + units explained.  Wilcoxon stars show: which models
deteriorate significantly when switching to 4h units, and which recover
when the explanation is added.

*Bottom panel (`full_treatment_bars`)*: four-condition pipeline from
(0) naive through (1) Nielsen, (2) Nielsen + 4h drop, (3) Nielsen + 4h + units
explanation.  Bars annotated with Fisher exact p-values between steps.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `dosing_n_sig_drop_tight` | Models with significant tight drop daily→4h |
| `dosing_mean_tight_drop_pp` | Mean tight accuracy drop daily→4h (pp) |
| `dosing_pooled_tight_p` | POOLED Wilcoxon p — daily vs 4h drop |
| `units_n_sig_recovery_tight` | Models with significant tight recovery with units explanation |
| `units_mean_tight_recovery_pp` | Mean tight accuracy recovery (pp) |
| `units_pooled_tight_p` | POOLED Wilcoxon p — units explanation recovery |

---

### 15. `nielsen_units_ranked`

**Title**: Best achievable accuracy after all prompt augmentations

**Build method**: `ComprehensiveFigureBuilder.build_nielsen_units_ranked()`
→ two calls to `naive_performance_bars()` stacked in a column: random doses
(4h + Nielsen + units) then vignettes (Nielsen + units).

**CSVs consumed**

| CSV | Role |
|-----|------|
| `results/ranking_random_doses_nielsen_units.csv` | CMH ranking for RD with Nielsen + explain_units; tight/broad accuracy and pairwise significance |
| `results/ranking_vignettes_nielsen_units.csv` | CMH ranking for vignettes with Nielsen + explain_units |

**What the code does**

Two ranked bar charts (same layout as §1 and §8).  Top panel shows the
best-achievable random-dose accuracy (OME pairs, Nielsen + 4h + units
explanation).  Bottom shows best-achievable vignette accuracy (all scenarios,
Nielsen + units explanation).  Together they set the ceiling against which naive
performance (§1, §8) is measured.

**Annotation fields from `figure_stats.json`**

| Field | Meaning |
|-------|---------|
| `best_rd_model_after_all` | Short model name: top random-dose after all augmentations |
| `best_rd_tight_after_all_pct` | Top random-dose tight accuracy after all augmentations (%) |
| `mean_rd_tight_after_all_pct` | Mean random-dose tight accuracy after all augmentations (%) |
| `best_vig_model_after_all` | Short model name: top vignette after all augmentations |
| `best_vig_tight_after_all_pct` | Top vignette tight accuracy after all augmentations (%) |
| `mean_vig_tight_after_all_pct` | Mean vignette tight accuracy after all augmentations (%) |

---

## Figure-to-CSV quick reference

| Figure slug | Primary CSVs |
|-------------|-------------|
| `naive_vignette_bars` | `ranking_naive_vignettes.csv` |
| `cv_strip_vignettes` | `reworded_vignettes_model_analysis.csv`, `cv_accuracy_chi2.csv` |
| `vignette_performance_profile` | `reworded_vignettes_model_analysis.csv`, `performance_profile_vignettes.csv`, `cv_accuracy_chi2.csv` |
| `variance_stacked` | `icc_analysis.csv`, `reworded_vignettes_model_analysis.csv`, `variance_icc_spearman.csv` |
| `spearman_variance_icc` | `variance_icc_spearman.csv`, `icc_analysis.csv`, `reworded_vignettes_model_analysis.csv` |
| `icc_bubble_grid` | `icc_analysis.csv`, `reworded_vignettes_model_analysis.csv` |
| `kendall_w_summary` | `kendall_w_vignettes.csv`, `icc_analysis.csv` |
| `naive_random_dose_bars` | `ranking_naive_random_doses.csv` |
| `cv_strip_random_doses` | `pair_model_analysis.csv`, `performance_profile_random_doses.csv` |
| `cross_experiment_bars` | `cross_experiment_matched_means.csv`, `cross_experiment_matched_wilcoxon.csv` |
| `nielsen_effect_combined` | `model_batch_means_vignettes.csv`, `wilcoxon_nielsen_effect_vignettes.csv`, `model_batch_means.csv`, `wilcoxon_nielsen_effect_random_dose.csv` |
| `nielsen_ranked_random_doses` | `ranking_random_doses_nielsen.csv` |
| `nielsen_by_ome_combined` | `model_batch_means_vignettes.csv`, `model_batch_means.csv`, `wilcoxon_nielsen_effect_vignettes_ome.csv`, `wilcoxon_nielsen_effect_vignettes_nonome.csv`, `wilcoxon_nielsen_effect_random_dose_ome.csv`, `wilcoxon_nielsen_effect_random_dose_nonome.csv` |
| `explain_units_full_treatment` | `pair_model_analysis.csv`, `explain_units_pair_model_analysis.csv`, `comparison_units_effect_random_dose.csv`, `wilcoxon_units_effect_random_dose.csv`, `comparison_dosing_unit_effect.csv`, `wilcoxon_dosing_unit_effect.csv` |
| `nielsen_units_ranked` | `ranking_random_doses_nielsen_units.csv`, `ranking_vignettes_nielsen_units.csv` |
