# orchestration/

Production scripts that replace the experiment-running and analysis cells in the Jupyter notebook. Run these in order to reproduce all results from scratch.

---

## Scripts

| Script | Purpose |
|---|---|
| `run_experiments.py` | Dispatch all LLM API calls and write raw results to `results/` |
| `run_analysis.py` | Process raw CSVs through the full statistical analysis pipeline |
| `generate_raw_figures.py` | Build intermediate exploratory figures in `raw_figures/` |
| `extract_figure_stats.py` | Read analysis CSVs and save key data-driven numbers to `summary_w_figures/figure_stats.json` |
| `generate_summaries.py` | Build the final deliverable documents in `summary_w_figures/` |

---

## Order of execution

```
1. run_experiments.py       →  results/experiment_results.csv
                               results/explain_units_experiment_results.csv
                               results/reworded_vignettes_results.csv

2. run_analysis.py          →  results/pair_model_analysis.csv
                               results/reworded_vignettes_model_analysis.csv
                               results/comparison_*.csv
                               results/wilcoxon_*.csv
                               results/icc_analysis.csv
                               results/ranking_*.csv
                               ... (all other derived CSVs)

3. generate_raw_figures.py  →  raw_figures/{name}_batch.html/.pdf        (6 pair-batch grids)
                               raw_figures/{name}_heatmaps.html/.pdf      (6 pair-batch heatmaps)
                               raw_figures/all_reworded_vignette_batch.html/.pdf
                               raw_figures/model_comparison_bar.html/.pdf
                               raw_figures/model_comparison_scatter.pdf
                               raw_figures/explain_units_bar.html/.pdf
                               raw_figures/icc_heatmap.html/.pdf/.png
                               raw_figures/icc_bubble_grid.html/.pdf/.png
                               raw_figures/kendall_w_summary.html/.pdf/.png
                               raw_figures/scenario_strip_{scenario}.html/.png
                               raw_figures/consistency_examples.png/.pdf
                               raw_figures/statistical_methods.html

4. generate_summaries.py    →  summary_w_figures/figure_stats.json
                               summary_w_figures/accuracy_summary_tables.html/.pdf
                               summary_w_figures/pub_figures/{slug}.html/.pdf/.png
                               summary_w_figures/all_comprehensive_figures.pdf
```

`generate_summaries.py` calls `extract_figure_stats.py` internally as its first step, so you do not need to run it separately. `generate_raw_figures.py` is optional — skip it if you only need the publication figures in `summary_w_figures/`.

---

## Usage

Run all scripts from the **repository root**:

```bash
# Full pipeline
python orchestration/run_experiments.py
python orchestration/run_analysis.py
python orchestration/generate_raw_figures.py
python orchestration/generate_summaries.py
```

Each script accepts `--section` to run only part of the pipeline:

```bash
# Run only the vignette experiment
python orchestration/run_experiments.py --section vignettes

# Run only the ICC analysis step
python orchestration/run_analysis.py --section icc

# Regenerate only the heatmaps
python orchestration/generate_raw_figures.py --section pair_batches

# Regenerate only the annotated PDF
python orchestration/generate_summaries.py --section pdf
```

### `run_experiments.py` sections

| Section | What it runs |
|---|---|
| `pair` | Random-dose opioid pair experiment (25 reps × all models × 3 conditions) |
| `explain_units` | mg/4-hours target unit experiment (25 reps, Nielsen + ask/explain units) |
| `vignettes` | Vignette scenario experiment v2–v81 (7 reps × all models × 3 conditions) |
| `empty_vignettes` | Empty-text baseline vignettes (calls `build_scenarios` then runs experiment) |
| `all` | All of the above in order *(default)* |

### `run_analysis.py` sections

| Section | What it produces |
|---|---|
| `processing` | `*_model_analysis.csv`, `*_consensus.csv`, `reworded_scenarios_model_analysis.csv` |
| `batch_means` | `model_batch_means*.csv` (4 files) |
| `comparisons` | `comparison_*.csv` (9 files) |
| `wilcoxon` | `wilcoxon_*.csv` (up to 12 files) |
| `text_effects` | `text_effect_differences.csv` |
| `ome_comparisons` | `comparison_ome_vs_nonome_*.csv` |
| `dosing` | `comparison_dosing_unit_effect.csv` |
| `cross_experiment` | `cross_experiment_matched_*.csv` |
| `icc` | `icc_analysis.csv` |
| `kendall_w` | `kendall_w_vignettes.csv` |
| `variance` | `variance_icc_spearman.csv` |
| `grid` | `grid_independence_tests.csv` |
| `profiles` | `performance_profile_*.csv` |
| `cv_chi2` | `cv_accuracy_chi2.csv` |
| `rankings` | `ranking_*.csv` (5 files) |
| `all` | All of the above in order *(default)* |

### `generate_raw_figures.py` sections

| Section | What it produces |
|---|---|
| `pair_batches` | `raw_figures/{all,ome,pairs}_{n0,n1}_batch.html/.pdf` and `_heatmaps.html/.pdf` (6 × 2 = 12 files) |
| `vignette_batch` | `raw_figures/all_reworded_vignette_batch.html/.pdf` |
| `model_comparison` | `raw_figures/model_comparison_bar.html/.pdf`, `model_comparison_scatter.pdf` |
| `explain_units` | `raw_figures/explain_units_bar.html/.pdf` |
| `icc` | `raw_figures/icc_heatmap.*`, `icc_bubble_grid.*`, `kendall_w_summary.*` (HTML+PDF+PNG each) |
| `scenario_strips` | `raw_figures/scenario_strip_{scenario_key}.html/.png` (one per vignette scenario) + combined PDF |
| `consistency` | `raw_figures/consistency_examples.png/.pdf` |
| `statistical_methods` | `raw_figures/statistical_methods.html` |
| `all` | All of the above in order *(default)* |

### `generate_summaries.py` sections

| Section | What it produces |
|---|---|
| `stats` | `summary_w_figures/figure_stats.json` |
| `tables` | `summary_w_figures/accuracy_summary_tables.html/.pdf` |
| `figures` | `summary_w_figures/pub_figures/{slug}.html/.pdf/.png` (15 figures) |
| `pdf` | `summary_w_figures/all_comprehensive_figures.pdf` |
| `all` | All of the above in order *(default)* |

---

## Design notes

- All scripts add the repository root to `sys.path` at startup, so `from analysis.xxx import ...` and `from visualize.xxx import ...` resolve correctly regardless of where Python is invoked from.
- All analysis steps are **idempotent**: they overwrite their output CSVs on each run.
- Experiment scripts **append** to existing CSVs and assign new `experiment_id` values, so partial runs accumulate rather than overwrite.
- The notebook (`opioid-conversion.ipynb`) remains useful for interactive exploration and visualising individual conversions or vignettes — it is no longer the source of truth for production runs.
