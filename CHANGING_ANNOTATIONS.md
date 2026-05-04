# Figure Annotation Change Log

Tracks all changes to the narrative annotation text in
`summary_w_figures/all_comprehensive_figures.pdf`, produced by
`visualize/comprehensive_figures.py → ComprehensiveFigureBuilder.save_annotated_pdf()`.

---

## How annotations are composed

Each of the 15 figures has a `(slug, method, title, description)` tuple in
`FIGURE_METADATA`. The `description` string may contain `{placeholder}` fields
that are substituted at PDF-build time from `summary_w_figures/figure_stats.json`,
which is populated by `orchestration/extract_figure_stats.py → compute_stats()`.

Numbers that are **not** expressed as `{placeholder}` fields are hardcoded in the
`description` string and are unaffected by changes to `extract_figure_stats.py`.

---

## Rewrite of `extract_figure_stats.py` (commit `1ec77f2`)

The script was rewritten from 26 computed stats to 78. **No existing figure
annotation values changed.** The table below lists every `{placeholder}` key
that appears in `FIGURE_METADATA`, its value before and after the rewrite, and
the figures it appears in.

| Placeholder | Pre-rewrite value | Post-rewrite value | Figures |
|---|---|---|---|
| `tight_lower_x` | 0.85 | 0.85 | 1, 2 |
| `tight_upper_x` | 1.15 | 1.15 | 1, 2 |
| `broad_lower_x` | 0.6 | 0.6 | 1, 2 |
| `broad_upper_x` | 1.7 | 1.7 | 1, 2 |
| `n_vignette_scenarios` | 20 | 20 | 1 (pre-session), 1 (post-session as inline label) |
| `n_model_scenario_cells` | 260 | 260 | 4 |
| `n_scenarios_per_model` | 20 | 20 | 5 |
| `icc_max_pairs` | 6 | 6 | 6 |
| `icc_total_sig_pairs` | 420 | 420 | 6 |
| `icc_total_pairs` | 1551 | 1551 | 6 |
| `n_opioid_pairs` | 49 | 49 | 8 |
| `n_matched_pairs` | 11 | 11 | 10 |

The 52 newly computed stats (e.g. `nielsen_pooled_tight_p_rd`, `dosing_mean_tight_drop_pp`, etc.)
are available in `figure_stats.json` but are **not yet referenced** in any
`FIGURE_METADATA` description. Annotations for those figures (11–15 and parts of
2–10) still contain hardcoded numbers.

---

## Deliberate annotation changes (this session)

### Figure 1 — `naive_vignette_bars`

**Reason**: The annotation said "20 realistic vignette-based prompts", which
understates the experiment size. There are 20 scenarios × 4 wordings = 80 prompts.

| | Before | After |
|---|---|---|
| Annotation text | `…on {n_vignette_scenarios} realistic vignette-based opioid conversion prompts without…` | `…on {n_vignette_prompts} realistic vignette-based opioid conversion prompts ({n_vignette_scenarios} clinical scenarios, {n_vignette_wordings} wordings each) without…` |
| Rendered value | "20 realistic vignette-based opioid conversion prompts" | "80 realistic vignette-based opioid conversion prompts (20 clinical scenarios, 4 wordings each)" |

New stats added to `extract_figure_stats.py` and `figure_stats.json`:
- `n_vignette_wordings` = 4
- `n_vignette_prompts` = 80

### Figure 6 — `icc_bubble_grid`

**Reason**: The annotation cited 1551 pairwise wording comparisons. The theoretical
maximum (13 models × 20 scenarios × C(4,2) = 1560) is not reached because
`qwen/qwen-2.5-72b-instruct` has only 3 wordings in 3 scenarios (see §
[Qwen missing wordings](#qwen-missing-wordings) below). A footnote dagger was added to explain the gap.

| | Before | After |
|---|---|---|
| End of annotation | `…{icc_total_sig_pairs}/{icc_total_pairs} comparisons significant).` | `…{icc_total_sig_pairs}/{icc_total_pairs}† comparisons significant). † qwen/qwen-2.5-72b-instruct has only 3 wordings (vs 4 for all other models) in 3 scenarios…` |

---

## Qwen missing wordings

`qwen/qwen-2.5-72b-instruct` is missing one wording in each of three scenarios:

| Scenario | Missing `vignette_id` | Present wordings |
|---|---|---|
| `elderly-hip-fracture` | v40 | v38, v39, v41 |
| `young-adult-sickle-cell-crisis` | v73 | v70, v71, v72 |
| `young-man-car-accident` | v67 | v66, v68, v69 |

**Root cause**: These three experiments were never run (or their results were not
saved) for Qwen. All other 12 `CONSISTENT_MODELS` have all 4 wordings per scenario.
The missing wordings are identifiable in `results/reworded_vignettes_results.csv`
by their absence in the Qwen rows.

**Impact**:
- `icc_analysis.csv` shows `k=3` (not 4) for these 3 Qwen cells.
- `total_pairs` for each affected cell is C(3,2)=3 instead of C(4,2)=6 → 3 × 3 = 9 fewer pairs.
- `icc_total_pairs` = 1551 rather than the expected 1560.
- ICC and Kendall's W for these 3 cells are computed with one fewer wording, making them
  less reliable than the other 257 cells.

**Remediation**: Re-run the 3 missing wording experiments for Qwen and append the
results to `results/reworded_vignettes_results.csv`, then re-run analysis.

---

## Figure 8 — `naive_random_dose_bars` (rank comparison panel added)

**Reason**: The figure title ("The same accuracy pattern holds on controlled random-dose
prompts") and the prior annotation ("The model ranking mirrors the vignette results")
made a factual claim that was not backed by any displayed data. A Spearman rank
correlation scatter panel was added to provide actual evidence.

| | Before | After |
|---|---|---|
| Build method returns | `naive_performance_bars(…)` (single bar chart) | `column(bar_chart, scatter_panel)` (bar chart + scatter) |
| Annotation | `"Horizontal bar chart of model accuracy on {n_opioid_pairs} random-dose opioid pair conversions without reference information. The model ranking mirrors the vignette results. Significance brackets…"` | `"Top: horizontal bar chart… Bottom: scatter plot of each model's tight accuracy on vignette prompts (x-axis) vs random-dose prompts (y-axis)… Spearman ρ = {naive_rank_spearman_rho} (p = {naive_rank_spearman_p}, n = {naive_rank_n_models} models) confirms…"` |
| Rendered Spearman ρ | (absent) | ρ = 0.68, p = 0.011, n = 13 |

New stats added to `extract_figure_stats.py` and `figure_stats.json`:
- `naive_rank_spearman_rho` = 0.68
- `naive_rank_spearman_p` = 0.011
- `naive_rank_n_models` = 13

The Spearman correlation is computed on `mean_tight_prop` (not on integer ranks) across
all 13 CONSISTENT_MODELS, using both experiments' full opioid-pair sets (not the 11
matched pairs used by §5 / figure 10). See `STATISTICAL_METHODS.md §5a` for details.

---

## Feb 2026 vs current annotation comparison

The Feb 2026 reference point is commit `c057501` (2026-02-25, "added missing opioids
from explain units experiments"). At that commit, `summary_w_figures/figure_stats.json`
**did not exist** — it was introduced in commit `1ec77f2`. All annotation numbers in
Feb 2026 were hardcoded strings in `FIGURE_METADATA`.

The table below lists every annotation string that differs between `c057501` and
the current HEAD. Figures whose annotation text is unchanged are omitted.

| Figure | Feb 2026 annotation (c057501) | Current annotation |
|--------|-------------------------------|-------------------|
| 1 `naive_vignette_bars` | `…on 20 realistic vignette-based opioid conversion prompts without any reference information. Dark bars show tight-range (0.85–1.15×) accuracy; translucent extensions show broad-range (0.6–1.7×) accuracy.` | `…on {n_vignette_prompts} realistic vignette-based opioid conversion prompts ({n_vignette_scenarios} clinical scenarios, {n_vignette_wordings} wordings each) without any reference information. Dark bars show tight-range ({tight_lower_x}–{tight_upper_x}×) accuracy; translucent extensions show broad-range ({broad_lower_x}–{broad_upper_x}×) accuracy.` → rendered: "…on **80** realistic … (**20** clinical scenarios, **4** wordings each)…" |
| 4 `variance_stacked` | `…across all 260 model × scenario cells.` | `…across all {n_model_scenario_cells} model × scenario cells.` → rendered: "260" (same value) |
| 5 `spearman_variance_icc` | `…and ICC (n = 20 scenarios per model), sorted…` | `…and ICC (n = {n_scenarios_per_model} scenarios per model), sorted…` → rendered: "20" (same value) |
| 6 `icc_bubble_grid` | `…Annotations show X/6 significant pairwise vignette comparisons (Bonferroni-corrected Mann–Whitney U).` | `…Annotations show the count of significant pairwise vignette comparisons out of {icc_max_pairs} per cell (Bonferroni-corrected Mann–Whitney U; overall {icc_total_sig_pairs}/{icc_total_pairs}† comparisons significant). † qwen/qwen-2.5-72b-instruct has only 3 wordings …` → rendered: "…out of **6** per cell (…; overall **420/1551**† …)" |
| 8 `naive_random_dose_bars` | `Horizontal bar chart of model accuracy on 49 random-dose opioid pair conversions without reference information. The model ranking mirrors the vignette results. Significance brackets compare adjacent-ranked models using Cochran–Mantel–Haenszel tests.` | Top bar + scatter panel; `…{n_opioid_pairs} random-dose… Spearman ρ = {naive_rank_spearman_rho} (p = {naive_rank_spearman_p}, n = {naive_rank_n_models} models) confirms that the model ranking is significantly correlated across the two prompt formats.` → rendered includes "**49**…ρ = **0.68** (p = **0.011**, n = **13** models)" |
| 10 `cross_experiment_bars` | `…restricted to the 11 opioid pairs present in both experiments.` | `…restricted to the {n_matched_pairs} opioid pairs present in both experiments.` → rendered: "**11**" (same value) |

**Summary of substantive changes** (values actually changed, not just converted to placeholders):

| Change | Feb 2026 rendered value | Current rendered value |
|--------|------------------------|----------------------|
| Fig 1 prompt count | 20 prompts | 80 prompts (20 scenarios × 4 wordings) |
| Fig 6 annotation detail | "X/6 sig pairs" (literal X) | "420/1551† sig pairs" with footnote explaining Qwen gap |
| Fig 8 rank evidence | claim only ("mirrors vignette results") | scatter panel + ρ = 0.68, p = 0.011 |

**No-op conversions** (value unchanged, only made data-driven):
- Fig 4: 260 → `{n_model_scenario_cells}` = 260
- Fig 5: 20 → `{n_scenarios_per_model}` = 20
- Fig 10: 11 → `{n_matched_pairs}` = 11
- Fig 1: 0.85/1.15/0.6/1.7 → `{tight_lower_x}` etc. (same values)
