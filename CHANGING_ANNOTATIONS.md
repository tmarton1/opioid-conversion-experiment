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
