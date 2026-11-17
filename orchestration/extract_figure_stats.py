"""
Extract key data-driven numbers from analysis CSVs and save to
summary_w_figures/figure_stats.json.

This JSON is consumed by ComprehensiveFigureBuilder.save_annotated_pdf()
to fill {placeholder} fields in FIGURE_METADATA descriptions so that every
number in the final annotated PDF comes from real data rather than being
hardcoded.

Run via generate_summaries.py (section "stats"), or directly:
    python orchestration/extract_figure_stats.py [--extended]

--extended also writes the Bonferroni-adjusted counts as *_bonf keys (revised baseline).

Key:   computed from → which CSV file(s) feed each stat block.
       See FIGURES.md §Annotation fields for per-figure usage of each key.
"""

from __future__ import annotations

import json
import os
import sys

# Allow running directly from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import RESULTS_DIR, FIGURES_DIR

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon as _wilcoxon, spearmanr


OUTPUT_PATH = f"{FIGURES_DIR}/figure_stats.json"
META_PATH = f"{FIGURES_DIR}/figure_stats_meta.json"


def _read(name: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [warn] {name} not found — skipping")
    return None


def _pct(val: float, decimals: int = 1) -> float:
    """Scale 0–1 proportion to percentage, rounded."""
    return round(float(val) * 100, decimals)


def _short(model_id: str) -> str:
    """Return the model name component after the vendor prefix (last path segment)."""
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _pooled_p_from_wilcoxon(fname: str, stats: dict, key: str) -> None:
    """
    Read a wilcoxon CSV and store the POOLED row's tight_p under stats[key].
    """
    wdf = _read(fname)
    if wdf is None:
        stats[key] = "?"
        return
    pooled_row = wdf[wdf["model"].str.upper() == "POOLED"]
    if not pooled_row.empty:
        stats[key] = round(float(pooled_row["tight_p"].iloc[0]), 6)
    else:
        stats[key] = "?"


def compute_stats() -> dict:
    stats: dict = {}

    # ── Model list ────────────────────────────────────────────────────────────
    # Source: common/models.py → CONSISTENT_MODELS
    _consistent_set: set | None = None
    try:
        from common.models import CONSISTENT_MODELS
        stats["n_models"] = len(CONSISTENT_MODELS)
        _consistent_set = set(CONSISTENT_MODELS)
    except ImportError:
        stats["n_models"] = "?"

    # ── Accuracy range constants ──────────────────────────────────────────────
    # Source: common/config.py
    from common.config import (
        TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER,
        BROAD_RANGE_LOWER, BROAD_RANGE_UPPER,
    )
    stats["tight_lower_x"] = f"{TIGHT_RANGE_LOWER:g}"
    stats["tight_upper_x"] = f"{TIGHT_RANGE_UPPER:g}"
    stats["broad_lower_x"] = f"{BROAD_RANGE_LOWER:g}"
    stats["broad_upper_x"] = f"{BROAD_RANGE_UPPER:g}"

    # ── Figs 1 & 8: Naive ranking bars ───────────────────────────────────────
    # Source: results/ranking_naive_vignettes.csv
    #         → mean_tight_prop, mean_broad_prop per model (CMH-ranked)
    rvig = _read("ranking_naive_vignettes.csv")
    if rvig is not None:
        if _consistent_set:
            rvig = rvig[rvig["model"].isin(_consistent_set)]
        tp = rvig["mean_tight_prop"]
        bp = rvig["mean_broad_prop"]
        stats["naive_vig_best_tight_pct"]  = _pct(tp.max())
        stats["naive_vig_worst_tight_pct"] = _pct(tp.min())
        stats["naive_vig_mean_tight_pct"]  = _pct(tp.mean())
        stats["naive_vig_best_broad_pct"]  = _pct(bp.max())
    else:
        for k in ("naive_vig_best_tight_pct", "naive_vig_worst_tight_pct",
                  "naive_vig_mean_tight_pct", "naive_vig_best_broad_pct"):
            stats[k] = "?"

    # Source: results/ranking_naive_random_doses.csv
    #         → mean_tight_prop, mean_broad_prop per model (CMH-ranked)
    rrd = _read("ranking_naive_random_doses.csv")
    if rrd is not None:
        if _consistent_set:
            rrd = rrd[rrd["model"].isin(_consistent_set)]
        tp = rrd["mean_tight_prop"]
        bp = rrd["mean_broad_prop"]
        stats["naive_rd_best_tight_pct"]  = _pct(tp.max())
        stats["naive_rd_worst_tight_pct"] = _pct(tp.min())
        stats["naive_rd_mean_tight_pct"]  = _pct(tp.mean())
        stats["naive_rd_best_broad_pct"]  = _pct(bp.max())
    else:
        for k in ("naive_rd_best_tight_pct", "naive_rd_worst_tight_pct",
                  "naive_rd_mean_tight_pct", "naive_rd_best_broad_pct"):
            stats[k] = "?"

    # ── Fig 8: Cross-experiment rank correlation (vignette vs random doses) ──
    # Source: results/ranking_naive_vignettes.csv + ranking_naive_random_doses.csv
    #         → Spearman ρ on mean_tight_prop across CONSISTENT_MODELS
    if rvig is not None and rrd is not None:
        merged_ranks = rvig[["model", "mean_tight_prop"]].merge(
            rrd[["model", "mean_tight_prop"]], on="model", suffixes=("_vig", "_rd")
        )
        if len(merged_ranks) >= 3:
            rho, p_val = spearmanr(
                merged_ranks["mean_tight_prop_vig"],
                merged_ranks["mean_tight_prop_rd"],
            )
            stats["naive_rank_spearman_rho"] = round(float(rho), 2)
            stats["naive_rank_spearman_p"]   = round(float(p_val), 3)
            stats["naive_rank_n_models"]     = int(len(merged_ranks))
        else:
            stats["naive_rank_spearman_rho"] = "?"
            stats["naive_rank_spearman_p"]   = "?"
            stats["naive_rank_n_models"]     = "?"
    else:
        stats["naive_rank_spearman_rho"] = "?"
        stats["naive_rank_spearman_p"]   = "?"
        stats["naive_rank_n_models"]     = "?"

    # ── Figs 2 & 3: CV strips / performance profile — vignettes ─────────────
    # Source: results/reworded_vignettes_model_analysis.csv
    #         → filtered to naive (include_nielsen=False, explain_units=False,
    #           has_vignette_text=True); scenario_key unique count
    vdf = _read("reworded_vignettes_model_analysis.csv")
    if vdf is not None:
        naive_vdf = vdf[
            (vdf.get("include_nielsen", vdf.get("include_nielson",
                pd.Series(False, index=vdf.index))) == False) &
            (vdf.get("explain_units", pd.Series(False, index=vdf.index)) == False) &
            (vdf.get("has_vignette_text", pd.Series(True, index=vdf.index)) == True)
        ]
        stats["n_vignette_scenarios"] = int(naive_vdf["scenario_key"].nunique())
        n_wordings = int(naive_vdf.groupby("scenario_key")["vignette_id"].nunique().mode()[0])
        stats["n_vignette_wordings"]  = n_wordings
        stats["n_vignette_prompts"]   = stats["n_vignette_scenarios"] * n_wordings
    else:
        stats["n_vignette_scenarios"] = "?"
        stats["n_vignette_wordings"]  = "?"
        stats["n_vignette_prompts"]   = "?"

    # Source: results/cv_accuracy_chi2.csv
    #         → vignette chi2 per model (significant col); see run_analysis.py §14
    #           and visualize/comprehensive_figures.py → build_cv_strip_vignettes()
    chi2df = _read("cv_accuracy_chi2.csv")
    if chi2df is not None:
        stats["n_sig_cv_chi2_vignettes"] = int(chi2df["significant"].sum())
    else:
        stats["n_sig_cv_chi2_vignettes"] = "?"

    # Source: results/performance_profile_vignettes.csv
    #         → correct, precise_wrong, imprecise columns per model
    ppv = _read("performance_profile_vignettes.csv")
    if ppv is not None:
        if _consistent_set:
            ppv = ppv[ppv["model"].isin(_consistent_set)]
        stats["vig_mean_correct_pct"]       = _pct(ppv["correct"].mean())
        stats["vig_mean_precise_wrong_pct"] = _pct(ppv["precise_wrong"].mean())
        stats["vig_mean_imprecise_pct"]     = _pct(ppv["imprecise"].mean())
    else:
        for k in ("vig_mean_correct_pct", "vig_mean_precise_wrong_pct",
                  "vig_mean_imprecise_pct"):
            stats[k] = "?"

    # ── Figs 4–7: ICC / variance decomposition ───────────────────────────────
    # Source: results/icc_analysis.csv
    #         → icc, var_between, var_within, var_total, sig_pairs, total_pairs,
    #           f_pvalue per (model, scenario_key)
    idf = _read("icc_analysis.csv")
    if idf is not None:
        stats["n_model_scenario_cells"] = int(len(idf))

        if "model" in idf.columns and "scenario_key" in idf.columns:
            per_model_counts = idf.groupby("model")["scenario_key"].nunique()
            stats["n_scenarios_per_model"] = int(per_model_counts.median())
        else:
            stats["n_scenarios_per_model"] = "?"

        if "sig_pairs" in idf.columns and "total_pairs" in idf.columns:
            stats["icc_total_sig_pairs"] = int(idf["sig_pairs"].sum())
            stats["icc_total_pairs"]     = int(idf["total_pairs"].sum())
            stats["icc_max_pairs"]       = int(idf["total_pairs"].mode().iloc[0])
        else:
            stats["icc_total_sig_pairs"] = "?"
            stats["icc_total_pairs"]     = "?"
            stats["icc_max_pairs"]       = 6  # C(4,2) default

        if "icc" in idf.columns:
            stats["mean_icc"]   = round(float(idf["icc"].mean()), 3)
            stats["median_icc"] = round(float(idf["icc"].median()), 3)
            if "model" in idf.columns:
                model_mean_icc = idf.groupby("model")["icc"].mean()
                stats["n_models_icc_gt_half"] = int((model_mean_icc > 0.5).sum())
            if "f_pvalue" in idf.columns and "model" in idf.columns:
                stats["n_models_with_sig_icc"] = int(
                    (idf.groupby("model")["f_pvalue"].min() < 0.05).sum()
                )

        if "var_between" in idf.columns and "var_total" in idf.columns:
            mask = idf["var_total"] > 0
            between_frac = idf.loc[mask, "var_between"] / idf.loc[mask, "var_total"]
            stats["mean_between_var_pct"] = _pct(between_frac.mean())
        else:
            stats["mean_between_var_pct"] = "?"
    else:
        for k in ("n_model_scenario_cells", "n_scenarios_per_model",
                  "icc_total_sig_pairs", "icc_total_pairs", "mean_icc", "median_icc",
                  "n_models_icc_gt_half", "n_models_with_sig_icc", "mean_between_var_pct"):
            stats[k] = "?"
        stats["icc_max_pairs"] = 6

    # Source: results/kendall_w_vignettes.csv
    #         → kendall_w, p_value per scenario
    kdf = _read("kendall_w_vignettes.csv")
    if kdf is not None:
        if "kendall_w" in kdf.columns:
            stats["mean_kendall_w"]   = round(float(kdf["kendall_w"].mean()), 3)
            stats["median_kendall_w"] = round(float(kdf["kendall_w"].median()), 3)
        if "p_value" in kdf.columns or "kendall_p" in kdf.columns:
            p_col = "p_value" if "p_value" in kdf.columns else "kendall_p"
            stats["n_sig_kendall_w"] = int((kdf[p_col] < 0.05).sum())

    # Source: results/variance_icc_spearman.csv
    #         → stat_type='variance_vs_icc'; rho, p_value per model + OVERALL row
    spdf = _read("variance_icc_spearman.csv")
    if spdf is not None:
        overall = spdf[
            (spdf.get("stat_type", pd.Series("", index=spdf.index)) == "variance_vs_icc") &
            (spdf.get("model",     pd.Series("", index=spdf.index)) == "OVERALL")
        ]
        if not overall.empty and "rho" in overall.columns:
            stats["spearman_rho_variance_icc"] = round(float(overall["rho"].iloc[0]), 3)
            stats["spearman_p_variance_icc"]   = round(float(overall["p_value"].iloc[0]), 4)

        per_model = spdf[
            (spdf.get("stat_type", pd.Series("", index=spdf.index)) == "variance_vs_icc") &
            (spdf.get("model",     pd.Series("", index=spdf.index)) != "OVERALL")
        ]
        if not per_model.empty and "rho" in per_model.columns:
            stats["n_positive_rho_models"] = int((per_model["rho"] > 0).sum())
            stats["n_total_rho_models"]    = int(len(per_model))
            stats["n_sig_rho_models"]      = int((per_model["p_value"] < 0.05).sum())

    # ── Fig 8 & 9: Naive random-dose bars / CV strip ─────────────────────────
    # Source: results/pair_model_analysis.csv
    #         → filtered to naive (include_nielsen=False, include_examples=False)
    pdf = _read("pair_model_analysis.csv")
    if pdf is not None:
        naive_pdf = pdf[
            (pdf.get("include_nielsen",
                     pdf.get("include_nielson", pd.Series(False, index=pdf.index))) == False) &
            (pdf.get("include_examples", pd.Series(False, index=pdf.index)) == False)
        ]
        stats["n_opioid_pairs"] = int(naive_pdf["opioid_pair"].nunique())

        # CV × accuracy chi-square for random doses (computed inline; see
        # analysis/variance_icc_stats.py → compute_cv_accuracy_chi2 for vignette version)
        if _consistent_set:
            naive_pdf = naive_pdf[naive_pdf["model"].isin(_consistent_set)]
        cv_col    = "observed_coefficient_cv"
        tight_col = "tight(0.85,1.15)"
        broad_col = "broad(0.6,1.7)"
        if all(c in naive_pdf.columns for c in [cv_col, tight_col, broad_col]):
            dfc = naive_pdf.dropna(subset=[cv_col, tight_col, broad_col]).copy()
            cv_bins      = [0, 0.1, 0.5, float("inf")]
            cv_bin_labels = ["<0.1", "0.1–0.5", "≥0.5"]
            dfc["_cv_bin"] = pd.cut(dfc[cv_col], bins=cv_bins,
                                    labels=cv_bin_labels, right=False)
            dfc["_acc"] = "neither"
            dfc.loc[dfc[broad_col] >= 0.5, "_acc"] = "broad"
            dfc.loc[dfc[tight_col] >= 0.5, "_acc"] = "tight"
            acc_labels = ["tight", "broad", "neither"]
            n_sig_rd = 0
            for m in sorted(dfc["model"].unique()):
                mdf = dfc[dfc["model"] == m]
                table = np.zeros((3, 3), dtype=int)
                for i, al in enumerate(acc_labels):
                    for j, bl in enumerate(cv_bin_labels):
                        table[i, j] = int(((mdf["_acc"] == al) &
                                           (mdf["_cv_bin"] == bl)).sum())
                try:
                    _, pval, _, _ = chi2_contingency(table)
                    if pval < 0.05:
                        n_sig_rd += 1
                except (ValueError, ZeroDivisionError):
                    pass
            stats["n_sig_cv_chi2_rd"] = n_sig_rd
        else:
            stats["n_sig_cv_chi2_rd"] = "?"
    else:
        stats["n_opioid_pairs"]    = "?"
        stats["n_sig_cv_chi2_rd"]  = "?"

    # Source: results/performance_profile_random_doses.csv
    #         → correct, precise_wrong, dose_dep_neg columns per model
    pprd = _read("performance_profile_random_doses.csv")
    if pprd is not None:
        if _consistent_set:
            pprd = pprd[pprd["model"].isin(_consistent_set)]
        stats["rd_mean_correct_pct"]       = _pct(pprd["correct"].mean())
        stats["rd_mean_precise_wrong_pct"] = _pct(pprd["precise_wrong"].mean())
        stats["rd_mean_dose_dep_neg_pct"]  = _pct(pprd["dose_dep_neg"].mean())
    else:
        for k in ("rd_mean_correct_pct", "rd_mean_precise_wrong_pct",
                  "rd_mean_dose_dep_neg_pct"):
            stats[k] = "?"

    # ── Fig 10: Cross-experiment bars ─────────────────────────────────────────
    # Source: results/cross_experiment_matched_means.csv
    #         → n_pairs column (number of matched opioid pairs per model × condition)
    cem = _read("cross_experiment_matched_means.csv")
    if cem is not None:
        if "n_pairs" in cem.columns:
            stats["n_matched_pairs"] = int(cem["n_pairs"].mode().iloc[0])
        else:
            pair_col = next(
                (c for c in ("opioid_pair", "pair", "unit") if c in cem.columns), None
            )
            stats["n_matched_pairs"] = (
                int(cem[pair_col].nunique()) if pair_col else int(len(cem))
            )
    else:
        stats["n_matched_pairs"] = "?"

    # Source: results/cross_experiment_matched_wilcoxon.csv
    #         → broad_diff per model × contrast; pooled Wilcoxon signed-rank test
    #           computed here across per-model diffs (no POOLED row in the CSV)
    cw = _read("cross_experiment_matched_wilcoxon.csv")
    if cw is not None:
        for ca, cb, key in [
            ("random_dose",    "text_vignette",  "cross_pooled_broad_p_rd_text"),
            ("random_dose",    "empty_vignette", "cross_pooled_broad_p_rd_empty"),
            ("empty_vignette", "text_vignette",  "cross_pooled_broad_p_empty_text"),
        ]:
            sub = cw[(cw["condition_a"] == ca) & (cw["condition_b"] == cb)]
            diffs = sub["broad_diff"].dropna().values if "broad_diff" in sub.columns else []
            if len(diffs) >= 3:
                try:
                    _, p = _wilcoxon(diffs)
                    stats[key] = round(float(p), 4)
                except Exception:
                    stats[key] = "?"
            else:
                stats[key] = "?"
    else:
        for k in ("cross_pooled_broad_p_rd_text", "cross_pooled_broad_p_rd_empty",
                  "cross_pooled_broad_p_empty_text"):
            stats[k] = "?"

    # Source: results/cross_experiment_matched_spearman.csv
    #         → spearman_rho, spearman_p per model for RD vs text_vignette, broad range
    cs = _read("cross_experiment_matched_spearman.csv")
    if cs is not None:
        rd_text_broad = cs[
            (cs["condition_a"] == "random_dose") &
            (cs["condition_b"] == "text_vignette") &
            (cs["range"] == "broad")
        ]
        if not rd_text_broad.empty:
            stats["cross_spearman_rho_rd_text_broad"] = round(
                float(rd_text_broad["spearman_rho"].mean()), 3
            )
            stats["cross_n_sig_spearman_rd_text"] = int(
                (rd_text_broad["spearman_p"] < 0.05).sum()
            )
        else:
            stats["cross_spearman_rho_rd_text_broad"] = "?"
            stats["cross_n_sig_spearman_rd_text"]     = "?"
    else:
        stats["cross_spearman_rho_rd_text_broad"] = "?"
        stats["cross_n_sig_spearman_rd_text"]     = "?"

    # ── Fig 11: Nielsen effect combined ──────────────────────────────────────
    # Source: results/wilcoxon_nielsen_effect_vignettes.csv
    #         results/wilcoxon_nielsen_effect_random_dose.csv
    #         → tight_p, tight_mean per model + POOLED row
    for exp_sfx, fname in (
        ("vignettes", "wilcoxon_nielsen_effect_vignettes.csv"),
        ("rd",        "wilcoxon_nielsen_effect_random_dose.csv"),
    ):
        wdf = _read(fname)
        if wdf is None:
            stats[f"nielsen_n_sig_tight_{exp_sfx}"]           = "?"
            stats[f"nielsen_mean_tight_improvement_{exp_sfx}"] = "?"
            stats[f"nielsen_pooled_tight_p_{exp_sfx}"]         = "?"
            continue
        per_model  = wdf[wdf["model"].str.upper() != "POOLED"]
        pooled_row = wdf[wdf["model"].str.upper() == "POOLED"]
        stats[f"nielsen_n_sig_tight_{exp_sfx}"] = int((per_model["tight_p"] < 0.05).sum())
        stats[f"nielsen_mean_tight_improvement_{exp_sfx}"] = (
            _pct(per_model["tight_mean"].mean()) if "tight_mean" in per_model.columns else "?"
        )
        stats[f"nielsen_pooled_tight_p_{exp_sfx}"] = (
            round(float(pooled_row["tight_p"].iloc[0]), 6) if not pooled_row.empty else "?"
        )

    # Source: results/ranking_random_doses_nielsen.csv
    #         → best model after Nielsen augmentation (rank 1 row)
    rnielsen = _read("ranking_random_doses_nielsen.csv")
    if rnielsen is not None:
        if _consistent_set:
            rnielsen = rnielsen[rnielsen["model"].isin(_consistent_set)]
        top = rnielsen.nsmallest(1, "adjusted_rank") if "adjusted_rank" in rnielsen.columns \
              else rnielsen.nlargest(1, "mean_tight_prop")
        stats["best_rd_model_after_nielsen"] = _short(str(top["model"].iloc[0]))
        tp = rnielsen["mean_tight_prop"]
        stats["nielsen_rd_best_tight_pct"] = _pct(tp.max())
        stats["nielsen_rd_mean_tight_pct"] = _pct(tp.mean())
    else:
        for k in ("best_rd_model_after_nielsen", "nielsen_rd_best_tight_pct",
                  "nielsen_rd_mean_tight_pct"):
            stats[k] = "?"

    # ── Fig 13: Nielsen improvement by OME stratum ───────────────────────────
    # Source: results/model_batch_means_vignettes.csv
    #         → mean_in_tight per model for batch_name in {ome_n0, ome_n1, pairs_n0, pairs_n1}
    bvig = _read("model_batch_means_vignettes.csv")
    if bvig is not None:
        if _consistent_set:
            bvig = bvig[bvig["model"].isin(_consistent_set)]
        def _batch_mean_tight(df: pd.DataFrame, batch: str) -> float | str:
            sub = df[df["batch_name"] == batch]
            if sub.empty or "mean_in_tight" not in df.columns:
                return "?"
            return _pct(sub["mean_in_tight"].mean())

        ome_n0  = _batch_mean_tight(bvig, "ome_n0")
        ome_n1  = _batch_mean_tight(bvig, "ome_n1")
        nonome_n0 = _batch_mean_tight(bvig, "pairs_n0")
        nonome_n1 = _batch_mean_tight(bvig, "pairs_n1")
        stats["vig_ome_naive_tight_pct"]     = ome_n0
        stats["vig_ome_nielsen_tight_pct"]   = ome_n1
        stats["vig_ome_tight_improvement_pp"] = (
            round(ome_n1 - ome_n0, 1) if isinstance(ome_n0, float) else "?"
        )
        stats["vig_nonome_tight_improvement_pp"] = (
            round(nonome_n1 - nonome_n0, 1)
            if isinstance(nonome_n0, float) else "?"
        )
    else:
        for k in ("vig_ome_naive_tight_pct", "vig_ome_nielsen_tight_pct",
                  "vig_ome_tight_improvement_pp", "vig_nonome_tight_improvement_pp"):
            stats[k] = "?"

    # Source: results/model_batch_means.csv
    #         → mean_in_tight per model for batch_name in {ome_n0, ome_n1, pairs_n0, pairs_n1}
    brd = _read("model_batch_means.csv")
    if brd is not None:
        if _consistent_set:
            brd = brd[brd["model"].isin(_consistent_set)]
        def _batch_mean_tight_rd(df: pd.DataFrame, batch: str) -> float | str:
            sub = df[df["batch_name"] == batch]
            if sub.empty or "mean_in_tight" not in df.columns:
                return "?"
            return _pct(sub["mean_in_tight"].mean())

        rd_ome_n0  = _batch_mean_tight_rd(brd, "ome_n0")
        rd_ome_n1  = _batch_mean_tight_rd(brd, "ome_n1")
        rd_nomo_n0 = _batch_mean_tight_rd(brd, "pairs_n0")
        rd_nomo_n1 = _batch_mean_tight_rd(brd, "pairs_n1")
        stats["rd_ome_tight_improvement_pp"] = (
            round(rd_ome_n1 - rd_ome_n0, 1) if isinstance(rd_ome_n0, float) else "?"
        )
        stats["rd_nonome_tight_improvement_pp"] = (
            round(rd_nomo_n1 - rd_nomo_n0, 1) if isinstance(rd_nomo_n0, float) else "?"
        )
    else:
        stats["rd_ome_tight_improvement_pp"]    = "?"
        stats["rd_nonome_tight_improvement_pp"] = "?"

    # Source: results/wilcoxon_nielsen_effect_vignettes_ome.csv   → POOLED tight_p
    #         results/wilcoxon_nielsen_effect_vignettes_nonome.csv → POOLED tight_p
    #         results/wilcoxon_nielsen_effect_random_dose_ome.csv  → POOLED tight_p
    #         results/wilcoxon_nielsen_effect_random_dose_nonome.csv → POOLED tight_p
    _pooled_p_from_wilcoxon("wilcoxon_nielsen_effect_vignettes_ome.csv",    stats, "nielsen_vig_ome_pooled_p")
    _pooled_p_from_wilcoxon("wilcoxon_nielsen_effect_vignettes_nonome.csv", stats, "nielsen_vig_nonome_pooled_p")
    _pooled_p_from_wilcoxon("wilcoxon_nielsen_effect_random_dose_ome.csv",  stats, "nielsen_rd_ome_pooled_p")
    _pooled_p_from_wilcoxon("wilcoxon_nielsen_effect_random_dose_nonome.csv", stats, "nielsen_rd_nonome_pooled_p")

    # ── Fig 14: Explain units / dosing-unit treatment ─────────────────────────
    # Source: results/wilcoxon_dosing_unit_effect.csv
    #         → tight_p, tight_mean per model + POOLED; daily→4h accuracy drop
    wdos = _read("wilcoxon_dosing_unit_effect.csv")
    if wdos is not None:
        per_model = wdos[wdos["model"].str.upper() != "POOLED"]
        pooled_row = wdos[wdos["model"].str.upper() == "POOLED"]
        stats["dosing_n_sig_drop_tight"]   = int((per_model["tight_p"] < 0.05).sum())
        stats["dosing_mean_tight_drop_pp"] = (
            _pct(per_model["tight_mean"].mean()) if "tight_mean" in per_model.columns else "?"
        )
        stats["dosing_pooled_tight_p"] = (
            round(float(pooled_row["tight_p"].iloc[0]), 6) if not pooled_row.empty else "?"
        )
    else:
        for k in ("dosing_n_sig_drop_tight", "dosing_mean_tight_drop_pp",
                  "dosing_pooled_tight_p"):
            stats[k] = "?"

    # Source: results/wilcoxon_units_effect_random_dose.csv
    #         → tight_p, tight_mean per model + POOLED; units-explanation recovery
    wunits = _read("wilcoxon_units_effect_random_dose.csv")
    if wunits is not None:
        per_model = wunits[wunits["model"].str.upper() != "POOLED"]
        pooled_row = wunits[wunits["model"].str.upper() == "POOLED"]
        stats["units_n_sig_recovery_tight"]   = int((per_model["tight_p"] < 0.05).sum())
        stats["units_mean_tight_recovery_pp"] = (
            _pct(per_model["tight_mean"].mean()) if "tight_mean" in per_model.columns else "?"
        )
        stats["units_pooled_tight_p"] = (
            round(float(pooled_row["tight_p"].iloc[0]), 6) if not pooled_row.empty else "?"
        )
    else:
        for k in ("units_n_sig_recovery_tight", "units_mean_tight_recovery_pp",
                  "units_pooled_tight_p"):
            stats[k] = "?"

    # ── Fig 15: Best accuracy after all augmentations ─────────────────────────
    # Source: results/ranking_random_doses_nielsen_units.csv
    #         → top model and accuracy (mean_tight_prop) after Nielsen + units
    rnall = _read("ranking_random_doses_nielsen_units.csv")
    if rnall is not None:
        if _consistent_set:
            rnall = rnall[rnall["model"].isin(_consistent_set)]
        top = rnall.nsmallest(1, "adjusted_rank") if "adjusted_rank" in rnall.columns \
              else rnall.nlargest(1, "mean_tight_prop")
        stats["best_rd_model_after_all"]    = _short(str(top["model"].iloc[0]))
        stats["best_rd_tight_after_all_pct"] = _pct(top["mean_tight_prop"].iloc[0])
        stats["mean_rd_tight_after_all_pct"] = _pct(rnall["mean_tight_prop"].mean())
    else:
        for k in ("best_rd_model_after_all", "best_rd_tight_after_all_pct",
                  "mean_rd_tight_after_all_pct"):
            stats[k] = "?"

    # Source: results/ranking_vignettes_nielsen_units.csv
    #         → top model and accuracy after Nielsen + units — vignettes
    rvall = _read("ranking_vignettes_nielsen_units.csv")
    if rvall is not None:
        if _consistent_set:
            rvall = rvall[rvall["model"].isin(_consistent_set)]
        top = rvall.nsmallest(1, "adjusted_rank") if "adjusted_rank" in rvall.columns \
              else rvall.nlargest(1, "mean_tight_prop")
        stats["best_vig_model_after_all"]     = _short(str(top["model"].iloc[0]))
        stats["best_vig_tight_after_all_pct"] = _pct(top["mean_tight_prop"].iloc[0])
        stats["mean_vig_tight_after_all_pct"] = _pct(rvall["mean_tight_prop"].mean())
    else:
        for k in ("best_vig_model_after_all", "best_vig_tight_after_all_pct",
                  "mean_vig_tight_after_all_pct"):
            stats[k] = "?"

    return stats


# ── Display formats and structured fields (decision 48) ──────────────────────
# figure_stats.json stays a flat key -> value map. figure_stats_meta.json (written next
# to it) gives, for every key, a display format, the formatted display value, and
# structured fields (no free text): measure, experiment, condition, reference,
# statistic, adjusted, figure.  Descriptions elsewhere are built from these fields only.
#
# Formats: "percent:N" (value already in %), "pp:N" (percentage points), "count",
# "ratio:N", "p" (JMIR style: P<.001, P=.03, no leading zero; italic capital P in
# text, <i>P</i> in display_html), "p_caption" (legacy caption style "p < 0.001"),
# "text".

BONF_SUFFIX = "_bonf"


def fmt_p(p: float) -> str:
    """JMIR style P value: 'P<.001', 'P=.007', 'P=.03', 'P>.99'; no leading zero."""
    p = float(p)
    if p < 0.001:
        return "P<.001"
    if p < 0.01:
        return "P=" + f"{p:.3f}"[1:]
    q = f"{p:.2f}"
    if float(q) >= 1.0:
        return "P>.99"
    return "P=" + q[1:]


def fmt_p_caption(p: float) -> str:
    """Caption style used in the figure captions: 'p < 0.001', 'p = 0.007', 'p = 0.24'."""
    p = float(p)
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def fmt_value(value, fmt: str) -> str:
    """Format a stats value; '?' (missing) is kept as '?'."""
    if isinstance(value, str):
        return value
    kind, _, arg = fmt.partition(":")
    n = int(arg) if arg else 0
    if kind == "percent":
        return f"{float(value):.{n}f}%"
    if kind == "pp":
        return f"{float(value):.{n}f} pp"
    if kind == "count":
        return str(int(round(float(value))))
    if kind == "ratio":
        return f"{float(value):.{n}f}"
    if kind == "p":
        return fmt_p(value)
    if kind == "p_caption":
        return fmt_p_caption(value)
    if kind == "text":
        return str(value)
    raise ValueError(f"unknown format {fmt!r}")


def fmt_html(value, fmt: str) -> str:
    """Display with the P italic (and < escaped) for P values; otherwise as fmt_value."""
    d = fmt_value(value, fmt)
    if fmt == "p" and d.startswith("P"):
        return "<i>P</i>" + d[1:].replace("<", "&lt;").replace(">", "&gt;")
    return d


# key -> (format, figure, measure, experiment, condition, reference, statistic)
# vocabulary (controlled; descriptions are assembled from these tokens in make_description):
#   measure: tight | broad | share_correct | share_precise_wrong | share_imprecise | share_dose_dep_neg |
#            icc | between_var | kendall_w | variance_icc_rho | rank_corr_tight | cv_chi2 | accuracy_gain_tight |
#            accuracy_drop_tight | accuracy_recovery_tight | design
#   experiment: vignettes | random doses | both | vignettes, oral morphine | ... | design
#   condition: naive | reference-grounded | unit-augmented | naive -> reference-grounded |
#              reference-grounded mg/day -> mg/4 h | reference-grounded mg/4 h -> unit-augmented | n/a
#   reference: Nielsen | n/a
#   statistic: mean | best | worst | median | count | count of significant models | p | rho | model | limit
_V, _R, _VO, _VN, _RO, _RN = ("vignettes", "random doses", "vignettes, oral morphine", "vignettes, other conversions",
                              "random doses, oral morphine", "random doses, other conversions")
KEY_META = {
    "n_models": ("count", "all figures", "design", "design", "n/a", "n/a", "count"),
    "tight_lower_x": ("ratio:2", "Figures 1, 8", "band", "design", "n/a", "Nielsen", "limit (lower, tight)"),
    "tight_upper_x": ("ratio:2", "Figures 1, 8", "band", "design", "n/a", "Nielsen", "limit (upper, tight)"),
    "broad_lower_x": ("ratio:1", "Figures 1, 8", "band", "design", "n/a", "Nielsen", "limit (lower, broad)"),
    "broad_upper_x": ("ratio:1", "Figures 1, 8", "band", "design", "n/a", "Nielsen", "limit (upper, broad)"),
    "naive_vig_best_tight_pct": ("percent:1", "Figure 1", "tight", _V, "naive", "Nielsen", "best model"),
    "naive_vig_worst_tight_pct": ("percent:1", "Figure 1", "tight", _V, "naive", "Nielsen", "worst model"),
    "naive_vig_mean_tight_pct": ("percent:1", "Figure 1", "tight", _V, "naive", "Nielsen", "mean of models"),
    "naive_vig_best_broad_pct": ("percent:1", "Figure 1", "broad", _V, "naive", "Nielsen", "best model"),
    "naive_rd_best_tight_pct": ("percent:1", "Figure 8", "tight", _R, "naive", "Nielsen", "best model"),
    "naive_rd_worst_tight_pct": ("percent:1", "Figure 8", "tight", _R, "naive", "Nielsen", "worst model"),
    "naive_rd_mean_tight_pct": ("percent:1", "Figure 8", "tight", _R, "naive", "Nielsen", "mean of models"),
    "naive_rd_best_broad_pct": ("percent:1", "Figure 8", "broad", _R, "naive", "Nielsen", "best model"),
    "naive_rank_spearman_rho": ("ratio:2", "Figure 8", "rank_corr_tight", "both", "naive", "Nielsen", "rho"),
    "naive_rank_spearman_p": ("p", "Figure 8", "rank_corr_tight", "both", "naive", "Nielsen", "p"),
    "naive_rank_n_models": ("count", "Figure 8", "rank_corr_tight", "both", "naive", "Nielsen", "count of models"),
    "n_vignette_scenarios": ("count", "Figures 1-3, 7", "design", "vignettes", "n/a", "n/a", "count of scenarios"),
    "n_vignette_wordings": ("count", "Figure 1", "design", "vignettes", "n/a", "n/a", "count of wordings per scenario"),
    "n_vignette_prompts": ("count", "Figure 1", "design", "vignettes", "n/a", "n/a", "count of prompts"),
    "n_sig_cv_chi2_vignettes": ("count", "Figure 2", "cv_chi2", _V, "naive", "Nielsen", "count of significant models"),
    "vig_mean_correct_pct": ("percent:1", "Figure 3", "share_correct", _V, "naive", "Nielsen", "mean of models"),
    "vig_mean_precise_wrong_pct": ("percent:1", "Figure 3", "share_precise_wrong", _V, "naive", "Nielsen", "mean of models"),
    "vig_mean_imprecise_pct": ("percent:1", "Figure 3", "share_imprecise", _V, "naive", "Nielsen", "mean of models"),
    "n_model_scenario_cells": ("count", "Figures 4, 6", "design", _V, "naive", "n/a", "count of model-scenario cells"),
    "n_scenarios_per_model": ("count", "Figure 5", "design", _V, "naive", "n/a", "count of scenarios per model"),
    "icc_total_sig_pairs": ("count", "Figure 6", "icc", _V, "naive", "Nielsen", "count of significant wording pairs"),
    "icc_total_pairs": ("count", "Figure 6", "icc", _V, "naive", "Nielsen", "count of wording pairs"),
    "icc_max_pairs": ("count", "Figure 6", "icc", _V, "naive", "Nielsen", "maximum wording pairs per cell"),
    "mean_icc": ("ratio:2", "Figure 4", "icc", _V, "naive", "Nielsen", "mean"),
    "median_icc": ("ratio:2", "Figure 4", "icc", _V, "naive", "Nielsen", "median"),
    "n_models_icc_gt_half": ("count", "Figure 4", "icc", _V, "naive", "Nielsen", "count of models above 0.5"),
    "n_models_with_sig_icc": ("count", "Figure 6", "icc", _V, "naive", "Nielsen", "count of significant models"),
    "mean_between_var_pct": ("percent:1", "Figure 4", "between_var", _V, "naive", "Nielsen", "mean of models"),
    "mean_kendall_w": ("ratio:3", "Figure 7", "kendall_w", _V, "naive", "Nielsen", "mean"),
    "median_kendall_w": ("ratio:3", "Figure 7", "kendall_w", _V, "naive", "Nielsen", "median"),
    "n_sig_kendall_w": ("count", "Figure 7", "kendall_w", _V, "naive", "Nielsen", "count of significant scenarios"),
    "spearman_rho_variance_icc": ("ratio:2", "Figure 5", "variance_icc_rho", _V, "naive", "Nielsen", "rho"),
    "spearman_p_variance_icc": ("p", "Figure 5", "variance_icc_rho", _V, "naive", "Nielsen", "p"),
    "n_positive_rho_models": ("count", "Figure 5", "variance_icc_rho", _V, "naive", "Nielsen", "count of models with positive rho"),
    "n_total_rho_models": ("count", "Figure 5", "variance_icc_rho", _V, "naive", "Nielsen", "count of models"),
    "n_sig_rho_models": ("count", "Figure 5", "variance_icc_rho", _V, "naive", "Nielsen", "count of significant models"),
    "n_opioid_pairs": ("count", "Figures 8, 9, 12", "design", _R, "n/a", "n/a", "count of opioid pairs"),
    "n_sig_cv_chi2_rd": ("count", "Figure 9", "cv_chi2", _R, "naive", "Nielsen", "count of significant models"),
    "rd_mean_correct_pct": ("percent:1", "Figure 9", "share_correct", _R, "naive", "Nielsen", "mean of models"),
    "rd_mean_precise_wrong_pct": ("percent:1", "Figure 9", "share_precise_wrong", _R, "naive", "Nielsen", "mean of models"),
    "rd_mean_dose_dep_neg_pct": ("percent:1", "Figure 9", "share_dose_dep_neg", _R, "naive", "Nielsen", "mean of models"),
    "n_matched_pairs": ("count", "Figure 10", "design", "both", "naive", "n/a", "count of matched conversions"),
    "cross_pooled_broad_p_rd_text": ("p", "Figure 10", "broad", "random doses vs vignettes", "naive", "Nielsen", "p"),
    "cross_pooled_broad_p_rd_empty": ("p", "Figure 10", "broad", "random doses vs empty prompts", "naive", "Nielsen", "p"),
    "cross_pooled_broad_p_empty_text": ("p", "Figure 10", "broad", "empty prompts vs vignettes", "naive", "Nielsen", "p"),
    "cross_spearman_rho_rd_text_broad": ("ratio:2", "Figure 10", "broad", "random doses vs vignettes", "naive", "Nielsen", "mean rho"),
    "cross_n_sig_spearman_rd_text": ("count", "Figure 10", "broad", "random doses vs vignettes", "naive", "Nielsen", "count of significant models"),
    "nielsen_n_sig_tight_vignettes": ("count", "Figure 11", "tight", _V, "naive -> reference-grounded", "Nielsen", "count of significant models"),
    "nielsen_mean_tight_improvement_vignettes": ("pp:1", "Figure 11", "accuracy_gain_tight", _V, "naive -> reference-grounded", "Nielsen", "mean of models"),
    "nielsen_pooled_tight_p_vignettes": ("p", "Figure 11", "tight", _V, "naive -> reference-grounded", "Nielsen", "p"),
    "nielsen_n_sig_tight_rd": ("count", "Figure 11", "tight", _R, "naive -> reference-grounded", "Nielsen", "count of significant models"),
    "nielsen_mean_tight_improvement_rd": ("pp:1", "Figure 11", "accuracy_gain_tight", _R, "naive -> reference-grounded", "Nielsen", "mean of models"),
    "nielsen_pooled_tight_p_rd": ("p", "Figure 11", "tight", _R, "naive -> reference-grounded", "Nielsen", "p"),
    "best_rd_model_after_nielsen": ("text", "Figure 12", "tight", _R, "reference-grounded", "Nielsen", "best model (name)"),
    "nielsen_rd_best_tight_pct": ("percent:1", "Figure 12", "tight", _R, "reference-grounded", "Nielsen", "best model"),
    "nielsen_rd_mean_tight_pct": ("percent:1", "Figure 12", "tight", _R, "reference-grounded", "Nielsen", "mean of models"),
    "vig_ome_naive_tight_pct": ("percent:1", "Figure 13", "tight", _VO, "naive", "Nielsen", "mean of models"),
    "vig_ome_nielsen_tight_pct": ("percent:1", "Figure 13", "tight", _VO, "reference-grounded", "Nielsen", "mean of models"),
    "vig_ome_tight_improvement_pp": ("pp:1", "Figure 13", "accuracy_gain_tight", _VO, "naive -> reference-grounded", "Nielsen", "mean of models"),
    "vig_nonome_tight_improvement_pp": ("pp:1", "Figure 13", "accuracy_gain_tight", _VN, "naive -> reference-grounded", "Nielsen", "mean of models"),
    "rd_ome_tight_improvement_pp": ("pp:1", "Figure 13", "accuracy_gain_tight", _RO, "naive -> reference-grounded", "Nielsen", "mean of models"),
    "rd_nonome_tight_improvement_pp": ("pp:1", "Figure 13", "accuracy_gain_tight", _RN, "naive -> reference-grounded", "Nielsen", "mean of models"),
    "nielsen_vig_ome_pooled_p": ("p", "Figure 13", "tight", _VO, "naive -> reference-grounded", "Nielsen", "p"),
    "nielsen_vig_nonome_pooled_p": ("p", "Figure 13", "tight", _VN, "naive -> reference-grounded", "Nielsen", "p"),
    "nielsen_rd_ome_pooled_p": ("p", "Figure 13", "tight", _RO, "naive -> reference-grounded", "Nielsen", "p"),
    "nielsen_rd_nonome_pooled_p": ("p", "Figure 13", "tight", _RN, "naive -> reference-grounded", "Nielsen", "p"),
    "dosing_n_sig_drop_tight": ("count", "Figure 14", "tight", _R, "reference-grounded mg/day -> mg/4 h", "Nielsen", "count of significant models"),
    "dosing_mean_tight_drop_pp": ("pp:1", "Figure 14", "accuracy_drop_tight", _R, "reference-grounded mg/day -> mg/4 h", "Nielsen", "mean of models"),
    "dosing_pooled_tight_p": ("p", "Figure 14", "tight", _R, "reference-grounded mg/day -> mg/4 h", "Nielsen", "p"),
    "units_n_sig_recovery_tight": ("count", "Figure 14", "tight", _R, "reference-grounded mg/4 h -> unit-augmented", "Nielsen", "count of significant models"),
    "units_mean_tight_recovery_pp": ("pp:1", "Figure 14", "accuracy_recovery_tight", _R, "reference-grounded mg/4 h -> unit-augmented", "Nielsen", "mean of models"),
    "units_pooled_tight_p": ("p", "Figure 14", "tight", _R, "reference-grounded mg/4 h -> unit-augmented", "Nielsen", "p"),
    "best_rd_model_after_all": ("text", "Figure 15", "tight", _R, "unit-augmented", "Nielsen", "best model (name)"),
    "best_rd_tight_after_all_pct": ("percent:1", "Figure 15", "tight", _R, "unit-augmented", "Nielsen", "best model"),
    "mean_rd_tight_after_all_pct": ("percent:1", "Figure 15", "tight", _R, "unit-augmented", "Nielsen", "mean of models"),
    "best_vig_model_after_all": ("text", "Figure 15", "tight", _V, "unit-augmented", "Nielsen", "best model (name)"),
    "best_vig_tight_after_all_pct": ("percent:1", "Figure 15", "tight", _V, "unit-augmented", "Nielsen", "best model"),
    "mean_vig_tight_after_all_pct": ("percent:1", "Figure 15", "tight", _V, "unit-augmented", "Nielsen", "mean of models"),
}

# Significance counts: key -> ([audit families], "tests" | "groups" | "same").  "tests" sums n_sig_bonf,
# "groups" n_groups_sig_bonf (results_revision/multiple_comparisons_audit.csv); "same" = already
# Bonferroni-adjusted (no _bonf key).  Families as in revision_private/figures_appendix/build_manuscript_numbers.py.
_N = "results_nielsen/"
BONF_FAMILIES = {
    "n_sig_cv_chi2_vignettes": ([_N + "cv_accuracy_chi2:Figure 2"], "tests"),
    "n_sig_cv_chi2_rd": (["figure9:cv_chi2"], "tests"),
    "icc_total_sig_pairs": ([], "same"),
    "n_models_with_sig_icc": ([_N + "icc_analysis:f_test"], "groups"),
    "n_sig_kendall_w": ([_N + "kendall_w_vignettes:kendall_p"], "tests"),
    "n_sig_rho_models": ([_N + "variance_icc_spearman:variance_vs_icc"], "tests"),
    "cross_n_sig_spearman_rd_text": ([_N + "cross_experiment_matched_spearman:random_dose-text_vignette:broad"], "tests"),
    "nielsen_n_sig_tight_vignettes": ([_N + "wilcoxon_nielsen_effect_vignettes:tight"], "tests"),
    "nielsen_n_sig_tight_rd": ([_N + "wilcoxon_nielsen_effect_random_dose:tight"], "tests"),
    "dosing_n_sig_drop_tight": ([_N + "wilcoxon_dosing_unit_effect:tight"], "tests"),
    "units_n_sig_recovery_tight": ([_N + "wilcoxon_units_effect_random_dose:tight"], "tests"),
}

AUDIT_CSV = "results_revision/multiple_comparisons_audit.csv"


def bonferroni_stats(audit_path: str = AUDIT_CSV) -> dict:
    """{key + '_bonf': count of significant models after Bonferroni within the family} for every
    significance count that multiple_comparisons gives. Empty if the audit CSV is absent."""
    if not os.path.exists(audit_path):
        return {}
    aud = pd.read_csv(audit_path).set_index("family")
    out = {}
    for key, (fams, how) in BONF_FAMILIES.items():
        if how == "same":
            continue
        col = "n_sig_bonf" if how == "tests" else "n_groups_sig_bonf"
        out[key + BONF_SUFFIX] = int(sum(aud.loc[f, col] for f in fams))
    return out


def build_meta(stats: dict, extra: dict | None = None) -> dict:
    """figure_stats_meta: for every key of ``stats`` (the flat map, with any *_bonf keys): value,
    format, display, display_html and the structured fields. Unknown keys raise (keep KEY_META complete)."""
    meta = {}
    extra = extra or {}
    for key, value in stats.items():
        if key in extra:        # caption-only keys: {format, figure}
            e = extra[key]
            meta[key] = dict(value=value, format=e["format"], display=fmt_value(value, e["format"]),
                             display_html=fmt_html(value, e["format"]), measure="caption quantity",
                             experiment="n/a", condition="n/a", reference="n/a", statistic="caption-derived",
                             adjusted="n/a", figure=e["figure"], scope="caption-only")
            continue
        base = key[:-len(BONF_SUFFIX)] if key.endswith(BONF_SUFFIX) and key[:-len(BONF_SUFFIX)] in BONF_FAMILIES else key
        if base not in KEY_META:
            raise KeyError(f"figure_stats key without metadata: {key}")
        fmt, figure, measure, experiment, condition, reference, statistic = KEY_META[base]
        if base != key:
            adjusted = "Bonferroni"
        elif base in BONF_FAMILIES:
            adjusted = "Bonferroni (already, within each cell)" if BONF_FAMILIES[base][1] == "same" else "unadjusted"
        elif fmt == "p":
            adjusted = "unadjusted"
        else:
            adjusted = "n/a"
        meta[key] = dict(value=value, format=fmt, display=fmt_value(value, fmt), display_html=fmt_html(value, fmt),
                         measure=measure, experiment=experiment, condition=condition, reference=reference,
                         statistic=statistic, adjusted=adjusted, figure=figure, scope="figure_stats")
    return meta


_MEASURE_WORDS = {
    "tight": "within ±15% of the expected dose", "broad": "within 0.6–1.7× of the expected dose",
    "share_correct": "answers within ±15% (correct)", "share_precise_wrong": "precise but wrong answers",
    "share_imprecise": "imprecise answers", "share_dose_dep_neg": "answers with negative dose dependence",
    "icc": "intraclass correlation (wording)", "between_var": "variance between wordings",
    "kendall_w": "Kendall's W", "variance_icc_rho": "Spearman correlation of variance with ICC",
    "rank_corr_tight": "Spearman correlation of model ranks (vignettes vs random doses)",
    "cv_chi2": "CV-by-accuracy chi-square test", "accuracy_gain_tight": "gain in accuracy within ±15%",
    "accuracy_drop_tight": "drop in accuracy within ±15%", "accuracy_recovery_tight": "recovery of accuracy within ±15%",
    "band": "accuracy band", "design": "design quantity",
}


def make_description(m: dict) -> str:
    """Description assembled from the structured fields only (no free text)."""
    if m.get("scope") == "caption-only":
        return f"caption-only quantity ({m['figure']} caption)"
    parts = [f"{m['statistic']}: {_MEASURE_WORDS.get(m['measure'], m['measure'])}"]
    for f in ("experiment", "condition", "reference"):
        if m[f] not in ("n/a", "design"):
            parts.append(m[f] if f != "reference" else f"vs {m[f]}")
    if m["adjusted"] not in ("n/a",):
        parts.append(m["adjusted"])
    return "; ".join(parts)


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("── Extracting figure stats ──")
    stats = compute_stats()
    if "--extended" in sys.argv:
        # flat map stays flat: the Bonferroni-adjusted counts are added as *_bonf keys
        # (results_revision/multiple_comparisons_audit.csv; revised-baseline runs only)
        stats.update(bonferroni_stats())

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # display formats and structured fields (adds the Bonferroni counts as *_bonf keys, meta file only)
    meta = build_meta(stats)
    write_json_atomic(META_PATH, meta)

    print(f"✓ Saved {len(stats)} stats to {OUTPUT_PATH}; {len(meta)} entries to {META_PATH}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run()
