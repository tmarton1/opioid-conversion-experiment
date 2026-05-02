"""
Extract key data-driven numbers from analysis CSVs and save to
summary_w_figures/figure_stats.json.

This JSON is consumed by ComprehensiveFigureBuilder.save_annotated_pdf()
to fill {placeholder} fields in FIGURE_METADATA descriptions so that every
number in the final annotated PDF comes from real data rather than being
hardcoded.

Run via generate_summaries.py (section "stats"), or directly:
    python orchestration/extract_figure_stats.py
"""

from __future__ import annotations

import json
import os
import sys

# Allow running directly from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


RESULTS_DIR = "results"
OUTPUT_PATH = "summary_w_figures/figure_stats.json"


def _read(name: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [warn] {name} not found — skipping")
    return None


def compute_stats() -> dict:
    stats: dict = {}

    # ── Accuracy range constants ──────────────────────────────────────
    from common.config import (
        TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER,
        BROAD_RANGE_LOWER, BROAD_RANGE_UPPER,
    )

    # Format as plain numbers for use in description strings (e.g. "0.85")
    stats["tight_lower_x"] = f"{TIGHT_RANGE_LOWER:g}"
    stats["tight_upper_x"] = f"{TIGHT_RANGE_UPPER:g}"
    stats["broad_lower_x"] = f"{BROAD_RANGE_LOWER:g}"
    stats["broad_upper_x"] = f"{BROAD_RANGE_UPPER:g}"

    # ── Vignette experiment: unique scenario keys ─────────────────────
    vdf = _read("reworded_vignettes_model_analysis.csv")
    if vdf is not None:
        naive_vdf = vdf[
            (vdf.get("include_nielsen", vdf.get("include_nielson", pd.Series(False, index=vdf.index))) == False) &
            (vdf.get("explain_units", pd.Series(False, index=vdf.index)) == False) &
            (vdf.get("has_vignette_text", pd.Series(True, index=vdf.index)) == True)
        ]
        stats["n_vignette_scenarios"] = int(naive_vdf["scenario_key"].nunique())
    else:
        stats["n_vignette_scenarios"] = "?"

    # ── Random-dose experiment: unique opioid pairs ───────────────────
    pdf = _read("pair_model_analysis.csv")
    if pdf is not None:
        naive_pdf = pdf[
            (pdf.get("include_nielsen", pdf.get("include_nielson", pd.Series(False, index=pdf.index))) == False) &
            (pdf.get("include_examples", pd.Series(False, index=pdf.index)) == False)
        ]
        stats["n_opioid_pairs"] = int(naive_pdf["opioid_pair"].nunique())
    else:
        stats["n_opioid_pairs"] = "?"

    # ── Cross-experiment matched pairs ────────────────────────────────
    cem = _read("cross_experiment_matched_means.csv")
    if cem is not None:
        pair_col = next(
            (c for c in ("opioid_pair", "pair", "unit") if c in cem.columns), None
        )
        if pair_col:
            stats["n_matched_pairs"] = int(cem[pair_col].nunique())
        else:
            stats["n_matched_pairs"] = int(len(cem))
    else:
        stats["n_matched_pairs"] = "?"

    # ── ICC analysis ──────────────────────────────────────────────────
    idf = _read("icc_analysis.csv")
    if idf is not None:
        stats["n_model_scenario_cells"] = int(len(idf))

        # Scenarios per model (assume same count across models)
        if "model" in idf.columns and "scenario_key" in idf.columns:
            per_model_counts = idf.groupby("model")["scenario_key"].nunique()
            stats["n_scenarios_per_model"] = int(per_model_counts.median())
        else:
            stats["n_scenarios_per_model"] = "?"

        # Pairwise sig_pairs stats
        if "sig_pairs" in idf.columns and "total_pairs" in idf.columns:
            stats["icc_total_sig_pairs"] = int(idf["sig_pairs"].sum())
            stats["icc_total_pairs"] = int(idf["total_pairs"].sum())
            # Modal value of total_pairs per cell (i.e. C(k,2) for k wordings)
            stats["icc_max_pairs"] = int(idf["total_pairs"].mode().iloc[0])
        else:
            stats["icc_total_sig_pairs"] = "?"
            stats["icc_total_pairs"] = "?"
            stats["icc_max_pairs"] = 6  # C(4,2) — default for 4 wordings per scenario

        # Mean and median ICC
        if "icc" in idf.columns:
            stats["mean_icc"] = round(float(idf["icc"].mean()), 3)
            stats["median_icc"] = round(float(idf["icc"].median()), 3)
            # Models with at least one significant F-test
            if "f_pvalue" in idf.columns and "model" in idf.columns:
                n_sig_models = int(
                    (idf.groupby("model")["f_pvalue"].min() < 0.05).sum()
                )
                stats["n_models_with_sig_icc"] = n_sig_models
    else:
        stats["n_model_scenario_cells"] = "?"
        stats["n_scenarios_per_model"] = "?"
        stats["icc_total_sig_pairs"] = "?"
        stats["icc_total_pairs"] = "?"
        stats["icc_max_pairs"] = 6

    # ── Model count ───────────────────────────────────────────────────
    try:
        from common.models import CONSISTENT_MODELS
        stats["n_models"] = len(CONSISTENT_MODELS)
    except ImportError:
        if pdf is not None and "model" in pdf.columns:
            stats["n_models"] = int(pdf["model"].nunique())
        else:
            stats["n_models"] = "?"

    # ── Kendall's W summary ───────────────────────────────────────────
    kdf = _read("kendall_w_vignettes.csv")
    if kdf is not None:
        if "kendall_w" in kdf.columns:
            stats["mean_kendall_w"] = round(float(kdf["kendall_w"].mean()), 3)
            stats["median_kendall_w"] = round(float(kdf["kendall_w"].median()), 3)
        if "p_value" in kdf.columns:
            stats["n_sig_kendall_w"] = int((kdf["p_value"] < 0.05).sum())

    # ── Variance–ICC Spearman summary ─────────────────────────────────
    spdf = _read("variance_icc_spearman.csv")
    if spdf is not None:
        overall = spdf[
            (spdf.get("stat_type", pd.Series("", index=spdf.index)) == "variance_vs_icc") &
            (spdf.get("model", pd.Series("", index=spdf.index)) == "OVERALL")
        ]
        if not overall.empty and "rho" in overall.columns:
            stats["spearman_rho_variance_icc"] = round(float(overall["rho"].iloc[0]), 3)
            stats["spearman_p_variance_icc"] = round(float(overall["p_value"].iloc[0]), 4)

        per_model = spdf[
            (spdf.get("stat_type", pd.Series("", index=spdf.index)) == "variance_vs_icc") &
            (spdf.get("model", pd.Series("", index=spdf.index)) != "OVERALL")
        ]
        if not per_model.empty and "rho" in per_model.columns:
            stats["n_positive_rho_models"] = int((per_model["rho"] > 0).sum())
            stats["n_total_rho_models"] = int(len(per_model))

    # ── Naive accuracy summaries (best model) ────────────────────────
    batch = _read("model_batch_means.csv")
    if batch is not None and "batch_name" in batch.columns:
        tight_col = next((c for c in batch.columns if "tight" in c.lower()), None)
        broad_col = next((c for c in batch.columns if "broad" in c.lower()), None)
        rd_naive = batch[batch["batch_name"] == "all_n0"]
        if not rd_naive.empty:
            if tight_col:
                stats["naive_rd_best_tight_pct"] = round(
                    float(rd_naive[tight_col].max()) * 100, 1
                )
            if broad_col:
                stats["naive_rd_best_broad_pct"] = round(
                    float(rd_naive[broad_col].max()) * 100, 1
                )

    batch_vig = _read("model_batch_means_vignettes.csv")
    if batch_vig is not None and "batch_name" in batch_vig.columns:
        tight_col = next((c for c in batch_vig.columns if "tight" in c.lower()), None)
        broad_col = next((c for c in batch_vig.columns if "broad" in c.lower()), None)
        vig_naive = batch_vig[batch_vig["batch_name"] == "all_n0"]
        if not vig_naive.empty:
            if tight_col:
                stats["naive_vig_best_tight_pct"] = round(
                    float(vig_naive[tight_col].max()) * 100, 1
                )
            if broad_col:
                stats["naive_vig_best_broad_pct"] = round(
                    float(vig_naive[broad_col].max()) * 100, 1
                )

    return stats


def run():
    os.makedirs("summary_w_figures", exist_ok=True)
    print("── Extracting figure stats ──")
    stats = compute_stats()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Saved {len(stats)} stats to {OUTPUT_PATH}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run()
