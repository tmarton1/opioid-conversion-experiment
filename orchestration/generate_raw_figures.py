"""
Generate all intermediate figures written to raw_figures/.

These are exploratory/diagnostic figures produced during analysis — batch grids,
heatmaps, model comparison bars, ICC plots, scenario strips, and the statistical
methods report.  They differ from the 15 publication figures in summary_w_figures/
pub_figures/, which are produced by generate_summaries.py.

Steps (each maps to a --section value)
---------------------------------------
1. pair_batches       — OpioidPairBatch grid + heatmap figures for all 6 conditions
2. vignette_batch     — VignetteBatch grid for all prompt conditions
3. model_comparison   — Model comparison bar + scatter (random-dose batch means)
4. explain_units      — Explain-units bar chart (mg/day vs mg/4h vs units explained)
5. icc                — ICC heatmap, bubble grid, Kendall W summary
6. scenario_strips    — Per-scenario strip plots with Kendall W annotations
7. consistency        — 2×3 consistency illustration figure (matplotlib)
8. statistical_methods — Full statistical methods HTML report

Usage
-----
    python orchestration/generate_raw_figures.py [--section SECTION]

    SECTION choices:
        pair_batches | vignette_batch | model_comparison | explain_units
        | icc | scenario_strips | consistency | statistical_methods | all
    Default: all

Run from the repository root:
    python orchestration/generate_raw_figures.py [--section SECTION]

Prerequisites
-------------
All of the following results/ CSVs must already exist (produced by run_analysis.py):
    experiment_results_by_ome.csv
    pair_model_analysis.csv
    opioid_pair_consensus.csv
    model_batch_means.csv
    model_batch_means_explain_units.csv
    reworded_vignettes_results_by_ome.csv
    reworded_vignettes_model_analysis.csv
    reworded_vignettes_model_consensus.csv
    icc_analysis.csv
    kendall_w_vignettes.csv
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow running directly from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import RESULTS_DIR, RAW_FIGURES_DIR


PAIR_BATCH_COMBINATIONS = [
    {"ome": None, "include_nielsen": False, "name": "all_n0"},
    {"ome": None, "include_nielsen": True,  "name": "all_n1"},
    {"ome": True,  "include_nielsen": False, "name": "ome_n0"},
    {"ome": True,  "include_nielsen": True,  "name": "ome_n1"},
    {"ome": False, "include_nielsen": False, "name": "pairs_n0"},
    {"ome": False, "include_nielsen": True,  "name": "pairs_n1"},
]

VIGNETTE_FILTER_SPECS = [
    {"include_nielsen": False, "include_examples": False,
     "ask_for_units": False, "explain_units": False, "batch_name": "naive"},
    {"include_nielsen": True,  "include_examples": False,
     "ask_for_units": False, "explain_units": False, "batch_name": "nielsen_only"},
    {"include_nielsen": True,  "include_examples": False,
     "ask_for_units": True,  "explain_units": True,  "batch_name": "nielsen_units_explained"},
]


def _parse_args():
    choices = [
        "pair_batches", "vignette_batch", "model_comparison", "explain_units",
        "icc", "scenario_strips", "consistency", "statistical_methods", "all",
    ]
    p = argparse.ArgumentParser(description="Generate raw_figures/ intermediate figures")
    p.add_argument(
        "--section",
        choices=choices,
        default="all",
        help="Which figure group to generate (default: all)",
    )
    return p.parse_args()


# ── 1. Opioid pair batch grids and heatmaps ───────────────────────────────

async def run_pair_batches():
    """Batch grids + heatmaps for all 6 opioid-pair conditions."""
    from analysis.batch import OpioidPairBatch
    from visualize.batch_visualizer import OpioidPairBatchVisualizer
    from visualize.opioid_pair_batch_heatmap_visualizer import OpioidPairBatchHeatmapVisualizer
    from visualize.visualization_config import convert_to_pdf_auto_size

    os.makedirs(RAW_FIGURES_DIR, exist_ok=True)

    for combo in PAIR_BATCH_COMBINATIONS:
        name = combo["name"]
        filter_spec: dict = {
            "include_nielsen": combo["include_nielsen"],
            "include_examples": False,
            "exclude_self_pairs": True,
        }
        if combo["ome"] is not None:
            filter_spec["ome"] = combo["ome"]

        try:
            batch = OpioidPairBatch.from_filter_specs(
                experiment_results_csv=f"{RESULTS_DIR}/experiment_results_by_ome.csv",
                variable_model_csv=f"{RESULTS_DIR}/pair_model_analysis.csv",
                consensus_csv=f"{RESULTS_DIR}/opioid_pair_consensus.csv",
                filter_specs=[filter_spec],
            )

            # ── Batch grid (one row per opioid pair) ──────────────────────
            vis = OpioidPairBatchVisualizer(batch, row_width=600, row_height=300)
            batch_html = os.path.join(RAW_FIGURES_DIR, f"{name}_batch.html")
            vis.create_grid(output_html=batch_html, show_inline=False)
            batch_pdf = os.path.join(RAW_FIGURES_DIR, f"{name}_batch.pdf")
            await convert_to_pdf_auto_size(batch_html, batch_pdf, scale=2.0)
            print(f"✓ {name}_batch")

            # ── Heatmaps (weight CV, broad range, tight range) ─────────────
            heatmap_vis = OpioidPairBatchHeatmapVisualizer(
                batch, cell_size=35,
                width_padding=0, height_padding=0,
                cell_text_size="6pt", label_text_size="7pt",
            )
            heatmap_html = os.path.join(RAW_FIGURES_DIR, f"{name}_heatmaps.html")
            heatmap_vis.create_heatmaps(output_html=heatmap_html)
            heatmap_pdf = os.path.join(RAW_FIGURES_DIR, f"{name}_heatmaps.pdf")
            await convert_to_pdf_auto_size(heatmap_html, heatmap_pdf, scale=3.0)
            print(f"✓ {name}_heatmaps")

        except Exception as e:
            print(f"✗ {name}: {e}")
            import traceback; traceback.print_exc()


# ── 2. Vignette batch grid ────────────────────────────────────────────────

async def run_vignette_batch():
    """Batch grid for all vignette prompt conditions."""
    from analysis.batch import VignetteBatch
    from visualize.batch_visualizer import VignetteBatchVisualizer
    from visualize.visualization_config import convert_to_pdf_auto_size

    os.makedirs(RAW_FIGURES_DIR, exist_ok=True)

    batch = VignetteBatch.from_filter_specs(
        experiment_results_csv=f"{RESULTS_DIR}/reworded_vignettes_results_by_ome.csv",
        variable_model_csv=f"{RESULTS_DIR}/reworded_vignettes_model_analysis.csv",
        consensus_csv=f"{RESULTS_DIR}/reworded_vignettes_model_consensus.csv",
        filter_specs=VIGNETTE_FILTER_SPECS,
    )

    vis = VignetteBatchVisualizer(batch, row_width=900, row_height=300)
    output_html = os.path.join(RAW_FIGURES_DIR, "all_reworded_vignette_batch.html")
    vis.create_grid(output_html=output_html, show_inline=False)
    output_pdf = os.path.join(RAW_FIGURES_DIR, "all_reworded_vignette_batch.pdf")
    await convert_to_pdf_auto_size(output_html, output_pdf, scale=2.0)
    print("✓ all_reworded_vignette_batch")


# ── 3. Model comparison bar chart ─────────────────────────────────────────

async def run_model_comparison():
    """Model comparison bar + scatter for random-dose batch means."""
    from visualize.opioid_model_comparison_bar_visualizer import ModelComparisonFigureBuilder

    os.makedirs(RAW_FIGURES_DIR, exist_ok=True)

    builder = ModelComparisonFigureBuilder(
        batch_means_csv=f"{RESULTS_DIR}/model_batch_means.csv",
    )
    await builder.save(
        os.path.join(RAW_FIGURES_DIR, "model_comparison_bar.html"),
        os.path.join(RAW_FIGURES_DIR, "model_comparison_bar.pdf"),
    )
    builder.save_scatter_figures_pdf(
        os.path.join(RAW_FIGURES_DIR, "model_comparison_scatter.pdf"),
    )
    print("✓ model_comparison_bar / scatter")


# ── 4. Explain-units bar chart ────────────────────────────────────────────

async def run_explain_units():
    """Explain-units bar: mg/day vs mg/4 h vs units-explained."""
    from visualize.explain_units_visualizer import ExplainUnitsVisualizer
    from common.models import CONSISTENT_MODELS

    os.makedirs(RAW_FIGURES_DIR, exist_ok=True)

    vis = ExplainUnitsVisualizer(
        batch_means_csv=f"{RESULTS_DIR}/model_batch_means.csv",
        batch_means_explain_units_csv=f"{RESULTS_DIR}/model_batch_means_explain_units.csv",
        models=CONSISTENT_MODELS,
    )
    await vis.save(
        os.path.join(RAW_FIGURES_DIR, "explain_units_bar.html"),
        os.path.join(RAW_FIGURES_DIR, "explain_units_bar.pdf"),
    )
    print("✓ explain_units_bar")


# ── 5. ICC figures ────────────────────────────────────────────────────────

async def run_icc():
    """ICC heatmap, bubble grid, and Kendall W summary."""
    from visualize.icc_visualizer import save_icc_figures

    await save_icc_figures(
        icc_csv=f"{RESULTS_DIR}/icc_analysis.csv",
        kendall_csv=f"{RESULTS_DIR}/kendall_w_vignettes.csv",
        figures_dir=RAW_FIGURES_DIR,
    )
    print("✓ icc_heatmap / icc_bubble_grid / kendall_w_summary")


# ── 6. Scenario strip plots ───────────────────────────────────────────────

async def run_scenario_strips():
    """Per-scenario strip plots with Kendall W annotations."""
    from visualize.scenario_strip_visualizer import save_all_scenario_strips

    await save_all_scenario_strips(
        results_csv=f"{RESULTS_DIR}/reworded_vignettes_results.csv",
        figures_dir=RAW_FIGURES_DIR,
    )
    print("✓ scenario_strips (all scenarios)")


# ── 7. Consistency illustration figure ───────────────────────────────────

def run_consistency():
    """2×3 matplotlib figure illustrating consistent/noisy/slope examples."""
    from visualize.consistency_figure import build_consistency_figure

    build_consistency_figure(
        results_csv=f"{RESULTS_DIR}/experiment_results.csv",
        vignette_csv=f"{RESULTS_DIR}/reworded_vignettes_results.csv",
        output_dir=RAW_FIGURES_DIR,
    )
    print("✓ consistency_examples")


# ── 8. Statistical methods HTML report ───────────────────────────────────

def run_statistical_methods():
    """Full statistical methods report written to raw_figures/statistical_methods.html."""
    import importlib
    import analysis.statistical_methods as sm_mod
    importlib.reload(sm_mod)
    # generate() writes to FIGURES_DIR defined in the module (raw_figures/).
    sm_mod.generate()
    print("✓ statistical_methods.html")


# ── Main ──────────────────────────────────────────────────────────────────

async def _run_async(section: str):
    if section in ("pair_batches", "all"):
        await run_pair_batches()
    if section in ("vignette_batch", "all"):
        await run_vignette_batch()
    if section in ("model_comparison", "all"):
        await run_model_comparison()
    if section in ("explain_units", "all"):
        await run_explain_units()
    if section in ("icc", "all"):
        await run_icc()
    if section in ("scenario_strips", "all"):
        await run_scenario_strips()


def main():
    args = _parse_args()
    section = args.section

    # Sync-only sections
    if section in ("consistency", "all"):
        run_consistency()
    if section in ("statistical_methods", "all"):
        run_statistical_methods()

    # Async sections (Bokeh + Playwright conversions)
    asyncio.run(_run_async(section))


if __name__ == "__main__":
    main()
