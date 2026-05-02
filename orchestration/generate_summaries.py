"""
Generate all summary documents in summary_w_figures/.

This script replaces the final summary-generation cells in opioid-conversion.ipynb.
Run this after run_analysis.py has completed all analysis steps.

Steps
-----
1. extract_figure_stats    — pull key numbers from analysis CSVs → JSON
2. accuracy_summary_tables — HTML/PDF accuracy comparison tables
3. comprehensive_figures   — all 15 publication figures (HTML/PDF/PNG)
4. annotated_pdf           — combined figure+text PDF report

Usage
-----
    python generate_summaries.py [--section SECTION]

    SECTION choices: stats | tables | figures | pdf | all
    Default: all

Run from the repository root:
    python orchestration/generate_summaries.py [--section SECTION]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio


def _parse_args():
    p = argparse.ArgumentParser(description="Generate summary documents")
    p.add_argument(
        "--section",
        choices=["stats", "tables", "figures", "pdf", "all"],
        default="all",
        help="Which summary step to run (default: all)",
    )
    return p.parse_args()


# ── 1. Extract figure stats ────────────────────────────────────────────────

def run_extract_stats():
    """Extract key data-driven numbers from analysis CSVs and save to JSON."""
    import orchestration.extract_figure_stats as extract_figure_stats
    extract_figure_stats.run()


# ── 2. Accuracy summary tables ────────────────────────────────────────────

async def run_tables():
    """Generate the accuracy summary HTML/PDF tables."""
    import importlib
    import visualize.summary_tables
    importlib.reload(visualize.summary_tables)
    from visualize.summary_tables import SummaryTablesBuilder

    os.makedirs("summary_w_figures", exist_ok=True)
    builder = SummaryTablesBuilder()
    await builder.save(
        "summary_w_figures/accuracy_summary_tables.html",
        "summary_w_figures/accuracy_summary_tables.pdf",
    )
    print("✓ Summary tables saved")


# ── 3. Comprehensive figures ───────────────────────────────────────────────

async def run_figures():
    """Generate all 15 publication figures as HTML/PDF/PNG."""
    import importlib
    import visualize.visualization_config
    import visualize.comprehensive_figures
    importlib.reload(visualize.visualization_config)
    importlib.reload(visualize.comprehensive_figures)
    from visualize.comprehensive_figures import ComprehensiveFigureBuilder

    os.makedirs("summary_w_figures", exist_ok=True)
    cfb = ComprehensiveFigureBuilder()
    await cfb.save_all()
    print("✓ Comprehensive figures saved")


# ── 4. Annotated PDF ──────────────────────────────────────────────────────

async def run_pdf():
    """Generate the combined annotated PDF (all figures + narrative text)."""
    import importlib
    import visualize.comprehensive_figures
    importlib.reload(visualize.comprehensive_figures)
    from visualize.comprehensive_figures import ComprehensiveFigureBuilder

    os.makedirs("summary_w_figures", exist_ok=True)
    cfb = ComprehensiveFigureBuilder()
    await cfb.save_annotated_pdf()
    print("✓ Annotated PDF saved")


# ── Main ───────────────────────────────────────────────────────────────────

async def _main(section: str):
    if section in ("stats", "all"):
        run_extract_stats()
    if section in ("tables", "all"):
        await run_tables()
    if section in ("figures", "all"):
        await run_figures()
    if section in ("pdf", "all"):
        await run_pdf()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_main(args.section))
