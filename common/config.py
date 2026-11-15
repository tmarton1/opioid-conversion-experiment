"""
Experiment-wide constants shared across analysis, visualisation, and reporting.

These are the canonical definitions — import from here, not from
visualize.visualization_config.
"""

import os

# ── Output directories ────────────────────────────────────────────────────
# Every results / figures path in the pipeline is built from these, so one
# run can be pointed at another score's results without mixing outputs:
#
#     RESULTS_DIR=results_all FIGURES_DIR=figures_all python orchestration/run_analysis.py
#
# Defaults reproduce the original layout. Paths are relative to the
# repository root (scripts are run from there) unless given as absolute.
#
#   RESULTS_DIR      analysis input and output CSVs           (default: results)
#   FIGURES_DIR      publication figures, figure_stats.json,
#                    summary tables and the annotated PDF     (default: summary_w_figures)
#   RAW_FIGURES_DIR  intermediate / exploratory figures       (default: raw_figures)

RESULTS_DIR: str = os.environ.get("RESULTS_DIR", "results")
FIGURES_DIR: str = os.environ.get("FIGURES_DIR", "summary_w_figures")
RAW_FIGURES_DIR: str = os.environ.get("RAW_FIGURES_DIR", "raw_figures")

# ── Accuracy range thresholds ─────────────────────────────────────────────
# A model's answer is accepted if (observed / expected) falls within the
# stated multiplicative bounds.  Both bounds are ratio values, not percentages.

TIGHT_RANGE_LOWER: float = 0.85   # observed/expected must be ≥ this
TIGHT_RANGE_UPPER: float = 1.15   # observed/expected must be ≤ this

BROAD_RANGE_LOWER: float = 0.6    # observed/expected must be ≥ this
BROAD_RANGE_UPPER: float = 1.7    # observed/expected must be ≤ this

# ── Weighting epsilon ─────────────────────────────────────────────────────
# Floor applied to proportions before computing inverse-variance weights,
# preventing division-by-zero when a model gets every conversion right or
# wrong in a batch.

EPSILON_FOR_WEIGHT: float = 5e-2
