"""
Experiment-wide constants shared across analysis, visualisation, and reporting.

These are the canonical definitions — import from here, not from
visualize.visualization_config.
"""

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
