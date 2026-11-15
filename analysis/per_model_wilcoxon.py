"""
Per-model and pooled Wilcoxon signed-rank tests on comparison CSVs.

Given a comparison CSV with columns ``model``, ``tight_diff``, ``broad_diff``,
computes:
  - Per-model Wilcoxon signed-rank test (H₀: diffs symmetric around zero)
  - Pooled Wilcoxon across all models

Output CSV columns
------------------
model           Model name (or "POOLED" for the overall test)
tight_p         Wilcoxon p-value for tight_diff
tight_median    Median of tight_diff
tight_n         Number of tight_diff values
broad_p         Wilcoxon p-value for broad_diff
broad_median    Median of broad_diff
broad_n         Number of broad_diff values
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common.models import CONSISTENT_MODELS


def _run_wilcoxon(diffs: pd.Series):
    """Run Wilcoxon signed-rank on a Series.  Returns (p, median, mean, n)."""
    diffs = diffs.dropna()
    n = len(diffs)
    if n < 2:
        return None, None, None, n
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 1:
        return 1.0, 0.0, 0.0, n
    _, p = wilcoxon(nonzero, alternative="two-sided")
    return float(p), float(diffs.median()), float(diffs.mean()), n


def compute_per_model_wilcoxon(
    comparison_csv: str,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Compute per-model and pooled Wilcoxon signed-rank tests.

    Parameters
    ----------
    comparison_csv : str
        Path to a CSV with columns ``model``, ``tight_diff``, ``broad_diff``.
    output_csv : str, optional
        If given, saves the result DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        One row per model plus a final ``POOLED`` row.
    """
    df = pd.read_csv(comparison_csv)
    if "tight_diff" not in df.columns or "broad_diff" not in df.columns:
        raise ValueError(
            f"comparison CSV must contain tight_diff and broad_diff columns; "
            f"found: {list(df.columns)}"
        )
    # Restrict to the 13 consistently-run models before any test (including POOLED).
    df = df[df["model"].isin(CONSISTENT_MODELS)]

    rows = []

    # Per model
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        t_p, t_med, t_mean, t_n = _run_wilcoxon(mdf["tight_diff"])
        b_p, b_med, b_mean, b_n = _run_wilcoxon(mdf["broad_diff"])
        rows.append({
            "model": model,
            "tight_p": t_p,
            "tight_median": t_med,
            "tight_mean": t_mean,
            "tight_n": t_n,
            "broad_p": b_p,
            "broad_median": b_med,
            "broad_mean": b_mean,
            "broad_n": b_n,
        })

    # Pooled across all models
    t_p, t_med, t_mean, t_n = _run_wilcoxon(df["tight_diff"])
    b_p, b_med, b_mean, b_n = _run_wilcoxon(df["broad_diff"])
    rows.append({
        "model": "POOLED",
        "tight_p": t_p,
        "tight_median": t_med,
        "tight_mean": t_mean,
        "tight_n": t_n,
        "broad_p": b_p,
        "broad_median": b_med,
        "broad_mean": b_mean,
        "broad_n": b_n,
    })

    result = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def lookup_per_model_p(
    wilcoxon_csv: str,
    model: str,
    range_type: str = "tight",
) -> tuple[float | None, float | None, int]:
    """Look up pre-computed per-model Wilcoxon results.

    Returns (p_value, median_diff, n_diffs) — same signature as the
    former ``_per_model_wilcoxon()`` in comprehensive_figures.py.
    """
    df = pd.read_csv(wilcoxon_csv)
    row = df[df["model"] == model]
    if row.empty:
        return None, None, 0
    row = row.iloc[0]
    p = row.get(f"{range_type}_p")
    med = row.get(f"{range_type}_median")
    n = int(row.get(f"{range_type}_n", 0))
    if pd.isna(p):
        p = None
    if pd.isna(med):
        med = None
    return p, med, n


def lookup_pooled_p(
    wilcoxon_csv: str,
    range_type: str = "tight",
) -> tuple[float | None, float | None, int]:
    """Look up pooled Wilcoxon results.

    Returns (p_value, median_diff, n_diffs) — same signature as the
    former ``_overall_wilcoxon()`` in comprehensive_figures.py.
    """
    return lookup_per_model_p(wilcoxon_csv, "POOLED", range_type)
