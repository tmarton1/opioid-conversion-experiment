"""
Performance-profile classification for opioid conversion results.

Pre-computes per-model proportions in each error category and saves
to CSV so visualisers only read — never compute — derived values.

Output CSVs
-----------
performance_profile_random_doses.csv
    Per-model proportions for random-dose experiments:
    correct, precise_wrong, other_wrong, dose_dep_neg, dose_dep_pos.

performance_profile_vignettes.csv
    Per-model proportions for vignette experiments:
    correct, precise_wrong, imprecise.
"""

import os
import numpy as np
import pandas as pd

from common.models import CONSISTENT_MODELS
from visualize.visualization_config import (
    TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER,
)

_CONSISTENT_SET = set(CONSISTENT_MODELS)


# ── Random-dose performance profile ────────────────────────────────

def compute_random_dose_profile(
    model_csv: str,
    output_csv: str | None = None,
    cv_col: str = "observed_coefficient_cv",
    ratio_num_col: str = "observed_coefficient_mean",
    ratio_den_col: str = "expected_coefficient",
    slope_col: str = "regression_slope",
    slope_p_col: str = "regression_slope_pvalue",
    slope_alpha: float = 0.05,
    cv_threshold: float = 0.2,
    naive_filter: dict | None = None,
) -> pd.DataFrame:
    """Classify each model's pair-level results into five categories.

    Categories
    ----------
    correct : tight accuracy (ratio within 0.85–1.15)
    precise_wrong : CV ≤ threshold, not tight, not dose-correlated
    other_wrong : not precise, not tight, not dose-correlated
    dose_dep_neg : significant negative regression slope
    dose_dep_pos : significant positive regression slope

    Parameters
    ----------
    model_csv : str
        Path to ``pair_model_analysis.csv``.
    output_csv : str, optional
        If given, saves to this path.

    Returns
    -------
    pd.DataFrame
        One row per model, columns: model + five category proportions + n_obs.
    """
    if naive_filter is None:
        naive_filter = {"include_nielsen": False, "include_examples": False}

    df = pd.read_csv(model_csv)
    for col, val in naive_filter.items():
        if col in df.columns:
            df = df[df[col] == val]
    df = df[df["model"].isin(_CONSISTENT_SET)].copy()

    # Compute ratio & classify
    df["_ratio"] = df[ratio_num_col] / df[ratio_den_col]
    df["_tight"] = df["_ratio"].between(TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER)
    df["_precise"] = df[cv_col] <= cv_threshold
    df["_dose_dep_neg"] = (df[slope_col] < 0) & (df[slope_p_col] < slope_alpha)
    df["_dose_dep_pos"] = (df[slope_col] > 0) & (df[slope_p_col] < slope_alpha)

    def _cat(row):
        if row["_dose_dep_neg"]:
            return "dose_dep_neg"
        if row["_dose_dep_pos"]:
            return "dose_dep_pos"
        if row["_tight"]:
            return "correct"
        if row["_precise"]:
            return "precise_wrong"
        return "other_wrong"

    df["_cat"] = df.apply(_cat, axis=1)

    cat_order = ["correct", "precise_wrong", "other_wrong",
                 "dose_dep_neg", "dose_dep_pos"]

    rows = []
    for m in sorted(df["model"].unique()):
        mdf = df[df["model"] == m]
        n = len(mdf)
        props = {"model": m, "n_obs": n}
        for c in cat_order:
            props[c] = (mdf["_cat"] == c).sum() / n if n else 0
        rows.append(props)

    result = pd.DataFrame(rows)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        result.to_csv(output_csv, index=False)

    return result


# ── Vignette performance profile ───────────────────────────────────

def compute_vignette_profile(
    model_csv: str,
    output_csv: str | None = None,
    cv_col: str = "observed_dose_cv",
    ratio_num_col: str = "observed_dose_mean",
    ratio_den_col: str = "expected_to_dose",
    naive_filter: dict | None = None,
) -> pd.DataFrame:
    """Classify each model's vignette results into three categories.

    Categories
    ----------
    correct : tight accuracy (ratio within 0.85–1.15)
    precise_wrong : CV=0 but NOT tight accuracy
    imprecise : CV > 0 and not tight

    Parameters
    ----------
    model_csv : str
        Path to ``reworded_vignettes_model_analysis.csv``.
    output_csv : str, optional
        If given, saves to this path.

    Returns
    -------
    pd.DataFrame
        One row per model, columns: model + three category proportions + n_obs.
    """
    if naive_filter is None:
        naive_filter = {"include_nielsen": False, "explain_units": False,
                        "has_vignette_text": True}

    df = pd.read_csv(model_csv)
    for col, val in naive_filter.items():
        if col in df.columns:
            df = df[df[col] == val]
    df = df[df["model"].isin(_CONSISTENT_SET)].copy()

    # Compute ratio & classify
    df["_ratio"] = df[ratio_num_col] / df[ratio_den_col]
    df["_tight"] = df["_ratio"].between(TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER)
    df["_cv_zero"] = df[cv_col] <= 1e-9

    def _cat(row):
        if row["_tight"]:
            return "correct"
        if row["_cv_zero"]:
            return "precise_wrong"
        return "imprecise"

    df["_cat"] = df.apply(_cat, axis=1)

    cat_order = ["correct", "precise_wrong", "imprecise"]

    rows = []
    for m in sorted(df["model"].unique()):
        mdf = df[df["model"] == m]
        n = len(mdf)
        props = {"model": m, "n_obs": n}
        for c in cat_order:
            props[c] = (mdf["_cat"] == c).sum() / n if n else 0
        rows.append(props)

    result = pd.DataFrame(rows)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        result.to_csv(output_csv, index=False)

    return result
