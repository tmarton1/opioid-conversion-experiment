"""
Kendall's W (coefficient of concordance) analysis.

Computes Kendall's W across models for each scenario, treating vignettes
as items and models as raters.  Saves results to a CSV with columns:

    scenario_key, kendall_w, kendall_p

The core ``kendall_w`` function can also be used standalone.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, chi2 as chi2_dist


def kendall_w(rankings_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Kendall's W (coefficient of concordance).

    Parameters
    ----------
    rankings_matrix : array of shape (n_judges, n_items)
        Each row is one judge's rankings of the items.

    Returns
    -------
    (W, chi2, p_value)
    """
    m, n = rankings_matrix.shape  # m judges, n items
    if m < 2 or n < 2:
        return (np.nan, np.nan, np.nan)

    # Rank each row (judge) independently
    ranked = np.array([rankdata(row) for row in rankings_matrix])

    # Sum of ranks for each item
    R = ranked.sum(axis=0)
    R_mean = R.mean()

    # SS of rank sums
    SS = np.sum((R - R_mean) ** 2)

    # W = 12 * SS / (m² * (n³ - n))
    W = 12 * SS / (m ** 2 * (n ** 3 - n))

    # Chi-squared approximation
    chi2_stat = m * (n - 1) * W
    p = 1 - chi2_dist.cdf(chi2_stat, n - 1)

    return (W, chi2_stat, p)


def compute_kendall_w_for_scenarios(
    results_csv: str,
    output_csv: str | None = None,
    filters: dict | None = None,
) -> pd.DataFrame:
    """Compute Kendall's W for every scenario in a vignette results CSV.

    Parameters
    ----------
    results_csv : str
        Path to the raw vignette experiment results CSV.  Must contain
        ``scenario_key``, ``vignette_id``, ``model``, and ``log_ratio``.
    output_csv : str, optional
        If given, saves the result to this path.
    filters : dict, optional
        Column-value pairs used to filter rows before analysis
        (e.g. ``{"include_nielsen": False}``).

    Returns
    -------
    pd.DataFrame
        One row per scenario with columns:
        ``scenario_key``, ``kendall_w``, ``kendall_p``.
    """
    from common.models import CONSISTENT_MODELS

    df = pd.read_csv(results_csv)

    # Apply filters
    if filters:
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]

    # Consistent models only
    df = df[df["model"].isin(set(CONSISTENT_MODELS))]

    scenarios = sorted(df["scenario_key"].unique())
    rows = []

    for scenario_key in scenarios:
        sdf = df[df["scenario_key"] == scenario_key]

        # Build mean log-ratio per (model, vignette_id)
        means = (
            sdf.groupby(["model", "vignette_id"])["log_ratio"]
            .mean()
            .reset_index()
        )

        models = sorted(means["model"].unique())
        vignettes = sorted(means["vignette_id"].unique())

        if len(models) < 2 or len(vignettes) < 2:
            rows.append({
                "scenario_key": scenario_key,
                "kendall_w": np.nan,
                "kendall_p": np.nan,
            })
            continue

        # Build (models × vignettes) matrix of mean log-ratios
        matrix = np.full((len(models), len(vignettes)), np.nan)
        vid_idx = {v: i for i, v in enumerate(vignettes)}
        for i, m in enumerate(models):
            mdf = means[means["model"] == m]
            for _, r in mdf.iterrows():
                j = vid_idx[r["vignette_id"]]
                matrix[i, j] = r["log_ratio"]

        # Drop vignettes with any NaN
        valid_cols = ~np.isnan(matrix).any(axis=0)
        matrix = matrix[:, valid_cols]

        if matrix.shape[1] < 2:
            rows.append({
                "scenario_key": scenario_key,
                "kendall_w": np.nan,
                "kendall_p": np.nan,
            })
            continue

        W, chi2_stat, p = kendall_w(matrix)
        rows.append({
            "scenario_key": scenario_key,
            "kendall_w": W,
            "kendall_p": p,
        })

    result = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result
