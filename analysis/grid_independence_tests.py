"""
Chi-square independence tests for 3×3 grid visualisations.

Tests three grid types per model:
  1. accuracy_x_variance  — accuracy category vs scenario variance tercile
  2. variance_x_cv        — scenario variance tercile vs within-vignette CV bin
  3. variance_x_icc       — scenario variance tercile vs ICC tercile

Also computes an overall Spearman correlation for variance vs ICC.

Moved from ``visualize.icc_visualizer`` so that all statistical tests
live in ``analysis/`` and results are persisted to CSV.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from common.models import CONSISTENT_MODELS

_CONSISTENT_SET = set(CONSISTENT_MODELS)

_CV_LABELS_VIGNETTE = ["CV=0", "0<CV≤0.2", "CV>0.2"]
_ACC_LABELS = ["tight", "broad", "neither"]


def _cv_bin_vignette(cv):
    """Assign a vignette CV value to one of three bins."""
    if cv <= 0:
        return _CV_LABELS_VIGNETTE[0]
    elif cv <= 0.2:
        return _CV_LABELS_VIGNETTE[1]
    return _CV_LABELS_VIGNETTE[2]


def compute_grid_independence_tests(
    icc_csv: str,
    vignette_model_csv: str,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Run chi-square tests of independence on each model's 3×3 grids.

    Tests three grid types per model:
      1. accuracy_x_variance  — accuracy category vs scenario variance tercile
      2. variance_x_cv        — scenario variance tercile vs within-vignette CV bin
      3. variance_x_icc       — scenario variance tercile vs ICC tercile
    Also computes overall Spearman correlation for variance vs ICC.

    Returns a DataFrame; optionally saves to *output_csv*.
    """
    icc_df = pd.read_csv(icc_csv)
    vma = pd.read_csv(vignette_model_csv)

    # ── Filter to naive, text vignettes, consistent models ──
    vma = vma[
        (vma["include_nielsen"] == False) &
        (vma["explain_units"] == False) &
        (vma["has_vignette_text"] == True)
    ]
    vma = vma[vma["model"].isin(_CONSISTENT_SET)]

    tight_col = "tight(0.85,1.15)"
    broad_col = "broad(0.6,1.7)"

    # Accuracy classification
    vma["_acc"] = "neither"
    vma.loc[vma[broad_col] >= 0.5, "_acc"] = "broad"
    vma.loc[vma[tight_col] >= 0.5, "_acc"] = "tight"

    # CV bins
    vma["_cv_bin"] = vma["observed_dose_cv"].apply(_cv_bin_vignette)

    # Scenario variance terciles
    scenario_var = icc_df.groupby("scenario_key")["var_total"].mean()
    t1, t2 = np.percentile(scenario_var.values, [33.3, 66.7])
    var_labels = [f"Low (<{t1:.2f})", f"Mid ({t1:.2f}–{t2:.2f})", f"High (≥{t2:.2f})"]

    def _var_bin(scenario):
        v = scenario_var.get(scenario, 0)
        if v < t1:
            return var_labels[0]
        elif v < t2:
            return var_labels[1]
        return var_labels[2]

    vma["_var_bin"] = vma["scenario_key"].apply(_var_bin)

    # ICC terciles
    icc_sub = icc_df[icc_df["model"].isin(_CONSISTENT_SET)].copy()
    icc_vals = icc_sub["icc"].dropna()
    icc_t1, icc_t2 = np.percentile(icc_vals.values, [33.3, 66.7])
    icc_labels = [
        f"Low (<{icc_t1:.2f})",
        f"Mid ({icc_t1:.2f}–{icc_t2:.2f})",
        f"High (≥{icc_t2:.2f})",
    ]

    def _v_bin_icc(row):
        v = scenario_var.get(row["scenario_key"], 0)
        if v < t1:
            return var_labels[0]
        elif v < t2:
            return var_labels[1]
        return var_labels[2]

    def _i_bin(row):
        icc = row["icc"]
        if np.isnan(icc):
            return icc_labels[0]
        if icc < icc_t1:
            return icc_labels[0]
        elif icc < icc_t2:
            return icc_labels[1]
        return icc_labels[2]

    icc_sub["_var_bin"] = icc_sub.apply(_v_bin_icc, axis=1)
    icc_sub["_icc_bin"] = icc_sub.apply(_i_bin, axis=1)

    # Ranked models
    from visualize.visualization_config import get_ranked_model_order
    ranked = get_ranked_model_order()
    models = [m for m in ranked if m in vma["model"].unique()]
    if not models:
        models = sorted(vma["model"].unique())

    # Grid specs: (grid_type, data, row_col, col_col, row_labels, col_labels)
    grid_specs = [
        ("accuracy_x_variance", vma, "_acc", "_var_bin", _ACC_LABELS, var_labels),
        ("variance_x_cv", vma, "_var_bin", "_cv_bin", var_labels, _CV_LABELS_VIGNETTE),
        ("variance_x_icc", icc_sub, "_var_bin", "_icc_bin", var_labels, icc_labels),
    ]

    rows = []
    for grid_type, data, row_col, col_col, row_labels, col_labels in grid_specs:
        for m in models:
            mdf = data[data["model"] == m]
            table = np.zeros((len(row_labels), len(col_labels)), dtype=int)
            for i, rl in enumerate(row_labels):
                for j, cl in enumerate(col_labels):
                    table[i, j] = int(
                        ((mdf[row_col] == rl) & (mdf[col_col] == cl)).sum()
                    )
            n_obs = int(table.sum())
            try:
                chi2, pval, dof, expected = chi2_contingency(table)
                min_exp = float(expected.min())
            except ValueError:
                chi2, pval, dof, min_exp = np.nan, np.nan, np.nan, 0.0
            rows.append({
                "grid_type": grid_type,
                "model": m,
                "chi2": chi2,
                "p_value": pval,
                "dof": dof,
                "min_expected": min_exp,
                "n_obs": n_obs,
                "significant": bool(pval < 0.05) if not np.isnan(pval) else False,
                "low_expected_warning": bool(min_exp < 5),
            })

    # Overall Spearman: variance vs ICC
    rho, sp_p = spearmanr(icc_sub["var_total"].values, icc_sub["icc"].values)
    rows.append({
        "grid_type": "spearman_variance_vs_icc",
        "model": "ALL",
        "chi2": rho,
        "p_value": sp_p,
        "dof": np.nan,
        "min_expected": np.nan,
        "n_obs": len(icc_sub),
        "significant": bool(sp_p < 0.05),
        "low_expected_warning": False,
    })

    result = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def lookup_grid_pvalues(
    grid_csv: str,
    grid_type: str,
) -> dict[str, float]:
    """Load pre-computed chi-square p-values for a specific grid type.

    Returns {model_name: p_value}.
    """
    df = pd.read_csv(grid_csv)
    sub = df[df["grid_type"] == grid_type]
    return dict(zip(sub["model"], sub["p_value"]))
