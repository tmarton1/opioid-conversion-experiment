"""
Variance–ICC statistics for vignette experiments.

Pre-computes all Spearman correlations, chi-square independence tests,
and the binomial sign test that ICC visualisations need.  Results are
saved to CSV so visualisers only read — never compute — statistics.

Output CSVs
-----------
variance_icc_spearman.csv
    Per-model and overall Spearman ρ  (variance vs ICC), plus per-model
    Spearman ρ  (accuracy vs total variance), and binomial sign-test
    summary.

grid_independence_tests.csv
    Per-model χ² independence tests for three 3×3 grids
    (accuracy × variance, variance × CV, variance × ICC) and an overall
    Spearman correlation for variance vs ICC.

cv_accuracy_chi2.csv
    Per-model χ² independence tests for accuracy × CV-bin grids
    (random-dose experiment).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest, chi2_contingency

from common.models import CONSISTENT_MODELS

_CONSISTENT_SET = set(CONSISTENT_MODELS)


# ── Spearman ρ  per-model and overall ───────────────────────────────

def compute_variance_icc_spearman(
    icc_csv: str,
    vignette_model_csv: str,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Per-model and overall Spearman ρ  (variance ↔ ICC).

    Also computes Spearman ρ (accuracy ↔ total variance) and a binomial
    sign test on the direction of per-model ρ.

    Parameters
    ----------
    icc_csv : str
        Path to ``icc_analysis.csv`` (one row per model × scenario).
    vignette_model_csv : str
        Path to ``reworded_vignettes_model_analysis.csv`` for accuracy.
    output_csv : str, optional
        If given, saves to this path.

    Returns
    -------
    pd.DataFrame
        Rows: one per model + ``OVERALL`` + ``SIGN_TEST`` +
        ``ACCURACY_VS_VARIANCE``.
    """
    icc_df = pd.read_csv(icc_csv)
    icc_df = icc_df[icc_df["model"].isin(_CONSISTENT_SET)]

    # Accuracy-sorted order
    vma = pd.read_csv(vignette_model_csv)
    vma = vma[
        (vma["include_nielsen"] == False) &
        (vma["explain_units"] == False) &
        (vma["has_vignette_text"] == True)
    ]
    vma = vma[vma["model"].isin(_CONSISTENT_SET)]
    tight_col = "tight(0.85,1.15)"
    model_acc = vma.groupby("model")[tight_col].mean()

    rows = []

    # Per-model Spearman ρ (variance vs ICC)
    rhos = {}
    for m in sorted(icc_df["model"].unique()):
        mdf = icc_df[icc_df["model"] == m]
        if len(mdf) < 3:
            rhos[m] = (np.nan, np.nan)
            continue
        rho, pval = spearmanr(mdf["var_total"], mdf["icc"])
        rhos[m] = (rho, pval)
        rows.append({
            "stat_type": "variance_vs_icc",
            "model": m,
            "rho": rho,
            "p_value": pval,
            "n": len(mdf),
        })

    # Overall Spearman (all model×scenario cells pooled)
    rho_all, p_all = spearmanr(icc_df["var_total"].values, icc_df["icc"].values)
    rows.append({
        "stat_type": "variance_vs_icc",
        "model": "OVERALL",
        "rho": rho_all,
        "p_value": p_all,
        "n": len(icc_df),
    })

    # Binomial sign test on per-model rho direction
    valid = [(m, r, p) for m, (r, p) in rhos.items() if not np.isnan(r)]
    n_pos = sum(1 for _, r, _ in valid if r > 0)
    n_neg = sum(1 for _, r, _ in valid if r < 0)
    n_test = n_pos + n_neg
    n_sig = sum(1 for _, _, p in valid if p < 0.05)
    if n_test > 0:
        k_majority = max(n_pos, n_neg)
        sign_p = binomtest(k_majority, n_test, 0.5, alternative="two-sided").pvalue
    else:
        sign_p = np.nan

    rows.append({
        "stat_type": "sign_test",
        "model": "ALL",
        "rho": np.nan,
        "p_value": sign_p,
        "n": n_test,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_individually_sig": n_sig,
    })

    # Spearman ρ (accuracy vs total variance)
    model_var = icc_df.groupby("model")["var_total"].mean()
    common = sorted(set(model_acc.index) & set(model_var.index))
    if len(common) >= 3:
        rho_av, p_av = spearmanr(
            model_acc.reindex(common).values,
            model_var.reindex(common).values,
        )
        rows.append({
            "stat_type": "accuracy_vs_variance",
            "model": "ALL",
            "rho": rho_av,
            "p_value": p_av,
            "n": len(common),
        })

    # Per-model variance means (for stacked bar annotation)
    model_var_full = icc_df.groupby("model").agg(
        mean_within=("var_within", "mean"),
        mean_between=("var_between", "mean"),
        mean_total=("var_total", "mean"),
    )
    overall_ratio = icc_df["var_between"].mean() / icc_df["var_total"].mean()
    rows.append({
        "stat_type": "between_ratio",
        "model": "ALL",
        "rho": overall_ratio,
        "p_value": np.nan,
        "n": len(icc_df),
    })

    result = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


# ── χ² independence tests for mosaic / circle grids ─────────────────

def compute_cv_accuracy_chi2(
    pair_model_csv: str,
    cv_col: str = "observed_dose_cv",
    tight_col: str = "tight(0.85,1.15)",
    broad_col: str = "broad(0.75,1.25)",
    cv_bins: list | None = None,
    cv_bin_labels: list | None = None,
    output_csv: str | None = None,
    naive_filter: dict | None = None,
) -> pd.DataFrame:
    """Per-model χ² independence: accuracy category × CV bin.

    Used by the ``_cv_accuracy_mosaic()`` grid.

    Returns one row per model with chi2, p_value, dof, n_obs.
    """
    df = pd.read_csv(pair_model_csv)
    if naive_filter:
        for col, val in naive_filter.items():
            if col in df.columns:
                df = df[df[col] == val]
    df = df[df["model"].isin(_CONSISTENT_SET)]

    if cv_bins is None:
        cv_bins = [0, 0.1, 0.5, float("inf")]
    if cv_bin_labels is None:
        cv_bin_labels = ["<0.1", "0.1–0.5", "≥0.5"]

    needed = [c for c in [cv_col, tight_col, broad_col] if c in df.columns]
    if len(needed) < 3:
        return pd.DataFrame()

    df = df.dropna(subset=needed).copy()
    df["_cv_bin"] = pd.cut(df[cv_col], bins=cv_bins, labels=cv_bin_labels, right=False)

    df["_acc"] = "neither"
    df.loc[df[broad_col] >= 0.5, "_acc"] = "broad"
    df.loc[df[tight_col] >= 0.5, "_acc"] = "tight"

    acc_labels = ["tight", "broad", "neither"]

    rows = []
    for m in sorted(df["model"].unique()):
        mdf = df[df["model"] == m]
        table = np.zeros((len(acc_labels), len(cv_bin_labels)), dtype=int)
        for i, al in enumerate(acc_labels):
            for j, bl in enumerate(cv_bin_labels):
                table[i, j] = int(((mdf["_acc"] == al) & (mdf["_cv_bin"] == bl)).sum())
        n_obs = int(table.sum())
        try:
            chi2, pval, dof, expected = chi2_contingency(table)
            min_exp = float(expected.min())
        except ValueError:
            chi2, pval, dof, min_exp = np.nan, np.nan, np.nan, 0.0

        rows.append({
            "model": m,
            "chi2": chi2,
            "p_value": pval,
            "dof": dof,
            "min_expected": min_exp,
            "n_obs": n_obs,
            "significant": bool(pval < 0.05) if not np.isnan(pval) else False,
            "low_expected_warning": bool(min_exp < 5),
        })

    result = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result
