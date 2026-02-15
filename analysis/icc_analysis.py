"""
Variance-components / ICC analysis for vignette experiments.

For each (model, scenario) pair, fits a one-way random-effects ANOVA where
the grouping factor is the *vignette* (4 text wordings of the same clinical
scenario) and the dependent variable is log(observed_dose / expected_dose).

Reports:
  - σ²_between  (between-vignette variance — how much wording matters)
  - σ²_within   (within-vignette variance — trial-to-trial stochasticity)
  - σ²_total    (σ²_between + σ²_within)
  - ICC(1,1)    = σ²_between / σ²_total
  - F-statistic and p-value for H₀: σ²_between = 0

Only the 4 text vignettes per scenario are used (the "empty" / bare-prompt
vignette is excluded because it represents a different construct).

Usage
-----
    from analysis.icc_analysis import ICCAnalysis

    icc = ICCAnalysis("results/reworded_vignettes_results.csv")
    icc.save("results/icc_analysis.csv")
"""

import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, mannwhitneyu
from common.models import CONSISTENT_MODELS


class ICCAnalysis:
    """
    Compute ICC(1,1) for each model × scenario from vignette experiment data.
    """

    def __init__(self, results_csv: str,
                 vignettes_json: str = "resources/reworded_vignettes.json",
                 filters: dict = None):
        """
        Parameters
        ----------
        filters : dict, optional
            Column filters applied before analysis, e.g.
            {"include_nielsen": False, "explain_units": False}
        """
        self.df = pd.read_csv(results_csv)
        self._filters = filters or {}
        self._load_text_vignettes(vignettes_json)
        self._results = self._compute_all()

    def _load_text_vignettes(self, path: str):
        """Load the vigntte JSON and identify the 4 text vignettes per scenario."""
        with open(path) as f:
            vigs = json.load(f)
        self._text_vignette_ids = {
            vid for vid, v in vigs.items()
            if v.get("has_vignette_text", True)
        }

    def _compute_icc(self, group_df: pd.DataFrame) -> dict:
        """
        Compute ICC(1,1) from a DataFrame with columns [vignette_id, log_ratio].
        Each row is one trial.  Group factor = vignette_id.

        Returns dict with: k, n_per_vignette, n_total, ms_between, ms_within,
                           var_between, var_within, var_total, icc, f_stat, f_pvalue
        """
        groups = group_df.groupby("vignette_id")["log_ratio"]
        group_means = groups.mean()
        group_sizes = groups.count()

        k = len(group_means)  # number of vignettes (should be 4)
        if k < 2:
            return None

        grand_mean = group_df["log_ratio"].mean()
        n_total = len(group_df)

        # For unbalanced designs, use harmonic mean of group sizes
        n_sizes = group_sizes.values
        if np.any(n_sizes == 0):
            return None
        n_0 = (1 / (k - 1)) * (n_total - np.sum(n_sizes ** 2) / n_total)

        # Sum of squares
        ss_between = np.sum(n_sizes * (group_means.values - grand_mean) ** 2)
        ss_within = np.sum(groups.apply(lambda g: np.sum((g - g.mean()) ** 2)))

        df_between = k - 1
        df_within = n_total - k

        if df_between == 0 or df_within == 0:
            return None

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        # Variance components
        var_within = ms_within
        var_between = max(0, (ms_between - ms_within) / n_0)
        var_total = var_between + var_within

        # ICC(1,1)
        icc = var_between / var_total if var_total > 0 else 0.0

        # F-test for H₀: σ²_between = 0
        f_stat = ms_between / ms_within if ms_within > 0 else np.inf
        f_pvalue = 1 - f_dist.cdf(f_stat, df_between, df_within)

        return {
            "k": k,
            "n_0": round(n_0, 2),
            "n_total": n_total,
            "ms_between": ms_between,
            "ms_within": ms_within,
            "var_between": var_between,
            "var_within": var_within,
            "var_total": var_total,
            "icc": icc,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
        }

    @staticmethod
    def _count_sig_pairs(group_df: pd.DataFrame, alpha: float = 0.05) -> dict:
        """
        For the 4 vignettes in a scenario, run Mann-Whitney U on all 6
        pairwise combinations of log_ratio distributions.

        Bonferroni correction: threshold = alpha / 6.

        Returns dict with sig_pairs (count), total_pairs, and the
        individual p-values.
        """
        vids = sorted(group_df["vignette_id"].unique())
        pairs = list(combinations(vids, 2))
        threshold = alpha / len(pairs) if pairs else alpha

        sig_count = 0
        pvals = []
        for v1, v2 in pairs:
            a = group_df.loc[group_df["vignette_id"] == v1, "log_ratio"].values
            b = group_df.loc[group_df["vignette_id"] == v2, "log_ratio"].values
            if len(a) < 2 or len(b) < 2:
                pvals.append(np.nan)
                continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                p = np.nan
            pvals.append(p)
            if not np.isnan(p) and p < threshold:
                sig_count += 1

        return {
            "sig_pairs": sig_count,
            "total_pairs": len(pairs),
            "pairwise_pvalues": pvals,
        }

    def _compute_all(self) -> pd.DataFrame:
        """Compute ICC and pairwise significance for every (model, scenario) pair."""
        df = self.df.copy()

        # Filter to consistent models only
        consistent_set = set(CONSISTENT_MODELS)
        df = df[df["model"].isin(consistent_set)]

        # Apply user-specified filters (e.g. naive condition)
        for col, val in self._filters.items():
            if col in df.columns:
                df = df[df[col] == val]

        # Filter to text vignettes only (exclude empty)
        df = df[df["vignette_id"].isin(self._text_vignette_ids)]

        # Filter to valid observed doses
        df = df[df["observed_to_dose"].notna() & (df["observed_to_dose"] > 0)]
        df = df[df["expected_to_dose"].notna() & (df["expected_to_dose"] > 0)]

        # Compute log ratio
        df["log_ratio"] = np.log(df["observed_to_dose"] / df["expected_to_dose"])

        rows = []
        for (model, scenario), group in df.groupby(["model", "scenario_key"]):
            result = self._compute_icc(group)
            if result is None:
                continue
            row = {"model": model, "scenario_key": scenario}
            row.update(result)

            # Pairwise significance
            pw = self._count_sig_pairs(group)
            row["sig_pairs"] = pw["sig_pairs"]
            row["total_pairs"] = pw["total_pairs"]
            rows.append(row)

        return pd.DataFrame(rows)

    @property
    def results(self) -> pd.DataFrame:
        return self._results.copy()

    def get_icc_matrix(self) -> pd.DataFrame:
        """Return a model × scenario pivot table of ICC values."""
        return self._results.pivot(
            index="model", columns="scenario_key", values="icc"
        )

    def get_var_total_matrix(self) -> pd.DataFrame:
        """Return a model × scenario pivot table of total variance."""
        return self._results.pivot(
            index="model", columns="scenario_key", values="var_total"
        )

    def get_var_between_matrix(self) -> pd.DataFrame:
        """Return a model × scenario pivot table of between-vignette variance."""
        return self._results.pivot(
            index="model", columns="scenario_key", values="var_between"
        )

    def get_var_within_matrix(self) -> pd.DataFrame:
        """Return a model × scenario pivot table of within-vignette variance."""
        return self._results.pivot(
            index="model", columns="scenario_key", values="var_within"
        )

    def save(self, path: str):
        """Save full results to CSV."""
        self._results.to_csv(path, index=False)
        print(f"Saved ICC analysis ({len(self._results)} rows) to {path}")
