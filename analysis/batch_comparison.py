"""
Model-level paired comparison between two batches.

For each model present in both batches, computes the difference in
mean tight and mean broad accuracy.  A Wilcoxon signed-rank test
across models determines whether one batch is systematically better
than the other.

Usage
-----
    from analysis.batch_comparison import BatchComparison

    comp = BatchComparison(
        batch_means_csv="results/model_batch_means.csv",
        batch_a="ome_n0",
        batch_b="pairs_n0",
        label_a="ome",
        label_b="non_ome",
    )
    comp.save("results/comparison_ome_vs_nonome_naive.csv")
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


class BatchComparison:
    """
    Compare two batches from model_batch_means.csv at the model level.

    Each row in the output represents one model.  The paired Wilcoxon
    signed-rank test treats models as paired observations, testing whether
    the differences in accuracy between batch A and batch B are symmetric
    around zero.

    Parameters
    ----------
    batch_means_csv : str
        Path to a model_batch_means CSV with columns
        ``model``, ``batch_name``, ``mean_in_tight``, ``mean_in_broad``.
    batch_a, batch_b : str
        The ``batch_name`` values for the two groups.
    label_a, label_b : str
        Human-readable labels for the output columns.
    models : list[str] | None
        If given, restrict to these models.
    """

    def __init__(
        self,
        batch_means_csv: str,
        batch_a: str,
        batch_b: str,
        label_a: str,
        label_b: str,
        models: list[str] | None = None,
    ):
        df = pd.read_csv(batch_means_csv)
        if models:
            df = df[df["model"].isin(models)]

        self.a_df = df[df["batch_name"] == batch_a].set_index("model")
        self.b_df = df[df["batch_name"] == batch_b].set_index("model")
        self.label_a = label_a
        self.label_b = label_b
        self.batch_a = batch_a
        self.batch_b = batch_b

    def build(self) -> pd.DataFrame:
        """Run the comparison.  Returns one row per model plus summary."""
        common = sorted(set(self.a_df.index) & set(self.b_df.index))
        if not common:
            raise ValueError("No models in common between the two batches")

        rows = []
        for m in common:
            a_t = self.a_df.loc[m, "mean_in_tight"]
            a_b = self.a_df.loc[m, "mean_in_broad"]
            b_t = self.b_df.loc[m, "mean_in_tight"]
            b_b = self.b_df.loc[m, "mean_in_broad"]
            rows.append({
                "model": m,
                f"{self.label_a}_tight": a_t,
                f"{self.label_a}_broad": a_b,
                f"{self.label_b}_tight": b_t,
                f"{self.label_b}_broad": b_b,
                "tight_diff": a_t - b_t,
                "broad_diff": a_b - b_b,
            })

        result = pd.DataFrame(rows)

        # Wilcoxon signed-rank on the differences across models
        for range_type in ("tight", "broad"):
            diffs = result[f"{range_type}_diff"].dropna().values
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 2:
                stat, p = wilcoxon(nonzero, alternative="two-sided")
                result[f"{range_type}_wilcoxon_p"] = p
                result[f"{range_type}_wilcoxon_stat"] = stat
            else:
                result[f"{range_type}_wilcoxon_p"] = np.nan
                result[f"{range_type}_wilcoxon_stat"] = np.nan

        result["batch_a"] = self.batch_a
        result["batch_b"] = self.batch_b

        return result

    def save(self, path: str) -> pd.DataFrame:
        """Build and write to CSV.  Returns the DataFrame."""
        df = self.build()
        df.to_csv(path, index=False)

        # Summary
        n = len(df)
        for rt in ("tight", "broad"):
            p = df[f"{rt}_wilcoxon_p"].iloc[0] if not df.empty else np.nan
            med = df[f"{rt}_diff"].median()
            print(f"  {self.label_a} vs {self.label_b} ({rt}): "
                  f"median diff = {med:+.4f}, Wilcoxon p = {p:.3e}  "
                  f"(n={n} models)")
        print(f"  → {path}")
        return df
