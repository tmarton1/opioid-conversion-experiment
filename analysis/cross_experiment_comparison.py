"""
Per-model significance comparison across experiments with different units.

Because random-dose and vignette experiments use different conversion
units (opioid pairs vs vignettes), a paired test is inappropriate.
Instead we use the Mann–Whitney U test per model to compare the
*distributions* of per-unit accuracy proportions between two conditions.

Usage
-----
    from analysis.cross_experiment_comparison import CrossExperimentComparison

    comp = CrossExperimentComparison(
        csv_a="results/pair_model_analysis.csv",
        csv_b="results/reworded_vignettes_model_analysis.csv",
        filter_a={"include_nielsen": False, "include_examples": False,
                   "explain_units": False},
        filter_b={"include_nielsen": False, "explain_units": False,
                   "has_vignette_text": False},
        label_a="random_dose",
        label_b="empty_vignette",
    )
    comp.save("results/comparison_cross_rd_vs_empty.csv")
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

TIGHT_COL = "tight(0.85,1.15)"
BROAD_COL = "broad(0.6,1.7)"


class CrossExperimentComparison:
    """
    Compare per-unit accuracy distributions across two experiments,
    per model, using the Mann–Whitney U test.

    Parameters
    ----------
    csv_a, csv_b : str
        Paths to per-unit model analysis CSVs (one row per model × unit).
    filter_a, filter_b : dict
        Column → value filters to select the relevant condition slice.
    label_a, label_b : str
        Human-readable labels for output column prefixes.
    models : list[str] | None
        Restrict to these models (default: all shared models).
    """

    def __init__(
        self,
        csv_a: str,
        csv_b: str,
        filter_a: dict,
        filter_b: dict,
        label_a: str,
        label_b: str,
        models: list[str] | None = None,
    ):
        self.df_a = self._load(csv_a, filter_a, models)
        self.df_b = self._load(csv_b, filter_b, models)
        self.label_a = label_a
        self.label_b = label_b

    @staticmethod
    def _load(csv: str, filt: dict, models=None) -> pd.DataFrame:
        df = pd.read_csv(csv)
        for col, val in filt.items():
            df = df[df[col] == val]
        if models:
            df = df[df["model"].isin(models)]
        return df

    def build(self) -> pd.DataFrame:
        common = sorted(set(self.df_a["model"].unique()) &
                        set(self.df_b["model"].unique()))
        if not common:
            raise ValueError("No models in common")

        rows = []
        for m in common:
            a = self.df_a[self.df_a["model"] == m]
            b = self.df_b[self.df_b["model"] == m]

            row = {
                "model": m,
                f"{self.label_a}_n": len(a),
                f"{self.label_b}_n": len(b),
                f"{self.label_a}_tight_mean": a[TIGHT_COL].mean(),
                f"{self.label_a}_broad_mean": a[BROAD_COL].mean(),
                f"{self.label_b}_tight_mean": b[TIGHT_COL].mean(),
                f"{self.label_b}_broad_mean": b[BROAD_COL].mean(),
            }

            for rng, col in [("tight", TIGHT_COL), ("broad", BROAD_COL)]:
                va = a[col].dropna().values
                vb = b[col].dropna().values
                row[f"{rng}_diff"] = float(np.mean(va)) - float(np.mean(vb))
                if len(va) >= 2 and len(vb) >= 2:
                    stat, p = mannwhitneyu(va, vb, alternative="two-sided")
                    row[f"{rng}_mannwhitney_p"] = p
                    row[f"{rng}_mannwhitney_stat"] = stat
                else:
                    row[f"{rng}_mannwhitney_p"] = np.nan
                    row[f"{rng}_mannwhitney_stat"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, path: str) -> pd.DataFrame:
        df = self.build()
        df.to_csv(path, index=False)
        n = len(df)
        for rng in ("tight", "broad"):
            p_col = f"{rng}_mannwhitney_p"
            sig = (df[p_col] < 0.05).sum()
            print(f"  {self.label_a} vs {self.label_b} ({rng}): "
                  f"{sig}/{n} models significant (p<0.05)")
        print(f"  → {path}")
        return df
