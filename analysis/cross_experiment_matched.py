"""
Matched-pair cross-experiment comparison.

Identifies opioid pair conversions that appear in BOTH the random-dose
and vignette experiments (same from→to opioid and units), then computes
per-model accuracy restricted to those matched pairs.

Text-vignette accuracy is averaged across the 4 wordings per scenario
so each model gets one value per overlapping pair, matching the
random-dose granularity.

Outputs
-------
    cross_experiment_matched_means.csv
        Per-model × condition accuracy on matched pairs.
    cross_experiment_matched_wilcoxon.csv
        Per-model paired Wilcoxon signed-rank tests across conditions.
    cross_experiment_matched_spearman.csv
        Spearman rank correlations of model accuracy across conditions.

Usage
-----
    from analysis.cross_experiment_matched import CrossExperimentMatched
    cem = CrossExperimentMatched(
        pair_csv="results/pair_model_analysis.csv",
        vignette_csv="results/reworded_vignettes_model_analysis.csv",
    )
    cem.save("results")
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from common.models import CONSISTENT_MODELS
from common.config import RESULTS_DIR

TIGHT = "tight(0.85,1.15)"
BROAD = "broad(0.6,1.7)"


class CrossExperimentMatched:
    """
    Compare random-dose, empty-vignette, and text-vignette experiments
    on their overlapping opioid pair conversions only.
    """

    CONDITIONS = ["random_dose", "empty_vignette", "text_vignette"]
    COMPARISONS = [
        ("random_dose", "text_vignette"),
        ("random_dose", "empty_vignette"),
        ("empty_vignette", "text_vignette"),
    ]

    def __init__(
        self,
        pair_csv: str,
        vignette_csv: str,
        models: list[str] | None = None,
    ):
        self.models = models or sorted(CONSISTENT_MODELS)
        self._load_and_filter(pair_csv, vignette_csv)

    # ── data loading ────────────────────────────────────────────

    def _load_and_filter(self, pair_csv: str, vignette_csv: str):
        """Load, filter to naive conditions, find overlap, and prepare."""
        rd = pd.read_csv(pair_csv)
        rd = rd[
            (rd["include_nielsen"] == False)
            & (rd["include_examples"] == False)
            & (rd["explain_units"] == False)
            & (rd["model"].isin(self.models))
        ]

        vt = pd.read_csv(vignette_csv)
        vt = vt[
            (vt["include_nielsen"] == False)
            & (vt["explain_units"] == False)
            & (vt["model"].isin(self.models))
            & (vt["multiple_from_opioids"] == False)
        ]

        # Discover overlapping (from, to, from_unit, to_unit) pairs
        rd_keys = set(zip(
            rd["from_opioid"], rd["to_opioid"],
            rd["from_opioid_unit"], rd["to_opioid_unit"],
        ))
        vt_uniq = vt[["scenario_key", "from_opioid", "to_opioid",
                       "from_opioid_unit", "to_opioid_unit"]].drop_duplicates()

        self.overlap_scenarios = []
        self.overlap_pair_keys = set()
        for _, r in vt_uniq.iterrows():
            key = (r.from_opioid, r.to_opioid,
                   r.from_opioid_unit, r.to_opioid_unit)
            if key in rd_keys:
                self.overlap_scenarios.append(r.scenario_key)
                self.overlap_pair_keys.add(key)

        self.n_overlap = len(self.overlap_pair_keys)

        # Filter both datasets to overlapping pairs
        def _in_overlap(df):
            return df.apply(
                lambda r: (r["from_opioid"], r["to_opioid"],
                           r["from_opioid_unit"], r["to_opioid_unit"])
                in self.overlap_pair_keys,
                axis=1,
            )

        self.rd = rd[_in_overlap(rd)].copy()
        self.rd["_pair_key"] = list(zip(
            self.rd["from_opioid"], self.rd["to_opioid"],
            self.rd["from_opioid_unit"], self.rd["to_opioid_unit"],
        ))

        vt = vt[vt["scenario_key"].isin(self.overlap_scenarios)]

        # Text vignettes: average across wordings per scenario per model
        vt_text = vt[vt["has_vignette_text"] == True]
        self.vt_text = (
            vt_text.groupby(["model", "scenario_key", "from_opioid",
                             "to_opioid", "from_opioid_unit", "to_opioid_unit"])
            .agg({TIGHT: "mean", BROAD: "mean"})
            .reset_index()
        )
        self.vt_text["_pair_key"] = list(zip(
            self.vt_text["from_opioid"], self.vt_text["to_opioid"],
            self.vt_text["from_opioid_unit"], self.vt_text["to_opioid_unit"],
        ))

        # Empty vignettes: 1 per scenario per model
        self.vt_empty = vt[vt["has_vignette_text"] == False].copy()
        self.vt_empty["_pair_key"] = list(zip(
            self.vt_empty["from_opioid"], self.vt_empty["to_opioid"],
            self.vt_empty["from_opioid_unit"], self.vt_empty["to_opioid_unit"],
        ))

    # ── per-model means ────────────────────────────────────────

    def build_means(self) -> pd.DataFrame:
        """One row per model × condition with mean accuracy on matched pairs."""
        rows = []
        for m in self.models:
            for cond, df in [("random_dose", self.rd),
                             ("empty_vignette", self.vt_empty),
                             ("text_vignette", self.vt_text)]:
                sub = df[df["model"] == m]
                rows.append({
                    "model": m,
                    "condition": cond,
                    "n_pairs": len(sub),
                    "mean_tight": sub[TIGHT].mean(),
                    "mean_broad": sub[BROAD].mean(),
                })
        return pd.DataFrame(rows)

    # ── per-model paired Wilcoxon ──────────────────────────────

    def _get_paired_vectors(self, model: str, cond_a: str, cond_b: str, col: str):
        """Return aligned vectors for two conditions on the matched pairs."""
        df_map = {
            "random_dose": self.rd,
            "empty_vignette": self.vt_empty,
            "text_vignette": self.vt_text,
        }
        a = df_map[cond_a]
        b = df_map[cond_b]
        a_m = a[a["model"] == model].set_index("_pair_key")[col].sort_index()
        b_m = b[b["model"] == model].set_index("_pair_key")[col].sort_index()
        common = a_m.index.intersection(b_m.index)
        return a_m.loc[common].values, b_m.loc[common].values, len(common)

    def build_wilcoxon(self) -> pd.DataFrame:
        """Paired Wilcoxon signed-rank tests per model × comparison."""
        rows = []
        for m in self.models:
            for cond_a, cond_b in self.COMPARISONS:
                row = {
                    "model": m,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                }
                for rng, col in [("tight", TIGHT), ("broad", BROAD)]:
                    va, vb, n = self._get_paired_vectors(m, cond_a, cond_b, col)
                    row[f"{rng}_diff"] = float(np.mean(va) - np.mean(vb)) if n else np.nan
                    if n >= 6:
                        try:
                            stat, p = wilcoxon(va, vb, alternative="two-sided")
                            row[f"{rng}_wilcoxon_stat"] = stat
                            row[f"{rng}_wilcoxon_p"] = p
                        except ValueError:
                            row[f"{rng}_wilcoxon_stat"] = np.nan
                            row[f"{rng}_wilcoxon_p"] = 1.0
                    else:
                        row[f"{rng}_wilcoxon_stat"] = np.nan
                        row[f"{rng}_wilcoxon_p"] = np.nan
                    row[f"{rng}_n_pairs"] = n
                rows.append(row)
        return pd.DataFrame(rows)

    # ── Per-model Spearman rank correlations ───────────────────

    def build_spearman(self) -> pd.DataFrame:
        """Per-model Spearman ρ correlating pair-level accuracy across conditions.

        For each model and condition pair, correlates the n matched-pair
        accuracy values to measure within-model consistency across
        experimental formats.
        """
        rows = []
        for m in self.models:
            for cond_a, cond_b in self.COMPARISONS:
                for rng, col in [("tight", TIGHT), ("broad", BROAD)]:
                    va, vb, n = self._get_paired_vectors(m, cond_a, cond_b, col)
                    if n >= 4:
                        rho, p = spearmanr(va, vb)
                    else:
                        rho, p = np.nan, np.nan
                    rows.append({
                        "model": m,
                        "condition_a": cond_a,
                        "condition_b": cond_b,
                        "range": rng,
                        "spearman_rho": rho,
                        "spearman_p": p,
                        "n_pairs": n,
                    })
        return pd.DataFrame(rows)

    # ── save all ───────────────────────────────────────────────

    def save(self, results_dir: str = RESULTS_DIR, suffix: str = ""):
        """Save all three CSVs to the results directory (``suffix`` is added
        before .csv, e.g. "_same_conversion" for a robustness variant)."""
        import os
        os.makedirs(results_dir, exist_ok=True)

        means = self.build_means()
        means_path = os.path.join(results_dir, f"cross_experiment_matched_means{suffix}.csv")
        means.to_csv(means_path, index=False)
        print(f"  → {means_path}  ({len(means)} rows, "
              f"{self.n_overlap} matched pairs)")

        wilc = self.build_wilcoxon()
        wilc_path = os.path.join(results_dir, f"cross_experiment_matched_wilcoxon{suffix}.csv")
        wilc.to_csv(wilc_path, index=False)
        for comp_a, comp_b in self.COMPARISONS:
            sub = wilc[(wilc["condition_a"] == comp_a)
                       & (wilc["condition_b"] == comp_b)]
            sig_t = (sub["tight_wilcoxon_p"] < 0.05).sum()
            sig_b = (sub["broad_wilcoxon_p"] < 0.05).sum()
            print(f"  {comp_a} vs {comp_b}: "
                  f"tight {sig_t}/{len(sub)} sig, "
                  f"broad {sig_b}/{len(sub)} sig")
        print(f"  → {wilc_path}")

        spear = self.build_spearman()
        spear_path = os.path.join(results_dir, f"cross_experiment_matched_spearman{suffix}.csv")
        spear.to_csv(spear_path, index=False)
        for comp_a, comp_b in self.COMPARISONS:
            sub = spear[(spear["condition_a"] == comp_a)
                        & (spear["condition_b"] == comp_b)
                        & (spear["range"] == "broad")]
            median_rho = sub["spearman_rho"].median()
            sig = (sub["spearman_p"] < 0.05).sum()
            print(f"  Spearman {comp_a} vs {comp_b} (broad): "
                  f"median ρ={median_rho:.3f}, "
                  f"{sig}/{len(sub)} models significant")
        print(f"  → {spear_path}")

        return means, wilc, spear
