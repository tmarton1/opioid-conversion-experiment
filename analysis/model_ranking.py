"""
Model-vs-model pairwise significance ranking.

For a given experimental condition (a filtered subset of a model-analysis CSV),
rank models by overall accuracy and then run **all-pairs** significance tests.
Models that are not significantly different from any higher-ranked model share
the same rank; a model drops to a new rank when *any* model above it is
significantly better.

Statistical method
------------------
Each row in the model-analysis CSV is one (model, unit) combination, where
*unit* is a scenario or opioid pair.  For each unit the tight/broad
proportion and ``n_observations`` give accuracy counts.

For every pair of models (A, B), a **one-sided Cochran–Mantel–Haenszel
(CMH) test** is run on the 2×2 tables stratified by unit:

    For each unit k:
               | correct | incorrect
        ───────┼─────────┼──────────
        model A|  a_k    | n_Ak − a_k
        model B|  b_k    | n_Bk − b_k

    H₀: common odds ratio = 1  (no difference across strata)
    H₁ (one-sided): P(correct | A) > P(correct | B)

The CMH test pools evidence across all shared units (strata) into a
single test statistic, producing **one p-value per model pair per range**
(tight and broad).  This respects the stratified structure of the data
— units are not independent, and the CMH test accounts for this by
treating each unit as a stratum rather than an independent observation.

**Bonferroni correction** is applied across all CMH tests
(n_model_pairs × 2 ranges).

Rank assignment
---------------
1. Sort models by mean tight accuracy (broad as tiebreaker).
2. The top model gets rank 1.
3. For each subsequent model *i* (in sorted order), if ANY model with a
   higher position is significantly better (Bonferroni-corrected p < 0.05
   on tight OR broad), model *i* gets rank = max(adjusted_rank of those
   significant superiors) + 1.  Otherwise it shares the rank of model i−1.

This avoids the "chain of insignificance" problem where A≈B and B≈C but
A>>C would incorrectly all share rank 1.

Usage
-----
    ranking = ModelRanking(
        csv_path="results/reworded_vignettes_model_analysis.csv",
        filters={"include_nielsen": False, "explain_units": False},
        unit_col="scenario_key",
        label="Naive Vignettes",
    )
    df = ranking.build()
    ranking.save("results/ranking_naive_vignettes.csv")
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from common.models import CONSISTENT_MODELS
from common.config import RESULTS_DIR


class ModelRanking:
    """
    Rank models by accuracy with Cochran–Mantel–Haenszel stratified tests.

    Parameters
    ----------
    csv_path : str
        Path to a model-analysis CSV with tight/broad proportion columns
        and an ``n_observations`` column.
    filters : dict
        Column-value pairs used to filter the data to a single experimental
        condition.
    unit_col : str
        Column identifying the observational unit / stratum
        (``"scenario_key"`` or ``"opioid_pair"``).
    label : str
        Human-readable label for this ranking (used in output).
    """

    def __init__(
        self,
        csv_path: str,
        filters: dict,
        unit_col: str,
        label: str = "",
    ):
        self.csv_path = csv_path
        self.filters = filters
        self.unit_col = unit_col
        self.label = label
        self.df_raw = self._load_and_filter()
        self._detect_range_cols()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_and_filter(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        # Restrict to the 13 consistently-run models before any test or summary.
        df = df[df["model"].isin(CONSISTENT_MODELS)]
        for col, val in self.filters.items():
            if col in df.columns:
                df = df[df[col] == val]
        return df

    def _detect_range_cols(self):
        tight = [c for c in self.df_raw.columns if c.startswith("tight(")]
        broad = [c for c in self.df_raw.columns if c.startswith("broad(")]
        if not tight or not broad:
            raise ValueError("Cannot find tight/broad columns")
        self.tight_col = tight[0]
        self.broad_col = broad[0]

    def _get_model_unit_props(self) -> pd.DataFrame:
        """Return per-(model, unit) tight and broad proportions.

        If multiple rows exist for the same (model, unit) — e.g. different
        vignette_ids sharing the same scenario_key — they are averaged.

        One row per (model, unit) with columns:
            model, <unit_col>, tight_prop, broad_prop
        """
        sub = self.df_raw[
            ["model", self.unit_col, self.tight_col, self.broad_col]
        ].rename(columns={
            self.tight_col: "tight_prop",
            self.broad_col: "broad_prop",
        })
        return (
            sub.groupby(["model", self.unit_col], as_index=False)
            [["tight_prop", "broad_prop"]]
            .mean()
        )

    def _get_model_unit_counts(self) -> pd.DataFrame:
        """Return per-(model, unit) pooled accuracy counts.

        If multiple rows share the same (model, unit) — e.g. different
        vignette_ids within one scenario_key — counts are summed.

        One row per (model, unit) with columns:
            model, <unit_col>, n_total, n_tight, n_broad
        """
        df = self.df_raw.copy()
        df["_n_obs"] = df["n_observations"].astype(int)
        df["_n_tight"] = (
            df[self.tight_col] * df["_n_obs"]
        ).round().clip(lower=0).astype(int)
        df["_n_broad"] = (
            df[self.broad_col] * df["_n_obs"]
        ).round().clip(lower=0).astype(int)

        return (
            df.groupby(["model", self.unit_col])
            .agg(
                n_total=("_n_obs", "sum"),
                n_tight=("_n_tight", "sum"),
                n_broad=("_n_broad", "sum"),
            )
            .reset_index()
        )

    def _aggregate_model(self, model_df: pd.DataFrame):
        """Pool counts for a single model across all units.

        Returns (n_tight, n_broad, n_total) — used only for display.
        """
        total_tight = total_broad = total_n = 0
        for _, row in model_df.iterrows():
            n = int(row["n_observations"])
            t = max(int(round(float(row[self.tight_col]) * n)), 0)
            b = max(int(round(float(row[self.broad_col]) * n)), 0)
            total_tight += t
            total_broad += b
            total_n += n
        return total_tight, total_broad, total_n

    @staticmethod
    def _cmh_one_sided(tables: list[np.ndarray]) -> float:
        """One-sided Cochran–Mantel–Haenszel test across K 2×2 tables.

        Each table has shape (2, 2):
            [[a_k, b_k], [c_k, d_k]]
        where row 0 = model A, row 1 = model B,
              col 0 = correct,  col 1 = incorrect.

        H₁: model A has a higher success proportion than model B.

        Returns the one-sided p-value (small ⟹ A > B is significant).
        Returns np.nan if degenerate (no usable strata).
        """
        numerator = 0.0
        variance = 0.0

        for tbl in tables:
            a_k = tbl[0, 0]   # A correct
            n1_k = tbl[0].sum()  # A total
            n2_k = tbl[1].sum()  # B total
            t_k = n1_k + n2_k    # stratum total
            m1_k = tbl[:, 0].sum()  # total correct in stratum

            if t_k <= 1 or n1_k == 0 or n2_k == 0:
                continue

            e_a = n1_k * m1_k / t_k
            v_a = n1_k * n2_k * m1_k * (t_k - m1_k) / (t_k ** 2 * (t_k - 1))

            numerator += (a_k - e_a)
            variance += v_a

        if variance <= 0:
            return np.nan

        z = numerator / np.sqrt(variance)
        # One-sided: H₁ = A > B  →  p = P(Z ≥ z) = 1 − Φ(z)
        return float(norm.sf(z))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """Compute rankings with CMH tests stratified by unit.

        Returns a DataFrame stored as self.result_df.  Also stores
        self.pairs_df (per-model-pair summary) and
        self.n_total_tests (total number of CMH tests).

        result_df columns:
          model, n_total, n_tight, n_broad,
          mean_tight_prop, mean_broad_prop, n_units,
          raw_rank, adjusted_rank

        pairs_df columns:
          model_a, model_b, n_strata,
          tight_p, broad_p,
          tight_p_bonf, broad_p_bonf, a_sig_better
        """
        unit_props = self._get_model_unit_props()
        unit_counts = self._get_model_unit_counts()

        # ── Step 1: aggregate per model ──────────────────────────────
        model_stats = []
        for model, grp in self.df_raw.groupby("model"):
            n_tight, n_broad, n_total = self._aggregate_model(grp)
            model_unit = unit_props[unit_props["model"] == model]
            model_stats.append({
                "model": model,
                "n_total": n_total,
                "n_tight": n_tight,
                "n_broad": n_broad,
                "mean_tight_prop": model_unit["tight_prop"].mean(),
                "mean_broad_prop": model_unit["broad_prop"].mean(),
                "n_units": len(model_unit),
            })
        stats = pd.DataFrame(model_stats)

        # ── Step 2: sort by mean tight desc, then mean broad desc ────
        stats = stats.sort_values(
            ["mean_tight_prop", "mean_broad_prop"], ascending=[False, False]
        ).reset_index(drop=True)
        stats["raw_rank"] = range(1, len(stats) + 1)
        models_sorted = list(stats["model"])
        n_models = len(models_sorted)

        # ── Step 3: CMH test for ALL model pairs ─────────────────────
        pairs = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_a = models_sorted[i]   # higher-ranked
                model_b = models_sorted[j]   # lower-ranked

                a_data = unit_counts[
                    unit_counts["model"] == model_a
                ].set_index(self.unit_col)
                b_data = unit_counts[
                    unit_counts["model"] == model_b
                ].set_index(self.unit_col)

                shared = sorted(a_data.index.intersection(b_data.index))

                if not len(shared):
                    continue

                # Build 2×2 tables for tight and broad ranges
                tight_tables = []
                broad_tables = []
                for unit in shared:
                    ar = a_data.loc[unit]
                    br = b_data.loc[unit]
                    a_n = int(ar["n_total"])
                    b_n = int(br["n_total"])

                    tight_tables.append(np.array([
                        [int(ar["n_tight"]), a_n - int(ar["n_tight"])],
                        [int(br["n_tight"]), b_n - int(br["n_tight"])],
                    ]))
                    broad_tables.append(np.array([
                        [int(ar["n_broad"]), a_n - int(ar["n_broad"])],
                        [int(br["n_broad"]), b_n - int(br["n_broad"])],
                    ]))

                p_tight = self._cmh_one_sided(tight_tables)
                p_broad = self._cmh_one_sided(broad_tables)

                pairs.append({
                    "idx_a": i, "idx_b": j,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_strata": len(shared),
                    "tight_p": p_tight,
                    "broad_p": p_broad,
                })

        pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame()

        # ── Step 4: Bonferroni correction across all CMH tests ───────
        if not pairs_df.empty:
            all_pvals = np.concatenate([
                pairs_df["tight_p"].values,
                pairs_df["broad_p"].values,
            ]).astype(float)
            valid_mask = ~np.isnan(all_pvals)
            corrected = np.full(len(all_pvals), np.nan)
            if valid_mask.any():
                _, pvals_corr, *_ = multipletests(
                    all_pvals[valid_mask],
                    method="bonferroni",
                    alpha=0.05,
                )
                corrected[valid_mask] = pvals_corr
            n_p = len(pairs_df)
            pairs_df["tight_p_bonf"] = corrected[:n_p]
            pairs_df["broad_p_bonf"] = corrected[n_p:]
            pairs_df["a_sig_better"] = (
                (pairs_df["tight_p_bonf"] < 0.05)
                | (pairs_df["broad_p_bonf"] < 0.05)
            )
        else:
            pairs_df = pd.DataFrame(columns=[
                "idx_a", "idx_b", "model_a", "model_b",
                "n_strata", "tight_p", "broad_p",
                "tight_p_bonf", "broad_p_bonf", "a_sig_better",
            ])

        # ── Step 5: assign adjusted ranks ────────────────────────────
        adjusted = [0] * n_models
        adjusted[0] = 1  # top model is always rank 1

        for j in range(1, n_models):
            max_superior_rank = 0
            for ii in range(j):
                if pairs_df.empty:
                    continue
                row = pairs_df[
                    (pairs_df["idx_a"] == ii) & (pairs_df["idx_b"] == j)
                ]
                if not row.empty and row.iloc[0]["a_sig_better"]:
                    max_superior_rank = max(max_superior_rank, adjusted[ii])

            if max_superior_rank > 0:
                adjusted[j] = max_superior_rank + 1
            else:
                adjusted[j] = adjusted[j - 1]

        stats["adjusted_rank"] = adjusted

        self.result_df = stats
        self.pairs_df = pairs_df
        self.n_total_tests = len(pairs_df) * 2 if not pairs_df.empty else 0
        return stats

    def save(self, output_csv: str) -> pd.DataFrame:
        df = self.build()
        df.to_csv(output_csv, index=False)
        # Also save the per-pair summary
        pairs_csv = output_csv.replace(".csv", "_pairs.csv")
        self.pairs_df.to_csv(pairs_csv, index=False)
        print(f"  → {output_csv}  ({len(df)} models, "
              f"{len(self.pairs_df)} model pairs, "
              f"{self.n_total_tests} CMH tests)")
        return df

    def summary_text(self) -> str:
        """Return a printable summary."""
        if not hasattr(self, "result_df"):
            self.build()
        df = self.result_df
        pdf = self.pairs_df if hasattr(self, "pairs_df") else pd.DataFrame()
        n_tests = getattr(self, "n_total_tests", 0)
        effective_alpha = 0.05 / n_tests if n_tests else float("nan")
        n_sig = (
            int(pdf["a_sig_better"].sum())
            if not pdf.empty and "a_sig_better" in pdf.columns
            else 0
        )

        lines = []
        lines.append(f"\n{'═' * 80}")
        lines.append(f"  MODEL RANKING: {self.label}")
        lines.append(
            f"  Cochran–Mantel–Haenszel, all pairs "
            f"(strata = {self.unit_col})"
        )
        lines.append(
            f"  {len(pdf)} model pairs × 2 ranges = "
            f"{n_tests} CMH tests, "
            f"Bonferroni α = {effective_alpha:.2e}"
        )
        lines.append(f"  {n_sig} significant superiority relationships")
        lines.append(f"{'═' * 80}")

        prev_rank = None
        for _, row in df.iterrows():
            rank = int(row["adjusted_rank"])
            separator = ""
            if prev_rank is not None and rank != prev_rank:
                separator = f"  {'─' * 72}\n"
            prev_rank = rank

            lines.append(
                f"{separator}"
                f"  Rank {rank:>2}  "
                f"tight={row['mean_tight_prop']:.3f}  "
                f"broad={row['mean_broad_prop']:.3f}  "
                f"({int(row['n_units']):>3} units, "
                f"n={int(row['n_total']):>5})  "
                f"{row['model']}"
            )
        lines.append("")
        return "\n".join(lines)

    def print_summary(self):
        print(self.summary_text())


def build_all_rankings() -> list["ModelRanking"]:
    """Build the standard set of 5 model rankings.

    Returns a list of ModelRanking objects (already built).
    """
    VIGNETTE_CSV = f"{RESULTS_DIR}/reworded_vignettes_model_analysis.csv"
    PAIR_CSV = f"{RESULTS_DIR}/pair_model_analysis.csv"
    EXPLAIN_PAIR_CSV = f"{RESULTS_DIR}/explain_units_pair_model_analysis.csv"

    specs = [
        {
            "csv_path": VIGNETTE_CSV,
            "filters": {"include_nielsen": False, "explain_units": False},
            "unit_col": "scenario_key",
            "label": "Naive Vignettes",
            "output": f"{RESULTS_DIR}/ranking_naive_vignettes.csv",
        },
        {
            "csv_path": VIGNETTE_CSV,
            "filters": {"include_nielsen": True, "explain_units": True},
            "unit_col": "scenario_key",
            "label": "Vignettes + Nielsen + Explain Units",
            "output": f"{RESULTS_DIR}/ranking_vignettes_nielsen_units.csv",
        },
        {
            "csv_path": PAIR_CSV,
            "filters": {"include_nielsen": False, "include_examples": False},
            "unit_col": "opioid_pair",
            "label": "Naive Random Doses",
            "output": f"{RESULTS_DIR}/ranking_naive_random_doses.csv",
        },
        {
            "csv_path": PAIR_CSV,
            "filters": {"include_nielsen": True, "include_examples": False},
            "unit_col": "opioid_pair",
            "label": "Random Doses + Nielsen",
            "output": f"{RESULTS_DIR}/ranking_random_doses_nielsen.csv",
        },
        {
            "csv_path": EXPLAIN_PAIR_CSV,
            "filters": {"explain_units": True},
            "unit_col": "opioid_pair",
            "label": "Random Doses + Nielsen + Explain Units",
            "output": f"{RESULTS_DIR}/ranking_random_doses_nielsen_units.csv",
        },
    ]

    rankings = []
    for spec in specs:
        r = ModelRanking(
            csv_path=spec["csv_path"],
            filters=spec["filters"],
            unit_col=spec["unit_col"],
            label=spec["label"],
        )
        r.save(spec["output"])
        r.print_summary()
        rankings.append(r)

    return rankings
