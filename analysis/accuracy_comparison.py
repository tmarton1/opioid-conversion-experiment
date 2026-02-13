"""
General-purpose directional significance tests comparing accuracy
proportions between two groups in a model-analysis CSV.

For each (model, unit, *hold_constant) combination, two separate 2×2
Fisher exact tests are run:

1. **Tight test** — is the proportion in the tight range significantly
   higher (or lower) in group A vs group B?
2. **Broad test** — is the proportion in the tight-or-broad range
   significantly higher (or lower) in group A vs group B?

Each test uses Fisher's exact test on a 2×2 table::

              | in_range | not_in_range |
    group_a   |    a     |   n_a − a    |
    group_b   |    b     |   n_b − b    |

One-sided p-values are computed in both directions (``greater`` = A better,
``less`` = B better).  Bonferroni correction is applied separately across
four families of one-sided tests (tight-A-better, tight-B-better,
broad-A-better, broad-B-better).  The output includes a combined
``tight_result`` / ``broad_result`` column whose value is one of
``"A sig. better"``, ``"B sig. better"``, or ``"n.s."``.

Usage
-----
    comp = AccuracyComparison(
        csv_path="results/reworded_vignettes_model_analysis.csv",
        group_col="has_vignette_text",
        group_a_value=True, group_b_value=False,
        group_a_label="text", group_b_label="empty",
        unit_col="scenario_key",
        hold_constant=["include_nielsen", "explain_units"],
    )
    df = comp.save("results/my_comparison.csv")
    comp.summarize(df)
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, binomtest, chisquare
from statsmodels.stats.multitest import multipletests


class AccuracyComparison:
    """
    Compare accuracy distributions between two groups defined by a single
    column value, across every (model, unit_col) combination, optionally
    sliced by hold_constant columns.

    Parameters
    ----------
    csv_path : str
        Path to a model-analysis CSV with tight/broad proportion columns
        and an ``n_observations`` column.
    group_col : str
        Column name whose values separate group A from group B.
    group_a_value, group_b_value
        The values of ``group_col`` that define the two groups.
    group_a_label, group_b_label : str
        Human-readable names for the groups (used in output columns and
        the summary).
    unit_col : str
        Column that identifies the observational unit for each test
        (e.g. ``"scenario_key"`` or ``"opioid_pair"``).
    hold_constant : list[str] | None
        Columns whose distinct value-combinations define separate slices
        (e.g. ``["include_nielsen", "explain_units"]``).  A separate test
        is run for every (model, unit_col, *hold_constant) tuple.
    """

    def __init__(
        self,
        csv_path: str,
        group_col: str,
        group_a_value,
        group_b_value,
        group_a_label: str,
        group_b_label: str,
        unit_col: str,
        hold_constant: list[str] | None = None,
        filters: dict | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        # Apply optional pre-filters (e.g. {"ome": True})
        if filters:
            for col, val in filters.items():
                if col in self.df.columns:
                    self.df = self.df[self.df[col] == val]
        self.group_col = group_col
        self.group_a_value = group_a_value
        self.group_b_value = group_b_value
        self.group_a_label = group_a_label
        self.group_b_label = group_b_label
        self.unit_col = unit_col
        self.hold_constant = hold_constant or []
        self._detect_range_cols()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _detect_range_cols(self):
        tight = [c for c in self.df.columns if c.startswith("tight(")]
        broad = [c for c in self.df.columns if c.startswith("broad(")]
        if not tight or not broad:
            raise ValueError("Cannot find tight/broad columns in data")
        self.tight_col = tight[0]
        self.broad_col = broad[0]

    def _pool_counts(self, sub_df: pd.DataFrame):
        """Pool tight and broad counts across rows.

        Returns (n_tight, n_broad, n_total).
        n_broad includes tight (i.e. proportion in broad range × n).
        """
        total_tight = total_broad = total_n = 0
        for _, row in sub_df.iterrows():
            n = int(row["n_observations"])
            t = max(int(round(float(row[self.tight_col]) * n)), 0)
            b = max(int(round(float(row[self.broad_col]) * n)), 0)
            total_tight += t
            total_broad += b
            total_n += n
        return total_tight, total_broad, total_n

    @staticmethod
    def _test_2x2(a_in: int, a_n: int, b_in: int, b_n: int):
        """
        Fisher exact test on::

            [[a_in, a_n - a_in],
             [b_in, b_n - b_in]]

        Returns (p_greater, p_less, diff_prop).
          p_greater = one-sided p for "A proportion > B proportion"
          p_less    = one-sided p for "A proportion < B proportion"
          diff_prop = (a_in/a_n) − (b_in/b_n)
        """
        if a_n == 0 or b_n == 0:
            return np.nan, np.nan, np.nan

        table = np.array([[a_in, a_n - a_in],
                          [b_in, b_n - b_in]])

        if (table.sum(axis=1) == 0).any():
            return np.nan, np.nan, np.nan

        _, p_greater = fisher_exact(table, alternative="greater")
        _, p_less = fisher_exact(table, alternative="less")
        diff = a_in / a_n - b_in / b_n
        return p_greater, p_less, diff

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """Run all tests.  Returns one row per (model, unit, *hold_constant)."""
        a_label = self.group_a_label
        b_label = self.group_b_label

        slice_cols = self.hold_constant
        group_keys = ["model", self.unit_col] + slice_cols

        results = []

        df_a = self.df[self.df[self.group_col] == self.group_a_value]
        df_b = self.df[self.df[self.group_col] == self.group_b_value]

        keys_a = set(df_a.groupby(group_keys).groups.keys())
        keys_b = set(df_b.groupby(group_keys).groups.keys())
        common_keys = sorted(keys_a & keys_b)

        grouped_a = df_a.groupby(group_keys)
        grouped_b = df_b.groupby(group_keys)

        for key in common_keys:
            grp_a = grouped_a.get_group(key)
            grp_b = grouped_b.get_group(key)

            a_tight, a_broad, a_n = self._pool_counts(grp_a)
            b_tight, b_broad, b_n = self._pool_counts(grp_b)

            # Tight range test (one-sided both directions)
            t_p_gt, t_p_lt, t_diff = self._test_2x2(a_tight, a_n, b_tight, b_n)
            # Broad range test (broad includes tight)
            b_p_gt, b_p_lt, b_diff = self._test_2x2(a_broad, a_n, b_broad, b_n)

            # Unpack key
            if isinstance(key, tuple):
                model, unit_val = key[0], key[1]
                hold_vals = key[2:]
            else:
                model, unit_val = key, None
                hold_vals = ()

            row = {
                "model": model,
                self.unit_col: unit_val,
                # Counts
                f"{a_label}_n": a_n,
                f"{a_label}_tight": a_tight,
                f"{a_label}_broad": a_broad,
                f"{b_label}_n": b_n,
                f"{b_label}_tight": b_tight,
                f"{b_label}_broad": b_broad,
                # Proportions
                f"{a_label}_tight_prop": a_tight / a_n if a_n else np.nan,
                f"{a_label}_broad_prop": a_broad / a_n if a_n else np.nan,
                f"{b_label}_tight_prop": b_tight / b_n if b_n else np.nan,
                f"{b_label}_broad_prop": b_broad / b_n if b_n else np.nan,
                # Tight range one-sided p-values
                "tight_diff": t_diff,
                "tight_p_a_better": t_p_gt,
                "tight_p_b_better": t_p_lt,
                # Broad range one-sided p-values
                "broad_diff": b_diff,
                "broad_p_a_better": b_p_gt,
                "broad_p_b_better": b_p_lt,
            }
            for col, val in zip(slice_cols, hold_vals):
                row[col] = val

            if "ome" in grp_a.columns and "ome" not in row:
                row["ome"] = grp_a["ome"].iloc[0]

            results.append(row)

        df = pd.DataFrame(results)
        if df.empty:
            return df

        # Bonferroni correction — separate for each one-sided test family
        p_cols = [
            "tight_p_a_better", "tight_p_b_better",
            "broad_p_a_better", "broad_p_b_better",
        ]
        for p_col in p_cols:
            corr_col = f"{p_col}_bonf"
            valid = df[p_col].notna()
            if valid.any():
                _, pvals_corr, *_ = multipletests(
                    df.loc[valid, p_col], method="bonferroni", alpha=0.05,
                )
                df.loc[valid, corr_col] = pvals_corr
            else:
                df[corr_col] = np.nan

        # Combined result columns (Bonferroni-corrected)
        for rng in ("tight", "broad"):
            a_corr = f"{rng}_p_a_better_bonf"
            b_corr = f"{rng}_p_b_better_bonf"
            result_col = f"{rng}_result"
            df[result_col] = "n.s."
            df.loc[df[a_corr] < 0.05, result_col] = f"{a_label} sig. better"
            df.loc[df[b_corr] < 0.05, result_col] = f"{b_label} sig. better"

        # Uncorrected result columns (raw p < 0.05)
        for rng in ("tight", "broad"):
            a_raw = f"{rng}_p_a_better"
            b_raw = f"{rng}_p_b_better"
            result_col = f"{rng}_result_uncorrected"
            df[result_col] = "n.s."
            df.loc[df[a_raw] < 0.05, result_col] = f"{a_label} sig. better"
            df.loc[df[b_raw] < 0.05, result_col] = f"{b_label} sig. better"

        return df

    def overall_significance(self, df: pd.DataFrame) -> dict:
        """Compute overall significance tests on the distribution of results.

        For each range (tight/broad) and correction level (corrected/uncorrected),
        computes:
        1. **Sign test**: Among significant results, is the direction non-random?
           Binomial test with H₀: P(A better | significant) = 0.5.
        2. **Rate test**: Is the total number of significant results more than
           expected by chance? Binomial test with H₀: P(significant) = α_per_test.

        Returns a dict keyed by (range, correction) with sub-dicts of test results.
        """
        a_label = self.group_a_label
        b_label = self.group_b_label
        results = {}

        for rng in ("tight", "broad"):
            for corrected, suffix in [(True, ""), (False, "_uncorrected")]:
                result_col = f"{rng}_result{suffix}"
                if result_col not in df.columns:
                    continue

                n_valid = int(df[f"{rng}_p_a_better"].notna().sum())
                n_a = int((df[result_col] == f"{a_label} sig. better").sum())
                n_b = int((df[result_col] == f"{b_label} sig. better").sum())
                n_sig = n_a + n_b

                # Sign test: among significant results, is direction skewed?
                if n_sig > 0:
                    # Two-sided binomial: k successes in n_sig trials, p=0.5
                    sign_result = binomtest(max(n_a, n_b), n_sig, p=0.5,
                                            alternative="two-sided")
                    sign_p = sign_result.pvalue
                else:
                    sign_p = np.nan

                # Rate test: more significant results than expected by chance?
                if corrected:
                    # After Bonferroni, per-test α = 0.05 / n_valid
                    alpha_per_test = 0.05 / n_valid if n_valid else 0.05
                else:
                    alpha_per_test = 0.05

                if n_valid > 0 and n_sig > 0:
                    rate_result = binomtest(n_sig, n_valid, p=alpha_per_test,
                                            alternative="greater")
                    rate_p = rate_result.pvalue
                else:
                    rate_p = np.nan

                key = (rng, "corrected" if corrected else "uncorrected")
                results[key] = {
                    "n_valid": n_valid,
                    "n_a_better": n_a,
                    "n_b_better": n_b,
                    "n_sig": n_sig,
                    "n_ns": n_valid - n_sig,
                    "sign_test_p": sign_p,
                    "rate_test_p": rate_p,
                    "alpha_per_test": alpha_per_test,
                }
        return results

    def save(self, output_csv: str) -> pd.DataFrame:
        df = self.build()
        df.to_csv(output_csv, index=False)
        a_label = self.group_a_label
        b_label = self.group_b_label
        n_valid = int(df["tight_p_a_better"].notna().sum()) if len(df) else 0
        for rng in ("tight", "broad"):
            col = f"{rng}_result"
            if col in df.columns:
                a_sig = int((df[col] == f"{a_label} sig. better").sum())
                b_sig = int((df[col] == f"{b_label} sig. better").sum())
                ns = int((df[col] == "n.s.").sum())
                print(f"  {rng}: {a_label} sig. better {a_sig}, "
                      f"{b_label} sig. better {b_sig}, n.s. {ns}")
        print(f"  → {output_csv}  ({n_valid} tests)")
        return df

    def summarize(self, df: pd.DataFrame):
        """Print a summary for both tight and broad range tests."""
        a_label = self.group_a_label
        b_label = self.group_b_label

        overall = self.overall_significance(df)

        for rng in ("tight", "broad"):
            diff_col = f"{rng}_diff"

            # --- Corrected results ---
            result_col = f"{rng}_result"
            result_col_unc = f"{rng}_result_uncorrected"

            if result_col not in df.columns or df.empty:
                continue

            n_valid = int(df[f"{rng}_p_a_better"].notna().sum())

            print(f"\n{'─'*60}")
            print(f"  {rng.upper()} RANGE: {a_label} vs {b_label}")
            print(f"  {n_valid} tests, Bonferroni-corrected α = {0.05/n_valid:.2e}")
            print(f"{'─'*60}")

            for label_tag, rc in [("CORRECTED", result_col),
                                   ("UNCORRECTED", result_col_unc)]:
                if rc not in df.columns:
                    continue
                a_sig = df[rc] == f"{a_label} sig. better"
                b_sig = df[rc] == f"{b_label} sig. better"
                ns = df[rc] == "n.s."

                n_a = int(a_sig.sum())
                n_b = int(b_sig.sum())
                n_ns = int(ns.sum())
                mean_d_a = df.loc[a_sig, diff_col].mean() if n_a else 0
                mean_d_b = df.loc[b_sig, diff_col].mean() if n_b else 0

                print(f"\n  [{label_tag}]")
                print(f"  {a_label} sig. better:  {n_a:>4}  "
                      f"({n_a/n_valid*100:5.1f}%)"
                      + (f"  mean Δ = {mean_d_a:+.3f}" if n_a else ""))
                print(f"  {b_label} sig. better:  {n_b:>4}  "
                      f"({n_b/n_valid*100:5.1f}%)"
                      + (f"  mean Δ = {mean_d_b:+.3f}" if n_b else ""))
                print(f"  n.s.:                  {n_ns:>4}  "
                      f"({n_ns/n_valid*100:5.1f}%)")

                # Overall significance tests
                corr_key = "corrected" if label_tag == "CORRECTED" else "uncorrected"
                ov = overall.get((rng, corr_key))
                if ov:
                    if not np.isnan(ov["sign_test_p"]):
                        print(f"  Sign test (direction):  p = {ov['sign_test_p']:.4f}"
                              + (" *" if ov["sign_test_p"] < 0.05 else ""))
                    if not np.isnan(ov["rate_test_p"]):
                        print(f"  Rate test (> chance):   p = {ov['rate_test_p']:.2e}"
                              + (" *" if ov["rate_test_p"] < 0.05 else ""))

            # Breakdown by hold_constant columns
            if self.hold_constant:
                print(f"\n  By {', '.join(self.hold_constant)}:")
                for vals, grp in df.groupby(self.hold_constant):
                    if not isinstance(vals, tuple):
                        vals = (vals,)
                    label = ", ".join(
                        f"{c}={v}" for c, v in zip(self.hold_constant, vals)
                    )
                    a_ct = int((grp[result_col] == f"{a_label} sig. better").sum())
                    b_ct = int((grp[result_col] == f"{b_label} sig. better").sum())
                    ns_ct = int((grp[result_col] == "n.s.").sum())
                    total = len(grp)
                    parts = []
                    if a_ct: parts.append(f"{a_label} {a_ct}")
                    if b_ct: parts.append(f"{b_label} {b_ct}")
                    parts.append(f"n.s. {ns_ct}")
                    print(f"    {label}: {total} pairs  ({', '.join(parts)})")

            # By model
            print(f"\n  By model:")
            model_summary = (
                df.groupby("model")[result_col]
                .value_counts()
                .unstack(fill_value=0)
            )
            # Ensure all three columns exist
            for col_name in [f"{a_label} sig. better",
                             f"{b_label} sig. better", "n.s."]:
                if col_name not in model_summary.columns:
                    model_summary[col_name] = 0
            model_summary = model_summary.sort_index()
            for m, row in model_summary.iterrows():
                a_ct = int(row.get(f"{a_label} sig. better", 0))
                b_ct = int(row.get(f"{b_label} sig. better", 0))
                ns_ct = int(row.get("n.s.", 0))
                total = a_ct + b_ct + ns_ct
                parts = []
                if a_ct: parts.append(f"{a_label} {a_ct}")
                if b_ct: parts.append(f"{b_label} {b_ct}")
                parts.append(f"n.s. {ns_ct}")
                print(f"    {m}: {total} pairs  ({', '.join(parts)})")
