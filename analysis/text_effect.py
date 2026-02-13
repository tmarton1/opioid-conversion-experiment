"""
Analysis of vignette text effect on opioid conversion accuracy.

TextEffectDifference: computes per-model differences (empty − text) in tight
  and broad accuracy across prompt conditions and OME levels.

TextEffectSignificance: thin wrapper around AccuracyComparison for the
  specific text-vs-empty comparison (kept for backward compatibility).
"""

import pandas as pd
import numpy as np
from analysis.accuracy_comparison import AccuracyComparison


# ---------------------------------------------------------------------------
# 1. Difference analysis  (batch-means level)
# ---------------------------------------------------------------------------

class TextEffectDifference:
    """
    Compute per-model Δ(empty − text) in tight and broad accuracy for each
    prompt condition × OME level.  Saves a flat CSV ready for plotting.

    Columns produced
    ----------------
    model, ome_level, prompt_key, prompt_label,
    text_tight, text_broad, empty_tight, empty_broad,
    diff_tight, diff_broad
    """

    PROMPT_KEYS = ["naive", "nielsen", "explain"]
    PROMPT_LABELS = {"naive": "Naive", "nielsen": "w/ Nielsen", "explain": "+ Explain Units"}
    OME_LEVELS = [("all", "All Opioid Pairs"), ("ome", "Oral Morphine"), ("nonome", "w/o Oral Morphine")]

    def __init__(self, batch_means_csv: str):
        self.df = pd.read_csv(batch_means_csv)

    def build(self) -> pd.DataFrame:
        """Return a DataFrame with one row per (model, ome_level, prompt_key)."""
        rows = []
        for ome_key, ome_title in self.OME_LEVELS:
            for pk in self.PROMPT_KEYS:
                empty_name = f"empty_{ome_key}_{pk}"
                text_name = f"text_{ome_key}_{pk}"

                empty_df = self.df[self.df["batch_name"] == empty_name].set_index("model")
                text_df = self.df[self.df["batch_name"] == text_name].set_index("model")

                models = sorted(set(empty_df.index) & set(text_df.index))
                for m in models:
                    et = float(empty_df.loc[m, "mean_in_tight"])
                    eb = float(empty_df.loc[m, "mean_in_broad"])
                    tt = float(text_df.loc[m, "mean_in_tight"])
                    tb = float(text_df.loc[m, "mean_in_broad"])
                    rows.append({
                        "model": m,
                        "ome_level": ome_key,
                        "ome_title": ome_title,
                        "prompt_key": pk,
                        "prompt_label": self.PROMPT_LABELS[pk],
                        "text_tight": tt,
                        "text_broad": tb,
                        "empty_tight": et,
                        "empty_broad": eb,
                        "diff_tight": et - tt,
                        "diff_broad": eb - tb,
                    })
        return pd.DataFrame(rows)

    def save(self, output_csv: str) -> pd.DataFrame:
        df = self.build()
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved text-effect differences to {output_csv}  ({len(df)} rows)")
        return df


# ---------------------------------------------------------------------------
# 2. χ² significance test  (per scenario) — thin wrapper
# ---------------------------------------------------------------------------

class TextEffectSignificance:
    """
    Backward-compatible wrapper around AccuracyComparison for the
    text-vs-empty comparison.  Delegates all logic to AccuracyComparison.
    """

    def __init__(self, model_analysis_csv: str):
        self._comp = AccuracyComparison(
            csv_path=model_analysis_csv,
            group_col="has_vignette_text",
            group_a_value=True,
            group_b_value=False,
            group_a_label="text",
            group_b_label="empty",
            unit_col="scenario_key",
            hold_constant=["include_nielsen", "explain_units"],
        )

    def build(self) -> pd.DataFrame:
        return self._comp.build()

    def save(self, output_csv: str) -> pd.DataFrame:
        return self._comp.save(output_csv)

    def summarize(self, df: pd.DataFrame):
        return self._comp.summarize(df)
