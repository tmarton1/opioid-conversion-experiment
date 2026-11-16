"""
Grounding effect on answers above / below the range of published standards, by
prompt type, on comparable items (author request after decision 35).

    python -m analysis.revision.range_rates_by_prompt

Naive -> Nielsen supplied, for:
  random doses   the conversions matched to vignettes (figure 10; subset
                 "matched"), and those matched to the 16 same-conversion
                 scenarios ("matched_same_conversion");
  empty prompts  all 20 scenarios, the 16 same-conversion scenarios, and the 4
                 reduction scenarios as a check;
  text vignettes the same three subsets.
Comparing random doses with empty prompts isolates the specific vignette dose;
empty prompts with text vignettes isolates the clinical narrative.

Outcomes: above_range, below_range (registry entries; lower is better) and, for
context, broad. Tests at three levels (stats.paired_effect): the item-level
paired Wilcoxon within each model (n models significantly better / worse,
p < 0.05) and a pooled Wilcoxon over all model x item pairs. Values are means
of item proportions over model x item pairs (the paired subset), in percent;
a positive diff means Nielsen supplied is better (fewer answers outside the
range; more within broad).

Writes results_revision/range_rates_by_prompt.csv (one row per prompt type x
subset x outcome x model, POOLED rows included).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.conditions import CONDITIONS

OUT = os.path.join("results_revision", "range_rates_by_prompt.csv")
COMPARISONS = [
    ("random_doses", "rd_naive", "rd_nielsen", ("matched", "matched_same_conversion")),
    ("empty_prompts", "vignette_empty_naive", "vignette_empty_nielsen", ("all", "same_conversion", "reduction")),
    ("text_vignettes", "vignette_naive", "vignette_nielsen", ("all", "same_conversion", "reduction")),
]
OUTCOMES = ["above_range", "below_range", "broad"]


def build(long=None) -> pd.DataFrame:
    long = stats.load_long() if long is None else long
    rows = []
    for prompt, a, b, subsets in COMPARISONS:
        for sub in subsets:
            for outcome in OUTCOMES:
                e = stats.paired_effect(long, "nielsen", a, b, sub, outcome)
                pm = e[e.model != "POOLED"]
                sig_b = (pm.p < 0.05) & (pm.mean_diff > 0)
                sig_w = (pm.p < 0.05) & (pm.mean_diff < 0)
                items = stats.paired_items(long, "nielsen", a, b, sub, outcome)["item"].nunique()
                for _, r in e.iterrows():
                    rows.append({"prompt": prompt, "experiment": CONDITIONS[a].experiment, "cond_a": a, "cond_b": b,
                                 "subset": sub, "outcome": outcome, "model": r.model, "n_items": int(items),
                                 "n_pairs": int(r.n_pairs),
                                 "pct_naive": 100 * r.mean_a, "pct_nielsen": 100 * r.mean_b,
                                 "mean_diff_pp": 100 * r.mean_diff, "p": r.p,
                                 "models_sig_better": int(sig_b.sum()) if r.model == "POOLED" else np.nan,
                                 "models_sig_worse": int(sig_w.sum()) if r.model == "POOLED" else np.nan})
    return pd.DataFrame(rows)


def main():
    df = build()
    df.to_csv(OUT + ".tmp", index=False, float_format="%.10g")
    os.replace(OUT + ".tmp", OUT)
    show = df[df.model == "POOLED"][["prompt", "subset", "outcome", "n_items", "pct_naive", "pct_nielsen",
                                     "mean_diff_pp", "p", "models_sig_better", "models_sig_worse"]]
    pd.set_option("display.width", 250)
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
