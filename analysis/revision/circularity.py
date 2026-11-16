"""
Task 3B: circularity test of the reference-grounding effect.

    python -m analysis.revision.circularity      # writes results_revision/circularity.csv

Grounding effect = naive vs Nielsen-grounded prompts (text vignettes; random
doses mg/day), re-scored under every score in the registry. Subsets: all, ome
(oral morphine is source or target, the pipeline flag), nonome, and
"disagree" (conversions where at least one current non-Nielsen standard differs
from Nielsen by more than 15%: any_outside_tight_current in
results_revision/references/by_conversion.csv; filtered here by item).

Outcomes (reported set, analysis/revision/reported_set.py): median_fold_error, tight and broad per
score and subset (geomean: all and disagree only); the standard-anchored entries (above_range,
below_range, overdose, severe_overdose, underdose) do not depend on the score, so they are written
once with score "any" (subset all; no threshold-free columns); scored without Nielsen only
above_range and below_range are kept (subset all).
Rows: one per experiment x subset x score x outcome x model, where model is
each of the 13 models plus ALL_MODELS (mean over models; pooled Wilcoxon p;
counts of models with a significant effect).
  before / after  per-model value under naive / Nielsen prompts (bands: % within,
                  mean over cells, as the batch means; fold error: median over
                  responses). ALL_MODELS: mean over the 13 models.
  gain            accuracy bands: after - before (pp); tails (overdose,
                  severe_overdose, underdose; lower is better) and median fold
                  error: before - after. Positive = grounding helps in every case.
  band_*          Wilcoxon on the item proportion within the band (NaN for the
                  median-fold-error outcome); tf_* Wilcoxon on the item's median
                  |log ratio| (threshold-free; the same for every outcome row of a
                  subset and score). Positive mean_diff = grounding better.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.reported_set import ANCHORED, ANCHORED_SOURCE_SCORE, ANY, REPORTED_SET, accuracy_subsets
from analysis.revision.scores import SCORES
from analysis.revision.tolerances import BANDS
from analysis.revision.distribution_metrics import write_csv

EXPERIMENTS = {"vignette": ("vignette_naive", "vignette_nielsen", "scenario_key"),
               "random_dose": ("rd_naive", "rd_nielsen", "opioid_pair")}
SUBSETS = ("all", "ome", "nonome", "disagree")
BY_CONV = os.path.join("results_revision", "references", "by_conversion.csv")
ALPHA = 0.05


def disagree_items() -> dict[str, set]:
    b = pd.read_csv(BY_CONV)
    b = b[b["any_outside_tight_current"].astype(str).eq("True")]
    return {e: set(g["conversion"]) for e, g in b.groupby("experiment")}


def _rows(d, ssub, score, ca, cb, exp, sub, label, outcomes, tables, ma, mb, tf):
    """Rows (one per outcome x model + ALL_MODELS) for one experiment x subset x score.
    ``outcomes`` maps outcome name -> band (None for median fold error); ``label`` is the score written
    ("any" for the score-independent entries, which carry no threshold-free columns)."""
    rows = []
    n_items = int(tf.loc["POOLED", "n_pairs"])
    for oc, band in outcomes.items():
        bp = stats.paired_effect(d, score, ca, cb, ssub, band, tables=tables).set_index("model") if band else None
        col = "median_fold_error" if band is None else f"pct_{band}"
        models = list(ma.index.intersection(mb.index))
        for model in models + ["ALL_MODELS"]:
            pm = "POOLED" if model == "ALL_MODELS" else model
            sel = models if model == "ALL_MODELS" else [model]
            before, after = float(ma.loc[sel, col].mean()), float(mb.loc[sel, col].mean())
            r = {"experiment": exp, "subset": sub, "score": label, "outcome": oc, "model": model,
                 "n_items": n_items if model == "ALL_MODELS" else int(tf.loc[pm, "n_pairs"]),
                 "n_responses_naive": int(ma.loc[sel, "n_responses"].sum()),
                 "n_responses_grounded": int(mb.loc[sel, "n_responses"].sum()),
                 "before": before, "after": after,
                 # gain > 0 = grounding helps: fold error and tails (lower is
                 # better) fall; accuracy bands rise (polarity from the registry)
                 "gain": (before - after) if (band is None or not BANDS[band].higher_is_better)
                 else (after - before)}
            if band is not None:
                r.update(band_n_pairs=int(bp.loc[pm, "n_pairs"]), band_mean_diff=bp.loc[pm, "mean_diff"],
                         band_p=bp.loc[pm, "p"])
            if label != ANY:
                r.update(tf_mean_diff=tf.loc[pm, "mean_diff"], tf_median_diff=tf.loc[pm, "median_diff"],
                         tf_p=tf.loc[pm, "p"])
            if model == "ALL_MODELS":
                pmods = list(models)
                if band is not None:
                    r["band_models_sig_better"] = int(((bp.loc[pmods, "p"] < ALPHA) & (bp.loc[pmods, "mean_diff"] > 0)).sum())
                    r["band_models_sig_worse"] = int(((bp.loc[pmods, "p"] < ALPHA) & (bp.loc[pmods, "mean_diff"] < 0)).sum())
                if label != ANY:
                    r["tf_models_sig_better"] = int(((tf.loc[pmods, "p"] < ALPHA) & (tf.loc[pmods, "mean_diff"] > 0)).sum())
                    r["tf_models_sig_worse"] = int(((tf.loc[pmods, "p"] < ALPHA) & (tf.loc[pmods, "mean_diff"] < 0)).sum())
            rows.append(r)
    return rows


def _prepare(d, score, ca, cb, ssub, bands):
    ma = stats.model_summary(d, score, ca, ssub, bands, average="cell").set_index("model")
    mb = stats.model_summary(d, score, cb, ssub, bands, average="cell").set_index("model")
    tables = {ca: stats.item_table(d, score, ca, ssub, bands), cb: stats.item_table(d, score, cb, ssub, bands)}
    tf = stats.paired_effect(d, score, ca, cb, ssub, stats.THRESHOLD_FREE, tables=tables).set_index("model")
    return ma, mb, tables, tf


def build(long: pd.DataFrame) -> pd.DataFrame:
    """Reported set only (analysis/revision/reported_set.py): accuracy outcomes per score and subset;
    standard-anchored entries once (score "any", subset all) and, scored without Nielsen, above / below range."""
    dis = disagree_items()
    cfg = REPORTED_SET["circularity"]
    acc = {"median_fold_error": None, **{b: b for b in cfg["accuracy_outcomes"] if b != "median_fold_error"}}
    rows = []
    for exp, (ca, cb, item_col) in EXPERIMENTS.items():
        for sub in SUBSETS:
            if sub == "disagree":
                d, ssub = long[long[item_col].isin(dis[exp])], "all"
            else:
                d, ssub = long, sub
            for score in SCORES:
                if sub not in accuracy_subsets(score):
                    continue
                bands = [b for b in acc.values() if b]
                ma, mb, tables, tf = _prepare(d, score, ca, cb, ssub, bands)
                rows += _rows(d, ssub, score, ca, cb, exp, sub, score, acc, tables, ma, mb, tf)
                if sub == "all" and score in cfg["anchored"]["coverage_restricted"]:
                    oc = {b: b for b in cfg["anchored"]["coverage_restricted_outcomes"]}
                    ma, mb, tables, tf = _prepare(d, score, ca, cb, ssub, list(oc))
                    rows += _rows(d, ssub, score, ca, cb, exp, sub, score, oc, tables, ma, mb, tf)
            if sub == "all":
                oc = {b: b for b in ANCHORED}
                ma, mb, tables, tf = _prepare(d, ANCHORED_SOURCE_SCORE, ca, cb, ssub, list(oc))
                rows += _rows(d, ssub, ANCHORED_SOURCE_SCORE, ca, cb, exp, sub, ANY, oc, tables, ma, mb, tf)
    return pd.DataFrame(rows)


def main() -> None:
    write_csv(build(stats.load_long()), "circularity.csv")
    print("wrote circularity.csv")


if __name__ == "__main__":
    main()
