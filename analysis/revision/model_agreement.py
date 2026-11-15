"""
Task 3D: agreement between models (cross-model consensus), DI 8.

Naive condition, Nielsen score, 13 models. Random-dose pairs (item = opioid
pair, experiment ``random_dose``) and TEXT vignettes (item = scenario_key,
experiment ``vignette``; note the old consensus file groups by vignette_id,
i.e. each scenario's four wordings separately; here wordings are pooled per
scenario, the pairing unit of every other revision analysis) are kept apart.

A *cell* is one model x item; its answer is the median log(observed/expected)
over that model's usable responses. A cell is *wrong* when its median ratio is
outside the tight band (0.85-1.15). Answers agree when they are within 10%
(max/min <= 1.10).

Per item (model_agreement.csv, row_type "item"):
  within_sd_median / within_cv_median  median over models of the SD of log ratio
                       / the CV of the raw answers (0 when a model repeats itself)
  within_*_wording_*   vignettes only: the same, per wording (vignette_id), the repeated-query
                       unit (the scenario-level within SD also contains wording differences)
  between_sd           SD (ddof=1) of the model medians (log)
  between_over_within  ratio; NaN when within is 0 (between_gt_within still set)
  consensus_*          the repository's weighted cross-model consensus
                       (analysis/opioid_pair_consensus.py): weights
                       1/(sd_log^2 + EPSILON_FOR_WEIGHT) over each model's mean
                       log ratio (the R^2 factor of the pair weights is omitted),
                       weighted variance with the 1 - sum(v^2) correction,
                       effective number of models, t-test vs Nielsen
  fold from Nielsen and from the nearest other standard
  wrong clusters       distinct values among the wrong models' medians (grouped
                       from the smallest upward, a group spans <= 10%), the size
                       of the largest group, and how many wrong cells share a
                       value with at least one other wrong model
Summary rows (row_type "summary", one per experiment) pool the items.
model_agreement_pairwise.csv: every model pair x experiment, share of items
with answers within 10% of each other, overall and where both are wrong.

    python -m analysis.revision.model_agreement
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from analysis.revision import stats
from common.config import EPSILON_FOR_WEIGHT, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER

SCORE = "nielsen"
EXPERIMENTS = {"random_dose": ("rd_naive", "opioid_pair"),
               "vignette": ("vignette_naive", "scenario_key")}
STANDARDS = ["cdc_2016", "cdc_2022", "nih_heal", "mcpherson_2018"]
AGREE = np.log(1.10)
OUT = "results_revision"


def _atomic(df: pd.DataFrame, name: str, out: str = OUT) -> None:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    df.to_csv(path + ".tmp", index=False, float_format="%.10g")
    os.replace(path + ".tmp", path)


def is_wrong(median_log) -> np.ndarray:
    r = np.exp(np.asarray(median_log, float))
    return (r < TIGHT_RANGE_LOWER) | (r > TIGHT_RANGE_UPPER)


def cells(long: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Model x item cells with median/mean/SD of log ratio and CV of answers."""
    cond, item = EXPERIMENTS[experiment]
    d = stats.responses(long, SCORE, cond)
    d = d[d["log_ratio"].notna()].copy()
    d["item"] = d[item]
    g = d.groupby(["item", "model"])
    c = g.agg(n=("log_ratio", "size"), median_log=("log_ratio", "median"),
              mean_log=("log_ratio", "mean"), sd_log=("log_ratio", lambda x: x.std(ddof=1)),
              mean_ans=("observed_to_dose", "mean"),
              sd_ans=("observed_to_dose", lambda x: x.std(ddof=1))).reset_index()
    c["cv"] = c["sd_ans"] / c["mean_ans"].abs()
    c["wrong"] = is_wrong(c["median_log"])
    c["experiment"] = experiment
    if experiment == "vignette":
        # within-wording SD / CV (model x scenario x wording, the repeated-query unit)
        w = d.groupby(["item", "model", "vignette_id"]).agg(
            sd=("log_ratio", lambda x: x.std(ddof=1)), m=("observed_to_dose", "mean"),
            s=("observed_to_dose", lambda x: x.std(ddof=1))).reset_index()
        w["cv"] = w["s"] / w["m"].abs()
        agg = w.groupby(["item", "model"]).agg(sd_wording=("sd", "median"), cv_wording=("cv", "median")).reset_index()
        c = c.merge(agg, on=["item", "model"], how="left")
    return c


def standard_rel(long: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Per item: each non-Nielsen standard's value relative to Nielsen (NaN
    where it does not cover the conversion), plus the between-standard spread."""
    cond, item = EXPERIMENTS[experiment]
    d = stats.responses(long, SCORE, cond)
    d = d.assign(item=d[item])
    rows = {}
    for it, g in d.groupby("item"):
        # one conversion per item; take the first response's expected values
        first = g.iloc[0]
        rows[it] = {s: (first[f"expected_{s}"] / first["expected_nielsen"]
                        if pd.notna(first[f"expected_{s}"]) else np.nan) for s in STANDARDS}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("item")


def group_values(vals: np.ndarray) -> list[int]:
    """Sizes of groups of log values, anchored at the smallest remaining value,
    each spanning at most 10% (log 1.10)."""
    v = np.sort(np.asarray(vals, float))
    sizes, i = [], 0
    while i < len(v):
        j = i
        while j < len(v) and v[j] - v[i] <= AGREE + 1e-12:
            j += 1
        sizes.append(j - i)
        i = j
    return sizes


def consensus(g: pd.DataFrame) -> dict:
    m = g[g["sd_log"].notna()]
    if len(m) < 2:
        return {}
    w = 1.0 / (m["sd_log"].values ** 2 + EPSILON_FOR_WEIGHT)
    v = w / w.sum()
    mu = m["mean_log"].values
    c = float(np.sum(v * mu))
    neff = 1.0 / float(np.sum(v ** 2))
    denom = 1.0 - float(np.sum(v ** 2))
    var = float(np.sum(v * (mu - c) ** 2) / denom) if denom > 0 else np.nan
    se = np.sqrt(var / neff) if np.isfinite(var) and neff > 1 else np.nan
    t = c / se if se and np.isfinite(se) and se > 0 else np.nan     # expected log ratio is 0
    p = 2 * (1 - sps.t.cdf(abs(t), df=neff - 1)) if np.isfinite(t) else np.nan
    return {"consensus_log_ratio": c, "consensus_ratio": float(np.exp(c)),
            "weighted_variance_log": var, "effective_n_models": neff, "p_value_log": p}


def item_table(c: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for it, g in c.groupby("item"):
        r = {"row_type": "item", "experiment": g["experiment"].iloc[0], "item": it,
             "n_models": len(g), "n_responses": int(g["n"].sum())}
        within = g["sd_log"].dropna()
        r["within_sd_median"] = within.median() if len(within) else np.nan
        r["within_cv_median"] = g["cv"].median()
        if "sd_wording" in g:
            r["within_sd_wording_median"] = g["sd_wording"].median()
            r["within_cv_wording_median"] = g["cv_wording"].median()
        r["between_sd"] = g["median_log"].std(ddof=1)
        r["between_over_within"] = (r["between_sd"] / r["within_sd_median"]
                                    if r["within_sd_median"] and r["within_sd_median"] > 0 else np.nan)
        r["between_gt_within"] = bool(r["between_sd"] > r["within_sd_median"])
        r["model_range_fold"] = float(np.exp(g["median_log"].max() - g["median_log"].min()))
        r.update(consensus(g))
        cl = r.get("consensus_log_ratio", np.nan)
        r["consensus_fold_from_nielsen"] = float(np.exp(abs(cl))) if np.isfinite(cl) else np.nan
        relv = rel.loc[it].dropna() if it in rel.index else pd.Series(dtype=float)
        r["n_other_standards"] = len(relv)
        if len(relv) and np.isfinite(cl):
            dist = (np.log(relv) - cl).abs()
            r["nearest_other_standard"] = dist.idxmin()
            r["consensus_fold_from_nearest_other"] = float(np.exp(dist.min()))
        w = g[g["wrong"]]["median_log"].values
        r["n_wrong"] = len(w)
        sizes = group_values(w)
        r["n_wrong_clusters"] = len(sizes)
        r["largest_wrong_cluster"] = max(sizes) if sizes else 0
        if len(w):
            share = [int((np.abs(w - x) <= AGREE + 1e-12).sum() >= 2) for x in w]
            r["n_wrong_sharing_value"] = int(sum(share))
        else:
            r["n_wrong_sharing_value"] = 0
        rows.append(r)
    return pd.DataFrame(rows)


def summary_row(t: pd.DataFrame, c: pd.DataFrame, exp: str) -> dict:
    w = c[c["wrong"]]
    return {"row_type": "summary", "experiment": exp, "item": "ALL",
            "n_items": len(t), "n_models": c["model"].nunique(), "n_responses": int(c["n"].sum()),
            "within_sd_median": t["within_sd_median"].median(),
            "within_cv_median": t["within_cv_median"].median(),
            "within_sd_wording_median": t.get("within_sd_wording_median", pd.Series(dtype=float)).median(),
            "within_cv_wording_median": t.get("within_cv_wording_median", pd.Series(dtype=float)).median(),
            "between_sd": t["between_sd"].median(),
            "between_over_within": t["between_over_within"].median(),
            "between_gt_within": t["between_gt_within"].mean(),      # share of items
            "model_range_fold": t["model_range_fold"].median(),
            "consensus_fold_from_nielsen": t["consensus_fold_from_nielsen"].median(),
            "consensus_fold_from_nearest_other": t["consensus_fold_from_nearest_other"].median(),
            "n_wrong": len(w),
            "n_wrong_clusters": t["n_wrong_clusters"].median(),
            "largest_wrong_cluster": t["largest_wrong_cluster"].median(),
            "n_wrong_sharing_value": int(t["n_wrong_sharing_value"].sum()),
            "share_wrong_sharing_value": t["n_wrong_sharing_value"].sum() / max(len(w), 1),
            "share_items_consensus_outside_tight": float(
                (t["consensus_fold_from_nielsen"] > TIGHT_RANGE_UPPER).mean()),
            "median_consensus_ratio": t["consensus_ratio"].median(),
            "median_effective_n_models": t["effective_n_models"].median(),
            "share_items_consensus_p_lt_005": float((t["p_value_log"] < 0.05).mean()),
            "share_cells_wrong": len(w) / len(c)}


def pairwise(c: pd.DataFrame, exp: str) -> pd.DataFrame:
    piv = c.pivot(index="item", columns="model", values="median_log")
    wr = c.pivot(index="item", columns="model", values="wrong")
    rows = []
    for a, b in itertools.combinations(sorted(piv.columns), 2):
        ok = piv[a].notna() & piv[b].notna()
        close = (piv[a] - piv[b]).abs() <= AGREE + 1e-12
        bw = ok & wr[a].fillna(False).astype(bool) & wr[b].fillna(False).astype(bool)
        rows.append({"experiment": exp, "model_a": a, "model_b": b, "n_items": int(ok.sum()),
                     "n_agree": int((close & ok).sum()), "share_agree": (close & ok).sum() / max(ok.sum(), 1),
                     "n_both_wrong": int(bw.sum()), "n_agree_when_wrong": int((close & bw).sum()),
                     "share_agree_when_wrong": (close & bw).sum() / bw.sum() if bw.sum() else np.nan})
    return pd.DataFrame(rows)


def build(long: pd.DataFrame | None = None):
    long = stats.load_long() if long is None else long
    items, pairs = [], []
    for exp in EXPERIMENTS:
        c = cells(long, exp)
        t = item_table(c, standard_rel(long, exp))
        items += [t, pd.DataFrame([summary_row(t, c, exp)])]
        pairs.append(pairwise(c, exp))
    return pd.concat(items, ignore_index=True), pd.concat(pairs, ignore_index=True)


def main():
    a, p = build()
    _atomic(a, "model_agreement.csv")
    _atomic(p, "model_agreement_pairwise.csv")
    print(a[a.row_type == "summary"].T.to_string())
    print(p.groupby("experiment")[["share_agree", "share_agree_when_wrong"]].agg(["mean", "median"]))


if __name__ == "__main__":
    main()
