"""
Model rankings by criterion and their rank correlations (FA.4, item 2).

    python -m analysis.revision.rankings_by_criterion

For each experiment (text vignettes, random doses), arm (naive; Nielsen supplied
as the grounded arm) and score (nielsen, all, non_nielsen, geomean), the 13
models are ranked on three criteria, one ranking per criterion and no composite.
Broad is the headline ranking.

  broad            share of responses within the broad band (0.6-1.7x). CMH
                   ranking on that one band (stats.rank_models, bands=("broad",)):
                   models sorted on the mean over items of the item's mean share,
                   one-sided Cochran-Mantel-Haenszel tests over all model pairs
                   stratified by item, Bonferroni over the pairs, a model drops a
                   tier when any model above it is significantly better.
  free_overdose    freedom from overdose: the same CMH ranking on the complement
                   of the overdose tail (1 - share > 2x the highest current
                   standard). Higher is better. The tail is anchored on the
                   standards, so the same under every score; only the conversions
                   a score covers differ.
  median_fold_error  per item (the vignette scenario / the opioid pair) the model's
                   median fold error (median |log ratio|); models are ranked within
                   each item (rank 1 = smallest error, ties averaged) and the mean
                   within-item rank is the value (Friedman style), lower is better.
                   Items where any model has no usable answer are dropped (complete
                   blocks; ``n_items_used`` vs ``n_items``).
                   Post-hoc tiers: the Friedman omnibus test is reported
                   (``friedman_p``); a model is significantly worse than a better
                   one when their mean ranks differ by more than the Nemenyi
                   critical difference CD = q(alpha, k, inf)/sqrt(2) *
                   sqrt(k(k+1)/(6n)) (alpha 0.05, k models, n items), and tiers
                   follow the CMH rule (a model drops a tier below the deepest
                   tier of any better model it is significantly worse than,
                   otherwise it shares the previous model's tier). Models with
                   equal mean rank are tied (same ``order``).

rankings_by_criterion.csv  one row per score x condition x criterion x model:
    value (broad, free_overdose: mean % over items; median_fold_error: mean
    within-item rank), median_fold_error_pooled (median fold error over all of the
    model's responses), order (1 = best by value; equal values share an order),
    tier (1 = best; models in one tier are tied), n_in_tier, tied (n_in_tier > 1),
    n_items, n_items_used, n_responses, tier_rule.
ranking_correlations.csv   Spearman rho on the criterion values and Kendall
    tau-b on the tiers (ties kept), between criteria within each score x condition
    (comparison = "criteria") and between scores within each condition x
    criterion (comparison = "scores"). Values are oriented so that better is
    higher (the mean rank enters as its negative) and tiers so that tier 1 is best.
    NaN where either side has no variation (all models in one tier).

Deterministic: no random numbers.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kendalltau, spearmanr, studentized_range

from analysis.revision import stats
from analysis.revision.scores import SCORES
from common.models import CONSISTENT_MODELS

OUT = "results_revision"
CONDS = ["vignette_naive", "rd_naive", "vignette_nielsen", "rd_nielsen"]
CRITERIA = ["broad", "free_overdose", "median_fold_error"]
ALPHA = 0.05
CMH_RULE = "CMH one-sided, Bonferroni over model pairs (alpha 0.05); a model drops a tier below any model significantly better"
NEMENYI_RULE = ("Nemenyi critical difference on mean within-item rank (alpha 0.05); a model drops a tier below any "
                "better model it differs from by more than CD")


def _write(df: pd.DataFrame, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    df.to_csv(path + ".tmp", index=False, float_format="%.10g")
    os.replace(path + ".tmp", path)


def tiers_from(order, sig) -> dict:
    """Tier rule of the Reanalysis rankings. ``order`` lists the models best first;
    ``sig(i, j)`` (i < j) is True when order[i] is significantly better than order[j].
    Tier 1 for the first; model j gets the deepest tier among the models significantly
    better than it, plus one, or else the previous model's tier."""
    t = [1] * len(order)
    for j in range(1, len(order)):
        sup = [t[i] for i in range(j) if sig(i, j)]
        t[j] = max(sup) + 1 if sup else t[j - 1]
    return dict(zip(order, t))


def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    return float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n)))


def friedman_tiers(w: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """w: items x models matrix of errors, complete blocks, lower is better. Returns the
    mean within-item rank per model (average ranks for ties), tiers by the Nemenyi rule,
    the critical difference, the number of items and the Friedman omnibus p."""
    ranks = w.rank(axis=1, method="average")
    mean_rank = ranks.mean()
    k, n = w.shape[1], w.shape[0]
    cd = nemenyi_cd(k, n, alpha)
    order = list(mean_rank.sort_values(kind="mergesort").index)
    tiers = tiers_from(order, lambda i, j: mean_rank[order[j]] - mean_rank[order[i]] > cd + 1e-12)
    p = float(friedmanchisquare(*[w[m].to_numpy() for m in w.columns]).pvalue) if n >= 2 and k >= 3 else np.nan
    return {"mean_rank": mean_rank.to_dict(), "tiers": tiers, "cd": cd, "n": n, "friedman_p": p}


def _cmh_rows(r, score, cond, criterion, scale=100.0) -> list[dict]:
    res = r.result_df
    return [{"score": score, "condition": cond, "criterion": criterion, "model": m,
             "value": scale * v, "tier": int(t), "n_items": int(nu), "n_items_used": int(nu),
             "n_responses": int(nt), "tier_rule": CMH_RULE}
            for m, v, t, nu, nt in zip(res["model"], res["mean_tight_prop"], res["adjusted_rank"],
                                       res["n_units"], res["n_total"])]


def criterion_rows(long, score: str, cond: str) -> list[dict]:
    rows = _cmh_rows(stats.rank_models(long, score, cond, "all", bands=("broad",)), score, cond, "broad")
    rows += _cmh_rows(stats.rank_models(long, score, cond, "all", bands=("overdose",)), score, cond, "free_overdose")
    d = stats.responses(long, score, cond)
    pooled = d.groupby("model")["fold_error"].median()
    nresp = d.groupby("model").size()
    it = stats.item_table(long, score, cond, "all", [])
    w = it.pivot(index="item", columns="model", values="median_abs_log_ratio")
    n_all = len(w)
    w = w.dropna()
    f = friedman_tiers(w)
    for m in w.columns:
        rows.append({"score": score, "condition": cond, "criterion": "median_fold_error", "model": m,
                     "value": f["mean_rank"][m], "tier": int(f["tiers"][m]), "n_items": n_all,
                     "n_items_used": f["n"], "n_responses": int(nresp[m]), "tier_rule": NEMENYI_RULE,
                     "critical_difference": f["cd"], "friedman_p": f["friedman_p"]})
    for r in rows:
        r["median_fold_error_pooled"] = float(pooled.get(r["model"], np.nan))
    return rows


def finish(rows: pd.DataFrame) -> pd.DataFrame:
    """Add order (1 = best by value, equal values share it), n_in_tier and tied."""
    out = []
    for _, g in rows.groupby(["score", "condition", "criterion"], sort=False):
        g = g.copy()
        lower = g["criterion"].iloc[0] == "median_fold_error"
        key = g["value"].round(9)
        g["order"] = key.rank(method="min", ascending=lower).astype(int)
        g["n_in_tier"] = g.groupby("tier")["model"].transform("size")
        g["tied"] = g["n_in_tier"] > 1
        out.append(g.sort_values(["order", "model"], kind="mergesort"))
    df = pd.concat(out, ignore_index=True)
    cols = ["score", "condition", "criterion", "model", "value", "median_fold_error_pooled", "order", "tier",
            "n_in_tier", "tied", "n_items", "n_items_used", "n_responses", "critical_difference", "friedman_p",
            "tier_rule"]
    return df.reindex(columns=cols)


def _oriented(g: pd.DataFrame) -> pd.DataFrame:
    """model-indexed (value, tier) oriented so that higher value is better; tier 1 stays best."""
    sign = -1.0 if g["criterion"].iloc[0] == "median_fold_error" else 1.0
    return pd.DataFrame({"value": sign * g["value"].to_numpy(), "tier": g["tier"].to_numpy()}, index=g["model"].to_numpy())


def _corr(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    if len(j) < 3:
        return {"n_models": len(j), "spearman_value": np.nan, "kendall_tau_b_tiers": np.nan}
    def safe(f, x, y):
        if x.nunique() < 2 or y.nunique() < 2:
            return np.nan
        return float(f(x, y).statistic)
    # tiers: smaller is better, so correlate the negated tiers (the sign is the same on both sides)
    return {"n_models": len(j), "spearman_value": safe(spearmanr, j["value_a"], j["value_b"]),
            "kendall_tau_b_tiers": safe(kendalltau, -j["tier_a"], -j["tier_b"])}


def correlations(rank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sc, cond), g in rank.groupby(["score", "condition"], sort=False):
        by = {c: _oriented(x) for c, x in g.groupby("criterion")}
        for a, b in itertools.combinations(CRITERIA, 2):
            rows.append({"comparison": "criteria", "score": "", "condition": cond, "criterion": "",
                         "a": a, "b": b, "within_score": sc, **_corr(by[a], by[b])})
    for (cond, crit), g in rank.groupby(["condition", "criterion"], sort=False):
        by = {s: _oriented(x) for s, x in g.groupby("score")}
        for a, b in itertools.combinations(list(SCORES), 2):
            rows.append({"comparison": "scores", "score": "", "condition": cond, "criterion": crit,
                         "a": a, "b": b, "within_score": "", **_corr(by[a], by[b])})
    df = pd.DataFrame(rows).drop(columns=["score"])
    return df[["comparison", "condition", "within_score", "criterion", "a", "b", "n_models",
               "spearman_value", "kendall_tau_b_tiers"]]


def build(long: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = stats.load_long() if long is None else long
    rows = []
    for cond in CONDS:
        for sc in SCORES:
            rows += criterion_rows(long, sc, cond)
    rank = finish(pd.DataFrame(rows))
    assert set(rank["model"]) <= set(CONSISTENT_MODELS)
    return rank, correlations(rank)


def main():
    rank, corr = build()
    _write(rank, "rankings_by_criterion.csv")
    _write(corr, "ranking_correlations.csv")
    pd.set_option("display.width", 250)
    c = corr[(corr.comparison == "criteria") & (corr.within_score == "nielsen")]
    print(c.round(2).to_string(index=False))
    s = corr[corr.comparison == "scores"].groupby(["condition", "criterion"])[["spearman_value", "kendall_tau_b_tiers"]].min()
    print(s.round(2))


if __name__ == "__main__":
    main()
