"""
Score- and band-parameterised accuracy statistics (Phase 2 framework).

Everything here reads ``results_revision/responses_long.csv`` (one row per
response, the ratio under every registered score) and takes the score, the
condition, the conversion subset and the bands as parameters, looping over
the registries in ``scores.py``, ``tolerances.py`` and ``conditions.py``.

Levels of aggregation (mirroring the original pipeline):
  response  one model answer; ratio = observed / reference dose under the score
  cell      model x (vignette wording | opioid pair): what the original
            *_model_analysis.csv files hold (n_observations, band proportions)
  item      model x (vignette scenario | opioid pair): the pairing unit of the
            paired tests and the stratum of the CMH ranking
  model     mean over items of the item's mean cell proportion (= the ranking's
            mean_tight_prop) or mean over cells (= the batch means), plus
            distribution summaries over responses

Outcomes for paired tests: any registered band (proportion within the band,
counts pooled over the item's responses, as in comparison_*.csv) or
``median_abs_log_ratio`` (threshold-free: the item's median |log ratio|).

    from analysis.revision import stats
    long = stats.load_long()
    stats.model_summary(long, "non_nielsen", "vignette_naive", bands=["tight", "broad"])
    stats.paired_effect(long, "nielsen", "rd_naive", "rd_nielsen", outcome="median_abs_log_ratio")
    stats.rank_models(long, "all", "rd_naive", bands=("tight", "broad"))
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.prefix import harmonize
from analysis.revision.conditions import CONDITIONS, select
from analysis.revision.scores import SCORES
from analysis.revision.tolerances import ANCHOR_COLUMNS, BANDS, within
from common.models import CONSISTENT_MODELS

LONG_CSV = os.path.join("results_revision", "responses_long.csv")
THRESHOLD_FREE = "median_abs_log_ratio"
OVERDOSE, UNDERDOSE = 2.0, 0.5          # DZ 7 extreme-error thresholds
FLAGS = ["include_nielsen", "include_examples", "ask_for_units", "explain_units", "has_vignette_text"]


# ── loading ──────────────────────────────────────────────────────────────

def prepare_long(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise model names (as the pipeline does), keep the 13 models,
    make the condition flags boolean, and add opioid_pair, ome and the
    conversion key used by per-conversion bands."""
    df = df.copy()
    df["model"] = df["model"].map(harmonize)
    df = df[df["model"].isin(CONSISTENT_MODELS)]
    for c in FLAGS:
        if c in df.columns:
            v = df[c].map({True: True, False: False, "True": True, "False": False})
            df[c] = v.where(v.notna(), False).astype(bool)
    df["opioid_pair"] = df["from_opioid"].astype(str) + " -> " + df["to_opioid"].astype(str)
    # the pipeline's `ome` flag: oral morphine is the source or the target
    df["ome"] = df["from_opioid"].eq("oral_morphine") | df["to_opioid"].eq("oral_morphine")
    multi = df["from_opioid"].astype(str).str.contains(",")
    # Bands depend on the dose only for multi-drug vignettes (dose-weighted mix)
    df["conv_dose"] = np.where(multi, df["from_dose"].astype(str), "1")
    return df


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> pd.DataFrame:
    return prepare_long(pd.read_csv(path, low_memory=False))


def load_long(path: str = LONG_CSV) -> pd.DataFrame:
    """The prepared long table (cached per file version)."""
    return _load_cached(path, os.path.getmtime(path))


def ratio_col(score: str) -> str:
    if score not in SCORES:
        raise KeyError(f"unknown score {score!r}; registered: {list(SCORES)}")
    return f"ratio_{score}"


# ── response level ───────────────────────────────────────────────────────

def responses(long: pd.DataFrame, score: str, condition: str, subset: str = "all",
              bands: Sequence[str] = ()) -> pd.DataFrame:
    """Scored responses: rows the score covers, with ratio, log ratio, fold
    error and a 0/1 (NaN where undefined) column ``in_<band>`` per band.

    A row the score covers but whose answer is unusable (ratio missing or not
    positive) stays in and counts as outside every band, as in the original
    pipeline; it has no log ratio."""
    d = select(long, condition, subset)
    rc = ratio_col(score)
    if rc not in d.columns:
        raise KeyError(f"{rc} missing from the long table: rerun the rescore step")
    d = d[d[f"expected_{score}"].notna()] if f"expected_{score}" in d.columns else d
    d = d[d[f"nearest_standard_{score}"].notna()] if f"nearest_standard_{score}" in d.columns else d
    d = d.copy()
    r = pd.to_numeric(d[rc], errors="coerce")
    d["ratio"] = r
    ok = r > 0
    d["log_ratio"] = np.where(ok, np.log(r.where(ok, 1.0)), np.nan)
    d["fold_error"] = np.exp(np.abs(d["log_ratio"]))
    frame = pd.DataFrame({"from_opioid": d["from_opioid"], "from_dose": d["conv_dose"],
                          "to_opioid": d["to_opioid"], "observed_to_dose": d["observed_to_dose"],
                          **{c: d[c] for c in ANCHOR_COLUMNS if c in d.columns}}, index=d.index)
    for b in bands:
        d[f"in_{b}"] = within(b, r.where(ok), frame)
    d.attrs.update(score=score, condition=condition, subset=subset)
    return d


# ── cell / item / model levels ───────────────────────────────────────────

def _band_list(bands) -> list[str]:
    bands = list(bands)
    for b in bands:
        if b not in BANDS:
            raise KeyError(f"unknown band {b!r}; registered: {list(BANDS)}")
    return bands


def cell_table(long, score, condition, subset="all", bands=("tight", "broad")) -> pd.DataFrame:
    """Model x cell: n_observations and proportion within each band (the
    layout of the original *_model_analysis.csv files)."""
    bands = _band_list(bands)
    c = CONDITIONS[condition]
    d = responses(long, score, condition, subset, bands)
    keys = ["model", c.item] + ([c.cell] if c.cell != c.item else [])
    agg = {"n_observations": ("ratio", "size")}
    agg.update({f"in_{b}": (f"in_{b}", "mean") for b in bands})
    return d.groupby(keys, as_index=False).agg(**agg)


def item_table(long, score, condition, subset="all", bands=("tight", "broad")) -> pd.DataFrame:
    """Model x item: n, k and proportion within each band (counts pooled over
    the item's responses), mean cell proportion (what rankings average),
    median |log ratio| (threshold-free outcome), median signed log2 ratio."""
    bands = _band_list(bands)
    c = CONDITIONS[condition]
    d = responses(long, score, condition, subset, bands)
    g = d.groupby(["model", c.item])
    out = g.agg(n=("ratio", "size"),
                median_abs_log_ratio=("log_ratio", lambda s: np.nanmedian(np.abs(s)) if s.notna().any() else np.nan),
                median_log2_ratio=("log_ratio", lambda s: np.nanmedian(s) / np.log(2) if s.notna().any() else np.nan),
                ome=("ome", "first")).reset_index()
    for b in bands:
        k = g[f"in_{b}"].agg(lambda s: s.sum(min_count=1))
        out[f"k_{b}"] = k.to_numpy()
        out[f"prop_{b}"] = g[f"in_{b}"].mean().to_numpy()
    cells = cell_table(long, score, condition, subset, bands)
    cm = cells.groupby(["model", c.item])[[f"in_{b}" for b in bands]].mean()
    cm.columns = [f"cellmean_{b}" for b in bands]
    out = out.merge(cm.reset_index(), on=["model", c.item], how="left")
    return out.rename(columns={c.item: "item"})


def model_summary(long, score, condition, subset="all", bands=("tight", "broad"),
                  average: str = "item") -> pd.DataFrame:
    """Per model: accuracy within each band, and distribution summaries over
    responses: median fold error, median signed log2 ratio, IQR of log2 ratio,
    % > 2x and % < 0.5x.

    ``average`` matches the two averages the original pipeline uses:
      "item"  mean over items of the item's mean cell proportion: the CMH
              rankings (ranking_*.csv; figures 1, 8, 12, 15)
      "cell"  mean over cells: the batch means (model_batch_means*.csv;
              figures 11, 13, 14 and the grounding gains in figure_stats)
    They differ only for vignettes with a missing wording cell; for random
    doses cell = item."""
    bands = _band_list(bands)
    if average not in ("item", "cell"):
        raise ValueError("average must be 'item' or 'cell'")
    d = responses(long, score, condition, subset, bands)
    it = item_table(long, score, condition, subset, bands)
    cells = cell_table(long, score, condition, subset, bands) if average == "cell" else None
    rows = []
    for m, g in d.groupby("model"):
        lr2 = g["log_ratio"].dropna() / np.log(2)
        row = {"score": score, "condition": condition, "subset": subset, "model": m,
               "n_responses": len(g), "n_items": int((it["model"] == m).sum()),
               "median_fold_error": float(np.nanmedian(g["fold_error"])) if g["fold_error"].notna().any() else np.nan,
               "median_log2_ratio": float(lr2.median()) if len(lr2) else np.nan,
               "iqr_log2_ratio": float(lr2.quantile(.75) - lr2.quantile(.25)) if len(lr2) else np.nan,
               "pct_over_2x": 100 * float((g["ratio"] > OVERDOSE).mean()),
               "pct_under_half": 100 * float((g["ratio"] < UNDERDOSE).mean()),
               }
        for b in bands:
            src = it.loc[it["model"] == m, f"cellmean_{b}"] if average == "item" else cells.loc[cells["model"] == m, f"in_{b}"]
            row[f"pct_{b}"] = 100 * float(src.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(long, score, condition, subset="all") -> dict:
    """Responses and items the score covers (for figure annotations)."""
    d = responses(long, score, condition, subset)
    c = CONDITIONS[condition]
    return {"n_responses": int(len(d)), "n_items": int(d[c.item].nunique())}


# ── helpers for the overdose exceedance curve (E8) and gross-error counts ─
# The safety tails themselves (overdose, severe_overdose, underdose) are
# registry entries in analysis/revision/tolerances.py, anchored on the current
# standards, and flow through model_summary / paired_effect like any band.
SAFETY = {"overdose_x": 2.0, "severe_overdose_x": 4.0, "underdose_x": 0.5, "gross_error_x": 10.0}
# log grid 1x-10x plus the exact tail thresholds, so the curve at 2x and 4x equals the
# pooled overdose and severe-overdose rates (strict >, same 1e-12 tolerance as tolerances.within)
EXCEEDANCE_MULTIPLES = tuple(sorted(set(np.round(np.exp(np.linspace(0, np.log(10), 61)), 6)) | {2.0, 4.0}))


def dose_ratios(d: pd.DataFrame) -> pd.DataFrame:
    """observed / highest and observed / lowest current standard per response."""
    e = d[[c for c in ANCHOR_COLUMNS if c in d.columns]].apply(pd.to_numeric, errors="coerce")
    obs = pd.to_numeric(d["observed_to_dose"], errors="coerce")
    obs = obs.where(obs > 0)
    return pd.DataFrame({"to_highest": obs / e.max(axis=1), "to_lowest": obs / e.min(axis=1)}, index=d.index)


def gross_error_count(long, condition, subset="all") -> pd.DataFrame:
    """Per model + POOLED: responses >= 10x the highest current standard
    (likely unit or decimal errors; reported as a count in the text)."""
    d = responses(long, "nielsen", condition, subset)
    th = dose_ratios(d)["to_highest"]
    rows = [{"model": m, "n_ge_10x_highest": int((th[g.index] >= SAFETY["gross_error_x"]).sum())} for m, g in d.groupby("model")]
    rows.append({"model": "POOLED", "n_ge_10x_highest": int((th >= SAFETY["gross_error_x"]).sum())})
    return pd.DataFrame(rows)


def overdose_exceedance(long, condition, subset="all", multiples=EXCEEDANCE_MULTIPLES) -> pd.DataFrame:
    """Share of responses above each multiple of the highest current standard
    (per model and POOLED), for the overdose exceedance curve."""
    d = responses(long, "nielsen", condition, subset)
    d = d.assign(_th=dose_ratios(d)["to_highest"])
    rows = []
    for m, g in list(d.groupby("model")) + [("POOLED", d)]:
        th = g["_th"].to_numpy()
        for k in multiples:
            rows.append({"condition": condition, "subset": subset, "model": m, "multiple_of_highest": float(k),
                         "pct_above": 100 * float(np.mean(th > k + 1e-12)) if len(th) else np.nan})
    return pd.DataFrame(rows)


# ── paired tests ─────────────────────────────────────────────────────────

def _wilcoxon(diffs: pd.Series):
    """As analysis/per_model_wilcoxon.py: zero differences dropped,
    two-sided; (p, median, mean, n)."""
    diffs = pd.Series(diffs).dropna()
    n = len(diffs)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    nz = diffs[diffs != 0]
    if len(nz) < 1:
        return 1.0, 0.0, 0.0, n
    _, p = wilcoxon(nz, alternative="two-sided")
    return float(p), float(diffs.median()), float(diffs.mean()), n


def paired_items(long, score, cond_a, cond_b, subset="all", outcome="tight") -> pd.DataFrame:
    """Model x item outcome under two conditions on the same items.
    diff = b - a for accuracy bands (positive = more within the band under b);
    a - b for tails (lower share is better) and for the threshold-free
    outcome (smaller error under b), so a positive diff always means b is better."""
    bands = [] if outcome == THRESHOLD_FREE else [outcome]
    col = THRESHOLD_FREE if outcome == THRESHOLD_FREE else f"prop_{outcome}"
    a = item_table(long, score, cond_a, subset, bands)[["model", "item", "ome", col]]
    b = item_table(long, score, cond_b, subset, bands)[["model", "item", col]]
    m = a.merge(b, on=["model", "item"], suffixes=("_a", "_b"))
    lower_better = outcome == THRESHOLD_FREE or not BANDS[outcome].higher_is_better
    m["diff"] = (m[f"{col}_a"] - m[f"{col}_b"]) if lower_better else (m[f"{col}_b"] - m[f"{col}_a"])
    return m


def paired_effect(long, score, cond_a, cond_b, subset="all", outcome="tight") -> pd.DataFrame:
    """Per-model and POOLED Wilcoxon signed-rank test of the item-level
    difference (b better than a when positive). One row per model + POOLED."""
    m = paired_items(long, score, cond_a, cond_b, subset, outcome)
    col = THRESHOLD_FREE if outcome == THRESHOLD_FREE else f"prop_{outcome}"
    rows = []
    for model, g in list(m.groupby("model")) + [("POOLED", m)]:
        p, med, mean, n = _wilcoxon(g["diff"])
        rows.append({"score": score, "cond_a": cond_a, "cond_b": cond_b, "subset": subset,
                     "outcome": outcome, "model": model, "n_pairs": n, "p": p,
                     "median_diff": med, "mean_diff": mean,
                     "mean_a": float(g[f"{col}_a"].mean()), "mean_b": float(g[f"{col}_b"].mean())})
    return pd.DataFrame(rows)


# ── CMH ranking for any score and pair of bands ──────────────────────────

def _free_of(cells: pd.DataFrame, band: str) -> str:
    """Add the complement column ``free_<band>`` = 1 - in_<band> (NaN stays NaN)."""
    cells[f"free_{band}"] = 1.0 - cells[f"in_{band}"]
    return f"free_{band}"


def rank_models(long, score, condition, subset="all", bands=("tight", "broad")):
    """CMH ranking (analysis/model_ranking.py) under any score, on one or two
    registered bands. Returns the built ModelRanking (``result_df``, ``pairs_df``).

    Two bands (primary, secondary; the original behaviour, unchanged): models
    are sorted on the primary band's mean share (secondary breaks ties), one-sided
    CMH tests are run for both bands over all model pairs, Bonferroni over
    2 x pairs, and a model drops a tier when any model above it is significantly
    better on either band. ``result_df`` keeps the columns ``mean_tight_prop`` and
    ``mean_broad_prop`` for the primary and secondary band.

    One band (``bands=("broad",)``): the same machinery on that band alone. Sorted on
    the band's mean share, one CMH test per model pair, Bonferroni over the pairs
    (not twice the pairs, as duplicating the column in ModelRanking would count), same
    tier rule. ``result_df`` has the band's mean share in both ``mean_tight_prop``
    and ``mean_broad_prop`` (same value); ``n_tight``/``n_broad`` are both the
    band's count. The ranking object gets ``bands`` (as passed) and ``criterion``.

    Freedom from a tail: a lower-is-better registry entry (overdose, severe_overdose,
    underdose) is ranked on its complement, 1 - in_tail, so a higher share of
    responses outside the tail is better; the same CMH machinery applies (the
    tail's counts per cell become n - k). Allowed in one-band form, or in the
    two-band form with accuracy bands and tails mixed (each tail enters as its
    complement); the share columns then hold the complement ("free") shares.
    """
    from analysis.model_ranking import ModelRanking
    bands = _band_list(bands)
    if len(bands) not in (1, 2):
        raise ValueError("rank_models takes one band or two bands (primary, secondary)")
    c = CONDITIONS[condition]
    cells = cell_table(long, score, condition, subset, sorted(set(bands)))
    cols = []
    for b in bands:
        cols.append(f"in_{b}" if BANDS[b].higher_is_better else _free_of(cells, b))
    cells = cells.dropna(subset=cols)
    two = len(bands) == 2
    use = tuple(cols) if two else (cols[0], "_dup")
    if not two:
        cells = cells.assign(_dup=cells[cols[0]])
    r = ModelRanking(csv_path=None, filters={}, unit_col=c.item,
                     label=f"{condition} [{score}]", df=cells, range_cols=use)
    r.build()
    r.bands = tuple(bands)
    r.criterion = "+".join(("" if BANDS[b].higher_is_better else "free_") + b for b in bands)
    if not two:
        _single_band_bonferroni(r)
    return r


def _single_band_bonferroni(r, alpha: float = 0.05) -> None:
    """Redo the multiple-test correction and the tier assignment of a ModelRanking built
    on one column duplicated as both ranges: Bonferroni over the model pairs of that one
    band (NaN p-values excluded from the count, as in ModelRanking), then the same tier
    rule as ModelRanking.build."""
    pairs, res = r.pairs_df, r.result_df
    n_models = len(res)
    if not pairs.empty:
        p = pairs["tight_p"].astype(float)
        m = int(p.notna().sum())
        corr = (p * m).clip(upper=1.0)
        pairs["tight_p_bonf"] = corr
        pairs["broad_p_bonf"] = corr
        pairs["a_sig_better"] = corr < alpha
    sig = set()
    if not pairs.empty:
        sig = {(int(i), int(j)) for i, j, s in zip(pairs["idx_a"], pairs["idx_b"], pairs["a_sig_better"]) if s}
    adjusted = [1] + [0] * (n_models - 1)
    for j in range(1, n_models):
        sup = [adjusted[i] for i in range(j) if (i, j) in sig]
        adjusted[j] = max(sup) + 1 if sup else adjusted[j - 1]
    res["adjusted_rank"] = adjusted
    r.n_total_tests = len(pairs) if not pairs.empty else 0
