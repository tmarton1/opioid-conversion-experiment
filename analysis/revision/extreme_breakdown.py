"""
Breakdown of extreme overdoses, and per-model tail tests (follow-up, 2026-11-15).

    python -m analysis.revision.extreme_breakdown

Population: responses in the severe-overdose tail (> 4x the highest current
standard for the conversion) and the subset >= 10x the highest standard, in
text vignettes (naive, Nielsen supplied, Nielsen + units) and random doses
(naive, Nielsen supplied). The tails are in dose terms, so the same under
every score (Nielsen-score rows, which cover every conversion).

Each response is classified by what its ratio to a current standard's expected
dose (observed / expected_s, for any of Nielsen, CDC 2022, NIH HEAL,
McPherson 2018) matches, within the tight tolerance (0.85-1.15x of the
candidate multiplier):
  factor_inversion  multiplied where it should divide (or the reverse), on the
                    whole conversion (expected x or / F^2, F = from-to factor)
                    or on one step through oral morphine (expected x or /
                    the source or target factor^2);
  unit_error        mg vs mcg (x1000; only where a unit in the conversion is in
                    mcg), mcg/h vs mg/day (x41.7 or x24; only where one is
                    mcg/hr), per 4 h vs per day (x6; only for mg/4hr targets);
  other             neither.
Inversion is checked first; a response matching both is counted as inversion
and flagged `also_unit`. The classes are a heuristic reading of the numbers;
the response is a structured JSON value, so the reasoning is not available.
`stated_unit` is the unit in the answer when the prompt asked for one,
otherwise "not asked" (the requested unit is `requested_unit`).

Writes:
  results_revision/extreme_errors_breakdown.csv   one row per response > 4x highest
  results_revision/extreme_errors_classes.csv     counts and class shares by
                                                  condition x tail x (model | item | ALL),
                                                  with n_models and n_cells contributing
  results_revision/safety_tests_by_model.csv      per-model and pooled paired Wilcoxon
                                                  (items within model; model x item
                                                  pairs pooled) of each tail, naive vs
                                                  Nielsen supplied (vignettes, random
                                                  doses) and naive vs Nielsen + units
                                                  (vignettes); positive diff = fewer in
                                                  the tail when grounded
"""
from __future__ import annotations

import ast
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.conditions import CONDITIONS
from analysis.revision.rescore import RAW_FILES, _factor, _parse_sources
from analysis.revision.reported_set import REPORTED_SET
from analysis.revision.scores import SCORING_INPUTS
from common.config import TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER

OUT = "results_revision"
CONDS = ["vignette_naive", "vignette_nielsen", "vignette_nielsen_units", "rd_naive", "rd_nielsen"]
TESTS = [("vignette_naive", "vignette_nielsen"), ("vignette_naive", "vignette_nielsen_units"),
         ("rd_naive", "rd_nielsen")]
UNIT_MULTIPLIERS = {"mg_vs_mcg_x1000": 1000.0, "mcg_per_h_vs_mg_per_day_x41.7": 1000.0 / 24,
                    "hourly_vs_daily_x24": 24.0, "per_4h_vs_per_day_x6": 6.0}


def _write(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path + ".tmp", index=False, float_format="%.10g")
    os.replace(path + ".tmp", path)


@lru_cache(maxsize=None)
def _inversion_multipliers(standard, from_opioid, from_dose, to_opioid):
    froms, doses = _parse_sources(from_opioid, from_dose)
    sf = [_factor(standard, f) for f in froms]
    st = _factor(standard, to_opioid)
    if st is None or any(v is None for v in sf):
        return {}
    base = sum(d * f for d, f in zip(doses, sf)) / st
    out = {"whole_conversion": sum(d * st / f for d, f in zip(doses, sf)) / base,
           "target_step": st * st,
           "source_step": sum(d / f for d, f in zip(doses, sf)) / st / base}
    return {k: v for k, v in out.items() if abs(np.log(v)) > np.log(1.5)}   # distinguishable from correct


def _unit_possible(name, row) -> bool:
    """A unit error is counted only where the prompt makes that confusion
    possible: mg/mcg where a unit in the conversion is in mcg; mcg/h vs mg/day
    where one is mcg/hr; per 4 h vs per day where the target is mg/4hr (or mg/4 hours)."""
    units = f"{row.get('from_opioid_unit', '')} {row.get('to_opioid_unit', '')}"
    if name.startswith("mg_vs_mcg"):
        return "mcg" in units
    if name.startswith(("mcg_per_h", "hourly")):
        return "mcg/hr" in units
    if name.startswith("per_4h"):
        to = str(row.get("to_opioid_unit", "")).replace(" ", "")
        return "4hr" in to or "4hours" in to     # "mg/4hr" (vignettes) and "mg/4 hours" (dosing-unit arms)
    return True


def _match(ratio, m):
    return TIGHT_RANGE_LOWER * m <= ratio <= TIGHT_RANGE_UPPER * m


def match_lists(row, unit_both_directions: bool = False) -> tuple[list, list]:
    """Matches of one response: (factor-inversion matches, unit-error matches),
    each a list of labels ``<pattern>@<standard>``; empty when none. Shared with
    analysis/revision/error_taxonomy.py. Unit errors are the multiplier only (the
    answer too large) unless ``unit_both_directions``, which also tries the
    reciprocal (the answer too small; label suffix ``_inverse``)."""
    inv, unit = [], []
    for s in SCORING_INPUTS:
        e = row.get(f"expected_{s}")
        if e is None or not np.isfinite(e) or e <= 0:
            continue
        r = row["observed_to_dose"] / e
        for name, m in _inversion_multipliers(s, row["from_opioid"], row["from_dose"], row["to_opioid"]).items():
            for mm, sign in ((m, "x"), (1 / m, "/")):
                if _match(r, mm):
                    inv.append(f"{name}{sign}F^2@{s}")
        for name, m in UNIT_MULTIPLIERS.items():
            if not _unit_possible(name, row):
                continue
            if _match(r, m):
                unit.append(f"{name}@{s}")
            if unit_both_directions and _match(r, 1 / m):
                unit.append(f"{name}_inverse@{s}")
    return inv, unit


def classify(row) -> dict:
    inv, unit = match_lists(row)
    cls = "factor_inversion" if inv else ("unit_error" if unit else "other")
    return {"error_class": cls, "also_unit": bool(inv and unit),
            "matched": ";".join(sorted(set(inv or unit)))}


def _stated_unit(resp, asked):
    if not asked:
        return "not asked"
    try:
        v = ast.literal_eval(str(resp))
        if isinstance(v, dict):
            for k in ("units", "unit"):
                if k in v:
                    return str(v[k])
    except (ValueError, SyntaxError):
        pass
    return "none given"


def _raw_responses():
    raw = {}
    for exp, fname in RAW_FILES.items():
        r = pd.read_csv(os.path.join("results", fname), low_memory=False)
        raw[exp] = r[["observed_to_dose", "response"]]
    return raw


def breakdown(long) -> pd.DataFrame:
    raw = _raw_responses()
    rows = []
    for cond in CONDS:
        c = CONDITIONS[cond]
        d = stats.responses(long, "nielsen", cond)
        th = stats.dose_ratios(d)["to_highest"]
        d = d.assign(ratio_to_highest=th)
        tail = d[th > stats.SAFETY["severe_overdose_x"] + 1e-12]
        for idx, r in tail.iterrows():
            rr = raw[r["experiment"]].iloc[int(r["row_in_raw"])]
            if not np.isclose(float(rr["observed_to_dose"]), float(r["observed_to_dose"])):
                raise ValueError(f"raw row mismatch at {r['experiment']} {r['row_in_raw']}")
            asked = bool(r.get("ask_for_units")) if pd.notna(r.get("ask_for_units")) else False
            item = r[c.item] if c.item in r else None
            rows.append({"experiment": c.experiment, "condition": cond, "model": r["model"],
                         "item": item if isinstance(item, str) else f"{r['from_opioid']} -> {r['to_opioid']}",
                         "from_opioid": r["from_opioid"], "from_dose": r["from_dose"], "to_opioid": r["to_opioid"],
                         "requested_unit": r["to_opioid_unit"], "stated_unit": _stated_unit(rr["response"], asked),
                         "observed_to_dose": r["observed_to_dose"],
                         **{f"expected_{s}": r.get(f"expected_{s}") for s in SCORING_INPUTS},
                         "ratio_to_highest": r["ratio_to_highest"],
                         "ge_10x_highest": bool(r["ratio_to_highest"] >= stats.SAFETY["gross_error_x"]),
                         **classify(r)})
    return pd.DataFrame(rows)


def classes(b: pd.DataFrame, long) -> pd.DataFrame:
    totals = {cond: stats.responses(long, "nielsen", cond).groupby("model").size() for cond in CONDS}
    out = []
    for cond in CONDS:
        for tail, sel in (("severe_gt4x", b.condition.eq(cond)),
                          ("gross_ge10x", b.condition.eq(cond) & b.ge_10x_highest)):
            g = b[sel]
            levels = [("ALL", "ALL", g)] + [("model", m, x) for m, x in g.groupby("model")] + \
                     [("item", i, x) for i, x in g.groupby("item")]
            for level, key, x in levels:
                n = len(x)
                row = {"condition": cond, "extreme": tail, "level": level, "key": key, "n_in_tail": n,
                       "n_responses": int(totals[cond].sum() if level != "model" else totals[cond].get(key, 0)),
                       "n_models": int(x.model.nunique()), "n_cells": int(x[["model", "item"]].drop_duplicates().shape[0])}
                for k in ("factor_inversion", "unit_error", "other"):
                    row[f"n_{k}"] = int((x.error_class == k).sum())
                    row[f"pct_{k}"] = 100 * row[f"n_{k}"] / n if n else np.nan
                row["n_also_unit"] = int(x.also_unit.sum())
                out.append(row)
    return pd.DataFrame(out)


def tail_tests(long) -> pd.DataFrame:
    """Paired tests for the reported set's tests and standard-anchored entries (Nielsen score; the entries
    are score-independent). One item table per condition holds every entry."""
    cfg = REPORTED_SET["safety_tests"]
    tails = list(cfg["outcomes"])
    tables = {c: stats.item_table(long, "nielsen", c, "all", tails) for c in {x for t in cfg["tests"] for x in t}}
    out = []
    for a, b_ in cfg["tests"]:
        for tail in tails:
            e = stats.paired_effect(long, "nielsen", a, b_, "all", tail, tables=tables)
            m = stats.paired_items(long, "nielsen", a, b_, "all", tail, tables=tables)
            nz = m[m[f"prop_{tail}_a"].gt(0) | m[f"prop_{tail}_b"].gt(0)]   # cells with any tail response
            cells = nz.groupby("model").size()
            e["n_cells_nonzero"] = e.model.map(cells).fillna(0).astype(int)
            e.loc[e.model == "POOLED", "n_cells_nonzero"] = len(nz)
            e["experiment"] = CONDITIONS[a].experiment
            e["sig_better"] = (e.p < 0.05) & (e.mean_diff > 0)
            e["sig_worse"] = (e.p < 0.05) & (e.mean_diff < 0)
            out.append(e)
    df = pd.concat(out, ignore_index=True)
    for c in ("mean_a", "mean_b", "median_diff", "mean_diff"):
        df[c] = 100 * df[c]          # percentages (item proportions x 100)
    return df


def main():
    long = stats.load_long()
    b = breakdown(long)
    _write(b, "extreme_errors_breakdown.csv")
    _write(classes(b, long), "extreme_errors_classes.csv")
    _write(tail_tests(long), "safety_tests_by_model.csv")
    print("wrote extreme_errors_breakdown.csv, extreme_errors_classes.csv, safety_tests_by_model.csv")


if __name__ == "__main__":
    main()
