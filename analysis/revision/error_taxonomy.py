"""
Error taxonomy of responses outside the broad band (FA.4, item 1).

    python -m analysis.revision.error_taxonomy

Population. Every response outside the broad band (0.6-1.7x) of the score's
reference dose, in both directions (below: ratio < 0.6; above: ratio > 1.7),
plus responses with no usable dose (missing or not positive; they count as
outside broad everywhere in the framework, direction "undefined", class
``no_usable_answer``). The primary score is ``nielsen`` (the published score,
covering every conversion). The population is also built under the other three
scores (``all``, ``non_nielsen``, ``geomean``); the class definitions below do
not depend on the score (they are computed against the current standards and the
dose itself), only the set of responses does. Under ``all`` and ``non_nielsen`` a
match to another standard is within broad by construction, so the
``standard_match`` share is near zero there.

Conditions: text vignettes (naive, Nielsen supplied, Nielsen + units explained),
random doses (naive, Nielsen supplied) and the dosing-unit arms (mg/4 h target,
Nielsen supplied, without and with the units explained). Not included: the
two-model Nielsen + examples arm and the empty-vignette controls.

Classes (a heuristic reading of the numbers; the response is a bare dose):
  factor_inversion  answer ~ expected x or / F^2 (whole conversion, or one step
                    through oral morphine; F = the factors, so for an
                    opioid-to-opioid conversion the ratio of the two opioids'
                    factors), within the tight tolerance (0.85-1.15x of the
                    multiplier), against the expected dose of any of the four
                    current standards. Classifier: extreme_breakdown.match_lists.
  unit_error        mg vs mcg (x1000), mcg/h vs mg/day (x41.7, x24), per 4 h vs
                    per day (x6), and the reciprocal of each (the answer too
                    small; extreme_breakdown tests only the too-large side), within
                    the same tolerance, counted only where
                    the prompt makes that confusion possible (a unit in mcg; a
                    mcg/hr unit; a mg/4hr or mg/4 hours target); same code as
                    extreme_breakdown (its "mg/4 hours" recognition was added
                    for the dosing-unit arms; its outputs are unchanged).
  standard_match    answer within 10% (as in 3E) of another published standard's
                    expected dose (CDC 2016, CDC 2022, NIH HEAL, McPherson 2018)
                    where that standard itself differs from Nielsen by more than
                    10% (otherwise it is a match to Nielsen and cannot lie outside
                    broad). Nielsen is excluded as the reference.
  other             none of the above.
  round number      a descriptive flag, not a class (author decision 35): the
                    dose is a multiple of a round step set by its size,
                    step = 5 x 10^(floor(log10 dose) - 1): 5 for 10-99, 50 for
                    100-999, 0.5 for 1-9.99, 500 for 1000-9999, and so on. Round
                    answers are common among correct answers too, so each summary
                    row reports ``pct_round_outside_broad`` next to
                    ``pct_round_within_broad`` (same group, responses within broad).
  no_usable_answer  no positive dose.

Precedence (one primary class per response): factor_inversion, unit_error,
standard_match, other (decision 35). ``pct_identifiable_mechanism`` is the share
of outside-broad responses in one of the three mechanism classes. Mechanism classes come first because each
names a specific wrong operation on the prompt's own numbers (inversion before
unit error as in extreme_breakdown); a match to another standard comes next
because it is a defensible alternative reference rather than a slip, but it is
a coincidence-prone 10% window. Because overlaps occur, every response also
carries the three class flags and the round-number flag independent of
precedence (``is_*``), the number of classes it matches, and the summary
reports, for each class, the count matching it at all (``n_any_*``), the
responses matching two or more (``n_multi``) and each pair (``n_<a>_and_<b>``).

Writes (atomically; deterministic, no random numbers):
  results_revision/error_taxonomy_responses.csv   one row per response outside
        broad under at least one of the four scores, with the flags, the primary
        class, and ``outside_<score>`` / ``direction_<score>`` columns
  results_revision/error_taxonomy_summary.csv     shares by score x condition
        (and ALL, pooled over conditions) x level (ALL | model) x direction
        (both | below | above), with n_models and n_cells (model x item cells
        with at least one such response; ``n_cells_total`` = cells with any
        scored response). Shares are percentages of the responses in that
        direction. ``n_responses`` is the group's scored responses.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.conditions import CONDITIONS
from analysis.revision.extreme_breakdown import match_lists
from analysis.revision.model_agreement import AGREE, STANDARDS
from analysis.revision.scores import SCORES
from analysis.revision.tolerances import BANDS

OUT = "results_revision"
PRIMARY_SCORE = "nielsen"
CONDS = ["vignette_naive", "vignette_nielsen", "vignette_nielsen_units",
         "rd_naive", "rd_nielsen", "eu_4h_nielsen", "eu_4h_nielsen_units"]
FLAGS = ["factor_inversion", "unit_error", "standard_match"]   # classes, precedence order (decision 35)
CLASSES = FLAGS + ["other", "no_usable_answer"]
DIRECTIONS = ("both", "below", "above")
ROUND_FACTOR = 5.0


def _write(df: pd.DataFrame, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    df.to_csv(path + ".tmp", index=False, float_format="%.10g")
    os.replace(path + ".tmp", path)


def is_round(dose) -> np.ndarray:
    """Multiple of step = 5 x 10^(floor(log10 dose) - 1) (relative tolerance 1e-9);
    False for missing or non-positive doses."""
    v = pd.to_numeric(pd.Series(dose), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v) & (v > 0)
    out = np.zeros(len(v), dtype=bool)
    if ok.any():
        x = v[ok]
        step = ROUND_FACTOR * 10.0 ** (np.floor(np.log10(x)) - 1)
        q = x / step
        out[ok] = np.abs(q - np.round(q)) <= 1e-9 * np.maximum(1.0, np.abs(q))
    return out


def standard_matches(row) -> list[str]:
    """Other published standards (non-Nielsen, incl. CDC 2016) whose expected dose the
    answer is within 10% of, where that standard differs from Nielsen by more than 10%."""
    obs, nie = row.get("observed_to_dose"), row.get("expected_nielsen")
    if obs is None or not np.isfinite(obs) or obs <= 0:
        return []
    out = []
    for s in STANDARDS:
        e = row.get(f"expected_{s}")
        if e is None or not np.isfinite(e) or e <= 0:
            continue
        if nie is not None and np.isfinite(nie) and nie > 0 and abs(np.log(e / nie)) <= AGREE + 1e-12:
            continue
        if abs(np.log(obs / e)) <= AGREE + 1e-12:
            out.append(s)
    return out


def primary_class(usable: bool, flags: dict) -> str:
    """Precedence: factor_inversion, unit_error, standard_match, other."""
    if not usable:
        return "no_usable_answer"
    for k in FLAGS:
        if flags[k]:
            return k
    return "other"


def _scored(long) -> pd.DataFrame:
    """All scored responses of the taxonomy conditions: one row per response (index of the
    long table; the Nielsen score covers every conversion, so it defines the rows), with the
    condition, the item, and per score covered / ratio / outside_<s> (outside broad) /
    direction_<s> ("below", "above", "undefined" for an unusable answer, "" if inside)."""
    lo, hi = BANDS["broad"].bounds("x", "1", "y")
    frames = []
    for cond in CONDS:
        d0 = stats.responses(long, PRIMARY_SCORE, cond, "all", ["broad"])
        base = d0.copy()
        base["condition"] = cond
        base["item"] = d0[CONDITIONS[cond].item].astype(str)
        for sc in SCORES:
            d = d0 if sc == PRIMARY_SCORE else stats.responses(long, sc, cond, "all", ["broad"])
            assert d.index.isin(d0.index).all(), f"{sc} covers rows the Nielsen score does not"
            ratio = d["ratio"].reindex(base.index)
            outside = d["in_broad"].eq(0).reindex(base.index, fill_value=False).astype(bool)
            base[f"covered_{sc}"] = base.index.isin(d.index)
            base[f"ratio_{sc}"] = ratio
            base[f"outside_{sc}"] = outside
            base[f"direction_{sc}"] = np.where(~outside, "", np.where(
                ratio.isna() | (ratio <= 0), "undefined", np.where(ratio < lo, "below", np.where(ratio > hi, "above", "undefined"))))
        frames.append(base)
    out = pd.concat(frames)
    assert out.index.is_unique
    return out


def build(long: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(classified responses outside broad under any score, all scored responses)."""
    long = stats.load_long() if long is None else long
    sc_df = _scored(long)
    sc_df["is_round"] = is_round(sc_df["observed_to_dose"])
    obs = pd.to_numeric(sc_df["observed_to_dose"], errors="coerce")
    sc_df["usable"] = ((obs > 0) & np.isfinite(obs)).to_numpy()
    any_out = np.zeros(len(sc_df), dtype=bool)
    for sc in SCORES:
        any_out |= sc_df[f"outside_{sc}"].to_numpy()
    cols = ["observed_to_dose", "from_opioid", "from_dose", "to_opioid", "from_opioid_unit", "to_opioid_unit",
            "expected_nielsen", "expected_cdc_2016", "expected_cdc_2022", "expected_nih_heal",
            "expected_mcpherson_2018"]
    recs = sc_df.loc[any_out, cols].to_dict("records")
    inv_l, unit_l, std_l = [], [], []
    for r in recs:
        inv, unit = match_lists(r, unit_both_directions=True)
        inv_l.append(inv); unit_l.append(unit); std_l.append(standard_matches(r))
    out = sc_df.loc[any_out].copy()
    out["factor_inversion"] = [bool(x) for x in inv_l]
    out["unit_error"] = [bool(x) for x in unit_l]
    out["standard_match"] = [bool(x) for x in std_l]
    out["matched_inversion"] = [";".join(sorted(set(x))) for x in inv_l]
    out["matched_unit"] = [";".join(sorted(set(x))) for x in unit_l]
    out["matched_standard"] = [";".join(x) for x in std_l]
    out["n_classes_matched"] = out[FLAGS].sum(axis=1)
    out["primary_class"] = [primary_class(bool(u), dict(zip(FLAGS, f)))
                            for u, f in zip(out["usable"], out[FLAGS].to_numpy())]
    return out, sc_df


def _responses_table(out: pd.DataFrame) -> pd.DataFrame:
    names = list(SCORES)
    cols = ["condition", "experiment", "row_in_raw", "model", "item", "from_opioid", "from_dose", "to_opioid",
            "from_opioid_unit", "to_opioid_unit", "observed_to_dose",
            "expected_nielsen", "expected_cdc_2016", "expected_cdc_2022", "expected_nih_heal",
            "expected_mcpherson_2018",
            *[f"ratio_{s}" for s in names], *[f"outside_{s}" for s in names], *[f"direction_{s}" for s in names],
            *FLAGS, "is_round", "n_classes_matched", "primary_class", "matched_inversion", "matched_unit", "matched_standard"]
    t = out[cols].rename(columns={**{f: f"is_{f}" for f in FLAGS}, "is_round": "is_round_number"})
    return t.sort_values(["condition", "experiment", "row_in_raw", "model"], kind="mergesort").reset_index(drop=True)


def _summ(g: pd.DataFrame, tot: pd.DataFrame, score: str, cond: str, level: str, key: str, direction: str) -> dict:
    """One summary row. g: outside-broad responses in the group and direction; tot: all
    scored responses in the group (denominator, cell count, within-broad round base rate)."""
    n = len(g)
    inside = tot[~tot[f"outside_{score}"]]
    row = {"score": score, "condition": cond, "level": level, "key": key, "direction": direction,
           "n_responses": len(tot), "n_outside_broad": n,
           "pct_outside_broad": 100 * n / len(tot) if len(tot) else np.nan,
           "n_models": int(g["model"].nunique()),
           "n_cells": int(g[["condition", "model", "item"]].drop_duplicates().shape[0]),
           "n_cells_total": int(tot[["condition", "model", "item"]].drop_duplicates().shape[0])}
    for k in CLASSES:
        row[f"n_{k}"] = int((g["primary_class"] == k).sum())
        row[f"pct_{k}"] = 100 * row[f"n_{k}"] / n if n else np.nan
    for k in FLAGS:
        row[f"n_any_{k}"] = int(g[k].sum())
    row["n_multi"] = int((g["n_classes_matched"] >= 2).sum())
    for a, b in itertools.combinations(FLAGS, 2):
        row[f"n_{a}_and_{b}"] = int((g[a] & g[b]).sum())
    row["n_identifiable_mechanism"] = int(sum(row[f"n_{k}"] for k in FLAGS))
    row["pct_identifiable_mechanism"] = 100 * row["n_identifiable_mechanism"] / n if n else np.nan
    row["n_round_number"] = int(g["is_round"].sum())
    row["pct_round_outside_broad"] = 100 * row["n_round_number"] / n if n else np.nan
    for k in FLAGS:
        row[f"n_round_and_{k}"] = int((g["is_round"] & g[k]).sum())
    row["n_within_broad"] = len(inside)
    row["pct_round_within_broad"] = 100 * float(inside["is_round"].mean()) if len(inside) else np.nan
    return row


def summary(out: pd.DataFrame, sc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCORES:
        cov = sc_df[sc_df[f"covered_{sc}"]]
        o = out[out[f"outside_{sc}"]]
        for cond in CONDS + ["ALL"]:
            tc = cov if cond == "ALL" else cov[cov["condition"] == cond]
            oc = o if cond == "ALL" else o[o["condition"] == cond]
            groups = [("ALL", "ALL", tc, oc)] + [("model", m, tc[tc["model"] == m], oc[oc["model"] == m])
                                               for m in sorted(tc["model"].unique())]
            for level, key, t, g in groups:
                for d in DIRECTIONS:
                    gd = g if d == "both" else g[g[f"direction_{sc}"] == d]
                    rows.append(_summ(gd, t, sc, cond, level, key, d))
    return pd.DataFrame(rows)


def main():
    long = stats.load_long()
    out, sc_df = build(long)
    resp = _responses_table(out)
    summ = summary(out, sc_df)
    _write(resp, "error_taxonomy_responses.csv")
    _write(summ, "error_taxonomy_summary.csv")
    pd.set_option("display.width", 250)
    show = summ[(summ.score == PRIMARY_SCORE) & (summ.level == "ALL")]
    cols = ["condition", "direction", "n_responses", "n_outside_broad", "n_models", "n_cells"] + \
           [f"pct_{k}" for k in CLASSES] + ["pct_round_within_broad"]
    print(show[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
