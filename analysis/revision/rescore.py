"""
Rescore every stored response under every registered score (task 1.3).

    python -m analysis.revision.rescore [--raw-dir results]

For each of the three raw CSVs and each registered score other than Nielsen,
writes ``results_<score>/<same name>.csv``: the raw rows with
``expected_to_dose`` replaced by the score's reference dose and
``ratio_observed_expected`` by observed / that dose, plus ``nearest_standard``,
``n_standards_available``, ``expected_nielsen`` and ``ratio_nielsen``.
Also writes ``results_revision/responses_long.csv``: one row per response with
the expected dose under every standard and the ratio under every score.

Expected doses use each standard's whole-conversion answer by factor swap
(REVISION_PLAN.md 1.2): expected_S = expected_nielsen x ratio_S with

    ratio_S = [sum_i(dose_i * S_i) / S_to] / [sum_i(dose_i * N_i) / N_to]

(for one source drug the doses cancel), so scenario multipliers stored in the
Nielsen expected dose (0.75, 0.8, 0.5, 1/6) are preserved. A standard is
available only if it has factors for every source drug and the target.

Deviations from the plan (recorded in REVISION_STATUS.md):
  * Rows whose score has no reference (e.g. non_nielsen on a conversion no
    alternative standard covers) are dropped from results_<score>/ rather than
    kept blank: the pipeline divides band proportions by all rows, so a blank
    row would count as an incorrect answer. They stay in responses_long.csv
    with blank values.
  * The prompt and response text columns are not copied into results_<score>/
    (the pipeline does not read them); the raw files in results/ keep them.
  * Empty-vignette prompts in the four scenarios whose text asks for a clinical
    dose reduction are scored against the unreduced conversion they ask for
    (author decision 2026-11-14; analysis/revision/vignette_targets.py).
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from analysis.revision.scores import SCORES, SCORING_INPUTS, rescored_names
from analysis.revision.vignette_targets import load_raw
from resources.reference_data import REFERENCES

RAW_FILES = {
    "random_dose": "experiment_results.csv",
    "explain_units": "explain_units_experiment_results.csv",
    "vignette": "reworded_vignettes_results.csv",
}
ALL_STANDARDS = list(REFERENCES)  # nielsen, cdc_2016, cdc_2022, nih_heal, mcpherson_2018
DROP_TEXT_COLS = ["prompt", "response"]
CONDITION_COLS = ["include_nielsen", "include_examples", "ask_for_units",
                  "explain_units", "has_vignette_text"]


def _factor(standard: str, route: str):
    e = REFERENCES[standard].get(route)
    return None if e is None else float(e["conversion_factor"])


def standard_ratio(standard: str, from_opioids: list[str], doses: list[float], to_opioid: str):
    """Expected dose under `standard` relative to Nielsen for one conversion, or None."""
    if standard == "nielsen":
        return 1.0
    s_from = [_factor(standard, f) for f in from_opioids]
    s_to = _factor(standard, to_opioid)
    if s_to is None or any(v is None for v in s_from):
        return None
    n_from = [_factor("nielsen", f) for f in from_opioids]
    n_to = _factor("nielsen", to_opioid)
    num = sum(d * s for d, s in zip(doses, s_from)) / s_to
    den = sum(d * n for d, n in zip(doses, n_from)) / n_to
    return num / den


def _parse_sources(from_opioid, from_dose):
    froms = [s.strip() for s in str(from_opioid).split(",")]
    if len(froms) == 1:
        return froms, [1.0]
    doses = [float(x) for x in str(from_dose).split(",")]
    if len(doses) != len(froms):
        raise ValueError(f"cannot pair doses {from_dose!r} with {from_opioid!r}")
    return froms, doses


def expected_by_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Columns expected_<standard> for every standard, one row per input row."""
    keys = df[["from_opioid", "from_dose", "to_opioid"]].astype(str).drop_duplicates()
    ratios = {}
    for f, d, t in keys.itertuples(index=False):
        froms, doses = _parse_sources(f, d)
        ratios[(f, d, t)] = {s: standard_ratio(s, froms, doses, t) for s in ALL_STANDARDS}
    k = list(zip(df["from_opioid"].astype(str), df["from_dose"].astype(str), df["to_opioid"].astype(str)))
    out = pd.DataFrame(index=df.index)
    base = pd.to_numeric(df["expected_to_dose"], errors="coerce")
    for s in ALL_STANDARDS:
        r = pd.Series([ratios[x][s] for x in k], index=df.index, dtype="float64")
        out[f"expected_{s}"] = base * r
    return out


def score_rows(observed: pd.Series, expected: pd.DataFrame, score_name: str):
    """Reference dose and its standard for every row under one score."""
    fn = SCORES[score_name].fn
    obs = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    exp_cols = {s: expected[f"expected_{s}"].to_numpy(dtype=float) for s in SCORING_INPUTS}
    ref = np.full(len(obs), np.nan)
    std = np.empty(len(obs), dtype=object)
    for i in range(len(obs)):
        e = {s: (None if math.isnan(v[i]) else float(v[i])) for s, v in exp_cols.items()}
        o = None if math.isnan(obs[i]) else float(obs[i])
        r, s = fn(o, e)
        if r is not None:
            ref[i] = r
        std[i] = s
    return pd.Series(ref, index=observed.index), pd.Series(std, index=observed.index)


def _ratio(observed, ref):
    obs = pd.to_numeric(observed, errors="coerce")
    r = obs / ref
    return r.where(obs.notna() & (ref > 0))


def _atomic_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def rescore(raw_dir: str = "results", out_root: str = ".", revision_dir: str = "results_revision",
            scores: list[str] | None = None) -> dict:
    scores = scores or rescored_names()
    long_parts, coverage = [], []
    for experiment, fname in RAW_FILES.items():
        # empty-vignette prompts scored against the conversion they ask for
        # (clinical dose reduction removed; analysis/revision/vignette_targets.py)
        raw = load_raw(raw_dir, fname)
        exp = expected_by_standard(raw)
        n_avail = exp[[f"expected_{s}" for s in SCORING_INPUTS]].notna().sum(axis=1)
        ratio_n = _ratio(raw["observed_to_dose"], exp["expected_nielsen"])

        long = pd.DataFrame({"experiment": experiment, "row_in_raw": np.arange(len(raw))})
        for c in ["experiment_id", "model", "scenario_key", "vignette_id", "from_opioid",
                  "from_opioid_unit", "from_dose", "to_opioid", "to_opioid_unit",
                  *CONDITION_COLS, "observed_to_dose"]:
            if c in raw.columns:
                long[c] = raw[c].to_numpy()
        for s in ALL_STANDARDS:
            long[f"expected_{s}"] = exp[f"expected_{s}"].to_numpy()

        for name in ["nielsen", *scores]:
            ref, std = score_rows(raw["observed_to_dose"], exp, name)
            long[f"ratio_{name}"] = _ratio(raw["observed_to_dose"], ref).to_numpy()
            long[f"nearest_standard_{name}"] = std.to_numpy()
            if name == "nielsen":
                continue
            out = raw.drop(columns=[c for c in DROP_TEXT_COLS if c in raw.columns]).copy()
            out["expected_to_dose"] = ref
            out["ratio_observed_expected"] = _ratio(raw["observed_to_dose"], ref)
            out["nearest_standard"] = std
            out["n_standards_available"] = n_avail
            out["expected_nielsen"] = exp["expected_nielsen"]
            out["ratio_nielsen"] = ratio_n
            keep = ref.notna()
            _atomic_csv(out[keep], os.path.join(out_root, f"results_{name}", fname))
            conv_cols = ["scenario_key"] if experiment == "vignette" else ["from_opioid", "to_opioid"]
            coverage.append({
                "score": name, "file": fname, "rows_total": len(raw), "rows_scored": int(keep.sum()),
                "conversions_total": raw.groupby(conv_cols).ngroups,
                "conversions_scored": raw[keep].groupby(conv_cols).ngroups,
            })
        long_parts.append(long)

    responses_long = pd.concat(long_parts, ignore_index=True)
    _atomic_csv(responses_long, os.path.join(revision_dir, "responses_long.csv"))
    cov = pd.DataFrame(coverage)
    _atomic_csv(cov, os.path.join(revision_dir, "rescore_coverage.csv"))
    return {"responses_long": responses_long, "coverage": cov}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rescore stored responses under every registered score.")
    ap.add_argument("--raw-dir", default="results")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--revision-dir", default="results_revision")
    a = ap.parse_args(argv)
    res = rescore(a.raw_dir, a.out_root, a.revision_dir)
    print(res["coverage"].to_string(index=False))
    print(f"responses_long: {len(res['responses_long'])} rows")


if __name__ == "__main__":
    main()
