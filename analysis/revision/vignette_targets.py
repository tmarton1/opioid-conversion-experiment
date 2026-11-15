"""
Score empty-vignette prompts against the conversion they actually ask for
(author decision 2026-11-14).

Each vignette's expected dose is the Nielsen conversion times its
``output_conversion`` in resources/reworded_vignettes.json. That expression
mixes two kinds of factor:

  * a **unit** factor (``/6``: the target is asked in mg/4 h): the empty prompt
    asks for the same unit, so it applies to both prompts;
  * a **clinical** factor (``* 0.75``, ``* 0.8``, ``* 0.5``: a dose reduction for
    incomplete cross-tolerance) that only the vignette text asks for.

The empty-vignette controls (v82-v101) have no text, but the stored expected
dose kept the clinical factor. This module removes it for the empty prompts
(expected / clinical factor; ratio recomputed), leaving the text prompts and
all unit factors unchanged. It reads the factors from the vignette definitions
and checks every scenario, so nothing is hard-coded. The raw CSV in results/
is never modified: the correction is applied when the raw file is loaded
(analysis/revision/rescore.py, orchestration/run_revision.py).

    from analysis.revision.vignette_targets import reduction_scenarios, correct_vignettes, load_raw
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

VIGNETTE_JSON = os.path.join("resources", "reworded_vignettes.json")
VIGNETTE_FILE = "reworded_vignettes_results.csv"
_TERM = re.compile(r"^\s*expected_to_dose\s*((?:[*/]\s*[0-9.]+\s*)*)$")


def _factors(expr: str) -> tuple[float, float]:
    """(clinical, unit) factors of an output_conversion expression:
    '* k' terms are clinical, '/ k' terms are unit conversions."""
    m = _TERM.match(expr or "expected_to_dose")
    if not m:
        raise ValueError(f"unrecognised output_conversion {expr!r}")
    clinical = unit = 1.0
    for op, val in re.findall(r"([*/])\s*([0-9.]+)", m.group(1)):
        if op == "*":
            clinical *= float(val)
        else:
            unit /= float(val)
    return clinical, unit


@lru_cache(maxsize=1)
def scenario_factors() -> pd.DataFrame:
    """One row per scenario: clinical and unit factor of the text vignettes and
    of the empty prompt as stored, with consistency checks."""
    d = json.load(open(VIGNETTE_JSON))
    rows = {}
    for v in d.values():
        sk = v["scenario_key"]
        c, u = _factors(v.get("output_conversion", "expected_to_dose"))
        r = rows.setdefault(sk, {"scenario_key": sk, "text_clinical": set(), "text_unit": set(),
                                 "empty_clinical": set(), "empty_unit": set(), "empty_has_text": False})
        kind = "text" if v["has_vignette_text"] else "empty"
        r[f"{kind}_clinical"].add(c)
        r[f"{kind}_unit"].add(u)
        if not v["has_vignette_text"] and str(v.get("text", "")).strip():
            r["empty_has_text"] = True
    out = []
    for r in rows.values():
        for k in ("text_clinical", "text_unit", "empty_clinical", "empty_unit"):
            if len(r[k]) != 1:
                raise ValueError(f"{r['scenario_key']}: wordings disagree on {k}: {r[k]}")
            r[k] = r[k].pop()
        if r["empty_has_text"]:
            raise ValueError(f"{r['scenario_key']}: empty-vignette prompt has text")
        if not np.isclose(r["text_unit"], r["empty_unit"]):
            raise ValueError(f"{r['scenario_key']}: text and empty prompts ask different units")
        out.append(r)
    df = pd.DataFrame(out).sort_values("scenario_key").reset_index(drop=True)
    # the empty prompt never asks for the clinical reduction: remove what it inherited
    df["empty_factor_removed"] = df["empty_clinical"]
    df["same_conversion"] = np.isclose(df["text_clinical"], 1.0)
    return df


@lru_cache(maxsize=1)
def _clinical_map() -> dict:
    f = scenario_factors()
    return {r.scenario_key: r.empty_factor_removed for r in f.itertuples() if not np.isclose(r.empty_factor_removed, 1.0)}


def reduction_scenarios() -> dict:
    """{scenario_key: clinical factor} for scenarios whose text asks for a dose reduction."""
    f = scenario_factors()
    return {r.scenario_key: r.text_clinical for r in f.itertuples() if not np.isclose(r.text_clinical, 1.0)}


def same_conversion_scenarios() -> list[str]:
    """Scenarios where the text and the empty prompt ask for the same conversion."""
    f = scenario_factors()
    return f.loc[f["same_conversion"], "scenario_key"].tolist()


def correct_vignettes(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw vignette rows with the empty prompts' expected dose (and ratio)
    corrected to the conversion they ask for. Text prompts are unchanged."""
    out = raw.copy()
    cmap = _clinical_map()
    empty = ~out["has_vignette_text"].astype(str).str.lower().eq("true")
    k = out["scenario_key"].map(cmap)
    m = empty & k.notna()
    out.loc[m, "expected_to_dose"] = out.loc[m, "expected_to_dose"] / k[m]
    obs = pd.to_numeric(out.loc[m, "observed_to_dose"], errors="coerce")
    out.loc[m, "ratio_observed_expected"] = obs / out.loc[m, "expected_to_dose"]
    return out


def load_raw(raw_dir: str, fname: str) -> pd.DataFrame:
    """Read a raw results CSV, applying the empty-vignette correction to the
    vignette file."""
    df = pd.read_csv(os.path.join(raw_dir, fname), low_memory=False)
    return correct_vignettes(df) if fname == VIGNETTE_FILE else df

