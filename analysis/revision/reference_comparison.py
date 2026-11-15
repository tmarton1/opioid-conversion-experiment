"""
Reference-standard comparison tables, independent of any model response (task 1.6).

    python -m analysis.revision.reference_comparison [--raw-dir results]

Writes to results_revision/references/:
  by_agent.csv       every agent: each standard's factor, ratio to Nielsen, number
                     of standards, max/min fold spread, Nielsen LQ/UQ
  by_conversion.csv  the 69 scored conversions (49 random-dose pairs, 20 vignette
                     scenarios): each standard's whole-conversion expected dose
                     relative to Nielsen, Nielsen interquartile envelope, spread
                     of the standards, tight/broad flags
  pairwise.csv       every pair of standards (CDC 2016 labelled superseded), over
                     shared agents and shared conversions
  coverage.csv       standards x study agents (1 = the standard gives a factor)

"current" columns use the scoring standards only (Nielsen, CDC 2022, NIH HEAL,
McPherson 2018); "all" columns also include the superseded CDC 2016.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np
import pandas as pd

from analysis.revision.rescore import _parse_sources, standard_ratio
from resources.reference_data import REFERENCES, load_index, load_reference_raw

TIGHT = (0.85, 1.15)
BROAD = (0.6, 1.7)
STANDARDS = list(REFERENCES)                     # nielsen first
ALTERNATIVES = [s for s in STANDARDS if s != "nielsen"]
CURRENT = [s for s, e in load_index().items() if e["role"] == "scoring"]
CURRENT_ALTS = [s for s in CURRENT if s != "nielsen"]
OUT_DIR = os.path.join("results_revision", "references")


def _fold(x: float) -> float:
    return math.exp(abs(math.log(x)))


def _outside(x, band):
    return x < band[0] or x > band[1]


def scored_conversions(raw_dir: str = "results") -> pd.DataFrame:
    """The 69 scored conversions with their source drugs, doses and target."""
    rows = []
    rd = pd.read_csv(os.path.join(raw_dir, "experiment_results.csv"),
                     usecols=["from_opioid", "to_opioid"]).drop_duplicates()
    for f, t in sorted(rd.itertuples(index=False)):
        rows.append(dict(experiment="random_dose", conversion=f"{f} -> {t}",
                         from_opioid=f, from_dose="1", to_opioid=t))
    vg = (pd.read_csv(os.path.join(raw_dir, "reworded_vignettes_results.csv"),
                      usecols=["scenario_key", "from_opioid", "from_dose", "to_opioid"])
          .groupby("scenario_key").first().reset_index())
    for r in vg.itertuples(index=False):
        rows.append(dict(experiment="vignette", conversion=r.scenario_key,
                         from_opioid=r.from_opioid, from_dose=str(r.from_dose), to_opioid=r.to_opioid))
    df = pd.DataFrame(rows)
    df["conversion_type"] = np.where(df.to_opioid == "oral_morphine", "to_OME", "opioid_to_opioid")
    return df


def study_agents(conv: pd.DataFrame) -> list[str]:
    agents = set(conv.to_opioid)
    for f in conv.from_opioid:
        agents |= set(f.split(","))
    return sorted(agents)


def by_agent(agents: list[str]) -> pd.DataFrame:
    nraw = load_reference_raw("nielsen")
    rows = []
    for route, n in nraw.items():
        row = {"opioid_route": route, "study_agent": route in agents,
               "unit": n["unit"], "Nielsen_LQ": n.get("LQ"), "Nielsen_UQ": n.get("UQ")}
        vals = {}
        for s in STANDARDS:
            e = REFERENCES[s].get(route)
            v = None if e is None else float(e["conversion_factor"])
            row[f"factor_{s}"] = v
            row[f"ratio_{s}_vs_nielsen"] = None if v is None else v / n["conversion_factor"]
            if v is not None:
                vals[s] = v
        for tag, names in [("all", STANDARDS), ("current", CURRENT)]:
            vs = [vals[s] for s in names if s in vals]
            row[f"n_standards_{tag}"] = len(vs)
            row[f"fold_spread_{tag}"] = max(vs) / min(vs) if len(vs) >= 2 else None
            alts = [vals[s] / n["conversion_factor"] for s in names if s in vals and s != "nielsen"]
            row[f"max_fold_from_nielsen_{tag}"] = max(_fold(a) for a in alts) if alts else None
            row[f"any_outside_tight_{tag}"] = any(_outside(a, TIGHT) for a in alts) if alts else None
        rows.append(row)
    return pd.DataFrame(rows)


def _iqr_envelope(froms, doses, to):
    """Nielsen interquartile envelope of the whole conversion, relative to the median."""
    nraw = load_reference_raw("nielsen")
    def q(route, which):
        v = nraw[route].get(which)
        return float(v) if v is not None else float(nraw[route]["conversion_factor"])
    med = sum(d * q(f, "conversion_factor") for f, d in zip(froms, doses)) / q(to, "conversion_factor")
    lo = sum(d * q(f, "LQ") for f, d in zip(froms, doses)) / q(to, "UQ")
    hi = sum(d * q(f, "UQ") for f, d in zip(froms, doses)) / q(to, "LQ")
    return min(lo, hi) / med, max(lo, hi) / med


def by_conversion(conv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in conv.itertuples(index=False):
        froms, doses = _parse_sources(r.from_opioid, r.from_dose)
        rec = dict(experiment=r.experiment, conversion=r.conversion, from_opioid=r.from_opioid,
                   from_dose=r.from_dose, to_opioid=r.to_opioid, conversion_type=r.conversion_type)
        rel = {}
        for s in ALTERNATIVES:
            v = standard_ratio(s, froms, doses, r.to_opioid)
            rec[f"{s}_vs_nielsen"] = v
            if v is not None:
                rel[s] = v
        lo, hi = _iqr_envelope(froms, doses, r.to_opioid)
        rec["nielsen_iqr_low"], rec["nielsen_iqr_high"] = lo, hi
        rec["nielsen_iqr_fold_width"] = hi / lo
        for tag, names in [("all", ALTERNATIVES), ("current", CURRENT_ALTS)]:
            alts = [rel[s] for s in names if s in rel]
            rec[f"n_alternatives_{tag}"] = len(alts)
            allv = alts + [1.0]
            rec[f"standards_low_{tag}"] = min(allv)
            rec[f"standards_high_{tag}"] = max(allv)
            # Fold spread between the two most extreme standards (k_c for reference_envelope).
            rec[f"standards_fold_spread_{tag}"] = max(allv) / min(allv) if alts else None
            rec[f"max_fold_from_nielsen_{tag}"] = max(_fold(a) for a in alts) if alts else None
            rec[f"any_outside_tight_{tag}"] = any(_outside(a, TIGHT) for a in alts) if alts else None
            rec[f"any_outside_broad_{tag}"] = any(_outside(a, BROAD) for a in alts) if alts else None
        rows.append(rec)
    return pd.DataFrame(rows)


def pairwise(agent_df: pd.DataFrame, conv_df: pd.DataFrame) -> pd.DataFrame:
    idx = load_index()
    rows = []
    agents = agent_df[agent_df.opioid_route != "oral_morphine"]
    for a, b in itertools.combinations(STANDARDS, 2):
        for scope in ("agents", "conversions"):
            if scope == "agents":
                fa, fb = agents[f"factor_{a}"], agents[f"factor_{b}"]
                m = fa.notna() & fb.notna()
                ratios = (fa[m] / fb[m]).astype(float)
            else:
                ca = conv_df["nielsen_vs_nielsen"] if a == "nielsen" else conv_df[f"{a}_vs_nielsen"]
                cb = conv_df[f"{b}_vs_nielsen"]
                m = ca.notna() & cb.notna()
                ratios = (ca[m] / cb[m]).astype(float)
            folds = np.exp(np.abs(np.log(ratios))) if len(ratios) else pd.Series(dtype=float)
            rows.append({
                "standard_a": a, "standard_b": b, "scope": scope,
                "superseded_involved": "recall_check_only" in (idx[a]["role"], idx[b]["role"]),
                "n_shared": int(m.sum()),
                "median_fold_difference": float(np.median(folds)) if len(folds) else None,
                "max_fold_difference": float(np.max(folds)) if len(folds) else None,
                "pct_differ_gt_15pct": float(100 * np.mean(folds > 1.15 + 1e-12)) if len(folds) else None,
            })
    return pd.DataFrame(rows)


def coverage(agent_df: pd.DataFrame) -> pd.DataFrame:
    st = agent_df[agent_df.study_agent]
    out = pd.DataFrame({"standard": STANDARDS})
    for route in st.opioid_route:
        row = st[st.opioid_route == route].iloc[0]
        out[route] = [int(pd.notna(row[f"factor_{s}"])) for s in STANDARDS]
    return out


def _atomic(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def build(raw_dir: str = "results") -> dict:
    conv = scored_conversions(raw_dir)
    agents = study_agents(conv)
    A = by_agent(agents)
    C = by_conversion(conv)
    C_for_pairs = C.assign(nielsen_vs_nielsen=1.0)
    P = pairwise(A, C_for_pairs)
    V = coverage(A)
    for df, name in [(A, "by_agent.csv"), (C, "by_conversion.csv"), (P, "pairwise.csv"), (V, "coverage.csv")]:
        _atomic(df, name)
    return {"by_agent": A, "by_conversion": C, "pairwise": P, "coverage": V}


def headline(res: dict) -> dict:
    A, C, P = res["by_agent"], res["by_conversion"], res["pairwise"]
    st = A[A.study_agent & (A.opioid_route != "oral_morphine")]
    withalt = C[C.n_alternatives_all > 0]
    withalt_cur = C[C.n_alternatives_current > 0]
    broad = C[C.any_outside_broad_all == True]
    heal_vs_cdc = A[(A.factor_cdc_2022.notna()) & (A.factor_nih_heal.notna())
                    & (A.factor_cdc_2022 != A.factor_nih_heal)].opioid_route.tolist()
    iv = st[st.opioid_route.str.startswith("iv_")]
    iv_alts = {r.opioid_route: [s for s in ALTERNATIVES if pd.notna(getattr(r, f"factor_{s}"))] for r in iv.itertuples()}
    lone = []
    for r in st.itertuples():
        alts = [getattr(r, f"factor_{s}") for s in CURRENT_ALTS if pd.notna(getattr(r, f"factor_{s}"))]
        if len(alts) >= 2 and len(set(alts)) == 1 and alts[0] != r.factor_nielsen:
            lone.append(r.opioid_route)
    return {
        "n_study_agents_excl_oral_morphine": len(st),
        "agents_with_standard_gt15pct_from_nielsen_all": int((st.any_outside_tight_all == True).sum()),
        "agents_with_standard_gt15pct_from_nielsen_current": int((st.any_outside_tight_current == True).sum()),
        "conversions_total": len(C),
        "conversions_with_alternative_all": len(withalt),
        "conversions_with_alternative_current": len(withalt_cur),
        "conversions_outside_tight_all": int((withalt.any_outside_tight_all == True).sum()),
        "conversions_outside_tight_current": int((withalt_cur.any_outside_tight_current == True).sum()),
        "conversions_outside_broad_all": broad.conversion.tolist(),
        "cdc_2022_vs_heal_differ_on": heal_vs_cdc,
        "iv_route_alternatives": iv_alts,
        "nielsen_lone_outlier_agents": lone,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--raw-dir", default="results")
    a = ap.parse_args(argv)
    res = build(a.raw_dir)
    h = headline(res)
    _atomic(pd.DataFrame([{"key": k, "value": json.dumps(v)} for k, v in h.items()]), "headline.csv")
    for k, v in h.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
