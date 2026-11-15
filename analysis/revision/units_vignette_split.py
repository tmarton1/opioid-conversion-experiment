"""3F: vignette units effect (text vignettes, Nielsen vs Nielsen + units) split
into the scenarios whose "+ units" prompt carries the explanatory sentences
about 4-hour dosing and all other scenarios. Loops over every registered score.
Output: results_revision/units_vignette_split.csv (deterministic, no randomness)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.scores import SCORES

RAW = os.path.join("results", "reworded_vignettes_results.csv")
OUT = os.path.join("results_revision", "units_vignette_split.csv")
COND_A, COND_B = "vignette_nielsen", "vignette_nielsen_units"
BANDS = ["tight", "broad"]
OUTCOMES = ["tight", "broad", stats.THRESHOLD_FREE]
MARKER = "over 4 hours"          # appears only in the added explanatory sentences


def explanation_scenarios(raw_path: str = RAW) -> set[str]:
    """Scenarios whose text 'Nielsen + units' prompts contain the 4-hour
    explanation and whose plain 'Nielsen' prompts do not (read from the stored prompts)."""
    d = pd.read_csv(raw_path, usecols=["scenario_key", "prompt", "include_nielsen",
                                       "explain_units", "has_vignette_text"], low_memory=False)
    for c in ["include_nielsen", "explain_units", "has_vignette_text"]:
        d[c] = d[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    d = d[d.has_vignette_text & d.include_nielsen & d.prompt.notna()]
    has = d.assign(m=d.prompt.str.contains(MARKER, regex=False)).groupby(["scenario_key", "explain_units"])["m"].any().unstack()
    return set(has.index[has[True] & ~has[False]])


def _acc_mfe(long, score, cond, scen):
    """Per model: accuracy (mean over cells, as the batch means) and median fold error."""
    out = {}
    for b_ in BANDS:
        pass
    d = stats.responses(long, score, cond, bands=BANDS)
    d = d[d["scenario_key"].isin(scen)]
    cells = d.groupby(["model", "vignette_id", "scenario_key"])[[f"in_{b}" for b in BANDS]].mean().reset_index()
    for m, g in d.groupby("model"):
        c = cells[cells.model == m]
        out[m] = dict(n=len(g), mfe=float(np.nanmedian(g["fold_error"])),
                      **{f"pct_{b}": 100 * float(c[f"in_{b}"].mean()) for b in BANDS})
    out["POOLED"] = dict(n=len(d), mfe=float(np.nanmedian(d["fold_error"])),
                         **{f"pct_{b}": float(np.mean([v[f"pct_{b}"] for v in out.values()])) for b in BANDS})
    return out


def compute(long: pd.DataFrame | None = None, raw_path: str = RAW) -> pd.DataFrame:
    long = stats.load_long() if long is None else long
    expl = explanation_scenarios(raw_path)
    all_sc = set(long.loc[long["has_vignette_text"], "scenario_key"].unique())
    groups = {"explanation_scenarios": expl, "other_scenarios": all_sc - expl, "all_scenarios": all_sc}
    rows = []
    for score in SCORES:
        paired = {o: stats.paired_items(long, score, COND_A, COND_B, outcome=o) for o in OUTCOMES}
        for gname, scen in groups.items():
            a = _acc_mfe(long, score, COND_A, scen)
            b = _acc_mfe(long, score, COND_B, scen)
            tests = {}
            for o, pi in paired.items():
                pi = pi[pi["item"].isin(scen)]
                for m, g in list(pi.groupby("model")) + [("POOLED", pi)]:
                    p, _, mean, n = stats._wilcoxon(g["diff"])
                    tests[(o, m)] = (p, mean, n)
            for m in a:
                if m not in b:
                    continue
                row = {"score": score, "group": gname, "n_scenarios": len(scen), "model": m,
                       "n_resp_nielsen": a[m]["n"], "n_resp_units": b[m]["n"],
                       "median_fold_error_nielsen": a[m]["mfe"], "median_fold_error_units": b[m]["mfe"],
                       "median_fold_error_diff": b[m]["mfe"] - a[m]["mfe"]}
                for bd in BANDS:
                    row[f"pct_{bd}_nielsen"] = a[m][f"pct_{bd}"]
                    row[f"pct_{bd}_units"] = b[m][f"pct_{bd}"]
                    row[f"pct_{bd}_diff"] = b[m][f"pct_{bd}"] - a[m][f"pct_{bd}"]
                for o in OUTCOMES:
                    p, mean, n = tests.get((o, m), (np.nan, np.nan, 0))
                    row[f"n_pairs_{o}"] = n
                    row[f"wilcoxon_p_{o}"] = p
                    row[f"mean_item_diff_{o}"] = mean
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> str:
    out = compute()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, OUT)
    return OUT


if __name__ == "__main__":
    print(main())
