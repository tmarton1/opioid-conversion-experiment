"""
Clinical-reduction scenarios, reported separately (author decision 2026-11-14).

In four scenarios the vignette text asks for a dose reduction for incomplete
cross-tolerance (factor k: 0.75, 0.8, 0.5, 0.75; read from the vignette
definitions, analysis/revision/vignette_targets.py); the empty prompt asks the
unreduced conversion. This script reports, for the text-vignette responses in
those scenarios, how often the requested reduction was applied, and compares
each model's text answer with its own empty-prompt answer.

Nielsen score (the question is whether the reduction was applied, not which
standard); per scenario x condition x model, plus POOLED rows:
  n                         responses (text vignettes)
  pct_closer_to_reduced     % whose answer is nearer (log scale) the reduced
                            target than the unreduced one: the reduction applied
  pct_tight_reduced         % within the tight band of the reduced target
  pct_tight_unreduced       % within the tight band of the unreduced target
  median_ratio_unreduced    median text answer / unreduced target (k = applied
                            exactly, 1 = not applied)
  empty_median_ratio        median empty-prompt answer / unreduced target
  implied_factor_vs_empty   median text answer / median empty answer (same model,
                            condition): the model's own implied reduction

    python -m analysis.revision.vignette_reductions
writes results_revision/vignette_reduction_check.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.vignette_targets import reduction_scenarios
from common.config import TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER

OUT = os.path.join("results_revision", "vignette_reduction_check.csv")
CONDS = {  # text condition -> empty condition
    "vignette_naive": "vignette_empty_naive",
    "vignette_nielsen": "vignette_empty_nielsen",
    "vignette_nielsen_units": "vignette_empty_nielsen_units",
}


def _row(g, e, k):
    obs = pd.to_numeric(g["observed_to_dose"], errors="coerce")
    reduced = g["expected_nielsen"]            # text target includes the reduction
    unreduced = reduced / k
    r_red, r_un = obs / reduced, obs / unreduced
    ok = obs > 0
    lr_red = np.log(r_red.where(ok)); lr_un = np.log(r_un.where(ok))
    inside = lambda r: (r >= TIGHT_RANGE_LOWER) & (r <= TIGHT_RANGE_UPPER)
    eobs = pd.to_numeric(e["observed_to_dose"], errors="coerce")
    e_ratio = (eobs / e["expected_nielsen"]).where(eobs > 0)
    t_med, e_med = obs.where(ok).median(), eobs.where(eobs > 0).median()
    return {
        "n": int(len(g)),
        "pct_closer_to_reduced": 100 * float(((lr_red.abs() < lr_un.abs()) & ok).mean()),
        "pct_tight_reduced": 100 * float(inside(r_red).mean()),
        "pct_tight_unreduced": 100 * float(inside(r_un).mean()),
        "median_ratio_unreduced": float(r_un.where(ok).median()),
        "empty_median_ratio": float(e_ratio.median()) if len(e) else np.nan,
        "implied_factor_vs_empty": float(t_med / e_med) if len(e) and e_med else np.nan,
    }


def build(long=None) -> pd.DataFrame:
    long = stats.load_long() if long is None else long
    red = reduction_scenarios()
    rows = []
    for tc, ec in CONDS.items():
        t_all = stats.responses(long, "nielsen", tc)
        e_all = stats.responses(long, "nielsen", ec)
        for sk, k in red.items():
            t, e = t_all[t_all.scenario_key == sk], e_all[e_all.scenario_key == sk]
            for m, g in list(t.groupby("model")) + [("POOLED", t)]:
                ee = e if m == "POOLED" else e[e.model == m]
                rows.append({"scenario_key": sk, "requested_factor": k, "condition": tc, "model": m,
                             **_row(g, ee, k)})
        t = t_all[t_all.scenario_key.isin(red)]
        e = e_all[e_all.scenario_key.isin(red)]
        if len(t):
            # pooled over the four scenarios: per-row targets, so k per row
            kk = t["scenario_key"].map(red)
            g = t.assign(expected_nielsen=t["expected_nielsen"])
            obs = pd.to_numeric(g["observed_to_dose"], errors="coerce")
            ok = obs > 0
            r_red = obs / g["expected_nielsen"]; r_un = obs / (g["expected_nielsen"] / kk)
            inside = lambda r: (r >= TIGHT_RANGE_LOWER) & (r <= TIGHT_RANGE_UPPER)
            rows.append({"scenario_key": "ALL_REDUCTION_SCENARIOS", "requested_factor": np.nan,
                         "condition": tc, "model": "POOLED", "n": int(len(g)),
                         "pct_closer_to_reduced": 100 * float(((np.log(r_red.where(ok)).abs()
                                                                 < np.log(r_un.where(ok)).abs()) & ok).mean()),
                         "pct_tight_reduced": 100 * float(inside(r_red).mean()),
                         "pct_tight_unreduced": 100 * float(inside(r_un).mean()),
                         "median_ratio_unreduced": np.nan, "empty_median_ratio": np.nan,
                         "implied_factor_vs_empty": np.nan})
    return pd.DataFrame(rows)


def main():
    df = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)
    print(df[df.model == "POOLED"].to_string(index=False))


if __name__ == "__main__":
    main()
