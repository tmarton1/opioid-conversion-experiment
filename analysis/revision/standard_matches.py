"""
Task 3E: model answers vs published standards (DI 8 recall check).

Same model x item cells as 3D (naive, Nielsen score, 13 models; median log
ratio). For every cell outside the tight band ("wrong"): is the answer within
10% of any NON-Nielsen standard's value for that conversion (CDC 2016 included),
and which one(s)? Standard values are relative to Nielsen
(expected_<std> / expected_nielsen); only standards that cover the conversion
count. Wrong cells in conversions that no other standard covers cannot match
and are reported separately (``eligible`` = at least one other standard covers
the conversion).

Chance baseline (seed SEED, N_PERM permutations, per experiment), two versions:
  shift   the plan's: each wrong cell's log ratio shifted by an offset drawn
          (with replacement) from the experiment's observed wrong log ratios
  redraw  sensitivity: the cell's value replaced by such a draw
          (wrong answers placed at random over the observed wrong values)
The match test is applied to the permuted value against the same conversion's
standards.

standard_matches.csv, row_type:
  cell         one row per wrong cell (median ratio, matched standards)
  conversion   per item: wrong cells, matches, expected matches under chance,
               and response-level % within tight of Nielsen vs of any other
  summary      per experiment x (eligible cells | all wrong cells)
spread_comparison.csv: per item, between-model spread (3D) next to the
between-standard spread (references/by_conversion.csv), for figures E6/AG1.

    python -m analysis.revision.standard_matches
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.model_agreement import (AGREE, EXPERIMENTS, STANDARDS, _atomic, cells,
                                               item_table, standard_rel)
from common.config import TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER

SEED = 20261114
N_PERM = 1000
LOG_TIGHT = (np.log(TIGHT_RANGE_LOWER), np.log(TIGHT_RANGE_UPPER))


def _match_matrix(vals: np.ndarray, rel_log: np.ndarray) -> np.ndarray:
    """vals (..., n_cells); rel_log (n_cells, n_std) with NaN where not covered
    -> boolean (..., n_cells) match to any standard within 10%."""
    d = np.abs(vals[..., None] - rel_log[None, ...] if vals.ndim == 2 else vals[:, None] - rel_log)
    return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=-1) <= AGREE + 1e-12


def build(long: pd.DataFrame | None = None, seed: int = SEED, n_perm: int = N_PERM):
    long = stats.load_long() if long is None else long
    rng = np.random.default_rng(seed)
    cell_rows, conv_rows, sum_rows, spread_rows = [], [], [], []
    refs = pd.read_csv("results_revision/references/by_conversion.csv")
    for exp, (cond, item) in EXPERIMENTS.items():
        c = cells(long, exp)
        rel = standard_rel(long, exp)
        w = c[c["wrong"]].copy()
        rl = np.log(rel.loc[w["item"]].values.astype(float))              # n_wrong x n_std
        v = w["median_log"].values
        covered = ~np.isnan(rl).all(axis=1)
        match = _match_matrix(v, rl)
        # which standards
        which = [";".join(s for s, x in zip(STANDARDS, row) if np.isfinite(x) and abs(v[i] - x) <= AGREE + 1e-12)
                 for i, row in enumerate(rl)]
        # distinct from Nielsen: the standard itself is >10% from Nielsen
        distinct = np.array([any(np.isfinite(x) and abs(v[i] - x) <= AGREE + 1e-12 and abs(x) > AGREE
                                 for x in row) for i, row in enumerate(rl)])
        # chance
        pool = v
        shift = v[None, :] + rng.choice(pool, size=(n_perm, len(v)))
        redraw = rng.choice(pool, size=(n_perm, len(v)))
        m_shift = _match_matrix(shift, rl)
        m_redraw = _match_matrix(redraw, rl)
        w = w.assign(ratio=np.exp(v), eligible=covered, matches_any=match & covered,
                     matches_distinct_standard=distinct, matched_standards=which,
                     chance_match_shift=m_shift.mean(axis=0), chance_match_redraw=m_redraw.mean(axis=0))
        for _, r in w.iterrows():
            cell_rows.append({"row_type": "cell", "experiment": exp, "item": r["item"], "model": r["model"],
                              "n": r["n"], "median_ratio": r["ratio"], "eligible": r["eligible"],
                              "matches_any": r["matches_any"], "matches_distinct_standard": r["matches_distinct_standard"],
                              "matched_standards": r["matched_standards"],
                              "chance_match_shift": r["chance_match_shift"], "chance_match_redraw": r["chance_match_redraw"]})
        # summary
        for scope, mask in (("eligible_wrong_cells", covered), ("all_wrong_cells", np.ones(len(w), bool))):
            sub = w[mask]
            ps, pr = m_shift[:, mask].mean(axis=1), m_redraw[:, mask].mean(axis=1)
            sum_rows.append({"row_type": "summary", "experiment": exp, "item": scope,
                             "n_cells": len(c), "n_wrong": len(sub),
                             "n_matches_any": int(sub["matches_any"].sum()),
                             "observed_match_rate": sub["matches_any"].mean(),
                             "n_matches_distinct_standard": int(sub["matches_distinct_standard"].sum()),
                             "observed_match_rate_distinct": sub["matches_distinct_standard"].mean(),
                             "chance_rate_shift": ps.mean(), "chance_shift_lo": np.quantile(ps, .025),
                             "chance_shift_hi": np.quantile(ps, .975),
                             "p_perm_shift": (1 + (ps >= sub["matches_any"].mean() - 1e-12).sum()) / (n_perm + 1),
                             "chance_rate_redraw": pr.mean(), "chance_redraw_lo": np.quantile(pr, .025),
                             "chance_redraw_hi": np.quantile(pr, .975),
                             "p_perm_redraw": (1 + (pr >= sub["matches_any"].mean() - 1e-12).sum()) / (n_perm + 1),
                             "seed": seed, "n_perm": n_perm})
        # per conversion
        resp = stats.responses(long, "nielsen", cond)
        resp = resp[resp["log_ratio"].notna()].assign(item=lambda x: x[item])
        for it, g in w.groupby("item"):
            pass
        for it in sorted(c["item"].unique()):
            g = w[w["item"] == it]
            rr = resp[resp["item"] == it]
            relv = np.log(rel.loc[it].values.astype(float))
            lr = rr["log_ratio"].values
            near_n = ((lr >= LOG_TIGHT[0]) & (lr <= LOG_TIGHT[1])).mean()
            lo, hi = LOG_TIGHT
            oth = np.zeros(len(lr), bool)
            for x in relv:
                if np.isfinite(x) and abs(x) > 1e-9:
                    oth |= (lr >= x + lo) & (lr <= x + hi)
            conv_rows.append({"row_type": "conversion", "experiment": exp, "item": it,
                              "n_other_standards": int(np.isfinite(relv).sum()),
                              "n_wrong": len(g), "n_matches_any": int(g["matches_any"].sum()),
                              "expected_matches_shift": g["chance_match_shift"].sum(),
                              "expected_matches_redraw": g["chance_match_redraw"].sum(),
                              "pct_responses_tight_of_nielsen": 100 * near_n,
                              "pct_responses_tight_of_other_standards": 100 * oth.mean()})
        # spread comparison
        t = item_table(c, rel)
        ref = refs[refs["experiment"] == exp].set_index("conversion")
        for _, r in t.iterrows():
            sp = ref.loc[r["item"]] if r["item"] in ref.index else None
            cur = sp["standards_fold_spread_current"] if sp is not None else np.nan
            al = sp["standards_fold_spread_all"] if sp is not None else np.nan
            spread_rows.append({"experiment": exp, "item": r["item"],
                                "n_other_standards": r["n_other_standards"],
                                "between_model_sd_log": r["between_sd"],
                                "model_range_fold": r["model_range_fold"],
                                "model_fold_sd": float(np.exp(r["between_sd"])),
                                "standards_fold_spread_current": cur,
                                "standards_fold_spread_all": al,
                                "model_range_over_standards_current": (r["model_range_fold"] / cur) if pd.notna(cur) else np.nan,
                                "model_range_over_standards_all": (r["model_range_fold"] / al) if pd.notna(al) else np.nan,
                                "models_wider_than_standards_current": bool(r["model_range_fold"] > cur) if pd.notna(cur) else np.nan,
                                "models_wider_than_standards_all": bool(r["model_range_fold"] > al) if pd.notna(al) else np.nan})
    matches = pd.concat([pd.DataFrame(cell_rows), pd.DataFrame(conv_rows), pd.DataFrame(sum_rows)], ignore_index=True)
    return matches, pd.DataFrame(spread_rows)


def main():
    m, s = build()
    _atomic(m, "standard_matches.csv")
    _atomic(s, "spread_comparison.csv")
    pd.set_option("display.width", 250)
    print(m[m.row_type == "summary"].dropna(axis=1, how="all").T.to_string())
    cv = m[m.row_type == "conversion"].sort_values("n_matches_any", ascending=False).head(12)
    print(cv.dropna(axis=1, how="all").to_string())
    for e, g in s.groupby("experiment"):
        sub = g[g.n_other_standards > 0]
        print(e, "conversions with other standards:", len(sub),
              "models wider (current):", sub.models_wider_than_standards_current.mean(),
              "median model range", sub.model_range_fold.median(),
              "median standards spread", sub.standards_fold_spread_current.median(),
              "all-model-range median", g.model_range_fold.median())


if __name__ == "__main__":
    main()
