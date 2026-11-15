"""
Task 3C: DI 2 subset analysis (methadone, buprenorphine, transdermal fentanyl).

Naive condition, Nielsen score, 13 models. Writes
  results_revision/di2_subset.csv              (metric, experiment, subset, value, note)
  results_revision/di2_vignette_responses.csv  (response distributions for the three DI 2 vignettes)

Run:  python -m analysis.revision.di2_subset
No randomness is used. Vignette numbering: see VIGNETTES below.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.prefix import harmonize
from analysis.revision import stats
from analysis.revision.conditions import CONDITIONS, select
from common.models import CONSISTENT_MODELS

OUT_DIR = "results_revision"
OUT = os.path.join(OUT_DIR, "di2_subset.csv")
OUT_RESP = os.path.join(OUT_DIR, "di2_vignette_responses.csv")
RAW_VIGNETTES = os.path.join("results", "reworded_vignettes_results.csv")
SCORE = "nielsen"
LETTER = {"change_pp": "<1", "rho": 0.87, "v20_pct_ge_2x": 40}

# Scenario keys the plan names; position in resources/original_vignettes.json (1-based) is noted.
VIGNETTES = {
    10: "elderly-female-chronic-bowel-obstruction",   # file position 12 (v46): oral -> IV methadone
    11: "elderly-male-abdominal-surgery",              # file position 13 (v50): methadone -> oral morphine
    20: "middle-aged-female-cancer",                   # file position 20 (v78): oral hydromorphone -> fentanyl patch
}
METHADONE_FACTORS = {"3:1": 3.0, "4:1": 4.0, "4.7 (Nielsen)": 4.7, "8:1": 8.0, "10:1": 10.0,
                     "12:1": 12.0, "20:1": 20.0}   # oral morphine mg per 1 mg oral methadone
DECLINE_PATTERNS = {
    "decline": r"\b(?:cannot|can't|can not|unable to|not able to|won't|refuse|decline|i'm sorry|i am sorry|sorry)\b",
    "consult": r"\b(?:consult|seek (?:medical|professional)|medical professional|healthcare (?:provider|professional)|"
               r"physician|clinician|pharmacist|specialist|palliative)\b",
    "no_dict": r"^(?!\s*\{)",   # response not starting with the requested dictionary
}
ROWS: list[dict] = []


def add(metric, experiment, subset, value, note=""):
    ROWS.append(dict(metric=metric, experiment=experiment, subset=subset,
                     value=value, note=note))


def check_vignette_mapping():
    """Confirm the three names against resources/original_vignettes.json and the di2 subset."""
    with open("resources/original_vignettes.json") as fh:
        d = json.load(fh)
    keys = [v["scenario_key"] for v in d.values()]
    for n, k in VIGNETTES.items():
        add("vignette_file_position", "vignette", k, keys.index(k) + 1,
            f"plan number {n}; 1-based position of the scenario in resources/original_vignettes.json "
            f"(vignette_id {[v['vignette_id'] for v in d.values()][keys.index(k)]})")
    long = stats.load_long()
    got = sorted(stats.responses(long, SCORE, "vignette_naive", "di2").scenario_key.unique())
    assert got == sorted(VIGNETTES.values()), got
    return long


def accuracy_blocks(long):
    out = {}
    for exp, cond in [("vignette", "vignette_naive"), ("random_dose", "rd_naive")]:
        s = {sub: stats.model_summary(long, SCORE, cond, sub, bands=["tight", "broad"]).set_index("model")
             for sub in ("all", "no_di2", "di2")}
        r = {sub: stats.rank_models(long, SCORE, cond, sub).result_df.set_index("model")
             for sub in ("all", "no_di2")}
        for sub, t in s.items():
            add("n_responses", exp, sub, int(t.n_responses.sum()), "summed over 13 models")
            add("n_items", exp, sub, int(t.n_items.max()))
            for b in ("tight", "broad"):
                add(f"mean_pct_{b}", exp, sub, t[f"pct_{b}"].mean(), "mean over 13 models of per-model % (mean over items)")
        for b in ("tight", "broad"):
            ch = s["no_di2"][f"pct_{b}"] - s["all"][f"pct_{b}"]
            add(f"change_pp_{b}_without_minus_with", exp, "no_di2_vs_all", s["no_di2"][f"pct_{b}"].mean() - s["all"][f"pct_{b}"].mean(),
                "percentage points, mean over models")
            add(f"max_abs_model_change_pp_{b}", exp, "no_di2_vs_all", ch.abs().max(), str(ch.abs().idxmax()))
            for m, v in ch.items():
                add(f"model_change_pp_{b}", exp, m, v, "without minus with, per model")
        m = s["all"].index
        rho_t = spearmanr(s["all"].loc[m, "pct_tight"], s["no_di2"].loc[m, "pct_tight"])
        rho_c = spearmanr(r["all"].loc[m, "adjusted_rank"], r["no_di2"].loc[m, "adjusted_rank"])
        add("spearman_rho_mean_tight_accuracy", exp, "all_vs_no_di2", rho_t.statistic, "13 models; with vs without DI 2 items")
        add("spearman_rho_cmh_adjusted_rank", exp, "all_vs_no_di2", rho_c.statistic, "13 models; CMH adjusted rank (tight, broad)")
        add("spearman_rho_di2_only_vs_no_di2_mean_tight", exp, "di2_vs_no_di2",
            spearmanr(s["di2"].loc[m, "pct_tight"], s["no_di2"].loc[m, "pct_tight"]).statistic,
            "agreement between model orderings on DI 2 items alone and on the rest")
        out[exp] = (s, r)
    return out


def vignette_details(long):
    r = stats.responses(long, SCORE, "vignette_naive", "di2", bands=["tight", "broad"])
    r["obs"] = pd.to_numeric(r["observed_to_dose"], errors="coerce")
    r["fd"] = pd.to_numeric(r["from_dose"], errors="coerce")
    r["factor"] = (r["obs"] / r["fd"]).round(3)
    dist = []
    for n, k in VIGNETTES.items():
        g = r[r.scenario_key == k]
        N = len(g)
        add("n_responses", "vignette", f"v{n}_{k}", N, "13 models, text vignettes, naive")
        add("pct_tight", "vignette", f"v{n}_{k}", 100 * g.in_tight.mean())
        add("pct_broad", "vignette", f"v{n}_{k}", 100 * g.in_broad.mean())
        add("expected_over_from_dose", "vignette", f"v{n}_{k}", (g.expected_nielsen / g.fd).median(),
            "Nielsen-expected output per unit input")
        vc = g.factor.value_counts()
        for rank, (f, c) in enumerate(vc.head(10).items(), 1):
            dist.append(dict(vignette=n, scenario_key=k, rank=rank, response_over_input=f,
                             n=int(c), share_pct=100 * c / N,
                             median_ratio_to_nielsen=g.loc[g.factor == f, "ratio"].median()))
        f0, c0 = vc.index[0], vc.iloc[0]
        add("modal_response_over_input", "vignette", f"v{n}_{k}", f0, f"{c0}/{N} = {100*c0/N:.1f}%")
        add("modal_response_share_pct", "vignette", f"v{n}_{k}", 100 * c0 / N)
        # per-model share of the modal answer
        add("pct_ge_2x_expected", "vignette", f"v{n}_{k}", 100 * (g.ratio >= 2).mean(), "ratio >= 2")
        add("pct_gt_2x_expected", "vignette", f"v{n}_{k}", 100 * (g.ratio > 2).mean(), "ratio > 2")
        add("pct_le_half_expected", "vignette", f"v{n}_{k}", 100 * (g.ratio <= 0.5).mean(), "ratio <= 0.5")
        if n == 10:
            add("pct_ratio_2_to_1_oral_to_iv", "vignette", f"v{n}_{k}", 100 * np.isclose(g.factor, 0.5, atol=0.005).mean(),
                "response = oral dose / 2 (2:1 oral:parenteral)")
            add("pct_ratio_3_to_1_oral_to_iv", "vignette", f"v{n}_{k}", 100 * np.isclose(g.factor, 1 / 3, atol=0.005).mean(),
                "response = oral dose / 3")
            add("nielsen_implied_oral_to_iv_ratio", "vignette", f"v{n}_{k}", (g.fd / g.expected_nielsen).median(),
                "Nielsen expected gives oral:IV ratio of this size")
        if n == 11:
            add("nielsen_factor_morphine_per_methadone", "vignette", f"v{n}_{k}", (g.expected_nielsen / g.fd).median())
            for lab, fv in METHADONE_FACTORS.items():
                pct = 100 * (np.abs(g.factor / fv - 1) <= 0.02).mean()
                add("pct_matching_methadone_factor", "vignette", f"v{n}_{lab}", pct,
                    f"response within 2% of {fv} mg oral morphine per mg methadone")
            other = ~np.any([np.abs(g.factor / fv - 1) <= 0.02 for fv in METHADONE_FACTORS.values()], axis=0)
            add("pct_matching_no_listed_factor", "vignette", f"v{n}_other", 100 * other.mean())
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                add("response_over_input_quantile", "vignette", f"v{n}_q{q}", g.factor.quantile(q))
    return pd.DataFrame(dist)


def refusal_search():
    raw = pd.read_csv(RAW_VIGNETTES, low_memory=False)
    raw["model"] = raw["model"].map(harmonize)
    raw = raw[raw.model.isin(CONSISTENT_MODELS)]
    for c in ("has_vignette_text", "include_nielsen", "explain_units", "ask_for_units"):
        raw[c] = raw[c].astype(str).eq("True")
    raw = raw[raw.has_vignette_text]
    raw["condition"] = np.where(raw.explain_units, "nielsen_units",
                                np.where(raw.include_nielsen, "nielsen", "naive"))
    raw["response"] = raw["response"].astype(str)
    raw["parsed_missing"] = pd.to_numeric(raw["observed_to_dose"], errors="coerce").isna()
    add("refusal_check_n_responses", "vignette", "all", len(raw),
        "text vignettes, 13 models, all conditions (raw results/reworded_vignettes_results.csv); "
        "random-dose responses have no stored text, so the check covers vignettes only")
    add("refusal_check_unparsed_responses", "vignette", "all", int(raw.parsed_missing.sum()))
    for name, pat in DECLINE_PATTERNS.items():
        hit = raw.response.str.contains(pat, flags=re.I, regex=True)
        add(f"refusal_pattern_{name}", "vignette", "all", int(hit.sum()), f"regex (case-insensitive): {pat}")
        for (m, c), n in raw[hit].groupby(["model", "condition"]).size().items():
            add(f"refusal_pattern_{name}", "vignette", f"{m}|{c}", int(n), "count by model|condition")
    return raw


def write_atomic(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    long = check_vignette_mapping()
    accuracy_blocks(long)
    dist = vignette_details(long)
    refusal_search()
    for k, v in LETTER.items():
        add(f"letter_claim_{k}", "draft_letter", "claim", str(v), "value stated in the draft response letter")
    write_atomic(pd.DataFrame(ROWS), OUT)
    write_atomic(dist, OUT_RESP)
    print(f"wrote {OUT} ({len(ROWS)} rows) and {OUT_RESP}")


if __name__ == "__main__":
    main()
