"""
Build the differences report (JMIR revision, Phase 4) from CSV and JSON outputs.

    python orchestration/build_differences_report.py            # stats + figures + HTML + PDF
    python orchestration/build_differences_report.py --no-figures
    python orchestration/build_differences_report.py --stats-only

Never calls any model API. Reads only files the pipeline wrote (results_*/,
results_revision/, figures_*/figure_stats.json, summary_w_figures/figure_stats.json).

The report itself is private: its template, numbers file and output live in
revision_private/differences_report/ (never committed). The script exits with a
message if the template is missing, so the public repository still works.

  1. Stats: every number the report uses is computed into
     differences_stats.json as {key: {value, display, source, description}}.
  2. Verdicts are rules, not judgments: the thresholds are in THRESHOLDS below,
     and every verdict word ("unchanged", "weakened", "strengthened",
     "reversed", "stable", "survives", ...) is computed from them.
  3. Template: template.html, in which every number is a placeholder
     {{key}} (filled with the stat's display string), tables are
     {{table:name}} and figures {{figure:E1}}. A paragraph written for a given
     verdict carries data-verdict="key=expected"; if the computed verdict
     differs, the script prints a warning and marks the paragraph
     "verdict changed since this text was written; review."
  4. Figures: regenerated from the current CSVs by the Phase 2 framework
     (visualize/revision/examples_scores.py, examples_references.py), so figures
     and numbers always match.
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from analysis.revision.scores import SCORES  # noqa: E402
from analysis.revision.tolerances import BANDS  # noqa: E402
from common.config import FIGURES_DIR  # noqa: E402
from common.models import CONSISTENT_MODELS  # noqa: E402

PRIVATE = os.path.join("revision_private", "differences_report")
TEMPLATE = os.path.join(PRIVATE, "template.html")
STATS_JSON = os.path.join(PRIVATE, "differences_stats.json")
OUT_HTML = os.path.join(PRIVATE, "differences_report.html")
OUT_PDF = os.path.join(PRIVATE, "differences_report.pdf")
EXAMPLES = os.path.join(FIGURES_DIR, "revision", "examples")
RR = "results_revision"

# ── Verdict thresholds (one place; stated in the report's methods box) ────
THRESHOLDS = {
    "material_pp": 5.0,         # an accuracy change of >= 5 percentage points is material
    "low_accuracy_pct": 50.0,   # "low naive accuracy" holds while mean tight accuracy < 50%
    "rank_stable_rho": 0.8,     # rankings are "stable" if Spearman rho >= 0.8
    "alpha": 0.05,              # significance level for "survives" / "significant"
    "material_fold": 0.05,      # a change in median fold error of >= 0.05 (5%) is material
}
SCORE_ORDER = list(SCORES)
MAIN_SCORES = ["nielsen", "all", "non_nielsen"]   # the three scores the plan compares
EXP_LABEL = {"vignette": "vignettes", "random_dose": "random doses"}


class Stats(dict):
    def add(self, key, value, source, description, fmt=None):
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        if value is None:
            disp = "n/a"
        elif fmt:
            disp = fmt(value) if callable(fmt) else format(value, fmt)
        elif isinstance(value, float):
            disp = f"{value:.1f}"
        else:
            disp = str(value)
        self[key] = {"value": value, "display": disp, "source": source, "description": description}
        return value

    def v(self, key):
        return self[key]["value"]


def pfmt(p):
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def fold(x):
    return f"{x:.2f}×"


def pp(x):
    return f"{x:+.1f}"


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _figstats(score):
    with open(os.path.join(f"figures_{score}", "figure_stats.json")) as f:
        return json.load(f)


def verdict_change(base, new, higher_is_better, material=THRESHOLDS["material_pp"], reversed_=False):
    if reversed_:
        return "reversed"
    d = new - base
    if abs(d) < material:
        return "unchanged"
    better = d > 0 if higher_is_better else d < 0
    return "strengthened" if better else "weakened"


# ── Section 1: standards ─────────────────────────────────────────────────
def sec_references(S):
    h = pd.read_csv(f"{RR}/references/headline.csv").set_index("key")["value"]
    src = f"{RR}/references/headline.csv"
    for k, d in [("n_study_agents_excl_oral_morphine", "study agents other than oral morphine"),
                 ("agents_with_standard_gt15pct_from_nielsen_all", "agents with a standard > 15% from Nielsen (incl. CDC 2016)"),
                 ("agents_with_standard_gt15pct_from_nielsen_current", "agents with a current standard > 15% from Nielsen"),
                 ("conversions_total", "scored conversions"),
                 ("conversions_with_alternative_current", "conversions with at least one non-Nielsen standard"),
                 ("conversions_outside_tight_all", "conversions outside tight under at least one standard (incl. CDC 2016)"),
                 ("conversions_outside_tight_current", "conversions outside tight under at least one current standard")]:
        S.add(f"ref_{k}", int(h[k]), src, d)
    S.add("ref_n_outside_broad", len(json.loads(h["conversions_outside_broad_all"])), src,
          "conversions outside the broad band under any standard (all CDC 2016 tramadol)")
    S.add("ref_nielsen_outliers", ", ".join(a.replace("_", " ") for a in json.loads(h["nielsen_lone_outlier_agents"])),
          src, "agents where Nielsen is the lone outlier")
    b = pd.read_csv(f"{RR}/references/band_justification_summary.csv").iloc[0]
    src = f"{RR}/references/band_justification_summary.csv"
    S.add("ref_disagree_median", b.disagreement_median, src, "median per-conversion fold spread between current standards", fold)
    S.add("ref_disagree_p95", b.disagreement_p95, src, "95th percentile of that spread", fold)
    S.add("ref_disagree_max", b.disagreement_max, src, "maximum of that spread (= k of reference_agreement)", fold)
    S.add("ref_k_without_codeine", b.k_without_codeine, src, "k without codeine", fold)
    bj = pd.read_csv(f"{RR}/references/band_justification.csv")
    src = f"{RR}/references/band_justification.csv"
    t = bj[bj.band == "tight"]
    S.add("ref_tight_pctile_upper", t[t.side == "upper"].percentile_of_disagreement.iloc[0], src,
          "percentile of the disagreement distribution at the tight upper bound", ".0f")
    S.add("ref_tight_pctile_lower", t[t.side == "lower"].percentile_of_disagreement.iloc[0], src,
          "percentile at the tight lower bound", ".0f")


# ── Section 1a: published vs new baseline ────────────────────────────────
DIRECTION_TESTS = [
    ("low_vig", "Low naive accuracy (vignettes): mean tight % < 50", lambda f: f["naive_vig_mean_tight_pct"] < 50),
    ("low_rd", "Low naive accuracy (random doses): mean tight % < 50", lambda f: f["naive_rd_mean_tight_pct"] < 50),
    ("precise_wrong", "Consistent but incorrect (vignettes): precise-but-wrong % > correct %",
     lambda f: f["vig_mean_precise_wrong_pct"] > f["vig_mean_correct_pct"]),
    ("wording", "Wording sensitivity: mean Kendall's W < 0.3", lambda f: f["mean_kendall_w"] < 0.3),
    ("ome_vig", "Grounding helps OME more than opioid-to-opioid (vignettes)",
     lambda f: f["vig_ome_tight_improvement_pp"] > f["vig_nonome_tight_improvement_pp"]),
    ("ome_rd", "Grounding helps OME more than opioid-to-opioid (random doses)",
     lambda f: f["rd_ome_tight_improvement_pp"] > f["rd_nonome_tight_improvement_pp"]),
    ("grounding", "Grounding improves tight accuracy (both experiments)",
     lambda f: f["nielsen_mean_tight_improvement_vignettes"] > 0 and f["nielsen_mean_tight_improvement_rd"] > 0),
    ("units", "mg/4 h dosing lowers accuracy; explaining units recovers part",
     lambda f: f["dosing_mean_tight_drop_pp"] > 0 and f["units_mean_tight_recovery_pp"] > 0),
]


def sec_baseline(S, tables):
    pub = json.load(open(os.path.join("summary_w_figures", "figure_stats.json")))
    new = _figstats("nielsen")
    src = "summary_w_figures/figure_stats.json vs figures_nielsen/figure_stats.json"
    changed = [k for k in new if str(pub.get(k)) != str(new[k])]
    S.add("base_n_keys", len(new), src, "published figure_stats keys", "d")
    S.add("base_n_changed", len(changed), src, "keys whose value changed", "d")
    rows, reversed_any = [], False
    for key, label, fn in DIRECTION_TESTS:
        a, b = bool(fn(pub)), bool(fn(new))
        reversed_any |= a != b
        rows.append((label, a, b, "same" if a == b else "REVERSED"))
    S.add("verdict_baseline_reversed_any", "yes" if reversed_any else "no", src,
          "does any headline direction test differ between published and new baseline")
    tables["baseline_changes"] = _html_table(
        ["Key", "Published", "New baseline", "Change"],
        [(k, pub.get(k), new[k], _delta(pub.get(k), new[k])) for k in changed])
    tables["baseline_directions"] = _html_table(["Headline direction test", "Published", "New baseline", "Direction"], rows)


def _delta(a, b):
    try:
        d = float(b) - float(a)
        return f"{d:+.3g}"
    except (TypeError, ValueError):
        return ""


# ── Section 2: headline findings under each score ────────────────────────
def _precision_shares(long, score):
    from visualize.revision.panels import precision_cells
    c = precision_cells(long, score, "vignette_naive")
    return c.groupby("model").category.value_counts(normalize=True).unstack(fill_value=0).mean() * 100


def _circ(c, exp, subset, score, outcome):
    r = c[(c.experiment == exp) & (c.subset == subset) & (c.score == score) & (c.outcome == outcome) & (c.model == "ALL_MODELS")]
    return r.iloc[0] if len(r) else None


def sec_headlines(S, tables, long):
    ds = pd.read_csv(f"{RR}/distribution_summary.csv")
    circ = pd.read_csv(f"{RR}/circularity.csv")
    src_fs = "figures_<score>/figure_stats.json"
    rows = []
    for sc in SCORE_ORDER:
        f = _figstats(sc)
        for exp, key in [("vig", "naive_vig_mean_tight_pct"), ("rd", "naive_rd_mean_tight_pct")]:
            S.add(f"h_low_{exp}_tight_{sc}", f[key], src_fs.replace("<score>", sc), f"mean naive tight % ({exp}), {sc}")
        for exp, cond in [("vig", "vignette_naive"), ("rd", "rd_naive")]:
            r = ds[(ds.score == sc) & (ds.condition == cond) & (ds.model == "POOLED")].iloc[0]
            S.add(f"h_low_{exp}_mfe_{sc}", r.median_fold_error, f"{RR}/distribution_summary.csv", f"median fold error, naive {exp}, {sc}", fold)
            S.add(f"h_low_{exp}_env_{sc}", r.pct_reference_envelope, f"{RR}/distribution_summary.csv",
                  f"% within reference_envelope, naive {exp}, {sc} (mean over models)")
        sh = _precision_shares(long, sc)
        S.add(f"h_prec_correct_{sc}", sh.get("correct", 0), "responses_long.csv via visualize/revision/panels.precision_cells",
              f"vignettes naive: % cells correct (mean ratio within tight), {sc}; precision from Nielsen CV")
        S.add(f"h_prec_pw_{sc}", sh.get("precise_wrong", 0), "same", f"vignettes naive: % precise (CV = 0) but wrong, {sc}")
        for exp, a, b in [("vig", "vig_ome_tight_improvement_pp", "vig_nonome_tight_improvement_pp"),
                          ("rd", "rd_ome_tight_improvement_pp", "rd_nonome_tight_improvement_pp")]:
            S.add(f"h_ome_{exp}_tight_ome_{sc}", f[a], src_fs.replace("<score>", sc), f"tight gain, OME, {exp}, {sc}", pp)
            S.add(f"h_ome_{exp}_tight_nonome_{sc}", f[b], src_fs.replace("<score>", sc), f"tight gain, opioid to opioid, {exp}, {sc}", pp)
        for exp, e in [("vig", "vignette"), ("rd", "random_dose")]:
            for out, tag in [("reference_envelope", "env"), ("median_fold_error", "mfe")]:
                for sub in ("ome", "nonome"):
                    r = _circ(circ, e, sub, sc, out)
                    fmtx = (lambda x: f"{x:+.2f}") if out == "median_fold_error" else pp
                    S.add(f"h_ome_{exp}_{tag}_{sub}_{sc}", None if r is None else r.gain, f"{RR}/circularity.csv",
                          f"grounding gain in {out} ({sub}), {exp}, {sc}", fmtx)
    S.add("h_kendall_w", _figstats("nielsen")["mean_kendall_w"], "figures_nielsen/figure_stats.json",
          "mean Kendall's W (wording effects; score-independent)", ".3f")
    # verdicts per score vs nielsen
    for sc in SCORE_ORDER:
        for exp in ("vig", "rd"):
            base, new = S.v(f"h_low_{exp}_tight_nielsen"), S.v(f"h_low_{exp}_tight_{sc}")
            S.add(f"verdict_low_{exp}_{sc}", verdict_change(base, new, higher_is_better=False,
                  reversed_=new >= THRESHOLDS["low_accuracy_pct"]), "rule", f"low naive accuracy ({exp}) under {sc} vs nielsen "
                  "(strengthened = accuracy lower by >= material_pp; weakened = higher; reversed = >= 50%)")
            g_ome, g_non = S.v(f"h_ome_{exp}_tight_ome_{sc}"), S.v(f"h_ome_{exp}_tight_nonome_{sc}")
            gb_ome, gb_non = S.v(f"h_ome_{exp}_tight_ome_nielsen"), S.v(f"h_ome_{exp}_tight_nonome_nielsen")
            S.add(f"verdict_ome_{exp}_{sc}", verdict_change(gb_ome - gb_non, g_ome - g_non, True,
                  reversed_=(g_ome - g_non) <= 0), "rule",
                  f"'grounding helps OME more' ({exp}, tight) under {sc}: OME minus opioid-to-opioid gain vs nielsen")
            e_ome, e_non = S.v(f"h_ome_{exp}_env_ome_{sc}"), S.v(f"h_ome_{exp}_env_nonome_{sc}")
            S.add(f"verdict_ome_env_{exp}_{sc}", "holds" if (e_ome is not None and e_non is not None and e_ome > e_non) else "does not hold",
                  "rule", f"OME gain > opioid-to-opioid gain on reference_envelope ({exp}), {sc}")
        c, w = S.v(f"h_prec_correct_{sc}"), S.v(f"h_prec_pw_{sc}")
        cb, wb = S.v("h_prec_correct_nielsen"), S.v("h_prec_pw_nielsen")
        S.add(f"verdict_prec_{sc}", verdict_change(wb - cb, w - c, True, reversed_=w <= c), "rule",
              f"'consistent but incorrect' under {sc}: precise-wrong minus correct vs nielsen")
        rows.append((SCORES[sc].label, S[f"h_low_vig_tight_{sc}"]["display"], S[f"h_low_rd_tight_{sc}"]["display"],
                     S[f"h_low_vig_mfe_{sc}"]["display"], S[f"h_low_rd_mfe_{sc}"]["display"],
                     S[f"h_prec_correct_{sc}"]["display"], S[f"h_prec_pw_{sc}"]["display"],
                     f'{S[f"h_ome_vig_tight_ome_{sc}"]["display"]} / {S[f"h_ome_vig_tight_nonome_{sc}"]["display"]}',
                     f'{S[f"h_ome_rd_tight_ome_{sc}"]["display"]} / {S[f"h_ome_rd_tight_nonome_{sc}"]["display"]}'))
    tables["headlines"] = _html_table(
        ["Score", "Naive tight %, vignettes", "Naive tight %, random doses", "Median fold error, vignettes",
         "Median fold error, random doses", "Correct % (vignettes)", "Precise but wrong %",
         "Tight gain OME / opioid-to-opioid, vignettes (pp)", "Same, random doses (pp)"], rows)


# ── Section 3: rankings across scores ────────────────────────────────────
def sec_rankings(S, tables):
    pairs = [("nielsen", "all"), ("nielsen", "non_nielsen"), ("nielsen", "geomean"), ("all", "non_nielsen")]
    rows = []
    for exp, f in [("vig", "ranking_naive_vignettes.csv"), ("rd", "ranking_naive_random_doses.csv")]:
        rk = {sc: pd.read_csv(os.path.join(f"results_{sc}", f)).set_index("model") for sc in SCORE_ORDER}
        for a, b in pairs:
            m = rk[a].join(rk[b], lsuffix="_a", rsuffix="_b", how="inner")
            m = m[m.index.isin(CONSISTENT_MODELS)]
            rho = spearmanr(m.mean_tight_prop_a, m.mean_tight_prop_b)[0]
            S.add(f"rank_rho_{exp}_{a}_{b}", rho, f"results_<score>/{f}", f"Spearman rho of mean naive tight accuracy, {exp}, {a} vs {b}", ".2f")
            S.add(f"verdict_rank_{exp}_{a}_{b}", "stable" if rho >= THRESHOLDS["rank_stable_rho"] else "not stable", "rule",
                  f"ranking stability {exp} {a} vs {b} (rho >= {THRESHOLDS['rank_stable_rho']})")
        base = rk["nielsen"]
        for sc in ("all", "non_nielsen"):
            pos_a = base.mean_tight_prop.rank(ascending=False); pos_b = rk[sc].mean_tight_prop.rank(ascending=False)
            d = (pos_b - pos_a).dropna()
            d = d[d.index.isin(CONSISTENT_MODELS)]
            top = d.abs().sort_values(ascending=False).head(2)
            S.add(f"rank_movers_{exp}_{sc}", "; ".join(
                f"{m.split('/')[-1]} ({'down' if d[m] > 0 else 'up'} {abs(d[m]):.0f} places)" for m in top.index),
                  f"results_<score>/{f}", f"models whose position (by mean tight accuracy) moves most, {exp}, nielsen → {sc}")
        for m in [x for x in base.index if x in CONSISTENT_MODELS]:
            rows.append((EXP_LABEL["vignette" if exp == "vig" else "random_dose"], m.split("/")[-1],
                         *[f"{100 * rk[sc].loc[m, 'mean_tight_prop']:.1f} ({int(rk[sc].loc[m, 'adjusted_rank'])})" for sc in SCORE_ORDER]))
    tables["rankings"] = _html_table(["Experiment", "Model", *[f"{SCORES[s].label}: tight % (CMH rank)" for s in SCORE_ORDER]], rows)


# ── Section 4: figure by figure ──────────────────────────────────────────
FIG_KEYS = [
    (1, "naive_vig_mean_tight_pct", "mean naive tight %, vignettes"),
    (1, "naive_vig_best_tight_pct", "best model naive tight %, vignettes"),
    (8, "naive_rd_mean_tight_pct", "mean naive tight %, random doses"),
    (8, "naive_rank_spearman_rho", "vignette vs random-dose rank rho"),
    (10, "cross_pooled_broad_p_rd_empty", "random dose vs empty prompt, pooled broad p"),
    (10, "cross_pooled_broad_p_empty_text", "empty vs text, pooled broad p"),
    (11, "nielsen_mean_tight_improvement_vignettes", "grounding gain, vignettes (pp)"),
    (11, "nielsen_mean_tight_improvement_rd", "grounding gain, random doses (pp)"),
    (11, "nielsen_n_sig_tight_vignettes", "models with significant vignette gain"),
    (12, "nielsen_rd_mean_tight_pct", "mean tight % with Nielsen, random doses"),
    (13, "vig_ome_tight_improvement_pp", "gain OME, vignettes (pp)"),
    (13, "vig_nonome_tight_improvement_pp", "gain opioid-to-opioid, vignettes (pp)"),
    (13, "rd_ome_tight_improvement_pp", "gain OME, random doses (pp)"),
    (13, "rd_nonome_tight_improvement_pp", "gain opioid-to-opioid, random doses (pp)"),
    (14, "dosing_mean_tight_drop_pp", "mg/4 h drop (pp)"),
    (14, "units_mean_tight_recovery_pp", "units recovery (pp)"),
    (15, "mean_rd_tight_after_all_pct", "mean tight % after all, random doses"),
    (15, "mean_vig_tight_after_all_pct", "mean tight % after all, vignettes"),
]


def sec_figures(S, tables):
    fs = {sc: _figstats(sc) for sc in SCORE_ORDER}
    rows = []
    for num, key, label in FIG_KEYS:
        vals = []
        for sc in SCORE_ORDER:
            v = fs[sc].get(key)
            S.add(f"fig_{key}_{sc}", v if not isinstance(v, str) or v != "?" else None,
                  f"figures_{sc}/figure_stats.json", f"figure {num}: {label}, {sc}",
                  (lambda x: f"{x:g}") if isinstance(v, (int, float)) else None)
            vals.append(S[f"fig_{key}_{sc}"]["display"])
        rows.append((num, label, *vals))
    tables["figure_numbers"] = _html_table(["Figure", "Number", *[SCORES[s].label for s in SCORE_ORDER]], rows)


# ── Section 5: circularity ───────────────────────────────────────────────
def sec_circularity(S, tables):
    c = pd.read_csv(f"{RR}/circularity.csv")
    src = f"{RR}/circularity.csv"
    outcomes = ["median_fold_error", *BANDS]
    rows = []
    for exp, e in [("vig", "vignette"), ("rd", "random_dose")]:
        for sub in ("all", "ome", "nonome", "disagree"):
            for sc in SCORE_ORDER:
                for out in outcomes:
                    r = _circ(c, e, sub, sc, out)
                    k = f"circ_{exp}_{sub}_{sc}_{out}"
                    if r is None:
                        continue
                    isf = out == "median_fold_error"
                    f = fold if isf else ".1f"
                    S.add(f"{k}_before", r.before, src, f"{out} before grounding, {exp}, {sub}, {sc}", f)
                    S.add(f"{k}_after", r.after, src, f"{out} after grounding", f)
                    S.add(f"{k}_gain", r.gain, src, "gain (fold error: before − after)", (lambda x: f"{x:+.2f}") if isf else pp)
                    S.add(f"{k}_tf_p", r.tf_p, src, "threshold-free Wilcoxon p (pooled)", pfmt)
                    if not isf:
                        S.add(f"{k}_band_p", r.band_p, src, "band-proportion Wilcoxon p (pooled)", pfmt)
                        S.add(f"{k}_nsig", int(r.band_models_sig_better), src, "models significantly better (band)", "d")
                    p = r.tf_p if isf else r.band_p
                    surv = "survives" if (r.gain > 0 and p is not None and p < THRESHOLDS["alpha"]) else \
                        ("reversed" if r.gain < 0 and p is not None and p < THRESHOLDS["alpha"] else "does not survive")
                    S.add(f"verdict_{k}", surv, "rule", f"grounding gain on {out} ({exp}, {sub}, {sc}): positive and p < alpha")
                    if sub in ("all", "disagree") and sc in MAIN_SCORES:
                        rows.append((EXP_LABEL[e], sub, SCORES[sc].label, out, S[f"{k}_before"]["display"],
                                     S[f"{k}_after"]["display"], S[f"{k}_gain"]["display"],
                                     S.get(f"{k}_band_p", S[f"{k}_tf_p"])["display"], surv))
    tables["circularity"] = _html_table(["Experiment", "Subset", "Score", "Outcome", "Naive", "Grounded", "Gain", "p", "Verdict"], rows)
    # headline verdict: fair tests under non_nielsen
    fair = [S.v(f"verdict_circ_{x}_all_non_nielsen_{o}") for x in ("vig", "rd") for o in ("median_fold_error", "reference_envelope")]
    S.add("verdict_circularity_overall", "survives" if all(v == "survives" for v in fair) else
          ("partly survives" if any(v == "survives" for v in fair) else "does not survive"), "rule",
          "grounding gain under non_nielsen on the fair tests (median fold error, reference_envelope), both experiments")


# ── Section 6: extreme errors ────────────────────────────────────────────
def sec_extremes(S, tables):
    ds = pd.read_csv(f"{RR}/distribution_summary.csv")
    rows = []
    for cond in ["vignette_naive", "rd_naive", "vignette_nielsen", "rd_nielsen", "eu_4h_nielsen", "eu_4h_nielsen_units"]:
        vals = []
        for sc in SCORE_ORDER:
            r = ds[(ds.score == sc) & (ds.condition == cond) & (ds.model == "POOLED")]
            if not len(r):
                vals += ["n/a", "n/a"]; continue
            r = r.iloc[0]
            S.add(f"ext_{cond}_over_{sc}", r.pct_over_2x, f"{RR}/distribution_summary.csv", f"% > 2x, {cond}, {sc}")
            S.add(f"ext_{cond}_under_{sc}", r.pct_under_half, f"{RR}/distribution_summary.csv", f"% < 0.5x, {cond}, {sc}")
            vals += [S[f"ext_{cond}_over_{sc}"]["display"], S[f"ext_{cond}_under_{sc}"]["display"]]
        spread = max(abs(S.v(f"ext_{cond}_{x}_{a}") - S.v(f"ext_{cond}_{x}_{b}")) for x in ("over", "under")
                     for a in SCORE_ORDER for b in SCORE_ORDER if f"ext_{cond}_{x}_{a}" in S and f"ext_{cond}_{x}_{b}" in S)
        S.add(f"verdict_ext_{cond}", "score-independent" if spread < THRESHOLDS["material_pp"] else "score-dependent",
              "rule", f"extreme-error rates for {cond} differ by < material_pp across scores")
        rows.append((cond, *vals))
    hdr = []
    for s in SCORE_ORDER:
        hdr += [f"{SCORES[s].label}: > 2×", f"< ½×"]
    tables["extremes"] = _html_table(["Condition", *hdr], rows)


# ── Section 7: agreement and recall ──────────────────────────────────────
def sec_agreement(S):
    ma = pd.read_csv(f"{RR}/model_agreement.csv")
    src = f"{RR}/model_agreement.csv"
    for exp, e in [("rd", "random_dose"), ("vig", "vignette")]:
        r = ma[(ma.row_type == "summary") & (ma.experiment == e)].iloc[0]
        S.add(f"agr_{exp}_between_gt_within", 100 * r.between_gt_within, src, f"% conversions with between > within model SD ({exp})", ".0f")
        S.add(f"agr_{exp}_within_sd", r.within_sd_median, src, f"median within-model SD of log ratio ({exp})", ".2f")
        S.add(f"agr_{exp}_between_sd", r.between_sd, src, f"median between-model SD of model median log ratios ({exp})", ".2f")
        S.add(f"agr_{exp}_share_wrong_sharing", 100 * r.share_wrong_sharing_value, src, f"% wrong answers sharing a value within 10% with another wrong model ({exp})", ".0f")
        S.add(f"agr_{exp}_largest_cluster", r.largest_wrong_cluster, src, f"median size of the largest group of wrong models ({exp})", "g")
        S.add(f"agr_{exp}_consensus_outside_tight", 100 * r.share_items_consensus_outside_tight, src, f"% conversions whose cross-model consensus is outside tight of Nielsen ({exp})", ".0f")
    pw = pd.read_csv(f"{RR}/model_agreement_pairwise.csv")
    src = f"{RR}/model_agreement_pairwise.csv"
    for exp, e in [("rd", "random_dose"), ("vig", "vignette")]:
        g = pw[pw.experiment == e]
        S.add(f"agr_{exp}_pair_agree", 100 * g.share_agree.mean(), src, f"mean % conversions on which two models agree within 10% ({exp})", ".0f")
        S.add(f"agr_{exp}_pair_agree_wrong", 100 * g.share_agree_when_wrong.mean(), src, f"same when both wrong ({exp})", ".0f")
    sm = pd.read_csv(f"{RR}/standard_matches.csv")
    src = f"{RR}/standard_matches.csv"
    for exp, e in [("rd", "random_dose"), ("vig", "vignette")]:
        r = sm[(sm.row_type == "summary") & (sm.experiment == e) & (sm.item == "eligible_wrong_cells")].iloc[0]
        S.add(f"rec_{exp}_obs", 100 * r.observed_match_rate, src, f"% eligible wrong cells matching a non-Nielsen standard ({exp})", ".1f")
        S.add(f"rec_{exp}_n", f"{int(r.n_matches_any)}/{int(r.n_wrong)}", src, f"matches / eligible wrong cells ({exp})")
        S.add(f"rec_{exp}_chance", 100 * r.chance_rate_shift, src, f"chance match rate, shift method ({exp})", ".1f")
        S.add(f"rec_{exp}_p", r.p_perm_shift, src, f"permutation p ({exp})", pfmt)
        S.add(f"rec_seed", int(r.seed), src, "random seed of the chance baseline", "d")
    sp = pd.read_csv(f"{RR}/spread_comparison.csv")
    src = f"{RR}/spread_comparison.csv"
    for exp, e in [("rd", "random_dose"), ("vig", "vignette")]:
        g = sp[(sp.experiment == e) & sp.standards_fold_spread_current.notna()]
        S.add(f"spr_{exp}_model_range", g.model_range_fold.median(), src, f"median fold range of model answers across 13 models ({exp})", fold)
        S.add(f"spr_{exp}_std_spread", g.standards_fold_spread_current.median(), src, f"median fold spread between current standards ({exp})", fold)
        S.add(f"spr_{exp}_pct_wider", 100 * g.models_wider_than_standards_current.mean(), src, f"% conversions where models are wider than standards ({exp})", ".0f")


# ── Section 8: DI 2 ──────────────────────────────────────────────────────
def sec_di2(S, tables):
    d = pd.read_csv(f"{RR}/di2_subset.csv")
    src = f"{RR}/di2_subset.csv"

    def g(metric, exp=None, subset=None):
        r = d[d.metric == metric]
        if exp: r = r[r.experiment == exp]
        if subset: r = r[r.subset == subset]
        return float(r.value.iloc[0]) if len(r) else None
    for exp, e in [("vig", "vignette"), ("rd", "random_dose")]:
        S.add(f"di2_{exp}_dtight", g("change_pp_tight_without_minus_with", e), src, f"tight change without minus with DI 2 items ({exp})", pp)
        S.add(f"di2_{exp}_dbroad", g("change_pp_broad_without_minus_with", e), src, f"broad change ({exp})", pp)
        S.add(f"di2_{exp}_rho", g("spearman_rho_mean_tight_accuracy", e), src, f"Spearman rho with vs without ({exp})", ".2f")
    S.add("di2_v20_ge2x", g("pct_ge_2x_expected", "vignette", "v20_middle-aged-female-cancer"), src, "vignette 20: % >= 2x expected", ".1f")
    S.add("di2_refusals", int(g("refusal_pattern_decline") or 0) + int(g("refusal_pattern_consult") or 0), src, "decline or consult responses", "d")
    S.add("di2_refusal_n", int(g("refusal_check_n_responses") or 0), src, "vignette responses searched", ",d")
    claims = [("Tight change < 1 pp (vignettes)", "< 1", S["di2_vig_dtight"]["display"], abs(S.v("di2_vig_dtight")) < 1),
              ("Tight change < 1 pp (random doses)", "< 1", S["di2_rd_dtight"]["display"], abs(S.v("di2_rd_dtight")) < 1),
              ("Broad change < 1 pp (random doses)", "< 1", S["di2_rd_dbroad"]["display"], abs(S.v("di2_rd_dbroad")) < 1),
              ("Spearman rho (vignettes)", "0.87", S["di2_vig_rho"]["display"], abs(S.v("di2_vig_rho") - 0.87) < 0.01),
              ("Vignette 20: % >= 2x expected", "40", S["di2_v20_ge2x"]["display"], abs(S.v("di2_v20_ge2x") - 40) < 1)]
    tables["di2_claims"] = _html_table(["Draft letter claim", "Letter", "Recomputed", "Holds"],
                                       [(a, b, c, "yes" if ok else "no") for a, b, c, ok in claims])


# ── Section 8a: empty prompts ────────────────────────────────────────────
def sec_empty(S, tables):
    from scipy.stats import wilcoxon
    for sfx, tag in [("", "all"), ("_same_conversion", "same")]:
        m = pd.read_csv(f"results_nielsen/cross_experiment_matched_means{sfx}.csv")
        m = m[m.model.isin(CONSISTENT_MODELS)].groupby("condition")[["mean_tight", "mean_broad"]].mean() * 100
        src = f"results_nielsen/cross_experiment_matched_means{sfx}.csv"
        for cond, c in [("random_dose", "rd"), ("empty_vignette", "empty"), ("text_vignette", "text")]:
            S.add(f"emp_{tag}_{c}_tight", m.loc[cond, "mean_tight"], src, f"matched mean tight %, {cond} ({tag})")
            S.add(f"emp_{tag}_{c}_broad", m.loc[cond, "mean_broad"], src, f"matched mean broad %, {cond} ({tag})")
        w = pd.read_csv(f"results_nielsen/cross_experiment_matched_wilcoxon{sfx}.csv")
        for a, b, k in [("random_dose", "empty_vignette", "rd_empty"), ("empty_vignette", "text_vignette", "empty_text"),
                        ("random_dose", "text_vignette", "rd_text")]:
            x = w[(w.condition_a == a) & (w.condition_b == b)].broad_diff.dropna().values
            x = x[x != 0]
            S.add(f"emp_{tag}_p_{k}", float(wilcoxon(x).pvalue) if len(x) >= 3 else None,
                  f"results_nielsen/cross_experiment_matched_wilcoxon{sfx}.csv", f"pooled broad Wilcoxon p, {a} vs {b} ({tag})", pfmt)
        s = pd.read_csv(f"results_nielsen/cross_experiment_matched_spearman{sfx}.csv")
        s = s[s.model.isin(CONSISTENT_MODELS)]
        for a, b, k in [("random_dose", "empty_vignette", "rd_empty"), ("empty_vignette", "text_vignette", "empty_text"),
                        ("random_dose", "text_vignette", "rd_text")]:
            g = s[(s.condition_a == a) & (s.condition_b == b)]
            S.add(f"emp_{tag}_rho_{k}", g.spearman_rho.median(), f"results_nielsen/cross_experiment_matched_spearman{sfx}.csv",
                  f"median per-model Spearman rho, {a} vs {b} ({tag})", ".2f")
            S.add(f"emp_{tag}_rho_{k}_nsig", int((g.spearman_p < THRESHOLDS["alpha"]).sum()), "same", f"of {len(g)} significant", "d")
        S.add(f"emp_{tag}_n_matched", int(m.shape[0] and pd.read_csv(f"results_nielsen/cross_experiment_matched_means{sfx}.csv").n_pairs.mode().iloc[0]),
              src, f"matched conversions ({tag})", "d")
        c = pd.read_csv(f"results_nielsen/comparison_text_vs_empty{sfx}.csv")
        c = c[c.model.isin(CONSISTENT_MODELS)]
        for (n1, eu), lab in [((False, False), "naive"), ((True, False), "nielsen"), ((True, True), "units")]:
            g = c[(c.include_nielsen == n1) & (c.explain_units == eu)]
            S.add(f"emp_{tag}_tve_{lab}_tight", 100 * g.tight_diff.mean(), f"results_nielsen/comparison_text_vs_empty{sfx}.csv",
                  f"text minus empty, tight (pp), {lab} ({tag})", pp)
        wt = pd.read_csv(f"results_nielsen/wilcoxon_text_vs_empty{sfx}.csv")
        r = wt[wt.model == "POOLED"].iloc[0]
        S.add(f"emp_{tag}_tve_p_tight", r.tight_p, f"results_nielsen/wilcoxon_text_vs_empty{sfx}.csv", f"text vs empty pooled tight p ({tag})", pfmt)
    S.add("verdict_fig10_consistent", "unchanged" if all((S.v(f"emp_all_p_{k}") or 1) >= THRESHOLDS["alpha"] for k in ("rd_empty", "empty_text", "rd_text"))
          else "changed", "rule", "figure 10 claim: no significant pooled difference between formats")
    rc = pd.read_csv(f"{RR}/vignette_reduction_check.csv")
    rows = []
    for sk in rc.scenario_key.unique():
        if sk.startswith("ALL"):
            continue
        g = rc[(rc.scenario_key == sk) & (rc.model == "POOLED")].set_index("condition")
        rows.append((sk, f"{g.requested_factor.iloc[0]:g}", *[f"{g.loc[c, 'implied_factor_vs_empty']:.2f}" for c in
                                                            ("vignette_naive", "vignette_nielsen", "vignette_nielsen_units")]))
    tables["reductions"] = _html_table(["Scenario", "Requested factor", "Implied factor: naive", "With Nielsen", "Nielsen + units"], rows)
    w = pd.read_csv("results_nielsen/wilcoxon_nielsen_effect_empty_vignettes.csv")
    r = w[w.model == "POOLED"].iloc[0]
    S.add("emp_grounding_p", r.tight_p, "results_nielsen/wilcoxon_nielsen_effect_empty_vignettes.csv", "grounding on empty prompts, pooled tight p", pfmt)
    S.add("emp_grounding_nsig", int((w[w.model != "POOLED"].tight_p < THRESHOLDS["alpha"]).sum()), "same", "models significant", "d")
    c = pd.read_csv("results_nielsen/comparison_nielsen_effect_empty_vignettes.csv")
    c = c[c.model.isin(CONSISTENT_MODELS)]
    S.add("emp_grounding_gain", 100 * c.tight_diff.mean(), "results_nielsen/comparison_nielsen_effect_empty_vignettes.csv", "mean tight gain on empty prompts (pp)", ".1f")


# ── Assembly ─────────────────────────────────────────────────────────────
def _html_table(header, rows):
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def compute():
    from analysis.revision import stats as st
    long = st.load_long()
    S, tables = Stats(), {}
    for k, v in THRESHOLDS.items():
        S.add(f"thr_{k}", v, "orchestration/build_differences_report.py THRESHOLDS", f"verdict threshold {k}", "g")
    sec_references(S)
    sec_baseline(S, tables)
    sec_headlines(S, tables, long)
    sec_rankings(S, tables)
    sec_figures(S, tables)
    sec_circularity(S, tables)
    sec_extremes(S, tables)
    sec_agreement(S)
    sec_di2(S, tables)
    sec_empty(S, tables)
    return S, tables


def build_figures():
    for mod in ("visualize.revision.examples_scores", "visualize.revision.examples_references"):
        path = os.path.join(*mod.split(".")) + ".py"
        if os.path.exists(path):
            subprocess.run([sys.executable, "-m", mod], check=True)
        else:
            print(f"  [warn] {path} missing: figures not regenerated")


def fill(template, S, tables):
    missing = set()

    def num(m):
        k = m.group(1).strip()
        if k.startswith("table:"):
            return tables.get(k[6:], f"<p class='missing'>[missing table {k[6:]}]</p>")
        if k.startswith("figure:"):
            name = k[7:]
            files = sorted(f for f in os.listdir(EXAMPLES) if f.startswith(name + "_") and f.endswith(".png")) \
                if os.path.isdir(EXAMPLES) else []
            if not files:
                missing.add(k)
                return f"<p class='missing'>[missing figure {name}]</p>"
            rel = os.path.relpath(os.path.join(EXAMPLES, files[0]), PRIVATE)
            return "".join(f"<img src='{os.path.relpath(os.path.join(EXAMPLES, f), PRIVATE)}' alt='{name}'>" for f in files)
        if k not in S:
            missing.add(k)
            return f"[[{k}]]"
        return html.escape(S[k]["display"])
    out = re.sub(r"\{\{\s*([^}]+?)\s*\}\}", num, template)
    # verdict checks
    changed = []

    def check(m):
        attrs, body = m.group(1), m.group(2)
        bad = []
        for spec in re.findall(r'data-verdict="([^"]+)"', attrs):
            for item in spec.split(";"):
                key, expected = [x.strip() for x in item.split("=", 1)]
                got = S.get(key, {}).get("value")
                if str(got) != expected:
                    bad.append(f"{key}: written for '{expected}', now '{got}'")
        if not bad:
            return m.group(0)
        changed.extend(bad)
        note = "<span class='changed'>Verdict changed since this text was written; review. (" + html.escape("; ".join(bad)) + ")</span> "
        return f"<p{attrs} class='verdict-changed'>{note}{body}</p>"
    out = re.sub(r"<p([^>]*data-verdict=[^>]*)>(.*?)</p>", check, out, flags=re.S)
    return out, sorted(missing), changed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the differences report.")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args(argv)
    if not os.path.isdir("revision_private"):
        raise SystemExit("revision_private/ not found: the differences report is private to the authors' computer.")
    os.makedirs(PRIVATE, exist_ok=True)
    S, tables = compute()
    tmp = STATS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(S, f, indent=1, default=str)
    os.replace(tmp, STATS_JSON)
    print(f"✓ {len(S)} numbers → {STATS_JSON}")
    if a.stats_only:
        return 0
    if not os.path.exists(TEMPLATE):
        raise SystemExit(f"{TEMPLATE} not found: the report template is private (revision_private/). Stats were written.")
    if not a.no_figures:
        build_figures()
    out, missing, changed = fill(open(TEMPLATE).read(), S, tables)
    with open(OUT_HTML, "w") as f:
        f.write(out)
    if missing:
        print("  [warn] missing placeholders:", ", ".join(missing))
    for c in changed:
        print("  [warn] VERDICT CHANGED:", c)
    from visualize.visualization_config import convert_to_pdf_auto_size
    asyncio.run(convert_to_pdf_auto_size(OUT_HTML, OUT_PDF, scale=1.0, max_width=1100))
    print(f"✓ {OUT_HTML}\n✓ {OUT_PDF}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
