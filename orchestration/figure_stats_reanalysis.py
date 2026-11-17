"""Decision 47: the reanalysis numbers of the revised figures, captions and appendices, computed by code from
public files, for orchestration/extract_figure_stats.py (--extended), which merges them into figure_stats.json.

Every number used in a figure, a caption, a repo PDF or a Multimedia Appendix that is not one of the 83 submitted
keys is a key here: other references and median fold error (Figure 11, circularity), range rates and extreme errors
(Figures 1, 8), consensus, the Bonferroni families (results_revision/multiple_comparisons_*.csv), the pooled tests
(Figure 9 sign test, Figure 13), Multimedia Appendix 1 counts, the extreme-error counts, the standard-matching test
and the DI 2 subset check.

``reanalysis_stats()`` returns (values, meta): values is the flat {key: value} map; meta has, per key, the display
format, structured fields (measure, experiment, condition, reference, statistic, adjusted, figure) and ``source``
(file and column / filter). No manuscript content is read or written here.

Inputs: results_revision/*.csv (+ references/), results_nielsen/wilcoxon_nielsen_effect_*.csv and
pair_model_analysis.csv, common.config band limits, visualize pooled slope test.
"""
from __future__ import annotations

import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RR = "results_revision"
NIELSEN_DIR = "results_nielsen"

DISPLAY_NAMES = {   # model id -> display name used in figures and captions
    "anthropic/claude-3.5-sonnet:20241022": "Claude 3.5 Sonnet", "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash", "openai/gpt-4.1": "GPT-4.1", "openai/gpt-4.1-mini": "GPT-4.1 mini",
    "openai/gpt-3.5-turbo": "GPT-3.5 Turbo", "deepseek/deepseek-chat": "DeepSeek Chat",
    "mistralai/mistral-large-2407": "Mistral Large", "mistralai/mixtral-8x22b-instruct": "Mixtral 8x22B",
    "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B", "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "ai21/jamba-large-1.7": "Jamba Large 1.7", "cohere/command-r-plus-08-2024": "Command R+",
}
EXPS = [("vignette", "vignettes", "vignette_naive"), ("random_dose", "random doses", "rd_naive")]


class _Reg:
    def __init__(self):
        self.values, self.meta = {}, {}

    def add(self, key, value, fmt, figure, measure, experiment, condition, reference, statistic, adjusted, source):
        if key in self.values:
            raise KeyError(f"duplicate figure_stats key {key}")
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        elif isinstance(value, (np.bool_,)):
            value = bool(value)
        self.values[key] = value
        self.meta[key] = dict(format=fmt, figure=figure, measure=measure, experiment=experiment, condition=condition,
                              reference=reference, statistic=statistic, adjusted=adjusted, source=source,
                              scope="reanalysis")


def _csv(name):
    return pd.read_csv(os.path.join(RR, name))


# ── multiple-comparison helpers (the same rules the figures use) ────────────────────────────────
def _level(p, pb):
    if pb is not None and np.isfinite(pb) and pb < 0.05:
        return "filled"
    if p is not None and np.isfinite(p) and p < 0.05:
        return "open"
    return ""


class _Tests:
    def __init__(self):
        self.t = _csv("multiple_comparisons_tests.csv")
        self.a = _csv("multiple_comparisons_audit.csv")

    def family(self, fam):
        d = self.t[self.t.family == fam]
        if d.empty:
            raise KeyError(f"family {fam!r} not in multiple_comparisons_tests.csv")
        return {u: (float(p), float(b)) for u, p, b in zip(d.unit, d.p, d.p_bonf)}

    def aud(self, fig, panel=None, measure=None, test=None):
        """(n_sig_unadj, n_sig_bonf, m, n_sig_bonf_panel, m_panel) of the first matching audit row."""
        d = self.a[self.a.figure == fig]
        if panel is not None:
            d = d[d.panel == panel]
        if measure is not None:
            d = d[d.measure == measure]
        if test is not None:
            d = d[d.test.str.contains(test, regex=False)]
        r = d.iloc[0]
        return int(r.n_sig_unadj), int(r.n_sig_bonf), int(r.m), int(r.n_sig_bonf_panel), int(r.m_panel)


def _r2(x):
    """Two decimals, rounding half up (1.725 -> 1.73), as printed in Multimedia Appendix 1."""
    return float(Decimal(repr(round(float(x), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def reanalysis_stats(models: list | None = None) -> tuple[dict, dict]:
    from analysis.revision import stats as S
    from common.config import BROAD_RANGE_LOWER, BROAD_RANGE_UPPER, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER
    from common.models import CONSISTENT_MODELS
    from visualize.visualization_config import MODEL_RANK_ORDER

    R = _Reg()
    T = _Tests()
    n_models = len(CONSISTENT_MODELS)
    order = [m for m in MODEL_RANK_ORDER if m in CONSISTENT_MODELS]
    CIRC, DIST = _csv("circularity.csv"), _csv("distribution_summary.csv")
    long = S.load_long()
    P_ = "multiple_comparisons_audit.csv"

    def circ(exp, score, outcome, subset="all"):
        d = CIRC[(CIRC.experiment == exp) & (CIRC.subset == subset) & (CIRC.score == score) & (CIRC.outcome == outcome)]
        return d.set_index("model")

    # ── Figures 1 and 8: naive accuracy, range rates, CMH groups and brackets ──────────────────
    for exp, word, cond in EXPS:
        t, fig = ("f1", "Figure 1") if exp == "vignette" else ("f8", "Figure 8")
        d = DIST[(DIST.score == "nielsen") & (DIST.condition == cond)].set_index("model")
        pooled, models_ = d.loc["POOLED"], d.drop("POOLED")
        A = DIST[(DIST.score == "any") & (DIST.condition == cond) & (DIST.model == "POOLED")].iloc[0]
        src = f"{RR}/distribution_summary.csv, score=nielsen, condition={cond}"
        add = lambda k, v, f, ms, stat, s, adj="n/a": R.add(f"{t}_{k}", v, f, fig, ms, word, "naive", "Nielsen", stat, adj, s)  # noqa: E731
        add("broad_min", models_.pct_broad.min(), "ratio:0", "within 0.6–1.7× (percent; no sign, used as the low end of a range)", "lowest model", src + ", pct_broad min over models")
        add("broad_max", models_.pct_broad.max(), "percent:0", "within 0.6–1.7×", "highest model", src + ", pct_broad max over models")
        add("mean_tight", pooled.pct_tight, "percent:0", "within ±15%", "mean of models", src + ", pct_tight, model=POOLED")
        add("mean_broad", pooled.pct_broad, "percent:0", "within 0.6–1.7×", "mean of models", src + ", pct_broad, model=POOLED")
        rs = f"{RR}/distribution_summary.csv, score=any, condition={cond}, model=POOLED"
        for k, col, lab in [("pct_below_range", "pct_below_range", "below the range of standards"),
                            ("pct_underdose", "pct_underdose", "below ½× the lowest standard"),
                            ("pct_above_range", "pct_above_range", "above the range of standards"),
                            ("pct_overdose", "pct_overdose", "above 2× the highest standard"),
                            ("pct_severe_overdose", "pct_severe_overdose", "above 4× the highest standard")]:
            R.add(f"{t}_{k}", A[col], "percent:1", fig, lab, word, "naive", "all current standards", "mean of models",
                  "n/a", rs + f", {col}")
        rk = S.rank_models(long, "nielsen", cond, "all", bands=("broad",)).result_df.set_index("model")
        grp = {m: int(rk.loc[m, "adjusted_rank"]) for m in rk.index}
        pb = d.pct_broad
        rows = sorted([m for m in rk.index if m in models_.index],
                      key=lambda m: (grp[m], -float(pb[m]), MODEL_RANK_ORDER.index(m)))
        fam = T.family(f"cmh_broad:{cond}")
        star = opn = 0
        for a, b in zip(rows[:-1], rows[1:]):
            lv = _level(*fam.get(f"{a} > {b}", (np.nan, np.nan)))
            star += lv == "filled"
            opn += lv == "open"
        u, bn, mm, _, _ = T.aud(fig, test="broad only")
        cs = "analysis.revision.stats.rank_models(bands=('broad',)) on results_revision/responses_long.csv"
        add("n_groups", len(set(grp.values())), "count", "CMH groups (0.6–1.7× only)", "count of groups", cs, "Bonferroni")
        add("n_pairs", mm, "count", "model pairs", "count of pairs", f"{RR}/{P_}, {fig}, test contains 'broad only', m", "Bonferroni")
        add("pairs_sig_bonf", bn, "count", "significant model pairs", "count", f"{RR}/{P_}, {fig}, n_sig_bonf", "Bonferroni")
        add("pairs_sig_unadj", u, "count", "significant model pairs", "count", f"{RR}/{P_}, {fig}, n_sig_unadj", "unadjusted")
        add("brackets_star", star, "count", "brackets between adjacent rows", "count of ★ brackets",
            f"{RR}/multiple_comparisons_tests.csv family cmh_broad:{cond}, adjacent rows by group then % within 0.6–1.7×", "Bonferroni")
        add("brackets_open", opn, "count", "brackets between adjacent rows", "count of ☆ brackets",
            f"{RR}/multiple_comparisons_tests.csv family cmh_broad:{cond}, adjacent rows", "unadjusted")

    # ── Figure 11 and circularity: before / after by reference, median fold error, marks ───────
    vn = {"vignette": "vig", "random_dose": "rd"}
    sn = {"nielsen": "nielsen", "non_nielsen": "nonn"}
    for exp, word, _ in EXPS:
        for sc, scl in sn.items():
            ref = "Nielsen" if sc == "nielsen" else "all except Nielsen"
            b = circ(exp, sc, "broad").loc["ALL_MODELS"]
            m = circ(exp, sc, "median_fold_error").loc["ALL_MODELS"]
            s = f"{RR}/circularity.csv, experiment={exp}, subset=all, score={sc}, model=ALL_MODELS"
            R.add(f"f11_{vn[exp]}_{scl}_before", b.before, "percent:1", "Figure 11", "within 0.6–1.7×", word, "naive", ref,
                  "mean of models", "n/a", s + ", outcome=broad, before")
            R.add(f"f11_{vn[exp]}_{scl}_after", b.after, "percent:1", "Figure 11", "within 0.6–1.7×", word,
                  "reference-grounded", ref, "mean of models", "n/a", s + ", outcome=broad, after")
            R.add(f"f11_mfe_{vn[exp]}_{scl}_before", m.before, "ratio:2", "Figure 11", "median fold error", word, "naive",
                  ref, "mean of models", "n/a", s + ", outcome=median_fold_error, before")
            R.add(f"f11_mfe_{vn[exp]}_{scl}_after", m.after, "ratio:2", "Figure 11", "median fold error", word,
                  "reference-grounded", ref, "mean of models", "n/a", s + ", outcome=median_fold_error, after")
            R.add(f"f11_mfe_{vn[exp]}_{scl}_p", m.tf_p, "p_caption", "Figure 11", "median fold error", word,
                  "naive -> reference-grounded", ref, "p (pooled threshold-free test)", "unadjusted",
                  s + ", outcome=median_fold_error, tf_p")
    R.add("f11_mfe_p_max", max(R.values[f"f11_mfe_{vn[e]}_{sn[s]}_p"] for e, _, _ in EXPS for s in sn), "p_caption",
          "Figure 11", "median fold error", "both", "naive -> reference-grounded", "Nielsen and all except Nielsen",
          "largest of the four pooled p values", "unadjusted", f"{RR}/circularity.csv tf_p, max of the four panels")
    # marks per table column: models with a gain that is Bonferroni-significant (filled) / unadjusted only (open)
    for exp, word, _ in EXPS:
        for sc in sn:
            for out in ("tight", "broad"):
                d = circ(exp, sc, out)
                fam = T.family(f"circularity:{exp}:all:{sc}:{out}:band")
                lv = [_level(*fam[m]) for m in d.index if m != "ALL_MODELS" and d.loc[m, "gain"] > 0]
                ref = "Nielsen" if sc == "nielsen" else "all except Nielsen"
                fc = f"{RR}/multiple_comparisons_tests.csv family circularity:{exp}:all:{sc}:{out}:band + circularity.csv gain>0"
                R.add(f"f11_{exp}_{sc}_{out}_star", lv.count("filled"), "count", "Figure 11", out, word,
                      "naive -> reference-grounded", ref, "count of models with ★", "Bonferroni", fc)
                R.add(f"f11_{exp}_{sc}_{out}_open", lv.count("open"), "count", "Figure 11", out, word,
                      "naive -> reference-grounded", ref, "count of models with ☆ only", "unadjusted", fc)
    # grayed ±15% panels: share of items where an exact copy of Nielsen fails the other standards
    oth = ["expected_cdc_2022", "expected_nih_heal", "expected_mcpherson_2018"]
    for exp, word, cond in EXPS:
        d = S.responses(long, "nielsen", cond)
        it = "scenario_key" if cond.startswith("vignette") else "opioid_pair"
        u = d.drop_duplicates(it)
        r = u[oth].apply(pd.to_numeric, errors="coerce").div(pd.to_numeric(u["expected_nielsen"], errors="coerce"), axis=0)
        near = np.log(r).abs().min(axis=1)
        near = near[r.notna().any(axis=1)]
        key = "f11_X" if exp == "vignette" else "f11_Y"
        R.add(key, float((near > np.log(1.15) + 1e-9).mean() * 100), "percent:0", "Figure 11", "copy of Nielsen scored wrong against another standard",
              word, "naive", "other current standards", "share of items", "n/a",
              f"{RR}/responses_long.csv expected_* columns: items whose nearest other standard is beyond ±15% of Nielsen")
        R.add(key + "_n", int(len(near)), "count", "Figure 11", "items with another standard", word, "naive",
              "other current standards", "count of items", "n/a", f"{RR}/responses_long.csv, {it} with another standard")

    # ── circularity figure ───────────────────────────────────────────────────────────────────
    agr, byc = _csv("model_agreement.csv"), pd.read_csv(os.path.join(RR, "references", "by_conversion.csv"))
    OTH = ["cdc_2022_vs_nielsen", "nih_heal_vs_nielsen", "mcpherson_2018_vs_nielsen"]
    for exp, word, _ in EXPS:
        c = vn[exp]
        db = S.responses(long, "nielsen", "vignette_nielsen" if exp == "vignette" else "rd_nielsen")
        lr = db["log_ratio"]
        a = agr[(agr.row_type == "item") & (agr.experiment == exp)][["item", "consensus_ratio"]]
        b = byc[byc.experiment == exp].rename(columns={"conversion": "item"})
        d = a.merge(b, on="item", how="left", validate="1:1")
        d["has_other"] = d[OTH].notna().any(axis=1)
        d["inside"] = (d.consensus_ratio >= d.standards_low_current * (1 - 1e-9)) & \
                      (d.consensus_ratio <= d.standards_high_current * (1 + 1e-9))
        d["within"] = (d.consensus_ratio >= BROAD_RANGE_LOWER * d.standards_low_current) & \
                      (d.consensus_ratio <= BROAD_RANGE_UPPER * d.standards_high_current)
        ho = d[d.has_other]
        sr = f"{RR}/responses_long.csv, condition={'vignette' if exp == 'vignette' else 'rd'}_nielsen, score=nielsen"
        cs = f"{RR}/model_agreement.csv (consensus_ratio) + {RR}/references/by_conversion.csv (standards_low/high_current)"
        R.add(f"circ_{c}_within_1pct", round(100 * float((lr.abs() <= np.log(1.01)).mean()), 1), "percent:0",
              "Circularity figure", "answers within 1% of Nielsen", word, "reference-grounded", "Nielsen",
              "share of answers", "n/a", sr + ", |log ratio| <= log 1.01")
        R.add(f"circ_{c}_exact", round(100 * float((lr.abs() < 1e-6).mean()), 1), "percent:0", "Circularity figure",
              "answers exactly equal to Nielsen", word, "reference-grounded", "Nielsen", "share of answers", "n/a",
              sr + ", |log ratio| < 1e-6")
        m = circ(exp, "non_nielsen", "median_fold_error").loc["ALL_MODELS"]
        R.add(f"circ_{c}_mfe_p", m.tf_p, "p_caption", "Circularity figure", "median fold error", word,
              "naive -> reference-grounded", "all except Nielsen", "p (pooled threshold-free test)", "unadjusted",
              f"{RR}/circularity.csv, score=non_nielsen, outcome=median_fold_error, tf_p")
        for k, v, lab in [("cons_inside", int(ho.inside.sum()), "consensus inside the range of current standards"),
                          ("cons_other", int(len(ho)), "conversions with another standard"),
                          ("cons_within", int(d.within.sum()), "consensus within 0.6–1.7× of the range of standards"),
                          ("cons_n", int(len(d)), "conversions")]:
            R.add(f"circ_{c}_{k}", v, "count", "Circularity figure", lab, word, "naive", "current standards",
                  "count", "n/a", cs)

    # ── Figure 9 pooled slope test, other audit counts, Figure 13 pooled tests ─────────────────
    from visualize.comprehensive_figures import _slope_pooled_stats
    st = _slope_pooled_stats(os.path.join(NIELSEN_DIR, "pair_model_analysis.csv"),
                             {"include_nielsen": False, "include_examples": False})
    s9 = f"{NIELSEN_DIR}/pair_model_analysis.csv (naive), regression_slope / regression_slope_pvalue"
    for k, v, f, stat, adj in [("slopes_neg", st["neg"], "count", "count of negative slopes", "n/a"),
                               ("slopes_nonzero", st["nonzero"], "count", "count of non-zero slopes", "n/a"),
                               ("sign_test_p", st["p"], "p_caption", "p (one-sided sign test over all slopes)", "unadjusted"),
                               ("neg_bonf", st["neg_bonf"], "count", "negative slopes significant", "Bonferroni"),
                               ("pos_bonf", st["pos_bonf"], "count", "positive slopes significant", "Bonferroni"),
                               ("family_slopes", st["family"], "count", "model x pair slopes with a p-value", "Bonferroni"),
                               ("slopes_zero", st["zero"], "count", "slopes exactly zero", "n/a"),
                               ("neg_unadj", st["neg_unadj"], "count", "negative slopes significant", "unadjusted"),
                               ("pos_unadj", st["pos_unadj"], "count", "positive slopes significant", "unadjusted")]:
        R.add(f"f9_{k}", v, f, "Figure 9", "slope of observed/expected ratio on dose", "random doses", "naive",
              "Nielsen", stat, adj, s9 + " (visualize.comprehensive_figures._slope_pooled_stats)")
    A9 = "strip chart (all models)"
    for k, i, lab, adj in [("m", 2, "model x pair slopes of the panel", "Bonferroni"),
                           ("slopes_bonf", 1, "slopes significant", "Bonferroni"),
                           ("slopes_unadj", 0, "slopes significant", "unadjusted")]:
        R.add(f"f9_{k}", T.aud("Figure 9", A9)[i], "count", "Figure 9", lab, "random doses", "naive", "Nielsen", "count",
              adj, f"{RR}/{P_}, Figure 9, panel '{A9}'")
    a2, a5, a7 = T.aud("Figure 2"), T.aud("Figure 5"), T.aud("Figure 7")
    for key, v, fig, lab, adj, row in [
            ("f2_n_tested", a2[2], "Figure 2", "models with a chi-square test", "n/a", "Figure 2, m"),
            ("f5_open", a5[0] - a5[1], "Figure 5", "models significant at p < 0.05 unadjusted only", "unadjusted",
             "Figure 5, n_sig_unadj - n_sig_bonf"),
            ("f7_n_scenarios", a7[2], "Figure 7", "scenarios tested (Kendall's W)", "n/a", "Figure 7, m"),
            ("f10_n_tests", T.aud("Figure 10")[4], "Figure 10", "tests in the panel", "Bonferroni", "Figure 10, m_panel"),
            ("f13_vig_ome_bonf", T.aud("Figure 13", "text vignettes, oral morphine", "tight")[1], "Figure 13",
             "models significant on ±15%", "Bonferroni", "Figure 13, text vignettes, oral morphine, tight, n_sig_bonf"),
            ("f13_vig_ome_unadj", T.aud("Figure 13", "text vignettes, oral morphine", "tight")[0], "Figure 13",
             "models significant on ±15%", "unadjusted", "Figure 13, text vignettes, oral morphine, tight, n_sig_unadj"),
            ("f14_drop_bonf", T.aud("Figure 14 (top)", "dosing-unit effect", "tight")[3], "Figure 14",
             "models with a drop on ±15% (mg/day to mg/4 h)", "Bonferroni", "Figure 14 (top), dosing-unit effect, tight, n_sig_bonf_panel")]:
        R.add(key, v, "count", fig, lab, "n/a", "n/a", "Nielsen", "count", adj, f"{RR}/{P_}, {row}")
    a9 = T.a[(T.a.figure == "Figure 9") & (T.a.panel == A9)].iloc[0]
    for key, col, adj in [("f9_neg_pct_unadj", "dose_dep_neg_pct_unadj", "unadjusted"),
                          ("f9_neg_pct_bonf", "dose_dep_neg_pct_bonf", "Bonferroni")]:
        R.add(key, float(a9[col]), "percent:1", "Figure 9", "share of model x pair slopes negative and significant",
              "random doses", "naive", "Nielsen", "percent of slopes", adj, f"{RR}/{P_}, Figure 9, panel '{A9}', {col}")
    a10 = T.a[T.a.figure == "Figure 10"]
    for key, col, adj in [("f10_brackets_unadj", "n_sig_unadj", "unadjusted"), ("f10_brackets_bonf", "n_sig_bonf", "Bonferroni")]:
        R.add(key, int(a10[col].sum()), "count", "Figure 10", "significant brackets (0.6–1.7×, matched Wilcoxon)", "n/a",
              "naive", "Nielsen", "count of brackets", adj, f"{RR}/{P_}, figure='Figure 10', sum of {col}")
    for key, i, adj in [("f13_vig_other_unadj", 0, "unadjusted"), ("f13_vig_other_bonf", 1, "Bonferroni")]:
        R.add(key, T.aud("Figure 13", "text vignettes, other conversions", "tight")[i], "count", "Figure 13",
              "models significant on ±15%", "text vignettes, other conversions", "reference-grounded", "Nielsen", "count", adj,
              f"{RR}/{P_}, Figure 13, text vignettes, other conversions, tight")
    R.add("f14_m_panel",T.aud("Figure 14 (top)", "dosing-unit effect", "tight")[4], "count", "Figure 14",
          "tests in the panel (family)", "random doses", "n/a", "Nielsen", "count", "Bonferroni",
          f"{RR}/{P_}, Figure 14 (top), dosing-unit effect, m_panel")
    for band in ("tight", "broad"):
        fam = f"results_nielsen/wilcoxon_units_effect_random_dose:{band}"
        u = T.a[T.a.family == fam].iloc[0]
        for key, col, adj in [(f"f14_units_{band}_unadj", "n_sig_unadj", "unadjusted"),
                              (f"f14_units_{band}_bonf", "n_sig_bonf", "Bonferroni")]:
            R.add(key, int(u[col]), "count", "Figure 14", f"models with a units-explained recovery ({band})",
                  "random doses", "unit-augmented", "Nielsen", "count of models", adj, f"{RR}/{P_}, family {fam}, {col}")
    R.add("f10_n_comparisons", R.values["f10_n_tests"] // n_models, "count", "Figure 10", "comparisons among random doses, empty prompts and text vignettes",
          "n/a", "naive", "Nielsen", "count", "n/a", f"{RR}/{P_}, Figure 10, m_panel / n_models")
    R.add("f12_n_pairs", n_models * (n_models - 1) // 2, "count", "Figure 12", "model pairs", "random doses", "reference-grounded",
          "Nielsen", "count of pairs", "Bonferroni", "common/models.py CONSISTENT_MODELS: n (n - 1) / 2")
    rnk = _csv("rankings_fig12_fig15_broad.csv")
    for key, fig_, lab in [("f1_changed", "Figure 1", "Figure 1"), ("f8_changed", "Figure 8", "Figure 8"),
                           ("f12_changed", "Figure 12", "Figure 12"),("f15_top_changed", "Figure 15 (top)", "Figure 15 (top)"),
                           ("f15_bottom_changed", "Figure 15 (bottom)", "Figure 15 (bottom)")]:
        R.add(key, int(rnk[rnk.figure == fig_].group_changed.sum()), "count", fig_.split(" (")[0],
              "models whose group changed against the submitted ranking", "n/a", "reference-grounded" if key == "f12_changed" else "unit-augmented",
              "Nielsen", "count of models", "n/a", f"{RR}/rankings_fig12_fig15_broad.csv, figure={fig_}, group_changed")
    for name, f_, lab in [("vig_ome", "vignettes_ome", "vignettes, oral morphine"), ("vig_other", "vignettes_nonome", "vignettes, other conversions"),
                          ("rd_ome", "random_dose_ome", "random doses, oral morphine"), ("rd_other", "random_dose_nonome", "random doses, other conversions")]:
        w = pd.read_csv(os.path.join(NIELSEN_DIR, f"wilcoxon_nielsen_effect_{f_}.csv")).set_index("model").loc["POOLED"]
        for band, col in (("tight", "tight_p"), ("broad", "broad_p")):
            R.add(f"f13_{name}_{band}_p", w[col], "p_caption", "Figure 13", "within ±15%" if band == "tight" else "within 0.6–1.7×",
                  lab, "naive -> reference-grounded", "Nielsen", "p (pooled Wilcoxon over model x item pairs)", "unadjusted",
                  f"{NIELSEN_DIR}/wilcoxon_nielsen_effect_{f_}.csv, model=POOLED, {col}")

    # ── supplement: extreme errors; Multimedia Appendix 2: models worse after grounding ────────
    SAFE = _csv("safety_tests_by_model.csv")
    for exp, word, ca, cb in [("vignette", "vignettes", "vignette_naive", "vignette_nielsen"), ("random_dose", "random doses", "rd_naive", "rd_nielsen")]:
        for o, lab in [("underdose", "below ½× the lowest standard"), ("overdose", "above 2× the highest standard"),
                       ("severe_overdose", "above 4× the highest standard")]:
            fam = T.family(f"safety_tests:{ca}>{cb}:{o}")
            cnt = {"better|filled": 0, "better|open": 0, "worse|filled": 0, "worse|open": 0}
            for m in order:
                t_ = SAFE[(SAFE.cond_b == cb) & (SAFE.outcome == o) & (SAFE.model == m)].iloc[0]
                lv = _level(*fam[m])
                if lv and t_.mean_b != t_.mean_a:
                    cnt[("better" if t_.mean_b < t_.mean_a else "worse") + "|" + lv] += 1
            for kk, ky in [("better|filled", "bonf"), ("better|open", "open")]:
                R.add(f"supp_{vn[exp]}_{o}_{ky}", cnt[kk], "count", "Supplement: extreme errors", lab, word,
                      "naive -> reference-grounded", "all current standards", "count of models with ★" if ky == "bonf" else "count of models with ☆ only",
                      "Bonferroni" if ky == "bonf" else "unadjusted",
                      f"{RR}/safety_tests_by_model.csv (mean_a, mean_b) + {RR}/multiple_comparisons_tests.csv family safety_tests:{ca}>{cb}:{o}")
    worse = []
    outs = ["tight", "broad", "median_fold_error", "above_range", "below_range", "underdose", "overdose", "severe_overdose"]
    for exp, word, _ in EXPS:
        for o in outs:
            for sc in (["any"] if o in outs[3:] else ["nielsen", "non_nielsen"]):
                d = circ(exp, sc, o)
                kind = "tf" if o == "median_fold_error" else "band"
                fam = T.family(f"circularity:{exp}:all:{sc}:{o}:{kind}")
                ms = [f"{DISPLAY_NAMES.get(m, m.split('/')[-1])} ({'Bonferroni' if _level(*fam[m]) == 'filled' else 'unadjusted only'})"
                      for m in order if m in fam and _level(*fam[m]) and d.loc[m, "gain"] < 0]
                if ms:
                    worse.append(f"{exp.replace('_', ' ')} {o.replace('_', ' ')}: {', '.join(ms)}")
    R.add("ma2_worse_text", "; ".join(worse), "text", "Multimedia Appendix 2", "models worse after grounding (names only)",
          "both", "naive -> reference-grounded", "all", "list of models", "unadjusted",
          f"{RR}/circularity.csv gain < 0 + {RR}/multiple_comparisons_tests.csv families circularity:*")

    # ── Multimedia Appendix 1: agreement between standards ─────────────────────────────────────
    d = byc[byc.n_alternatives_current >= 1].copy()
    cols = ["cdc_2022_vs_nielsen", "nih_heal_vs_nielsen", "mcpherson_2018_vs_nielsen"]
    g = d.groupby(["from_opioid", "to_opioid"] + cols, dropna=False, sort=False).size().reset_index()
    v = g[cols]
    far = np.exp(np.log(v).abs().max(axis=1))

    def out(x, lo, hi):
        return pd.notna(x) and (x < lo - 1e-9 or x > hi + 1e-9)
    ot = v.apply(lambda r: any(out(x, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER) for x in r), axis=1)
    ob = v.apply(lambda r: any(out(x, BROAD_RANGE_LOWER, BROAD_RANGE_UPPER) for x in r), axis=1)
    allv = v.to_numpy(float)
    gmin, gmax = float(np.nanmin(allv)), float(np.nanmax(allv))
    s1 = f"{RR}/references/by_conversion.csv, rows with n_alternatives_current >= 1, distinct (from, to, standards)"
    for k, val, f, lab, stat in [("ma1_n", len(g), "count", "conversions with Nielsen and another current standard", "count"),
                                 ("ma1_farthest", float(far.max()), "ratio:2", "farthest standard from Nielsen (fold)", "maximum"),
                                 ("ma1_n_outside_tight", int(ot.sum()), "count", "conversions with a standard outside ±15% of Nielsen", "count"),
                                 ("ma1_n_outside_broad", int(ob.sum()), "count", "conversions with a standard outside 0.6–1.7× of Nielsen", "count"),
                                 ("ma1_lowest_standard", gmin, "ratio:2", "lowest current standard relative to Nielsen", "minimum"),
                                 ("ma1_highest_standard", gmax, "ratio:2", "highest current standard relative to Nielsen", "maximum"),
                                 ("ma1_extended_lowest", _r2(gmin / TIGHT_RANGE_UPPER), "ratio:2", "lowest standard extended by ±15%", "lowest / 1.15"),
                                 ("ma1_extended_highest", _r2(gmax * TIGHT_RANGE_UPPER), "ratio:2", "highest standard extended by ±15%", "highest x 1.15")]:
        R.add(k, val, f, "Multimedia Appendix 1", lab, "both", "n/a", "Nielsen", stat, "n/a", s1)
    R.add("ma1_farthest_conversion", str(g.apply(lambda r: f"{r.from_opioid.replace('_', ' ')} → {r.to_opioid.replace('_', ' ')}", axis=1).iloc[int(np.argmax(far.to_numpy()))]),
          "text", "Multimedia Appendix 1", "conversion with the farthest standard", "both", "n/a", "Nielsen", "name", "n/a", s1)

    # ── standard matching (3E) ──────────────────────────────────────────────────────────────────
    sm = _csv("standard_matches.csv")
    sm = sm[(sm.row_type == "summary") & (sm.item == "eligible_wrong_cells")].set_index("experiment")
    for exp, word, _ in EXPS:
        r = sm.loc[exp]
        s3 = f"{RR}/standard_matches.csv, row_type=summary, item=eligible_wrong_cells, experiment={exp}"
        R.add(f"std_match_{vn[exp]}_observed_pct", r.observed_match_rate * 100, "percent:0", "3E (standard matching)",
              "wrong answers matching another standard", word, "naive", "all current standards", "observed share", "n/a", s3 + ", observed_match_rate")
        R.add(f"std_match_{vn[exp]}_chance_pct", r.chance_rate_shift * 100, "percent:0", "3E (standard matching)",
              "wrong answers matching another standard", word, "naive", "all current standards", "share expected by chance (shift)", "n/a", s3 + ", chance_rate_shift")

    # ── DI 2 subset check ────────────────────────────────────────────────────────────────────────
    di = _csv("di2_subset.csv")

    def dv(metric, exp, subset):
        return float(di[(di.metric == metric) & (di.experiment == exp) & (di.subset == subset)].value.iloc[0])
    sd = f"{RR}/di2_subset.csv"
    for exp, word, _ in EXPS:
        for band in ("tight", "broad"):
            R.add(f"di2_{vn[exp]}_change_pp_{band}", dv(f"change_pp_{band}_without_minus_with", exp, "no_di2_vs_all"), "pp:1",
                  "DI 2 subset check", "change in mean accuracy without the DI 2 scenarios", word, "naive", "Nielsen",
                  "mean over models (without minus with)", "n/a", f"{sd}, metric=change_pp_{band}_without_minus_with, subset=no_di2_vs_all")
    R.add("di2_vig_rho_mean_tight", dv("spearman_rho_mean_tight_accuracy", "vignette", "all_vs_no_di2"), "ratio:2",
          "DI 2 subset check", "rank correlation of mean tight accuracy, with vs without DI 2", "vignettes", "naive", "Nielsen",
          "rho", "n/a", f"{sd}, metric=spearman_rho_mean_tight_accuracy, subset=all_vs_no_di2")
    R.add("di2_refusal_n_responses", int(dv("refusal_check_n_responses", "vignette", "all")), "count", "DI 2 subset check",
          "responses searched for refusals", "vignettes", "all conditions", "n/a", "count", "n/a", f"{sd}, metric=refusal_check_n_responses")
    R.add("di2_refusal_decline", int(dv("refusal_pattern_decline", "vignette", "all")), "count", "DI 2 subset check",
          "responses matching a refusal pattern", "vignettes", "all conditions", "n/a", "count", "n/a", f"{sd}, metric=refusal_pattern_decline")
    return R.values, R.meta
