"""
Multiple comparisons: one method for the whole paper (author decision 45, DZ 4).

    python -m analysis.revision.multiple_comparisons

Bonferroni within each family of tests (one panel or one table column): the
adjusted p-value is min(1, p x m), m = the number of tests with a p-value in the
family. This is the method the submitted analysis already used for the CMH
rankings, the per-cell Fisher tests (Figure 6) and the Mann-Whitney tests; here it
is applied to the per-model tests that drive a significance mark or an in-text
count. Pooled tests (POOLED / ALL_MODELS / OVERALL rows) are unchanged and not
listed. There is no "smaller of tight and broad p" rule: tight and broad are
separate families, and each mark refers to one measure.

No new tests are run and no model is called: every p-value is read from an
existing result CSV. Two stored-only counts are recomputed from the same data with
the same test: Figure 9's chi-square of CV by accuracy (per model, as in
orchestration/extract_figure_stats.py) and Figure 9's per-pair regression slopes
(read from pair_model_analysis.csv).

Writes (atomically, deterministic, no randomness):
  results_revision/multiple_comparisons_tests.csv   one row per test
      family, scope (published | revision), figure, panel, test, measure, comparison,
      source, unit (model, item or pair), group, p, effect (signed effect; positive =
      second condition better where a direction exists), m, p_bonf, sig_unadj
      (p < 0.05), sig_bonf (p_bonf < 0.05), m_panel, p_bonf_panel (m over every test
      of the same figure, panel and measure: 'one panel' read as all comparisons
      drawn in the panel), p_bonf_published (the correction already in the source,
      CMH pairs only), already_adjusted
  results_revision/multiple_comparisons_audit.csv   one row per family
      family definition, m, n_sig_unadj, n_sig_bonf, n_open (marks that change from
      filled to open), n_sig_bonf_panel, plus n_groups_unadj / n_groups_bonf for
      families split by group (count of models with at least one significant test)
  results_revision/rankings_fig12_fig15_broad.csv   CMH rankings on 0.6-1.7x only
      (Figures 1, 8, 12 and 15), sorted by group then %, next to the published
      tight-or-broad groups

Scope of "family": the unit is one table column or one panel for one measure: the
13 per-model tests of a comparison and measure (tight and broad separately); the
13 per-model chi-square or Spearman tests of a figure; the 20 per-scenario Kendall
tests; the 78 CMH model pairs of a ranking; every model x pair slope of the Figure 9
panel (and, as an alternative, within each model's 49 pairs).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.revision import stats
from common.models import CONSISTENT_MODELS

RES = "results_nielsen"
RR = "results_revision"
ALPHA = 0.05
MODELS = set(CONSISTENT_MODELS)
SCORE_DIRS = ["results_nielsen", "results_all", "results_non_nielsen", "results_geomean"]

TEST_COLS = ["family", "scope", "figure", "panel", "test", "measure", "comparison", "source", "unit", "group", "p",
             "effect", "family_scope", "m", "p_bonf", "sig_unadj", "sig_bonf", "m_panel", "p_bonf_panel", "p_bonf_published",
             "already_adjusted"]


def _write(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(RR, name)
    df.to_csv(path + ".tmp", index=False, float_format="%.10g")
    os.replace(path + ".tmp", path)


def _models(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"].isin(MODELS)].copy()


# ── readers: each returns unit, group, p, effect ────────────────────────────

def _frame(unit, p, effect=None, group=None) -> pd.DataFrame:
    unit = pd.Series(list(unit))
    return pd.DataFrame({"unit": unit, "group": pd.Series(list(group)) if group is not None else "",
                         "p": pd.to_numeric(pd.Series(list(p)), errors="coerce"),
                         "effect": pd.to_numeric(pd.Series(list(effect)), errors="coerce") if effect is not None else np.nan})


def read_wilcoxon(path: str, measure: str) -> pd.DataFrame:
    d = _models(pd.read_csv(path))
    return _frame(d["model"], d[f"{measure}_p"], d[f"{measure}_mean"])


def read_mannwhitney(path: str, measure: str) -> pd.DataFrame:
    d = _models(pd.read_csv(path))
    return _frame(d["model"], d[f"{measure}_mannwhitney_p"], d[f"{measure}_diff"])


def read_cross(path: str, a: str, b: str, measure: str) -> pd.DataFrame:
    d = _models(pd.read_csv(path))
    d = d[(d.condition_a == a) & (d.condition_b == b)]
    return _frame(d["model"], d[f"{measure}_wilcoxon_p"], d[f"{measure}_diff"])


def read_spearman_cross(path: str, a: str, b: str, rng: str) -> pd.DataFrame:
    d = _models(pd.read_csv(path))
    d = d[(d.condition_a == a) & (d.condition_b == b) & (d.range == rng)]
    return _frame(d["model"], d["spearman_p"], d["spearman_rho"])


def read_chi2_vignettes(path: str) -> pd.DataFrame:
    d = _models(pd.read_csv(path))
    return _frame(d["model"], d["p_value"])


def read_rd_chi2() -> pd.DataFrame:
    """Figure 9's chi-square of CV bin by accuracy class, per model (the inline computation of
    orchestration/extract_figure_stats.py, same data and test)."""
    from scipy.stats import chi2_contingency
    d = pd.read_csv(os.path.join(RES, "pair_model_analysis.csv"))
    d = d[(d.include_nielsen == False) & (d.include_examples == False) & d.model.isin(MODELS)]  # noqa: E712
    cv, tight, broad = "observed_coefficient_cv", "tight(0.85,1.15)", "broad(0.6,1.7)"
    d = d.dropna(subset=[cv, tight, broad]).copy()
    labels = ["<0.1", "0.1–0.5", "≥0.5"]
    d["_bin"] = pd.cut(d[cv], bins=[0, 0.1, 0.5, float("inf")], labels=labels, right=False)
    d["_acc"] = "neither"
    d.loc[d[broad] >= 0.5, "_acc"] = "broad"
    d.loc[d[tight] >= 0.5, "_acc"] = "tight"
    rows = []
    for m in sorted(d.model.unique()):
        g = d[d.model == m]
        table = np.array([[int(((g._acc == a) & (g._bin == b)).sum()) for b in labels] for a in ["tight", "broad", "neither"]])
        try:
            p = float(chi2_contingency(table)[1])
        except (ValueError, ZeroDivisionError):
            p = np.nan
        rows.append((m, p))
    return _frame([r[0] for r in rows], [r[1] for r in rows])


def read_slopes() -> pd.DataFrame:
    """Figure 9: regression slope of observed/expected on dose, per model x opioid pair (naive random doses)."""
    d = pd.read_csv(os.path.join(RES, "pair_model_analysis.csv"))
    d = d[(d.include_nielsen == False) & (d.include_examples == False) & d.model.isin(MODELS)]  # noqa: E712
    d = d.sort_values(["model", "opioid_pair"])
    return _frame(d["model"] + " | " + d["opioid_pair"], d["regression_slope_pvalue"], -d["regression_slope"], d["model"])


def read_kendall() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(RES, "kendall_w_vignettes.csv"))
    return _frame(d["scenario_key"], d["kendall_p"], d["kendall_w"])


def read_spearman_models() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(RES, "variance_icc_spearman.csv"))
    d = _models(d[d.stat_type == "variance_vs_icc"])
    return _frame(d["model"], d["p_value"], d["rho"])


def read_icc_f() -> pd.DataFrame:
    d = _models(pd.read_csv(os.path.join(RES, "icc_analysis.csv"))).sort_values(["model", "scenario_key"])
    return _frame(d["model"] + " | " + d["scenario_key"], d["f_pvalue"], d["icc"], d["model"])


def read_cmh_pairs(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    f = _frame(d["model_a"] + " > " + d["model_b"], d["broad_p"])
    f["p_bonf_published"] = pd.to_numeric(d["broad_p_bonf"], errors="coerce").to_numpy()
    f["_tight_p"] = pd.to_numeric(d["tight_p"], errors="coerce").to_numpy()
    return f


# ── the families ─────────────────────────────────────────────────────────────

def _wilcoxon_families():
    """(figure, panel, comparison label, file stem under each results dir) for the per-model Wilcoxon tests."""
    F = []
    for exp, lab in (("vignettes", "text vignettes"), ("random_dose", "random doses")):
        F.append(("Figure 11", f"{lab}", f"naive vs reference-grounded, {lab}", f"wilcoxon_nielsen_effect_{exp}"))
    for exp, lab in (("vignettes", "text vignettes"), ("random_dose", "random doses")):
        for sub, sl in (("ome", "oral morphine"), ("nonome", "other conversions")):
            F.append(("Figure 13", f"{lab}, {sl}", f"naive vs reference-grounded, {lab}, {sl}", f"wilcoxon_nielsen_effect_{exp}_{sub}"))
    F.append(("Figure 14 (top)", "dosing-unit effect", "mg/day vs mg/4 h, reference-grounded", "wilcoxon_dosing_unit_effect"))
    F.append(("Figure 14 (top)", "dosing-unit effect", "mg/4 h vs mg/4 h with units explained", "wilcoxon_units_effect_random_dose"))
    F.append(("Figure 14 (bottom)", "full treatment", "naive vs reference-grounded (mg/day), random doses, oral morphine",
              "wilcoxon_nielsen_effect_random_dose_ome"))
    F.append(("Figure 14 (bottom)", "full treatment", "mg/4 h vs mg/4 h with units explained", "wilcoxon_units_effect_random_dose"))
    F.append(("text (no figure)", "units explanation, vignettes", "reference-grounded vs unit-augmented, vignettes", "wilcoxon_units_effect_vignettes"))
    F.append(("text (no figure)", "empty prompts", "naive vs reference-grounded, empty prompts", "wilcoxon_nielsen_effect_empty_vignettes"))
    F.append(("text (no figure)", "text vs empty", "text vs empty vignettes", "wilcoxon_text_vs_empty"))
    F.append(("text (no figure)", "text vs empty, same conversion", "text vs empty vignettes, same-conversion scenarios",
              "wilcoxon_text_vs_empty_same_conversion"))
    return F


def build_tests() -> pd.DataFrame:
    parts = []
    seen: set = set()

    def add(family, scope, figure, panel, test, measure, comparison, source, frame, by_group=False, adjusted=False,
            panel_group=None):
        f = frame.copy()
        f["family"], f["scope"], f["figure"], f["panel"], f["test"] = family, scope, figure, panel, test
        f["measure"], f["comparison"], f["source"] = measure, comparison, source
        f["already_adjusted"] = adjusted
        f["_panel_group"] = panel_group if panel_group is not None else family
        f["_by_group"] = by_group
        parts.append(f)

    # published pipeline: per-model Wilcoxon (tight and broad are separate families)
    for d in SCORE_DIRS:
        for figure, panel, comp, stem in _wilcoxon_families():
            path = os.path.join(d, f"{stem}.csv")
            if not os.path.exists(path):
                continue
            if d != RES and not stem.startswith("wilcoxon_nielsen_effect_") or (d != RES and "empty" in stem):
                continue
            for measure in ("tight", "broad"):
                fam = f"{d}/{stem}:{measure}"
                if fam in seen and d != RES:         # other scores: each table once
                    continue
                if fam in seen:                      # the same table drawn in a second figure: its own family
                    fam += "@" + figure.replace(" ", "").replace("(", "").replace(")", "").lower()
                seen.add(fam)
                scope = "published" if d == RES else "robustness (other scores)"
                add(fam, scope, figure if d == RES else f"{figure} under {d.replace('results_', '')} score", panel,
                    "per-model Wilcoxon signed-rank (items within model)", measure, comp, path, read_wilcoxon(path, measure),
                    panel_group=f"{d}|{figure}|{panel}|{measure}")
    # Mann-Whitney per model (oral morphine vs other conversions; table only)
    for tag in ("naive", "nielsen"):
        path = os.path.join(RES, f"comparison_ome_vs_nonome_{tag}_permodel.csv")
        if os.path.exists(path):
            for measure in ("tight", "broad"):
                add(f"{path[:-4]}:{measure}", "published", "text (no figure)", f"oral morphine vs other, {tag}",
                    "per-model Mann-Whitney U (items within model)", measure, f"oral morphine vs other conversions, {tag}",
                    path, read_mannwhitney(path, measure))
    # Figure 10: paired Wilcoxon per model, broad accuracy, three comparisons
    for sfx, lab in (("", ""), ("_same_conversion", " (same-conversion scenarios)")):
        path = os.path.join(RES, f"cross_experiment_matched_wilcoxon{sfx}.csv")
        if not os.path.exists(path):
            continue
        for a, b, comp in (("random_dose", "empty_vignette", "random doses vs empty prompts"),
                           ("empty_vignette", "text_vignette", "empty prompts vs text vignettes"),
                           ("random_dose", "text_vignette", "random doses vs text vignettes")):
            add(f"{path[:-4]}:{a}-{b}:broad", "published", "Figure 10" if not sfx else "text (no figure)",
                "matched conversions" + lab, "per-model paired Wilcoxon signed-rank", "broad", comp + lab, path,
                read_cross(path, a, b, "broad"), panel_group=f"{path}|broad")
        sp = os.path.join(RES, f"cross_experiment_matched_spearman{sfx}.csv")
        for a, b, comp in (("random_dose", "empty_vignette", "random doses vs empty prompts"),
                           ("empty_vignette", "text_vignette", "empty prompts vs text vignettes"),
                           ("random_dose", "text_vignette", "random doses vs text vignettes")):
            for rng in ("tight", "broad"):
                add(f"{sp[:-4]}:{a}-{b}:{rng}", "published", "Figure 10 (text)" if not sfx else "text (no figure)",
                    "matched conversions" + lab, "per-model Spearman correlation across matched conversions", rng,
                    comp + lab, sp, read_spearman_cross(sp, a, b, rng))
    # Figures 2 and 3: chi-square of CV by accuracy, per model (vignettes)
    path = os.path.join(RES, "cv_accuracy_chi2.csv")
    for fig in ("Figure 2", "Figure 3"):
        add(f"{path[:-4]}:{fig}", "published", fig, "bottom grid (CV bins x accuracy)", "per-model chi-square", "accuracy class",
            "CV bin by accuracy class, naive vignettes", path, read_chi2_vignettes(path))
    # Figure 5: Spearman of variance against ICC, per model
    path = os.path.join(RES, "variance_icc_spearman.csv")
    add(f"{path[:-4]}:variance_vs_icc", "published", "Figure 5", "per-model bars", "per-model Spearman correlation",
        "variance vs ICC", "scenario variance vs ICC (20 scenarios per model)", path, read_spearman_models())
    # Figure 6: ICC F-test per model x scenario (text: models with a significant ICC); sig_pairs already Bonferroni
    path = os.path.join(RES, "icc_analysis.csv")
    add(f"{path[:-4]}:f_test", "published", "Figure 6 (text)", "model x scenario grid", "one-way ANOVA F-test (wording effect)",
        "ICC", "wording effect within model, per scenario (Bonferroni within model over its scenarios)", path, read_icc_f(),
        by_group=True)
    # Figure 7: Kendall's W per scenario
    path = os.path.join(RES, "kendall_w_vignettes.csv")
    add(f"{path[:-4]}:kendall_p", "published", "Figure 7", "per-scenario bars", "Kendall's W test", "agreement across models",
        "20 scenarios", path, read_kendall())
    # Figure 9: chi-square per model, and per model x pair regression slopes (dose-dependent bias)
    add("figure9:cv_chi2", "published", "Figure 9", "CV bins x accuracy", "per-model chi-square", "accuracy class",
        "CV bin by accuracy class, naive random doses", os.path.join(RES, "pair_model_analysis.csv"), read_rd_chi2())
    sl = read_slopes()
    add("figure9:slope_panel", "published", "Figure 9", "strip chart (all models)", "per-pair regression slope t-test",
        "dose-dependent bias", "slope of observed/expected on dose, family = every model x pair of the panel",
        os.path.join(RES, "pair_model_analysis.csv"), sl)
    add("figure9:slope_per_model", "alternative (per model)", "Figure 9", "strip chart, within each model",
        "per-pair regression slope t-test", "dose-dependent bias", "same tests, family = the model's pairs",
        os.path.join(RES, "pair_model_analysis.csv"), sl, by_group=True)
    # CMH pair tests of the published rankings (already Bonferroni over 2 x pairs: tight and broad)
    for stem, fig, cond in (("ranking_naive_vignettes", "Figure 1", "naive vignettes"),
                            ("ranking_naive_random_doses", "Figure 8", "naive random doses"),
                            ("ranking_random_doses_nielsen", "Figure 12", "random doses, reference-grounded"),
                            ("ranking_random_doses_nielsen_units", "Figure 15 (top)", "random doses mg/4 h, unit-augmented"),
                            ("ranking_vignettes_nielsen_units", "Figure 15 (bottom)", "vignettes, unit-augmented")):
        path = os.path.join(RES, f"{stem}_pairs.csv")
        add(f"{path[:-4]}:broad", "published", fig, "model pairs", "one-sided CMH test over items, all model pairs",
            "broad", cond, path, read_cmh_pairs(path).drop(columns=["_tight_p"]), adjusted=True)
    # revision: CMH on 0.6-1.7x only
    for cond, fig, panel in RANK_CONDS:
        r = stats.rank_models(_long(), "nielsen", cond, "all", bands=("broad",))
        p = r.pairs_df
        f = _frame(p["model_a"] + " > " + p["model_b"], p["broad_p"])
        f["p_bonf_published"] = p["broad_p_bonf"].to_numpy()
        add(f"cmh_broad:{cond}", "revision", fig, panel, "one-sided CMH test over items, all model pairs (broad only)", "broad",
            cond, "analysis/revision/stats.rank_models(bands=('broad',))", f, adjusted=True)
    # revision tables and figures: circularity (per-model band and threshold-free tests)
    c = pd.read_csv(os.path.join(RR, "circularity.csv"))
    c = c[c.model.isin(MODELS)]
    for (exp, sub, score, outcome), g in c.groupby(["experiment", "subset", "score", "outcome"], sort=True):
        figs = "Figure 11 / Circularity / Multimedia Appendix 2" if sub == "all" else "supplement"
        if outcome != "median_fold_error":
            add(f"circularity:{exp}:{sub}:{score}:{outcome}:band", "revision", figs, f"{exp}, {sub}", "per-model Wilcoxon (items)",
                outcome, f"naive vs reference-grounded, scored against {score}", RR + "/circularity.csv",
                _frame(g["model"], g["band_p"], g["band_mean_diff"]))
        elif score != "any":
            add(f"circularity:{exp}:{sub}:{score}:median_fold_error:tf", "revision", figs, f"{exp}, {sub}", "per-model Wilcoxon (items)",
                "median fold error", f"naive vs reference-grounded, scored against {score}", RR + "/circularity.csv",
                _frame(g["model"], g["tf_p"], g["tf_mean_diff"]))
    # revision: range rates and extreme errors, per model
    s = pd.read_csv(os.path.join(RR, "safety_tests_by_model.csv"))
    s = s[s.model.isin(MODELS)]
    for (a, b, outcome), g in s.groupby(["cond_a", "cond_b", "outcome"], sort=True):
        figs = "Figure 11 / supplement extreme errors"
        add(f"safety_tests:{a}>{b}:{outcome}", "revision", figs, f"{a} to {b}", "per-model Wilcoxon (items)", outcome,
            f"{a} vs {b}", RR + "/safety_tests_by_model.csv", _frame(g["model"], g["p"], g["mean_diff"]))
    rp = pd.read_csv(os.path.join(RR, "range_rates_by_prompt.csv"))
    rp = rp[rp.model.isin(MODELS)]
    for (prompt, sub, outcome), g in rp.groupby(["prompt", "subset", "outcome"], sort=True):
        add(f"range_rates:{prompt}:{sub}:{outcome}", "revision", "report (prompt types)", f"{prompt}, {sub}",
            "per-model Wilcoxon (items)", outcome, f"{prompt}: naive vs reference-grounded", RR + "/range_rates_by_prompt.csv",
            _frame(g["model"], g["p"], g["mean_diff_pp"]))
    w = pd.read_csv(os.path.join(RR, "wilcoxon_threshold_free.csv"))
    w = w[w.model.isin(MODELS)]
    for (score, comp, sub), g in w.groupby(["score", "comparison", "subset"], sort=True):
        add(f"threshold_free:{score}:{comp}:{sub}:tf", "revision", "tables (threshold-free)", f"{comp}, {sub}",
            "per-model Wilcoxon (items)", "median |log ratio|", f"{comp}, scored against {score}",
            RR + "/wilcoxon_threshold_free.csv", _frame(g["model"], g["tf_p"], g["tf_mean_diff"]))
        if g["tight_p"].notna().any():
            add(f"threshold_free:{score}:{comp}:{sub}:tight", "revision", "tables (threshold-free)", f"{comp}, {sub}",
                "per-model Wilcoxon (items)", "tight", f"{comp}, scored against {score}", RR + "/wilcoxon_threshold_free.csv",
                _frame(g["model"], g["tight_p"], g["tight_mean_diff"]))
    t = pd.concat(parts, ignore_index=True)
    return _adjust(t)


def _adjust(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    key = t["family"].where(~t["_by_group"], t["family"] + "|" + t["group"].astype(str))
    t["family_scope"] = np.where(t["_by_group"], "within group", "all tests listed")
    t["m"] = t["p"].notna().groupby(key).transform("sum").astype(int)
    t["p_bonf"] = (t["p"] * t["m"]).clip(upper=1.0)
    t["m_panel"] = t["p"].notna().groupby(t["_panel_group"]).transform("sum").astype(int)
    t["p_bonf_panel"] = (t["p"] * t["m_panel"]).clip(upper=1.0)
    t["sig_unadj"] = t["p"] < ALPHA
    t["sig_bonf"] = t["p_bonf"] < ALPHA
    if "p_bonf_published" not in t.columns:
        t["p_bonf_published"] = np.nan
    return t[TEST_COLS + ["_panel_group", "_by_group"]].drop(columns=["_panel_group", "_by_group"])


def build_audit(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in t.groupby("family", sort=True):
        r0 = g.iloc[0]
        valid = g[g.p.notna()]
        row = {"family": fam, "scope": r0.scope, "figure": r0.figure, "panel": r0.panel, "test": r0.test,
               "measure": r0.measure, "comparison": r0.comparison,
               "family_scope": r0.family_scope,
               "n_tests": len(valid), "m": int(valid.m.max()) if len(valid) else 0,
               "m_min": int(valid.m.min()) if len(valid) else 0,
               "n_sig_unadj": int(valid.sig_unadj.sum()), "n_sig_bonf": int(valid.sig_bonf.sum()),
               "n_open": int((valid.sig_unadj & ~valid.sig_bonf).sum()),
               "m_panel": int(valid.m_panel.max()) if len(valid) else 0,
               "n_sig_bonf_panel": int((valid.p_bonf_panel < ALPHA).sum()),
               "already_adjusted": bool(r0.already_adjusted),
               "n_sig_published_correction": int((valid.p_bonf_published < ALPHA).sum()) if valid.p_bonf_published.notna().any() else np.nan}
        eff = valid.effect
        row["n_sig_unadj_positive"] = int((valid.sig_unadj & (eff > 0)).sum()) if eff.notna().any() else np.nan
        row["n_sig_bonf_positive"] = int((valid.sig_bonf & (eff > 0)).sum()) if eff.notna().any() else np.nan
        if (g.group.astype(str) != "").any():
            gr = valid.groupby("group")
            row["n_groups"] = int(gr.ngroups)
            row["n_groups_sig_unadj"] = int(gr.sig_unadj.any().sum())
            row["n_groups_sig_bonf"] = int(gr.sig_bonf.any().sum())
        if fam.startswith("figure9:slope"):
            # mean over models of the share of pairs classed as a significant negative slope (Figure 9 performance
            # profile, category dose_dep_neg; orchestration/extract_figure_stats.py rd_mean_dose_dep_neg_pct)
            neg = g.effect > 0
            for tag, sig in (("unadj", g.sig_unadj), ("bonf", g.sig_bonf)):
                row[f"dose_dep_neg_pct_{tag}"] = float(100 * (sig & neg).groupby(g.group).mean().mean())
        # marks that change filled -> open, by unit
        row["units_filled_to_open"] = "; ".join(sorted(valid.loc[valid.sig_unadj & ~valid.sig_bonf, "unit"].astype(str)))[:600]
        rows.append(row)
    return pd.DataFrame(rows)


# ── rankings on 0.6-1.7x only ───────────────────────────────────────────────

RANK_CONDS = [("vignette_naive", "Figure 1", "naive vignettes"), ("rd_naive", "Figure 8", "naive random doses"),
              ("rd_nielsen", "Figure 12", "random doses, reference-grounded"),
              ("eu_4h_nielsen_units", "Figure 15 (top)", "random doses mg/4 h, unit-augmented"),
              ("vignette_nielsen_units", "Figure 15 (bottom)", "vignettes, unit-augmented")]
PUBLISHED = {"vignette_naive": "ranking_naive_vignettes", "rd_naive": "ranking_naive_random_doses",
             "rd_nielsen": "ranking_random_doses_nielsen", "eu_4h_nielsen_units": "ranking_random_doses_nielsen_units",
             "vignette_nielsen_units": "ranking_vignettes_nielsen_units"}
_LONG = None


def _long():
    global _LONG
    if _LONG is None:
        _LONG = stats.load_long()
    return _LONG


def build_rankings() -> pd.DataFrame:
    rows = []
    for cond, fig, panel in RANK_CONDS:
        r = stats.rank_models(_long(), "nielsen", cond, "all", bands=("broad",))
        res = r.result_df.copy()
        pub = pd.read_csv(os.path.join(RES, PUBLISHED[cond] + ".csv")).set_index("model")
        res["group_broad"] = res["adjusted_rank"].astype(int)
        res["pct_broad"] = 100 * res["mean_broad_prop"]
        res = res.sort_values(["group_broad", "pct_broad"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
        n_in = res.groupby("group_broad")["model"].transform("size")
        pairs = r.pairs_df
        for i, x in res.iterrows():
            pr = pub.loc[x.model]
            rows.append({"figure": fig, "panel": panel, "condition": cond, "model": x.model, "row_order": i + 1,
                         "pct_broad": x.pct_broad, "group_broad": int(x.group_broad), "n_in_group": int(n_in[i]),
                         "n_groups": int(res.group_broad.max()), "n_responses": int(x.n_total), "n_items": int(x.n_units),
                         "published_group": int(pr.adjusted_rank), "published_order": int(pr.raw_rank),
                         "published_pct_tight": 100 * pr.mean_tight_prop, "published_pct_broad": 100 * pr.mean_broad_prop,
                         "group_changed": int(x.group_broad) != int(pr.adjusted_rank),
                         "n_pairs_tested": int(pairs.broad_p.notna().sum()),
                         "n_pairs_sig_bonf_broad_only": int((pairs.broad_p_bonf < ALPHA).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    t = build_tests()
    _write(t.sort_values(["family", "group", "unit"], kind="mergesort"), "multiple_comparisons_tests.csv")
    a = build_audit(t)
    _write(a, "multiple_comparisons_audit.csv")
    _write(build_rankings(), "rankings_fig12_fig15_broad.csv")
    pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 500)
    print(a[a.scope == "published"][["figure", "panel", "measure", "m", "n_sig_unadj", "n_sig_bonf"]].to_string(index=False))


if __name__ == "__main__":
    main()
