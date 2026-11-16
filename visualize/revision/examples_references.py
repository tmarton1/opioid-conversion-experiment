"""
Phase 4 example figures E5-E7 (reference standards and agreement).

  E5  Reference standards compared, by study agent (RS1; model-independent)
  E6  Agreement between models: within- vs between-model spread per conversion,
      with the spread between published standards overlaid (AG1)
  E7  Worked scoring examples: how the Nielsen, all-standards and
      all-except-Nielsen scores assign error to real naive random-dose
      responses for four conversions to oral morphine (RS5)

Inputs (never results/):
  results_revision/references/by_agent.csv, by_conversion.csv     (task 1.6)
  results_revision/model_agreement.csv                              (task 3D)
  results_revision/responses_long.csv via analysis.revision.stats   (task 1.3)
  resources/reference_standards_index.json (standard labels)

Outputs: $FIGURES_DIR/revision/examples/E5_*, E6_*, E7_* (PNG 300 dpi + PDF)
and the draft legends results_revision/figure_legends_E5-E7.md. Every number
in titles and legends is computed here. Deterministic; never queries a model.

    python -m visualize.revision.examples_references
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from visualize.revision import style  # noqa: E402  (backend, rcParams)
from visualize.revision.style import BAND_FILL, GRID, INK, MUTED, clean_axes, fold_axis, save, shade_bands  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from analysis.revision.tolerances import BANDS  # noqa: E402
from common import config  # noqa: E402

REF_DIR = os.path.join("results_revision", "references")
LEGENDS = os.path.join("results_revision", "figure_legends_E5-E7.md")
# Approval status of this legend (author decision, part 2): accepted for use without author
# review. Any change to the text needs author approval (scripts/revision/check_approvals.py).
ACCEPTED_NOTE = "Accepted from part 1 without author review."
OUT_DIR = os.path.join(config.FIGURES_DIR, "revision", "examples")

TIGHT = BANDS["tight"].bounds("x", "1", "y")
BROAD = BANDS["broad"].bounds("x", "1", "y")

# Categorical markers for the standards: a small palette distinct from the green
# band fills, with shape as a second encoding; CDC 2016 open and grey (superseded).
CURRENT = ["cdc_2022", "nih_heal", "mcpherson_2018"]
ALTERNATIVES = ["cdc_2016"] + CURRENT
MARKER = {
    "nielsen": dict(marker="o", mfc=INK, mec=INK, ms=6.5, mew=0.6),
    "cdc_2016": dict(marker="o", mfc="white", mec="#8a8984", ms=7.0, mew=1.4),
    "cdc_2022": dict(marker="s", mfc="#2a78d6", mec=INK, ms=6.8, mew=0.5),
    "nih_heal": dict(marker="^", mfc="#eb6834", mec=INK, ms=7.8, mew=0.5),
    "mcpherson_2018": dict(marker="D", mfc="#7b4fb3", mec=INK, ms=6.2, mew=0.5),
}
IQR_COLOR = "#9a9993"

# Score curves in E7 (scores, not models or standards: neutral ink + two hues
# not used by the standards, line style as the second encoding).
SCORE_STYLE = {
    "nielsen": dict(color=INK, ls="-", lw=2.0),
    "all": dict(color="#0f766e", ls="--", lw=2.0),
    "non_nielsen": dict(color="#b8327a", ls=":", lw=2.4),
}
SCORE_NAME = {"nielsen": "Nielsen", "all": "all standards", "non_nielsen": "all except Nielsen"}


def _labels() -> dict:
    with open(os.path.join("resources", "reference_standards_index.json")) as f:
        idx = json.load(f)
    lab = {k: v.get("label", k) for k, v in idx.items()}
    lab["nielsen"] = "Nielsen 2016"
    lab["_unverified"] = [k for k, v in idx.items() if not v.get("verified", False)]
    return lab


def _fold(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") + "×"


def _outside(v, band=TIGHT) -> bool:
    return bool(pd.notna(v) and (v < band[0] - 1e-9 or v > band[1] + 1e-9))


def nice_agent(k: str) -> str:
    route, drug = k.split("_", 1)
    route = {"oral": "oral", "iv": "IV", "sublingual": "sublingual", "transdermal": "patch",
             "sc": "SC", "im": "IM", "parenteral": "parenteral"}.get(route, route)
    return f"{drug.capitalize()} ({route})"


def nice_conv(c: str) -> str:
    if " -> " not in c:
        return c.replace("-", " ").replace("middle aged", "middle-aged").replace("young adult", "young-adult").capitalize()
    a, b = c.split(" -> ")

    def one(k):
        route, drug = k.split("_", 1)
        route = {"oral": "oral", "iv": "IV", "sublingual": "SL", "transdermal": "patch",
                 "sc": "SC", "im": "IM", "parenteral": "parenteral"}.get(route, route)
        return f"{route} {drug}"
    return f"{one(a)} → {one(b)}"


# ── E5 ──────────────────────────────────────────────────────────────────────

def e5_data():
    A = pd.read_csv(os.path.join(REF_DIR, "by_agent.csv"))
    P = pd.read_csv(os.path.join(REF_DIR, "by_conversion.csv"))
    study = set(P.from_opioid.str.split(",").explode()) | set(P.to_opioid)
    flagged = set(A.loc[A.study_agent.astype(bool), "opioid_route"])
    if study != flagged:
        print(f"note: by_agent study_agent flag differs from by_conversion agents: {sorted(study ^ flagged)}")
    a = A[A.opioid_route.isin(study) & (A.opioid_route != "oral_morphine")].copy()
    rel = a[[f"ratio_{s}_vs_nielsen" for s in ALTERNATIVES]]
    rel.columns = ALTERNATIVES
    a = a.join(rel)
    has_alt = rel.notna().any(axis=1)
    noalt = sorted(a.loc[~has_alt, "opioid_route"], key=nice_agent)
    a = a[has_alt].copy()
    absl = np.log(a[ALTERNATIVES]).abs()
    a["dev_current"] = absl[CURRENT].max(axis=1).fillna(0)
    a["dev_all"] = absl.max(axis=1)
    a = a.sort_values(["dev_current", "dev_all", "opioid_route"], ascending=[False, False, True]).reset_index(drop=True)
    out_all = a[ALTERNATIVES].apply(lambda r: any(_outside(v) for v in r), axis=1)
    out_cur = a[CURRENT].apply(lambda r: any(_outside(v) for v in r), axis=1)
    out_broad = a[ALTERNATIVES].apply(lambda r: any(_outside(v, BROAD) for v in r), axis=1)
    lone = []
    for _, r in a.iterrows():   # Nielsen is the lone outlier: >=2 current alternatives that all agree, and differ from Nielsen
        v = [r[s] for s in CURRENT if pd.notna(r[s])]
        if len(v) >= 2 and max(v) / min(v) < 1.001 and abs(np.log(v[0])) > 1e-6:
            lone.append(r.opioid_route)
    stats = dict(n=len(a), n_study=len(a) + len(noalt), n_out_all=int(out_all.sum()), n_out_cur=int(out_cur.sum()),
                 broad_agents=[a.opioid_route[i] for i in a.index[out_broad]], noalt=noalt, lone=lone)
    return a, stats


def build_e5(lab) -> tuple[list[str], dict]:
    a, st = e5_data()
    n = st["n"]
    fig, ax = plt.subplots(figsize=(8.4, 0.46 * n + 2.9))
    ax.set_ylim(n - 0.5, -0.5)
    fold_axis(ax, "x", lim=(0.45, 2.25), ticks=[0.5, 0.6, 0.85, 1, 1.15, 1.7, 2])
    shade_bands(ax, "x")
    for y in range(n - 1):
        ax.axhline(y + 0.5, color=GRID, lw=0.6, zorder=0)
    offs = dict(zip(ALTERNATIVES, np.linspace(-0.27, 0.27, 4)))
    for i, r in a.iterrows():
        if pd.notna(r.Nielsen_LQ) and pd.notna(r.Nielsen_UQ) and r.Nielsen_UQ > r.Nielsen_LQ:
            ax.plot([r.Nielsen_LQ / r.factor_nielsen, r.Nielsen_UQ / r.factor_nielsen], [i, i], color=IQR_COLOR,
                    lw=5, alpha=0.7, solid_capstyle="butt", zorder=2)
        for s in ALTERNATIVES:
            v = r[s]
            if pd.notna(v):
                ax.plot(v, i + offs[s], ls="none", zorder=4 if s != "cdc_2016" else 3, **MARKER[s])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{nice_agent(k)}  ·  Nielsen {v:g}" for k, v in zip(a.opioid_route, a.factor_nielsen)],
                       fontsize=8.6)
    clean_axes(ax)
    ax.set_xlabel("Conversion factor relative to the Nielsen median (log scale; right = more potent than Nielsen)",
                  fontsize=8.8)
    fig.suptitle(f"Reference standards: published factors differ from Nielsen by more than 15% "
                 f"for {st['n_out_all']} of {n} study agents",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.955,
             f"{st['n_out_cur']} of {n} excluding the superseded CDC 2016 factors. Oral morphine equivalent factor for each study "
             f"agent under each standard, divided by\nthe Nielsen median. Model-independent (no model responses). "
             f"Agents sorted by the largest difference under a current standard.",
             va="top", fontsize=8.2, color=MUTED)
    h = [Line2D([0], [0], color=INK, lw=0.9, alpha=0.6, label="Nielsen median (1×)"),
         Line2D([0], [0], color=IQR_COLOR, lw=5, alpha=0.7, label="Nielsen interquartile range")]
    h += [Line2D([0], [0], ls="none", label=lab[s], **MARKER[s]) for s in CURRENT + ["cdc_2016"]]
    h += [Patch(color=BAND_FILL["tight"], label=f"Tight band ({TIGHT[0]:g}–{TIGHT[1]:g}×)"),
          Patch(color=BAND_FILL["broad"], label=f"Broad band ({BROAD[0]:g}–{BROAD[1]:g}×)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=4, frameon=False, fontsize=8,
               handlelength=1.6, columnspacing=1.3)
    fig.text(0.01, 0.01, "No alternative standard for: " + ", ".join(nice_agent(k) for k in st["noalt"])
             + ". Only Nielsen gives an interquartile range.\nNot yet verified against the original publications: "
             + ", ".join(lab[k] for k in lab["_unverified"]) + ".", fontsize=7.4, color=MUTED)
    fig.subplots_adjust(left=0.3, right=0.975, top=0.835, bottom=0.12)
    return save(fig, os.path.join(OUT_DIR, "E5_reference_standards_by_agent")), st


# ── E6 ──────────────────────────────────────────────────────────────────────

def e6_data():
    m = pd.read_csv(os.path.join("results_revision", "model_agreement.csv"))
    m = m[m.row_type == "item"]
    c = pd.read_csv(os.path.join(REF_DIR, "by_conversion.csv"))
    d = m.merge(c[["experiment", "conversion", "standards_fold_spread_current", "conversion_type"]],
                left_on=["experiment", "item"], right_on=["experiment", "conversion"], how="left", validate="1:1")
    if d.conversion.isna().any():
        raise ValueError("model_agreement items missing from by_conversion.csv")
    d["within_fold"] = np.exp(d.within_sd_median)
    d["between_fold"] = np.exp(d.between_sd)
    d["std_fold"] = d.standards_fold_spread_current
    return d


def build_e6() -> tuple[list[str], dict]:
    d = e6_data()
    panels = [("random_dose", "Random-dose conversions (naive)", (0.97, 2.9), [1, 1.15, 1.5, 2, 2.5]),
              ("vignette", "Text vignettes (naive)", (0.97, 1.6), [1, 1.15, 1.3, 1.5])]
    ylim = (0.97, 9.0)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.6), sharey=True)
    PT, ST = "#2a5c8a", "#c2410c"
    st = {}
    for ax, (e, title, xlim, xt) in zip(axs, panels):
        g = d[d.experiment == e]
        fold_axis(ax, "x", lim=xlim, ticks=xt)
        fold_axis(ax, "y", lim=ylim, ticks=[1, 1.15, 1.5, 2, 3, 5, 8])
        ax.grid(color=GRID, lw=0.5, zorder=0)
        ax.plot([1, xlim[1]], [1, xlim[1]], color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1)
        xi = xlim[1] ** 0.93
        ax.text(xi, xi * 1.06, "identity", fontsize=7, color=MUTED, ha="right", va="bottom")
        gs = g[g.std_fold.notna()]
        for _, r in gs.iterrows():
            ax.plot([r.within_fold, r.within_fold], [r.std_fold, r.between_fold], color="#c9c8c3", lw=0.8, zorder=2)
        ax.scatter(gs.within_fold, gs.std_fold, marker="D", s=22, facecolor="white", edgecolor=ST, lw=1.1, zorder=3)
        ax.scatter(g.within_fold, g.between_fold, s=30, color=PT, edgecolor="white", lw=0.6, zorder=4)
        lx = lambda v: np.log(v / xlim[0]) / np.log(xlim[1] / xlim[0])   # data x -> axes fraction
        for j, (_, r) in enumerate(g.nlargest(2, "between_fold").iterrows()):
            # labels in the empty top band, beside their own point, with leader lines that cannot cross
            fx = lx(r.within_fold)
            left_half = fx < 0.5
            ax.annotate(nice_conv(r["item"]), (r.within_fold, r.between_fold),
                        xytext=(fx + (0.03 if left_half else -0.03), 0.95 - 0.07 * j), textcoords="axes fraction",
                        ha="left" if left_half else "right", va="center", fontsize=6.8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=4,
                                        relpos=(0, 0.5) if left_half else (1, 0.5)))
        n_bw = int((g.between_sd > g.within_sd_median).sum())
        n_bs = int((gs.between_fold > gs.std_fold).sum())
        st[e] = dict(n=len(g), n_bw=n_bw, n_std=len(gs), n_bs=n_bs,
                     med_within=float(g.within_fold.median()), med_between=float(g.between_fold.median()),
                     med_std=float(gs.std_fold.median()))
        ax.set_title(title, loc="left", pad=30)
        ax.text(0, 1.015,
                f"Between > within models: {n_bw} of {len(g)};  between models > between standards: {n_bs} of {len(gs)}\n"
                f"Medians: within {_fold(st[e]['med_within'])}, between models {_fold(st[e]['med_between'])}, "
                f"between standards {_fold(st[e]['med_std'])}",
                transform=ax.transAxes, va="bottom", fontsize=7.2, color=MUTED)
        ax.set_xlabel("Within-model spread: median over models of the SD of log ratio,\nshown as a fold factor exp(SD)",
                      fontsize=8)
        clean_axes(ax, keep=("bottom", "left"))
    axs[0].set_ylabel("Spread between models (SD of model medians, as exp(SD))\nor between standards (max/min fold)",
                      fontsize=8)
    tot_bw = sum(s["n_bw"] for s in st.values())
    tot = sum(s["n"] for s in st.values())
    tot_bs = sum(s["n_bs"] for s in st.values())
    tot_std = sum(s["n_std"] for s in st.values())
    fig.suptitle(f"Agreement between models (Nielsen score): models disagree with each other more than with themselves "
                 f"on {tot_bw} of {tot}\nconversions, and more than the published standards disagree on {tot_bs} of "
                 f"{tot_std}", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.995)
    h = [Line2D([0], [0], ls="none", marker="o", ms=6, color=PT, mec="white", label="Between models (one point per conversion)"),
         Line2D([0], [0], ls="none", marker="D", ms=5, mfc="white", mec=ST, mew=1.1,
                label="Between current standards, same conversion"),
         Line2D([0], [0], color=MUTED, ls=(0, (4, 3)), lw=1, label="Identity (between = within)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=3, frameon=False, fontsize=7.8)
    fig.text(0.01, 0.0,
             "Spreads are of the natural-log ratio observed/expected, plotted as fold factors (log axes; panels have different "
             "x ranges). Current standards: Nielsen, CDC 2022, NIH HEAL, McPherson 2018.\nThe between-standard "
             "spread is a range (max/min), never smaller than the corresponding SD, so comparing it with the between-model SD "
             "favors the standards. Vignette within-model\nspread pools a scenario's four wordings; repeated queries of one "
             "wording vary much less. Conversions with no alternative standard show no diamond. Grey lines join the two "
             "spreads\nof one conversion.",
             fontsize=7, color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.76, bottom=0.2, wspace=0.08)
    return save(fig, os.path.join(OUT_DIR, "E6_model_agreement_spread")), st


# ── E7 ──────────────────────────────────────────────────────────────────────

EXAMPLES = ["oral_hydrocodone", "iv_hydromorphone", "oral_oxycodone", "oral_codeine"]
TO = "oral_morphine"


def _err(x, S):
    """Fold error a nearest-standard score assigns to answer x (relative to Nielsen) given standards S."""
    return np.exp(np.min(np.abs(np.log(np.asarray(x)[:, None] / np.asarray(S)[None, :])), axis=1))


def build_e7(lab) -> tuple[list[str], dict]:
    from analysis.revision import stats
    P = pd.read_csv(os.path.join(REF_DIR, "by_conversion.csv"))
    long = stats.load_long()
    scored = {s: stats.responses(long, s, "rd_naive", bands=["tight"]) for s in SCORE_STYLE}
    x = np.exp(np.linspace(np.log(0.25), np.log(4), 900))
    lo_x, hi_x = 0.25, 4.0
    fig = plt.figure(figsize=(11, 11.6))
    outer = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.24, left=0.07, right=0.985, top=0.83, bottom=0.075)
    st = {}
    for k, f in enumerate(EXAMPLES):
        conv = f"{f} -> {TO}"
        p = P[(P.experiment == "random_dose") & (P.conversion == conv)].iloc[0]
        rel = {s: float(p[f"{s}_vs_nielsen"]) for s in CURRENT if pd.notna(p[f"{s}_vs_nielsen"])}
        sets = {"nielsen": [1.0], "all": sorted(set(rel.values()) | {1.0}), "non_nielsen": sorted(set(rel.values()))}
        inner = outer[k // 2, k % 2].subgridspec(3, 1, height_ratios=[3, 0.8, 1.1], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        sx = fig.add_subplot(inner[1], sharex=ax)
        hx = fig.add_subplot(inner[2], sharex=ax)
        # curves: error assigned to every possible answer, bands as horizontal shading
        fold_axis(ax, "y", lim=(1, 4.2), ticks=[1, 1.15, 1.7, 2, 3, 4])
        ax.set_xscale("log")
        shade_bands(ax, "y")
        for s, S in sets.items():
            ax.plot(x, _err(x, S), zorder=3, label=SCORE_NAME[s], **SCORE_STYLE[s])
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        clean_axes(ax, keep=("left",))
        plt.setp(ax.get_xticklabels(), visible=False)
        if k % 2 == 0:
            ax.set_ylabel("Fold error assigned", fontsize=8.5)
        title = nice_conv(conv)
        title = title[0].upper() + title[1:]
        ax.set_title(title, loc="left", fontsize=10, pad=18)
        groups = {}
        for s, v in rel.items():
            groups.setdefault(round(v, 3), []).append(lab[s])
        sub = "; ".join(f"{_fold(v)} Nielsen ({', '.join(n)})" for v, n in sorted(groups.items()))
        ax.text(0, 1.03, "Other standards: " + sub, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
        # standards strip
        sx.set_xscale("log")
        sx.set_ylim(-0.7, 3.7)
        sx.set_yticks([])
        clean_axes(sx, keep=())
        plt.setp(sx.get_xticklabels(), visible=False)
        sx.plot(1.0, 0, ls="none", clip_on=False, **MARKER["nielsen"])
        for j, s in enumerate(CURRENT):
            if s in rel:
                sx.plot(rel[s], j + 1, ls="none", clip_on=False, **MARKER[s])
        for v in set(rel.values()) | {1.0}:
            for a_ in (ax, sx, hx):
                a_.axvline(v, color=INK, lw=0.6, alpha=0.18, zorder=0)
        sx.text(0.0, 0.5, "standards", transform=sx.transAxes, fontsize=7, color=MUTED, ha="left", va="center")
        # real naive responses (relative to the Nielsen expected dose)
        r = scored["nielsen"]
        r = r[(r.from_opioid == f) & (r.to_opioid == TO)]
        vals = r.ratio[r.ratio > 0].values
        n_lo, n_hi = int((vals < lo_x).sum()), int((vals > hi_x).sum())
        bins = np.exp(np.linspace(np.log(lo_x), np.log(hi_x), 49))
        hx.hist(np.clip(vals, lo_x * 1.001, hi_x * 0.999), bins=bins, color="#b9b8b2", edgecolor="white", lw=0.5)
        fold_axis(hx, "x", lim=(lo_x, hi_x), ticks=[0.25, 0.5, 1, 2, 4])
        hx.set_yticks([])
        clean_axes(hx)
        within = {}
        for s in SCORE_STYLE:
            q = scored[s]
            q = q[(q.from_opioid == f) & (q.to_opioid == TO)]
            within[s] = (float(q["in_tight"].mean()), len(q))
        n = len(r)
        st[conv] = dict(n=n, n_unusable=int(n - len(vals)), n_lo=n_lo, n_hi=n_hi,
                        **{f"tight_{s}": v[0] for s, v in within.items()}, rel=rel)
        edge = f"; {n_lo} below ¼× and {n_hi} above 4× piled at the edges" if (n_lo or n_hi) else ""
        hx.text(0.0, -0.42, f"Naive responses, 13 models (n = {n}{edge})\nWithin tight band (error ≤ {TIGHT[1]:g}×):  "
                + "   ".join(f"{SCORE_NAME[s].capitalize() if s == 'nielsen' else SCORE_NAME[s]} {within[s][0]:.0%}"
                             for s in SCORE_STYLE),
                transform=hx.transAxes, fontsize=7.8, color=INK, va="top")
    fig.suptitle("Worked scoring examples: error assigned by the Nielsen, all-standards and all-except-Nielsen scores",
                 x=0.01, ha="left", y=0.995, fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.962,
             "Under a nearest-standard score, an answer that matches any standard in the set scores 1× (no error); otherwise "
             "its error is the fold distance to the nearest\nstandard in the set. Curves: error each score assigns to every "
             "possible answer (where curves coincide they overlay). Histograms: the actual naive random-dose responses.\nScoring sets: Nielsen; "
             "all standards = Nielsen, CDC 2022, NIH HEAL, McPherson 2018; all except Nielsen = the last three "
             "(CDC 2016 is never used for scoring).",
             fontsize=8.2, color=MUTED, va="top")
    h = [Line2D([0], [0], label=f"Score: {SCORE_NAME[s]}", **SCORE_STYLE[s]) for s in SCORE_STYLE]
    h += [Line2D([0], [0], ls="none", label=lab[s], **MARKER[s]) for s in ["nielsen"] + CURRENT]
    h += [Patch(color=BAND_FILL["tight"], label=f"Tight band (≤ {TIGHT[1]:g}× error)"),
          Patch(color=BAND_FILL["broad"], label=f"Broad band (≤ {BROAD[1]:g}× error)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=5, frameon=False, fontsize=8,
               columnspacing=1.4)
    fig.text(0.53, 0.012, "Model response relative to the Nielsen expected dose (log scale; right = higher dose)",
             ha="center", fontsize=9, color=INK)
    return save(fig, os.path.join(OUT_DIR, "E7_scoring_examples")), st


# ── legends ─────────────────────────────────────────────────────────────────

def write_legends(s5, s6, s7, paths):
    rd, vg = s6["random_dose"], s6["vignette"]
    lines = [ACCEPTED_NOTE, "", "# Draft figure legends: example figures E5-E7", "",
             "Generated by `python -m visualize.revision.examples_references`; every number is computed from the CSVs.", ""]
    lines += [f"## E5. Reference standards compared, by agent (`{os.path.basename(paths['E5'])}`)", "",
              f"Oral morphine equivalent factor of each study agent under each published standard, relative to the "
              f"Nielsen median (log scale), for the {s5['n']} of {s5['n_study']} study agents other than oral morphine that "
              f"at least one alternative standard covers. Grey bars: Nielsen interquartile range; shading: tight "
              f"({TIGHT[0]:g}–{TIGHT[1]:g}×) and broad ({BROAD[0]:g}–{BROAD[1]:g}×) bands. {s5['n_out_all']} of "
              f"{s5['n']} agents have at least one standard outside the tight band ({s5['n_out_cur']} excluding the "
              f"superseded CDC 2016 factors, open circles); outside the broad band: "
              f"{', '.join(nice_agent(a) for a in s5['broad_agents']) or 'none'} (CDC 2016 only). Nielsen is the "
              f"lone outlier (all current alternatives agree with each other and differ from Nielsen) for "
              f"{', '.join(nice_agent(a) for a in s5['lone'])}. No alternative standard: "
              f"{', '.join(nice_agent(a) for a in s5['noalt'])}. Model-independent.", ""]
    lines += [f"## E6. Agreement between models (`{os.path.basename(paths['E6'])}`)", "",
              f"One point per conversion (naive condition, Nielsen score, 13 models). x: within-model spread (median "
              f"over models of the SD of the log ratio); y: between-model spread (SD of the 13 model medians), both "
              f"plotted as exp(SD) on log axes; dashed line: identity. Open diamonds: the spread between the current "
              f"standards for the same conversion (max/min fold), joined to the model point by a grey line. "
              f"Random-dose pairs: between > within for {rd['n_bw']} of {rd['n']}; between-model spread exceeds the "
              f"between-standard range for {rd['n_bs']} of {rd['n_std']} pairs with an alternative standard (medians: "
              f"within {_fold(rd['med_within'])}, between models {_fold(rd['med_between'])}, between standards "
              f"{_fold(rd['med_std'])}). Text vignettes: {vg['n_bw']} of {vg['n']} and {vg['n_bs']} of {vg['n_std']} "
              f"(medians {_fold(vg['med_within'])}, {_fold(vg['med_between'])}, {_fold(vg['med_std'])}). The "
              f"vignette within-model spread pools each scenario's four wordings. Comparing a range (standards) with "
              f"an SD (models) favors the standards.", ""]
    ex = []
    for conv, s in s7.items():
        ex.append(f"{nice_conv(conv)} (n = {s['n']}): Nielsen {s['tight_nielsen']:.0%}, all standards "
                  f"{s['tight_all']:.0%}, all except Nielsen {s['tight_non_nielsen']:.0%}")
    lines += [f"## E7. Worked scoring examples (`{os.path.basename(paths['E7'])}`)", "",
              "For four conversions to oral morphine, the fold error each score assigns to every possible answer "
              "(curves; x = answer relative to the Nielsen expected dose, log scale) and the actual naive random-dose "
              "responses of the 13 models (histograms). Markers: the expected dose under each current standard; "
              "shading: tight and broad bands on the assigned error. Under the all-standards and all-except-Nielsen "
              "scores, an answer matching any standard in the set scores 1×; otherwise its error is the fold distance "
              "to the nearest. Responses within the tight band: " + "; ".join(ex) + ". Unusable answers count as "
              "outside every band.", ""]
    os.makedirs(os.path.dirname(LEGENDS), exist_ok=True)
    tmp = LEGENDS + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines))
    os.replace(tmp, LEGENDS)


def main():
    os.chdir(ROOT)
    lab = _labels()
    p5, s5 = build_e5(lab)
    p6, s6 = build_e6()
    p7, s7 = build_e7(lab)
    write_legends(s5, s6, s7, dict(E5=p5[0], E6=p6[0], E7=p7[0]))
    for p in p5 + p6 + p7 + [LEGENDS]:
        print("✓", p)
    print("E5:", {k: v for k, v in s5.items()})
    print("E6:", s6)
    print("E7:", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in s7.items()})


if __name__ == "__main__":
    main()
