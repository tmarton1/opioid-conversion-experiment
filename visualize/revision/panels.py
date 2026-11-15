"""
Panel kinds for the revision figures. Each panel is drawn for one score from
the long table (analysis/revision/stats.py), so the same code draws a single
figure or one column of a multi-score comparison grid.

Kinds
  accuracy   one condition; bands view: per-model bars, one per band;
             distribution view: per-model violin + responses on a log fold axis,
             annotated with median fold error, % > 2x and % < 0.5x
  sequence   two or more conditions on the same models (e.g. naive -> Nielsen);
             bands view: per-model connected dots per condition (one sub-panel
             per band); distribution view: per-model median, IQR and 10-90%
             range per condition, annotated with median fold error per condition
  precision  one condition; precision (CV of repeated answers) is always
             computed from the Nielsen responses (observed doses only), accuracy
             from the plotted score; bands view: per-model shares of correct /
             precise-but-wrong / imprecise cells; distribution view: CV vs the
             cell's mean ratio under the score
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from analysis.revision import stats
from analysis.revision.conditions import CONDITIONS, SUBSETS
from analysis.revision.tolerances import BANDS, within
from visualize.revision import style as S

XLIM = (0.2, 5.0)
SEQ_MARKERS = ["o", "s", "^", "D", "v"]
SEQ_ALPHA = {2: [0.35, 1.0], 3: [0.3, 0.6, 1.0], 4: [0.25, 0.5, 0.75, 1.0], 5: [0.2, .4, .6, .8, 1.0]}
MAX_POINTS = 250            # responses drawn per model in a violin (random subsample, seed 0)
CV_PRECISE = {"vignette": 1e-9, "random_dose": 0.2, "explain_units": 0.2}   # as analysis/performance_profile.py


@dataclass(frozen=True)
class Panel:
    kind: str                         # accuracy | sequence | precision
    conditions: tuple                 # condition names (one, or several for sequence)
    subset: str = "all"
    title: str = ""
    subsets: Optional[tuple] = None   # per-condition subsets for sequence (default: subset)

    def cond_subsets(self):
        subs = self.subsets or tuple(self.subset for _ in self.conditions)
        return list(zip(self.conditions, subs))


def n_rows(panel: Panel, view: str, bands) -> int:
    """Sub-rows a panel needs in a grid (one per band for sequence/bands)."""
    return len(bands) if (panel.kind == "sequence" and view == "bands") else 1


# ── helpers ──────────────────────────────────────────────────────────────

def _rows(ax, models, show_labels):
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([S.short(m) for m in models] if show_labels else [], fontsize=7.5)
    for y in range(len(models) - 1):
        ax.axhline(y + 0.5, color=S.GRID, lw=0.5, zorder=0)


def _coverage_note(ax, n_resp, n_items, item_word, above=False):
    """Responses and items scored, under the axis (or above it when the panel
    has an x-axis label)."""
    ax.text(1.0, 1.01 if above else -0.06, f"n = {n_resp:,} responses, {n_items} {item_word}",
            transform=ax.transAxes, fontsize=6.3, color=S.MUTED, ha="right",
            va="bottom" if above else "top")


def _item_word(cond):
    return "scenarios" if CONDITIONS[cond].item == "scenario_key" else "pairs"


def _empty(ax, text="no alternative standard covers these conversions"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color=S.MUTED)
    ax.set_xticks([]); ax.set_yticks([])


# ── accuracy ─────────────────────────────────────────────────────────────

def accuracy_bands(ax, long, score, panel, bands, models, show_labels=True):
    cond = panel.conditions[0]
    summ = stats.model_summary(long, score, cond, panel.subset, bands)
    if summ.empty:
        return _empty(ax)
    summ = summ.set_index("model")
    h = 0.8 / len(bands)
    alphas = np.linspace(1.0, 0.3, len(bands))
    for i, m in enumerate(models):
        if m not in summ.index:
            continue
        for j, b in enumerate(bands):
            v = summ.loc[m, f"pct_{b}"]
            y = i - 0.4 + h * (j + 0.5)
            ax.barh(y, v, height=h * 0.9, color=S.color(m), alpha=alphas[j], lw=0,
                    hatch="////" if BANDS[b].per_conversion else None, edgecolor="white")
    _rows(ax, models, show_labels)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=S.GRID, lw=0.5); ax.set_axisbelow(True)
    S.clean_axes(ax)
    _coverage_note(ax, int(summ.n_responses.sum()), int(summ.n_items.max()), _item_word(cond))
    return [Patch(facecolor="#555555", alpha=a, hatch="////" if BANDS[b].per_conversion else None,
                  edgecolor="white", label=f"% within {BANDS[b].label}") for b, a in zip(bands, alphas)]


def _violin(ax, y, logs, c, width=0.42):
    logs = logs[np.isfinite(logs)]
    if len(logs) < 2:
        return
    lo, hi = np.log(XLIM)
    clipped = np.clip(logs, lo, hi)
    if np.ptp(clipped) < 1e-9:
        ax.plot([np.exp(clipped[0])] * 2, [y - width, y + width], color=c, lw=2)
        return
    grid = np.linspace(lo, hi, 300)
    try:
        dens = gaussian_kde(clipped, bw_method=0.15)(grid)
    except np.linalg.LinAlgError:
        return
    dens = dens / dens.max() * width
    ax.fill_between(np.exp(grid), y - dens, y + dens, color=c, alpha=0.28, lw=0, zorder=2)


def accuracy_distribution(ax, long, score, panel, bands, models, show_labels=True, annotate=True):
    cond = panel.conditions[0]
    d = stats.responses(long, score, cond, panel.subset)
    if d.empty:
        return _empty(ax)
    summ = stats.model_summary(long, score, cond, panel.subset, []).set_index("model")
    S.shade_bands(ax, "x", bands)
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        g = d[d.model == m]
        if g.empty:
            continue
        logs = g["log_ratio"].to_numpy()
        _violin(ax, i, logs, S.color(m))
        pts = logs[np.isfinite(logs)]
        if len(pts) > MAX_POINTS:
            pts = rng.choice(pts, MAX_POINTS, replace=False)
        jit = rng.uniform(-0.28, 0.28, len(pts))
        ax.scatter(np.exp(np.clip(pts, *np.log(XLIM))), i + jit, s=2.2, color=S.color(m),
                   alpha=0.45, lw=0, zorder=3)
        med = np.nanmedian(logs)
        if np.isfinite(med):
            ax.plot([np.exp(med)] * 2, [i - 0.36, i + 0.36], color=S.INK, lw=1.3, zorder=4)
        if annotate:
            r = summ.loc[m]
            ax.text(1.02, i, f"{r.median_fold_error:.2f}×  {r.pct_over_2x:.0f}%  {r.pct_under_half:.0f}%",
                    transform=ax.get_yaxis_transform(), fontsize=6.4, color=S.INK, va="center")
    if annotate:
        ax.text(1.02, -0.75, "MFE   >2×  <½×", transform=ax.get_yaxis_transform(),
                fontsize=6.2, color=S.MUTED, va="center")
    _rows(ax, models, show_labels)
    S.fold_axis(ax, "x", XLIM)
    S.clean_axes(ax)
    _coverage_note(ax, int(summ.n_responses.sum()), int(summ.n_items.max()), _item_word(cond))
    return [Line2D([0], [0], color=S.INK, lw=1.3, label="median"),
            Patch(color=S.BAND_FILL["tight"], label="tight band"),
            Patch(color=S.BAND_FILL["broad"], label="broad band")]


# ── sequence ─────────────────────────────────────────────────────────────

def _seq_labels(panel):
    """Condition names without the experiment prefix ("naive", "Nielsen"),
    unless the sequence spans experiments (then the full label)."""
    exps = {CONDITIONS[c].experiment for c in panel.conditions}
    labs = [CONDITIONS[c].label for c in panel.conditions]
    return [l.split(", ", 1)[-1] for l in labs] if len(exps) == 1 else labs


def sequence_bands(ax, long, score, panel, band, models, show_labels=True):
    cs = panel.cond_subsets()
    # sequence figures (11, 13, 14) report batch means: average over cells
    summs = [stats.model_summary(long, score, c, s, [band], average="cell") for c, s in cs]
    if any(x.empty for x in summs):
        return _empty(ax)
    summs = [x.set_index("model") for x in summs]
    al = SEQ_ALPHA[len(cs)]
    for i, m in enumerate(models):
        xs = [x.loc[m, f"pct_{band}"] if m in x.index else np.nan for x in summs]
        ax.plot(xs, [i] * len(xs), color=S.color(m), lw=1.2, alpha=0.7, zorder=2)
        for k, x in enumerate(xs):
            ax.scatter([x], [i], s=26, marker=SEQ_MARKERS[k], color=S.color(m), alpha=al[k],
                       edgecolor=S.INK, linewidths=0.4, zorder=3)
    _rows(ax, models, show_labels)
    ax.set_xlim(-2, 102); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=S.GRID, lw=0.5); ax.set_axisbelow(True)
    S.clean_axes(ax)
    n = sum(int(x.n_responses.sum()) for x in summs)
    _coverage_note(ax, n, int(max(x.n_items.max() for x in summs)), _item_word(cs[0][0]) + ", all conditions")
    return [Line2D([0], [0], ls="none", marker=SEQ_MARKERS[k], color="#555555", alpha=al[k],
                   markeredgecolor=S.INK, markeredgewidth=0.4, label=lab)
            for k, lab in enumerate(_seq_labels(panel))]


def sequence_distribution(ax, long, score, panel, bands, models, show_labels=True, annotate=True):
    cs = panel.cond_subsets()
    ds = [stats.responses(long, score, c, s) for c, s in cs]
    if any(x.empty for x in ds):
        return _empty(ax)
    S.shade_bands(ax, "x", bands)
    k = len(cs)
    offs = np.linspace(-0.3, 0.3, k)
    al = SEQ_ALPHA[k]
    for i, m in enumerate(models):
        mfe = []
        for j, d in enumerate(ds):
            lr = d.loc[d.model == m, "log_ratio"].dropna().to_numpy()
            if not len(lr):
                mfe.append(np.nan); continue
            q10, q25, q50, q75, q90 = np.exp(np.clip(np.percentile(lr, [10, 25, 50, 75, 90]), *np.log(XLIM)))
            y = i + offs[j]
            ax.plot([q10, q90], [y, y], color=S.color(m), lw=0.8, alpha=max(al[j], .5), zorder=2)
            ax.plot([q25, q75], [y, y], color=S.color(m), lw=3.2, alpha=al[j], solid_capstyle="butt", zorder=3)
            ax.scatter([q50], [y], s=16, marker=SEQ_MARKERS[j], color="white", edgecolor=S.INK,
                       linewidths=0.6, zorder=4)
            mfe.append(float(np.exp(np.median(np.abs(lr)))))
        if annotate:
            ax.text(1.02, i, " → ".join("–" if not np.isfinite(v) else f"{v:.2f}×" for v in mfe),
                    transform=ax.get_yaxis_transform(), fontsize=6.4, color=S.INK, va="center")
    if annotate:
        ax.text(1.02, -0.75, "median fold error", transform=ax.get_yaxis_transform(),
                fontsize=6.2, color=S.MUTED, va="center")
    _rows(ax, models, show_labels)
    S.fold_axis(ax, "x", XLIM)
    S.clean_axes(ax)
    n = sum(len(d) for d in ds)
    _coverage_note(ax, n, int(max(d[CONDITIONS[c].item].nunique() for d, (c, _) in zip(ds, cs))),
                   _item_word(cs[0][0]) + ", all conditions")
    h = [Line2D([0], [0], ls="none", marker=SEQ_MARKERS[j], markerfacecolor="white",
                markeredgecolor=S.INK, label=f"{lab} (median)") for j, lab in enumerate(_seq_labels(panel))]
    h += [Line2D([0], [0], color="#555555", lw=3.2, label="IQR"),
          Line2D([0], [0], color="#555555", lw=0.8, label="10–90%")]
    return h


# ── precision ────────────────────────────────────────────────────────────

def precision_cells(long, score, cond, subset="all", band="tight") -> pd.DataFrame:
    """Model x cell: CV of the observed answers (from the Nielsen responses,
    score-independent) and the mean ratio under the score, with a flag for the
    mean ratio inside ``band`` and the precision category."""
    c = CONDITIONS[cond]
    keys = ["model", c.item] + ([c.cell] if c.cell != c.item else [])
    nz = stats.responses(long, "nielsen", cond, subset)
    if c.experiment == "vignette":
        nz = nz.assign(_v=pd.to_numeric(nz["observed_to_dose"], errors="coerce"))
    else:
        nz = nz.assign(_v=pd.to_numeric(nz["observed_to_dose"], errors="coerce")
                       / pd.to_numeric(nz["from_dose"], errors="coerce"))
    # CV over positive answers (sample SD / mean), as the pipeline's observed_*_cv
    def _cv(v):
        v = v[v > 0]
        return v.std() / v.mean() if len(v) and v.mean() else np.nan
    cv = nz.groupby(keys)["_v"].agg(_cv).rename("cv")
    d = stats.responses(long, score, cond, subset)
    agg = d.groupby(keys).agg(mean_ratio=("ratio", "mean"), from_opioid=("from_opioid", "first"),
                              conv_dose=("conv_dose", "first"), to_opioid=("to_opioid", "first"))
    out = agg.join(cv, how="left").reset_index()
    frame = pd.DataFrame({"from_opioid": out.from_opioid, "from_dose": out.conv_dose, "to_opioid": out.to_opioid})
    out["correct"] = within(band, out["mean_ratio"], frame) == 1
    precise = out["cv"] <= CV_PRECISE[c.experiment]
    out["category"] = np.where(out["correct"], "correct", np.where(precise, "precise_wrong", "imprecise"))
    return out


PREC_CATS = [("correct", 1.0, None), ("precise_wrong", 0.6, "////"), ("imprecise", 0.22, None)]


def precision_bands(ax, long, score, panel, bands, models, show_labels=True):
    cond = panel.conditions[0]
    band = bands[0]
    cells = precision_cells(long, score, cond, panel.subset, band)
    if cells.empty:
        return _empty(ax)
    share = cells.groupby("model")["category"].value_counts(normalize=True).unstack(fill_value=0) * 100
    for i, m in enumerate(models):
        left = 0
        for cat, a, hatch in PREC_CATS:
            v = share.loc[m, cat] if (m in share.index and cat in share.columns) else 0
            ax.barh(i, v, left=left, height=0.7, color=S.color(m), alpha=a, hatch=hatch,
                    edgecolor="white", lw=0.6)
            left += v
    _rows(ax, models, show_labels)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    S.clean_axes(ax)
    thr = CV_PRECISE[CONDITIONS[cond].experiment]
    ptxt = "CV = 0" if thr < 1e-6 else f"CV ≤ {thr:g}"
    _coverage_note(ax, int(len(stats.responses(long, score, cond, panel.subset))),
                   int(cells[CONDITIONS[cond].item].nunique()), _item_word(cond))
    return [Patch(facecolor="#555555", alpha=1.0, label=f"correct (mean within {band})"),
            Patch(facecolor="#555555", alpha=0.6, hatch="////", edgecolor="white", label=f"precise ({ptxt}) but wrong"),
            Patch(facecolor="#555555", alpha=0.22, label="imprecise and wrong")]


def precision_distribution(ax, long, score, panel, bands, models, show_labels=True):
    cond = panel.conditions[0]
    cells = precision_cells(long, score, cond, panel.subset, bands[0])
    if cells.empty:
        return _empty(ax)
    S.shade_bands(ax, "y", bands)
    rng = np.random.default_rng(0)
    for m in models:
        g = cells[cells.model == m]
        x = g["cv"].to_numpy(dtype=float)
        x = np.where(x <= 1e-9, 10 ** rng.uniform(-3.3, -2.9, len(x)), x)   # CV = 0 drawn in a strip at left
        y = np.clip(g["mean_ratio"].to_numpy(dtype=float), 0.21, 4.9)
        ax.scatter(x, y, s=7, color=S.color(m), alpha=0.6, lw=0, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(4e-4, 3)
    ax.set_xticks([7e-4, 0.01, 0.1, 1]); ax.set_xticklabels(["CV = 0", "0.01", "0.1", "1"])
    ax.axvline(1.5e-3, color=S.GRID, lw=0.8)
    S.fold_axis(ax, "y", XLIM)
    ax.set_xlabel("CV of repeated answers (Nielsen responses)", fontsize=7.5)
    ax.set_ylabel("Mean ratio to reference", fontsize=7.5) if show_labels else None
    S.clean_axes(ax, keep=("bottom", "left"))
    _coverage_note(ax, int(len(stats.responses(long, score, cond, panel.subset))),
                   int(cells[CONDITIONS[cond].item].nunique()), _item_word(cond), above=True)
    return [Line2D([0], [0], ls="none", marker="o", color=S.color(m), label=S.short(m)) for m in models]


DRAW = {
    ("accuracy", "bands"): accuracy_bands,
    ("accuracy", "distribution"): accuracy_distribution,
    ("sequence", "distribution"): sequence_distribution,
    ("precision", "bands"): precision_bands,
    ("precision", "distribution"): precision_distribution,
}
