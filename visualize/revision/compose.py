"""
Build one figure for a selection of scores, a view and a list of bands.

Two engines:
  legacy  the original Bokeh builder (visualize/comprehensive_figures.py) run on
          results_<score>/ (one score, bands view, default bands). Titles get the
          score's suffix and, for rescored scores, a coverage note; figures 2, 3
          and 9 get Nielsen precision through an adapter (see ``adapt_inputs``).
          With score nielsen this is exactly the published code path.
  grid    matplotlib panels (visualize/revision/panels.py) from the long table:
          the distribution view, multi-score comparisons (columns = scores,
          shared axes and model order) and any registered bands.

Output paths encode figure, scores, view and (if not the default) bands:
  <out>/figNN_<slug>/figNN_<slug>__<score[+score...]>__<view>[__<band+band>].{png,pdf[,html]}
"""
from __future__ import annotations

import asyncio
import os
import shutil

import numpy as np
import pandas as pd

from visualize.revision import style as S
from analysis.revision import stats
from analysis.revision.scores import SCORES
from analysis.revision.tolerances import BANDS, DEFAULT_BANDS
from visualize.revision.panels import DRAW, Panel, band_axis_label, n_rows, sequence_bands
from visualize.revision.registry import FIGURES, FigureSpec

ADAPTER_ROOT = os.path.join("results_revision", "figure_inputs")


def out_stem(out_dir: str, spec: FigureSpec, scores, view: str, bands) -> str:
    name = f"{spec.stem}__{'+'.join(scores)}__{view}"
    if list(bands) != list(DEFAULT_BANDS):
        name += "__" + "+".join(bands)
    return os.path.join(out_dir, spec.stem, name)


def engine_for(spec: FigureSpec, scores, view, bands, compare) -> str:
    if view == "bands" and len(scores) == 1 and not compare and list(bands) == list(DEFAULT_BANDS):
        return "legacy"
    if not spec.panels:
        raise ValueError(f"figure {spec.num} is score-independent: only nielsen, bands view")
    return "grid"


# ── grid engine ──────────────────────────────────────────────────────────

def build_grid(spec: FigureSpec, scores, view, bands, long=None, out_dir=None) -> list[str]:
    import matplotlib.pyplot as plt
    long = stats.load_long() if long is None else long
    models = S.model_order()
    rows = []                                     # (panel, band or None)
    for p in spec.panels:
        if p.kind == "sequence" and view == "bands":
            rows += [(p, b) for b in bands]
        else:
            rows.append((p, None))
    ncol = len(scores)
    annot = view == "distribution"
    col_w = 3.3 + (1.15 if annot else 0)
    row_h = 0.25 * len(models) + 0.9
    fig, axes = plt.subplots(len(rows), ncol, figsize=(col_w * ncol + 1.2, row_h * len(rows) + 1.5),
                             squeeze=False, sharex="row")
    handles = {}
    for r, (p, band) in enumerate(rows):
        for c, sc in enumerate(scores):
            ax = axes[r][c]
            first = c == 0
            if p.kind == "sequence" and view == "bands":
                h = sequence_bands(ax, long, sc, p, band, models, show_labels=first)
            elif p.kind == "precision" and view == "distribution":
                h = DRAW[(p.kind, view)](ax, long, sc, p, bands, models, show_labels=first)
            elif view == "distribution":
                h = DRAW[(p.kind, view)](ax, long, sc, p, bands, models, show_labels=first, annotate=True)
            else:
                h = DRAW[(p.kind, view)](ax, long, sc, p, bands, models, show_labels=first)
            if h and (p.kind, band) not in handles:
                handles[(p.kind, band)] = h
            if r == 0:
                ax.set_title(S.score_title(sc), loc="left", pad=14)
            if first:
                ax.text(-0.02 if p.kind != "precision" else 0.0, 1.0 + (0.12 if r == 0 else 0.06),
                        p.title + ("" if band is None else f" — {band_axis_label(band)}"),
                        transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=S.MUTED,
                        ha="right" if p.kind != "precision" else "left", va="bottom")
    seen, hs = set(), []
    for h in handles.values():
        for x in h:
            if x.get_label() not in seen:
                seen.add(x.get_label()); hs.append(x)
    view_txt = {"bands": "proportion within bands", "distribution": "distribution of observed / reference (log scale)"}[view]
    who = " vs ".join(S.score_label(s) for s in scores)
    fig.suptitle(f"Figure {spec.num}. {spec.title} — {view_txt}; scored against {who}",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=S.INK)
    ncol = min(len(hs), 5)
    fig.legend(handles=hs, loc="lower center", ncol=ncol, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, 0.0))
    # reserve room for the legend (about 0.2 in per legend row) and the x labels
    legend_rows = -(-len(hs) // ncol) if hs else 0
    bottom = (0.35 + 0.2 * legend_rows) / fig.get_figheight()
    top = 1 - 0.45 / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, top), h_pad=2.2, w_pad=(4.5 if annot else 3.0))
    from common.config import FIGURES_DIR
    out_dir = out_dir or os.path.join(FIGURES_DIR, "revision")
    return S.save(fig, out_stem(out_dir, spec, scores, view, bands))


# ── legacy engine ────────────────────────────────────────────────────────

_VIG_KEYS = ["vignette_id", "model", "include_nielsen", "include_examples", "ask_for_units", "explain_units"]
_PAIR_KEYS = ["from_opioid", "to_opioid", "model", "include_nielsen", "include_examples"]
_VIG_NIELSEN_COLS = ["observed_dose_mean", "observed_dose_std", "observed_dose_cv",
                     "observed_log_dose_mean", "observed_log_dose_std", "observed_log_dose_cv"]
_PAIR_NIELSEN_COLS = ["observed_coefficient_mean", "observed_coefficient_std", "observed_coefficient_cv",
                      "observed_log_coefficient_mean", "observed_log_coefficient_std",
                      "r_squared", "regression_slope", "regression_slope_pvalue"]


def _cell_mean_ratios(long, experiment, keys):
    d = long[long["experiment"].eq(experiment)]
    cols = {f"mr_{s}": (f"ratio_{s}", "mean") for s in SCORES}
    return d.groupby(keys).agg(**cols).reset_index()


def adapt_inputs(score: str, long=None) -> str:
    """Model-analysis CSVs for figures 2, 3 and 9 under ``score``: precision
    (CV, observed means, dose-dependence regression) from results_nielsen/,
    accuracy (band proportions) from results_<score>/, and the expected column
    replaced by an effective reference so that observed mean / expected equals
    the cell's mean ratio under the score (expected_nielsen x mean ratio
    Nielsen / mean ratio score; an identity under nielsen). Derived files
    (CV chi-square, performance profiles) are rebuilt from these by the
    builders. Returns the adapter directory."""
    long = stats.load_long() if long is None else long
    d = os.path.join(ADAPTER_ROOT, score)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for fname, exp, keys, ncols, ecol in [
        ("reworded_vignettes_model_analysis.csv", "vignette", _VIG_KEYS, _VIG_NIELSEN_COLS, "expected_to_dose"),
        ("pair_model_analysis.csv", "random_dose", _PAIR_KEYS, _PAIR_NIELSEN_COLS, "expected_coefficient"),
    ]:
        sc = pd.read_csv(os.path.join(f"results_{score}", fname))
        nz = pd.read_csv(os.path.join("results_nielsen", fname))
        for k in [k for k in keys if k.startswith(("include_", "ask_", "explain_"))]:
            sc[k] = sc[k].astype(bool); nz[k] = nz[k].astype(bool)
        nz = nz[keys + ncols + [ecol]].rename(columns={ecol: "_exp_n"})
        out = sc.drop(columns=[c for c in ncols if c in sc.columns]).merge(nz, on=keys, how="left")
        out = out.merge(_cell_mean_ratios(long, exp, keys), on=keys, how="left")
        factor = out["mr_nielsen"] / out[f"mr_{score}"]
        out[ecol] = np.where(factor.notna() & np.isfinite(factor), out["_exp_n"] * factor, out[ecol])
        out = out.drop(columns=["_exp_n"] + [c for c in out.columns if c.startswith("mr_")])
        out = out[sc.columns.tolist()]
        out.to_csv(os.path.join(d, fname), index=False)
    return d


def _coverage_text(spec, score, long) -> str:
    parts = []
    for p in spec.panels:
        n = sum(stats.coverage(long, score, c, s)["n_responses"] for c, s in p.cond_subsets())
        k = max(stats.coverage(long, score, c, s)["n_items"] for c, s in p.cond_subsets())
        parts.append(f"{p.title}: {n:,} responses, {k} conversion items")
    head = SCORES[score].title_suffix.strip(" —")
    return (f"<b>{head[0].upper() + head[1:]}</b>. Scored: " + "; ".join(parts) +
            ". Conversions no standard in this score covers are excluded.")


def build_legacy(spec: FigureSpec, score: str, out_dir=None, long=None) -> list[str]:
    from bokeh.io import output_file, save
    from bokeh.layouts import column
    from bokeh.models import Div, Plot
    from bokeh.resources import INLINE
    from visualize.comprehensive_figures import ComprehensiveFigureBuilder
    from visualize.visualization_config import convert_to_pdf_auto_size, convert_to_png_auto_size

    if not spec.rescorable and score != "nielsen":
        raise ValueError(f"figure {spec.num} is score-independent: built from Nielsen only")
    method = {n: m for n, m, _t, _d in ComprehensiveFigureBuilder.FIGURE_METADATA}[spec.slug]
    from common.config import FIGURES_DIR
    out_dir = out_dir or os.path.join(FIGURES_DIR, "revision")
    stem = out_stem(out_dir, spec, [score], "bands", DEFAULT_BANDS)
    os.makedirs(os.path.dirname(stem), exist_ok=True)

    class _Builder(ComprehensiveFigureBuilder):
        adapter_dir = None

        def _p(self, *parts):
            if self.adapter_dir and parts:
                cand = os.path.join(self.adapter_dir, *parts)
                if parts[-1] in ADAPTED_FILES:
                    return cand
            return super()._p(*parts)

    b = _Builder(results_dir=f"results_{score}", figures_dir=os.path.dirname(stem))
    if score != "nielsen" and spec.num in (2, 3, 9):
        long = stats.load_long() if long is None else long
        b.adapter_dir = adapt_inputs(score, long)
    fig = getattr(b, method)()
    suffix = SCORES[score].title_suffix
    if suffix:
        for p in fig.select({"type": Plot}):
            if p.title is not None and p.title.text:
                # second line, so long titles are not clipped at the plot width
                p.title.text = p.title.text + "\n" + suffix.strip(" —")
        long = stats.load_long() if long is None else long
        width = max([getattr(p, "width", 900) or 900 for p in fig.select({"type": Plot})] + [600])
        fig = column(fig, Div(text=f"<p style='font-size:12px;color:#555'>{_coverage_text(spec, score, long)}</p>",
                              width=width))
    html = stem + ".html"
    output_file(html, title=f"Figure {spec.num} [{score}]")
    save(fig, resources=INLINE)

    async def _convert():
        await convert_to_pdf_auto_size(html, stem + ".pdf", scale=2.0)
        await convert_to_png_auto_size(html, stem + ".png", scale=2.0)
    asyncio.run(_convert())
    return [stem + ".png", stem + ".pdf", html]


ADAPTED_FILES = {"reworded_vignettes_model_analysis.csv", "pair_model_analysis.csv",
                 "cv_accuracy_chi2.csv", "performance_profile_vignettes.csv",
                 "performance_profile_random_doses.csv"}


def build(num: int, scores, view: str, bands=DEFAULT_BANDS, compare=False, out_dir=None, long=None) -> list[list[str]]:
    """Build figure ``num``. With several scores and compare=False, one figure
    per score; with compare=True, one comparison figure. Returns output paths."""
    spec = FIGURES[num]
    if view not in spec.views:
        raise ValueError(f"figure {num} supports views {spec.views}")
    scores = list(scores)
    if not spec.rescorable:
        scores = ["nielsen"]
    groups = [scores] if compare else [[s] for s in scores]
    outs = []
    for g in groups:
        if engine_for(spec, g, view, bands, compare and len(g) > 1) == "legacy":
            outs.append(build_legacy(spec, g[0], out_dir, long))
        else:
            outs.append(build_grid(spec, g, view, list(bands), long, out_dir))
    return outs
