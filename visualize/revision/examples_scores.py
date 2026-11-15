"""
Phase 4 example figures E1-E4: comparing the three scores (Nielsen, all
standards, all standards except Nielsen) and the two views (bands,
distribution). Prototypes for the authors to iterate on.

  E1   Figure 1 as a comparison grid: rows = view, columns = score
       (naive text vignettes).
  E2a  Figure 11 (reference-grounding effect) in the same grid, text vignettes:
       one bands row per registered band (BANDS), then the distribution row
       annotated with median fold error (naive -> Nielsen).
  E2b  The same for random doses.
  E3   Ranking shifts across the three scores (slope chart), vignettes and
       random doses; rank by mean tight accuracy (the ranking's sort key) with
       the CMH tier in each marker, and rank by median fold error.
  E4   Tolerance curves: % of responses within each fold-error tolerance
       (1x-4x), naive, one panel per score, vignettes and random doses as rows.

Built from results_revision/responses_long.csv through the Phase 2 framework
(analysis/revision/stats.py, visualize/revision/panels.py and style.py).
Deterministic; never queries any model.

    python -m visualize.revision.examples_scores            # all of E1-E4
    python -m visualize.revision.examples_scores E1 E3      # a selection
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from visualize.revision import style as S  # noqa: F401  (backend and rcParams first)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analysis.revision import stats  # noqa: E402
from analysis.revision.conditions import CONDITIONS  # noqa: E402
from analysis.revision.tolerances import BANDS, reference_agreement_k, scored_spreads  # noqa: E402
from visualize.revision import panels as P  # noqa: E402

SCORE_LIST = ["nielsen", "all", "non_nielsen"]
EXPERIMENTS = [("vignette_naive", "Text vignettes"), ("rd_naive", "Random doses")]


def out_dir() -> str:
    from common.config import FIGURES_DIR
    return os.path.join(FIGURES_DIR, "revision", "examples")


def _dedupe(handles):
    seen, out = set(), []
    for h in handles:
        if h.get_label() not in seen:
            seen.add(h.get_label())
            out.append(h)
    return out


def _legend_bottom(fig, handles, ncol, h_in=0.22):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.5, 0.0))
    rows = -(-len(handles) // ncol)
    return (0.3 + h_in * rows) / fig.get_figheight()


def _row_label(ax, text):
    ax.set_ylabel(text, fontsize=9, fontweight="bold", color=S.INK, labelpad=8)


# ── E1 ───────────────────────────────────────────────────────────────────

def build_e1(long) -> list[str]:
    models = S.model_order()
    panel = P.Panel("accuracy", ("vignette_naive",), title="Vignettes, naive")
    bands = ["tight", "broad"]
    row_h = 0.25 * len(models) + 1.0
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 2 * row_h + 1.6), squeeze=False)
    handles = []
    for c, sc in enumerate(SCORE_LIST):
        first = c == 0
        handles += P.accuracy_bands(axes[0][c], long, sc, panel, bands, models, show_labels=first) or []
        handles += P.accuracy_distribution(axes[1][c], long, sc, panel, bands, models,
                                           show_labels=first, annotate=True) or []
        axes[0][c].set_title(S.score_title(sc), loc="left", pad=8)
        axes[1][c].set_title(S.score_title(sc), loc="left", pad=16)
        axes[1][c].set_xlabel("Observed / reference dose (log scale)", fontsize=7.5, labelpad=12)
    _row_label(axes[0][0], "Bands view\n(% of responses within band)")
    _row_label(axes[1][0], "Distribution view\n(each response, log fold scale)")
    fig.suptitle("E1. Figure 1, naive accuracy on the text vignettes: bands and distribution views "
                 "scored against Nielsen, all standards and all except Nielsen",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=S.INK)
    hs = _dedupe(handles)
    hs.append(Line2D([], [], ls="none", label="MFE = median fold error; >2× and <½× = % of responses"))
    bottom = _legend_bottom(fig, hs, 3)
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.45 / fig.get_figheight()), h_pad=3.0, w_pad=4.5)
    return S.save(fig, os.path.join(out_dir(), "E1_fig01_naive_vignettes_score_view_grid"))


# ── E2 ───────────────────────────────────────────────────────────────────

def _pooled_band_change(long, score, panel, band):
    vals = []
    for c, s in panel.cond_subsets():
        m = stats.model_summary(long, score, c, s, [band], average="cell")
        vals.append(float(m[f"pct_{band}"].mean()) if not m.empty else np.nan)
    return vals


def _pooled_mfe(long, score, panel):
    out = []
    for c, s in panel.cond_subsets():
        d = stats.responses(long, score, c, s)
        out.append(float(np.nanmedian(d["fold_error"])) if len(d) else np.nan)
    return out


def _band_label(b):
    """Registry label, with the computed k written out for reference_agreement."""
    if b == "reference_agreement":
        k = reference_agreement_k()
        return f"Reference agreement ({1 / k:.2f}–{k:.2g}×)"
    return BANDS[b].label


def build_e2(long, which: str) -> list[str]:
    models = S.model_order()
    if which == "vignette":
        panel = P.Panel("sequence", ("vignette_naive", "vignette_nielsen"), title="Text vignettes")
        tag, name = "E2a", "text vignettes"
    else:
        panel = P.Panel("sequence", ("rd_naive", "rd_nielsen"), title="Random doses")
        tag, name = "E2b", "random doses"
    bands = list(BANDS)                      # every registered band
    row_h = 0.24 * len(models) + 1.0
    nrow = len(bands) + 1
    fig, axes = plt.subplots(nrow, 3, figsize=(15.5, nrow * row_h + 1.6), squeeze=False)
    handles = []
    for c, sc in enumerate(SCORE_LIST):
        first = c == 0
        for r, b in enumerate(bands):
            ax = axes[r][c]
            handles += P.sequence_bands(ax, long, sc, panel, b, models, show_labels=first) or []
            a, z = _pooled_band_change(long, sc, panel, b)
            ax.set_title(S.score_title(sc), loc="left", pad=16)
            ax.text(0.0, 1.015, f"{_band_label(b)}: mean of models {a:.0f}% → {z:.0f}% "
                    f"({z - a:+.0f} points)", transform=ax.transAxes, fontsize=6.8,
                    color=S.MUTED, va="bottom")
            if r == len(bands) - 1:
                ax.set_xlabel("% of responses within band (batch means)", fontsize=7.5, labelpad=12)
            if first:
                _row_label(ax, f"Bands view\n{_band_label(b)}")
        ax = axes[-1][c]
        handles += P.sequence_distribution(ax, long, sc, panel, ["tight", "broad"], models,
                                           show_labels=first, annotate=True) or []
        a, z = _pooled_mfe(long, sc, panel)
        ax.set_title(S.score_title(sc), loc="left", pad=24)
        ax.annotate(f"Median fold error, all responses: {a:.2f}× → {z:.2f}×", (0.0, 1.0),
                    xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                    fontsize=6.8, color=S.MUTED, va="bottom")
        ax.set_xlabel("Observed / reference dose (log scale)", fontsize=7.5, labelpad=12)
        if first:
            _row_label(ax, "Distribution view\nmedian, IQR, 10–90%")
    fig.suptitle(f"{tag}. Figure 11, reference-grounding effect (naive → Nielsen supplied), {name}: "
                 "every band and the distribution, scored against Nielsen, all standards and all except Nielsen",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=S.INK)
    hs = _dedupe(handles) + [Patch(color=S.BAND_FILL["tight"], label="tight band"),
                             Patch(color=S.BAND_FILL["broad"], label="broad band")]
    hs = _dedupe(hs)
    hs.append(Line2D([], [], ls="none", label="right of distribution panels: median fold error, naive → Nielsen"))
    bottom = _legend_bottom(fig, hs, 4)
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.45 / fig.get_figheight()), h_pad=2.6, w_pad=4.5)
    stem = "E2a_fig11_grounding_vignettes_score_view_grid" if which == "vignette" \
        else "E2b_fig11_grounding_random_doses_score_view_grid"
    return S.save(fig, os.path.join(out_dir(), stem))


# ── E3 ───────────────────────────────────────────────────────────────────

def ranking_table(long) -> pd.DataFrame:
    """Model x experiment x score: mean tight accuracy, its rank (1 = best,
    the CMH ranking's sort order), the CMH adjusted rank and median fold error
    with its rank (ties share the average rank)."""
    rows = []
    for cond, _lab in EXPERIMENTS:
        for sc in SCORE_LIST:
            r = stats.rank_models(long, sc, cond).result_df
            mfe = stats.model_summary(long, sc, cond, bands=[]).set_index("model")["median_fold_error"]
            t = r[["model", "mean_tight_prop", "raw_rank", "adjusted_rank"]].copy()
            t["median_fold_error"] = t["model"].map(mfe)
            t["mfe_rank"] = t["median_fold_error"].round(6).rank(method="average")
            t["condition"], t["score"] = cond, sc
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _slope(ax, tab, value_col, models, tier_col=None):
    xs = np.arange(len(SCORE_LIST))
    n = len(models)
    for m in models:
        g = tab[tab.model == m].set_index("score").reindex(SCORE_LIST)
        y = g[value_col].to_numpy(dtype=float)
        ax.plot(xs, y, color=S.color(m), lw=1.6, alpha=0.85, zorder=2)
        if tier_col:
            ax.scatter(xs, y, s=95, color="white", edgecolor=S.color(m), linewidths=1.6, zorder=3)
            for x, yy, t in zip(xs, y, g[tier_col].to_numpy()):
                ax.text(x, yy, f"{int(t)}", ha="center", va="center", fontsize=5.8, color=S.INK, zorder=4)
        else:
            ax.scatter(xs, y, s=28, color=S.color(m), edgecolor="white", linewidths=0.5, zorder=3)
    # end labels in ordinal slots (ties broken by the global model order), with leaders
    for end, x, ha, dx in [(SCORE_LIST[0], xs[0], "right", -0.16), (SCORE_LIST[-1], xs[-1], "left", 0.16)]:
        g = tab[tab.score == end].set_index("model").reindex(models)
        g["_o"] = range(n)
        g = g.sort_values([value_col, "_o"])
        for slot, (m, row) in enumerate(g.iterrows(), start=1):
            ax.plot([x + np.sign(dx) * 0.07, x + dx * 0.85], [row[value_col], slot], color=S.GRID, lw=0.6, zorder=1)
            ax.text(x + dx, slot, S.short(m), ha=ha, va="center", fontsize=6.8, color=S.color(m),
                    fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([S.score_label(s)[0].upper() + S.score_label(s)[1:] for s in SCORE_LIST], fontsize=8)
    ax.set_xlim(-1.25, len(SCORE_LIST) - 1 + 1.25)
    ax.set_ylim(n + 0.6, 0.4)
    ax.set_yticks(range(1, n + 1))
    ax.tick_params(axis="y", labelsize=7)
    for x in xs:
        ax.axvline(x, color=S.GRID, lw=0.6, zorder=0)
    S.clean_axes(ax, keep=())


def build_e3(long) -> list[str]:
    models = S.model_order()
    tab = ranking_table(long)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.8), squeeze=False)
    for c, (cond, lab) in enumerate(EXPERIMENTS):
        t = tab[tab.condition == cond]
        _slope(axes[0][c], t, "raw_rank", models, tier_col="adjusted_rank")
        axes[0][c].set_title(f"{lab}, naive: rank by mean tight accuracy (number = CMH rank)",
                             loc="left", fontsize=9)
        _slope(axes[1][c], t, "mfe_rank", models)
        axes[1][c].set_title(f"{lab}, naive: rank by median fold error (ties share a rank)",
                             loc="left", fontsize=9)
        n_items = {sc: stats.coverage(long, sc, cond)["n_items"] for sc in SCORE_LIST}
        axes[1][c].text(0.5, -0.07, "Items scored: " + ", ".join(
            f"{S.score_label(s)} {n_items[s]}" for s in SCORE_LIST) + f" {P._item_word(cond)}",
            transform=axes[1][c].transAxes, ha="center", va="top", fontsize=6.6, color=S.MUTED)
    for r in range(2):
        axes[r][0].set_ylabel("Rank (1 = best)", fontsize=8)
    fig.suptitle("E3. Ranking shifts across scores (Nielsen, all standards, all except Nielsen), "
                 "naive condition, text vignettes and random doses",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=S.INK)
    hs = [Line2D([0], [0], color=S.color(m), lw=2, label=S.short(m)) for m in models]
    hs.append(Line2D([0], [0], ls="none", marker="o", markerfacecolor="white", markeredgecolor="#555555",
                     markersize=8, label="number in marker: CMH adjusted rank (significance tier)"))
    bottom = _legend_bottom(fig, hs, 5)
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.45 / fig.get_figheight()), h_pad=3.0, w_pad=3.0)
    tab.to_csv(os.path.join(out_dir(), "E3_ranking_shifts_data.csv"), index=False)
    return S.save(fig, os.path.join(out_dir(), "E3_ranking_shifts_three_scores"))


# ── E4 ───────────────────────────────────────────────────────────────────

TOL = np.exp(np.linspace(0, np.log(4.0), 400))


def tolerance_curve(d: pd.DataFrame) -> np.ndarray:
    """% of responses with fold error <= t for t in TOL. Unusable answers (no
    log ratio) count as outside every tolerance, as in the band statistics."""
    fe = d["fold_error"].to_numpy(dtype=float)
    fe = np.where(np.isfinite(fe), fe, np.inf)
    fe = np.sort(fe)
    return 100 * np.searchsorted(fe, TOL * (1 + 1e-9), side="right") / max(len(fe), 1)


def build_e4(long) -> list[str]:
    models = S.model_order()
    k_agree = reference_agreement_k()
    spreads = scored_spreads()
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), squeeze=False, sharey=True)
    for r, (cond, lab) in enumerate(EXPERIMENTS):
        exp = CONDITIONS[cond].experiment
        env = spreads.loc[spreads["experiment"].eq(exp), "spread"].dropna()
        for c, sc in enumerate(SCORE_LIST):
            ax = axes[r][c]
            ax.axvspan(1.0, 1.7, color=S.BAND_FILL["broad"], lw=0, zorder=0)
            ax.axvspan(1.0, 1.15, color=S.BAND_FILL["tight"], lw=0, zorder=0)
            for v, name, ls, y in [(1.15, "tight 1.15×", "-", 101.5), (1.7, "broad 1.7×", "-", 101.5),
                                   (k_agree, f"ref. agreement {k_agree:.2g}×", ":", 106.5)]:
                ax.axvline(v, color=S.MUTED, lw=0.9, ls=ls, zorder=1)
                ax.text(v, y, name, rotation=0, fontsize=6.3, color=S.MUTED,
                        ha="center" if name.startswith("ref") else ("right" if v < 1.3 else "left"),
                        va="bottom", clip_on=False)
            d = stats.responses(long, sc, cond)
            for m in models:
                g = d[d.model == m]
                if g.empty:
                    continue
                ax.plot(TOL, tolerance_curve(g), color=S.color(m), lw=1.2, alpha=0.9,
                        drawstyle="steps-post", zorder=3)
            S.fold_axis(ax, "x", (1.0, 4.0), ticks=[1, 1.5, 2, 3, 4])
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_yticklabels(["0", "25", "50", "75", "100%"])
            ax.grid(axis="y", color=S.GRID, lw=0.5)
            S.clean_axes(ax, keep=("bottom", "left"))
            ax.set_title(f"{lab}, naive — {S.score_title(sc)}", loc="left", pad=24, fontsize=9)
            cov = stats.coverage(long, sc, cond)
            ax.text(0.98, 0.04, f"n = {cov['n_responses']:,} responses, {cov['n_items']} {P._item_word(cond)}\n"
                    f"reference envelope (per conversion): median {env.median():.2g}×, max {env.max():.2g}×",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=S.MUTED,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=5)
            if r == 1:
                ax.set_xlabel("Fold-error tolerance t (answer within t-fold of the reference either way)",
                              fontsize=7.5)
            if c == 0:
                ax.set_ylabel("% of responses within tolerance", fontsize=8)
    fig.suptitle("E4. Tolerance curves, naive condition: % of responses within each fold-error tolerance, "
                 "scored against Nielsen, all standards and all except Nielsen",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=S.INK)
    hs = [Line2D([0], [0], color=S.color(m), lw=2, label=S.short(m)) for m in models]
    hs += [Patch(color=S.BAND_FILL["tight"], label="tight (fold error ≤ 1.15×)"),
           Patch(color=S.BAND_FILL["broad"], label="broad (fold error ≤ 1.7×)"),
           Line2D([0], [0], color=S.MUTED, ls=":", label="reference-agreement band k")]
    bottom = _legend_bottom(fig, hs, 6)
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.45 / fig.get_figheight()), h_pad=2.5, w_pad=2.0)
    return S.save(fig, os.path.join(out_dir(), "E4_tolerance_curves_three_scores"))


BUILDERS = {
    "E1": build_e1,
    "E2a": lambda long: build_e2(long, "vignette"),
    "E2b": lambda long: build_e2(long, "random_dose"),
    "E3": build_e3,
    "E4": build_e4,
}


def main(argv=None) -> list[str]:
    argv = sys.argv[1:] if argv is None else argv
    sel = argv or list(BUILDERS)
    sel = [k for s in sel for k in BUILDERS if k == s or (s == "E2" and k.startswith("E2"))]
    long = stats.load_long()
    out = []
    for k in sel:
        paths = BUILDERS[k](long)
        print("✓", k, paths[0], flush=True)
        out += paths
    return out


if __name__ == "__main__":
    main()
