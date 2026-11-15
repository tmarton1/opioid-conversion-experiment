"""
Example figure E8: overdose exceedance curves, per condition (author decision
2026-11-15). Safety is defined in dose terms against the current standards, so
it is the same under every score: x = multiple of the highest current standard
for the conversion (Nielsen, CDC 2022, NIH HEAL, McPherson 2018), y = % of
responses above it (pooled over the 13 models). 2x (overdose) and 4x (severe
overdose) are marked. Data: results_revision/safety_exceedance.csv (task 3A).

    python -m visualize.revision.examples_safety
"""
from __future__ import annotations

import os

import pandas as pd

from visualize.revision import style as S  # noqa: F401  (backend, styling)
import matplotlib.pyplot as plt  # noqa: E402

from analysis.revision.stats import SAFETY  # noqa: E402
from common.config import FIGURES_DIR  # noqa: E402

SRC = os.path.join("results_revision", "safety_exceedance.csv")
PANELS = [
    ("Vignettes (text)", [("vignette_naive", "naive", "-"), ("vignette_nielsen", "Nielsen supplied", "--"),
                          ("vignette_nielsen_units", "Nielsen + units", ":")]),
    ("Random doses (mg/day)", [("rd_naive", "naive", "-"), ("rd_nielsen", "Nielsen supplied", "--")]),
    ("Dosing-unit experiment (mg/4 h, Nielsen supplied)", [("eu_4h_nielsen", "units not explained", "-"),
                                                          ("eu_4h_nielsen_units", "units explained", "--")]),
]
SHADES = ["#2b2b2a", "#5b7fa6", "#a3a29c"]


def build(out_dir=None):
    d = pd.read_csv(SRC)
    d = d[d.model == "POOLED"]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.4), sharey=True)
    for ax, (title, conds) in zip(axes, PANELS):
        for k, (cond, lab, ls) in enumerate(conds):
            g = d[d.condition == cond].sort_values("multiple_of_highest")
            ax.plot(g.multiple_of_highest, g.pct_above, color=SHADES[k], ls=ls, lw=1.8, label=lab)
        for x, name in [(SAFETY["overdose_x"], "overdose > 2×"), (SAFETY["severe_overdose_x"], "severe > 4×")]:
            ax.axvline(x, color=S.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
            ax.text(x * 1.03, 0.97, name, transform=ax.get_xaxis_transform(), fontsize=7, color=S.MUTED, va="top")
        ax.set_xscale("log"); ax.set_xlim(1, 10)
        ax.set_xticks([1, 2, 4, 10]); ax.set_xticklabels(["1×", "2×", "4×", "10×"]); ax.minorticks_off()
        ax.grid(axis="y", color=S.GRID, lw=0.5)
        S.clean_axes(ax, keep=("bottom", "left"))
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Response as a multiple of the highest current standard", fontsize=8)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    axes[0].set_ylabel("% of responses above", fontsize=8.5)
    fig.suptitle("E8. Overdose exceedance: share of responses above each multiple of the highest current standard "
                 "(same under every score; 13 models pooled)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold", color=S.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    out_dir = out_dir or os.path.join(FIGURES_DIR, "revision", "examples")
    return S.save(fig, os.path.join(out_dir, "E8_overdose_exceedance_by_condition"))


if __name__ == "__main__":
    for p in build():
        print("✓", p)
