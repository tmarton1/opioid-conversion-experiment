"""
Shared styling for the revision figures (matplotlib).

- Model colors: the fixed map MODEL_COLOR_DICT (visualize/visualization_config.py),
  the same in every figure under every score and view.
- Model order: the fixed global order (MODEL_RANK_ORDER, the order of the color
  map, from the all-standards rankings), the same for every score, so panels
  for different scores line up and the color gradient runs in sequence.
- Ratio axes: log scale, ticks labeled as fold values, overdosing to the right.
- Bands: fixed-width bands are shaded and labeled; per-conversion bands
  (reference_envelope) cannot be one region, so they are named in the legend.
"""
from __future__ import annotations

import os

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.revision.scores import SCORES  # noqa: E402
from analysis.revision.tolerances import BANDS  # noqa: E402
from common.models import CONSISTENT_MODELS  # noqa: E402
from visualize.visualization_config import MODEL_COLOR_DICT  # noqa: E402

INK = "#2b2b2a"
MUTED = "#6b6a65"
GRID = "#e4e3df"
SURFACE = "#ffffff"
BAND_FILL = {"tight": "#d6e9d6", "broad": "#eef5ee", "reference_agreement": "#e4e9f4"}
FOLD_TICKS = [0.25, 0.5, 1, 2, 4]          # band edges are labeled on the shading itself
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def model_order() -> list[str]:
    """Fixed global model order (best first), the same under every score."""
    from visualize.visualization_config import get_ranked_model_order
    order = [m for m in get_ranked_model_order() if m in CONSISTENT_MODELS]
    return order + [m for m in CONSISTENT_MODELS if m not in order]


def color(model: str) -> str:
    return MODEL_COLOR_DICT.get(model, "#888888")


def short(model: str) -> str:
    return model.split("/")[-1].replace(":20241022", "")


def fold_label(x: float) -> str:
    if x == 0.25:
        return "¼×"
    if x == 0.5:
        return "½×"
    return f"{x:g}×"


def score_label(score: str) -> str:
    """Score name for running text ("all standards", "geometric mean of standards")."""
    return "Nielsen" if score == "nielsen" else SCORES[score].label[0].lower() + SCORES[score].label[1:]


def score_title(score: str) -> str:
    """Panel heading naming the score."""
    return f"Scored against {score_label(score)}"


def fold_axis(ax, axis="x", lim=(0.2, 5.0), ticks=FOLD_TICKS):
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale
    set_scale("log")
    (ax.set_xlim if axis == "x" else ax.set_ylim)(*lim)
    t = [v for v in ticks if lim[0] <= v <= lim[1]]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(t)
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([fold_label(v) for v in t])
    ax.minorticks_off()


def shade_bands(ax, axis="x", bands=("tight", "broad"), label=True):
    """Shade fixed-width bands (widest first) and mark 1x."""
    fixed = [b for b in bands if b in BANDS and not BANDS[b].per_conversion]
    fixed.sort(key=lambda b: -np.log(BANDS[b].bounds("x", "1", "y")[1]))
    span = ax.axvspan if axis == "x" else ax.axhspan
    line = ax.axvline if axis == "x" else ax.axhline
    for b in fixed:
        lo, hi = BANDS[b].bounds("x", "1", "y")
        span(lo, hi, color=BAND_FILL.get(b, "#eeeeee"), lw=0, zorder=0)
        if label:
            if axis == "x":
                ax.text(hi, 1.0, f" {b}", transform=ax.get_xaxis_transform(), fontsize=6.5,
                        color=MUTED, ha="left", va="bottom", clip_on=False)
            else:
                ax.text(1.0, hi, f" {b}", transform=ax.get_yaxis_transform(), fontsize=6.5,
                        color=MUTED, ha="left", va="center", clip_on=False)
    line(1.0, color=INK, lw=0.9, alpha=0.6, zorder=1)


def clean_axes(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def save(fig, path_stem: str) -> list[str]:
    """Save PNG (300 dpi) and PDF next to each other; returns the paths."""
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = f"{path_stem}.{ext}"
        tmp = f"{path_stem}.tmp.{ext}"
        # no creation date in the PDF, so reruns give byte-identical files
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(tmp, dpi=DPI, bbox_inches="tight", metadata=meta)
        os.replace(tmp, p)
        out.append(p)
    plt.close(fig)
    return out
