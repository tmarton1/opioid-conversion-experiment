"""
Consistency illustration figure.

Six-panel matplotlib figure showing by example how naive LLMs can produce
inaccurate yet *consistent* opioid-conversion coefficients, and how Nielsen
reference data rescues accuracy.

Layout (2 rows × 3 columns):
  Row 0 – Three "good" examples: low CV, wrong coefficient, Nielsen rescues
  Row 1 – Noisy example | Negative-slope example | Vignette-wording sensitivity

Each random-dose panel plots from_dose vs derived coefficient (obs/from),
with a dashed line at the expected coefficient.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Selected examples ────────────────────────────────────────────────

GOOD_EXAMPLES = [
    {
        "from_opioid": "oral_tramadol",
        "to_opioid": "oral_morphine",
        "model": "anthropic/claude-3.5-sonnet:20241022",
        "label": "Oral Tramadol → Oral Morphine\nClaude 3.5 Sonnet",
    },
    {
        "from_opioid": "oral_tapentadol",
        "to_opioid": "oral_morphine",
        "model": "mistralai/mixtral-8x22b-instruct",
        "label": "Oral Tapentadol → Oral Morphine\nMixtral 8×22B",
    },
    {
        "from_opioid": "iv_fentanyl",
        "to_opioid": "oral_morphine",
        "model": "anthropic/claude-3.5-sonnet:20241022",
        "label": "IV Fentanyl → Oral Morphine\nClaude 3.5 Sonnet",
    },
]

NOISY_EXAMPLE = {
    "from_opioid": "iv_fentanyl",
    "to_opioid": "oral_morphine",
    "model": "openai/gpt-4.1",
    "label": "IV Fentanyl → Oral Morphine\nGPT-4.1",
}

NEG_SLOPE_EXAMPLE = {
    "from_opioid": "transdermal_buprenorphine",
    "to_opioid": "oral_morphine",
    "model": "meta-llama/llama-3.3-70b-instruct",
    "label": "Transdermal Buprenorphine → Oral Morphine\nLlama 3.3 70B",
}

VIGNETTE_EXAMPLE = {
    "scenario_key": "pediatric-burn-patient",
    "model": "google/gemini-2.5-flash",
    "label": "Pediatric Burn Patient\nGemini 2.5 Flash",
}


# ── Helper: load trials ─────────────────────────────────────────────

def _load_rd_trials(raw: pd.DataFrame, ex: dict, *, nielsen: bool = False):
    """Return (from_dose, observed_coeff, expected_coeff) arrays."""
    mask = (
        (raw["from_opioid"] == ex["from_opioid"]) &
        (raw["to_opioid"] == ex["to_opioid"]) &
        (raw["model"] == ex["model"]) &
        (raw["include_nielsen"] == nielsen) &
        (raw["include_examples"] == False)
    )
    sub = raw.loc[mask].copy()
    from_dose = sub["from_dose"].values
    obs_coeff = sub["observed_to_dose"].values / from_dose
    exp_coeff = (sub["expected_to_dose"].values / from_dose)[0]  # constant
    return from_dose, obs_coeff, exp_coeff


# ── Panel plotters ───────────────────────────────────────────────────

_NAIVE_COLOR = "#3182bd"      # blue
_NIELSEN_COLOR = "#e6550d"    # orange
_EXPECTED_COLOR = "#2ca02c"   # green
_NOISY_COLOR = "#9467bd"      # purple
_NEG_SLOPE_COLOR = "#d62728"  # red


def _plot_good(ax, raw: pd.DataFrame, ex: dict, panel_letter: str):
    """Coefficient scatter (naive + Nielsen) for a consistent-but-wrong pair."""
    fd_n, oc_n, ec = _load_rd_trials(raw, ex, nielsen=False)
    fd_r, oc_r, _ = _load_rd_trials(raw, ex, nielsen=True)

    ax.scatter(fd_n, oc_n, c=_NAIVE_COLOR, s=28, alpha=0.7, zorder=3,
               edgecolors="white", linewidths=0.3, label="Naive")
    ax.scatter(fd_r, oc_r, c=_NIELSEN_COLOR, s=28, alpha=0.7, zorder=3,
               edgecolors="white", linewidths=0.3, marker="D", label="Nielsen")

    xlim = ax.get_xlim()
    ax.axhline(ec, color=_EXPECTED_COLOR, ls="--", lw=1.4, zorder=2, label="Expected")

    # Annotate naive coefficient (median)
    naive_med = np.median(oc_n)
    ax.axhline(naive_med, color=_NAIVE_COLOR, ls=":", lw=1.0, alpha=0.6, zorder=1)

    ax.set_title(ex["label"], fontsize=8.5, fontweight="bold", pad=6)
    ax.set_xlabel("From dose", fontsize=8)
    ax.set_ylabel("Observed coefficient", fontsize=8)
    ax.tick_params(labelsize=7)

    # Panel letter
    ax.text(-0.12, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


def _plot_noisy(ax, raw: pd.DataFrame, ex: dict, panel_letter: str):
    """Coefficient scatter for a noisy (high-CV) pair."""
    fd, oc, ec = _load_rd_trials(raw, ex, nielsen=False)
    ax.scatter(fd, oc, c=_NOISY_COLOR, s=28, alpha=0.7, zorder=3,
               edgecolors="white", linewidths=0.3)
    ax.axhline(ec, color=_EXPECTED_COLOR, ls="--", lw=1.4, zorder=2)

    ax.set_title(ex["label"], fontsize=8.5, fontweight="bold", pad=6)
    ax.set_xlabel("From dose", fontsize=8)
    ax.set_ylabel("Observed coefficient", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(-0.12, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


def _plot_neg_slope(ax, raw: pd.DataFrame, ex: dict, panel_letter: str):
    """Coefficient scatter for a negative-slope pair, with regression line."""
    fd, oc, ec = _load_rd_trials(raw, ex, nielsen=False)
    ax.scatter(fd, oc, c=_NEG_SLOPE_COLOR, s=28, alpha=0.7, zorder=3,
               edgecolors="white", linewidths=0.3)
    ax.axhline(ec, color=_EXPECTED_COLOR, ls="--", lw=1.4, zorder=2)

    # Regression line
    if len(fd) > 2:
        coeffs = np.polyfit(fd, oc, 1)
        xs = np.linspace(fd.min(), fd.max(), 100)
        ax.plot(xs, np.polyval(coeffs, xs), color=_NEG_SLOPE_COLOR,
                ls="-", lw=1.4, alpha=0.7, zorder=2)

    ax.set_title(ex["label"], fontsize=8.5, fontweight="bold", pad=6)
    ax.set_xlabel("From dose", fontsize=8)
    ax.set_ylabel("Observed coefficient", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(-0.12, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


def _plot_vignette(ax, vr: pd.DataFrame, ex: dict, panel_letter: str):
    """Strip plot of responses by vignette wording."""
    mask = (
        (vr["scenario_key"] == ex["scenario_key"]) &
        (vr["model"] == ex["model"]) &
        (vr["include_nielsen"] == False) &
        (vr["explain_units"] == False) &
        (vr["has_vignette_text"] == True)
    )
    sub = vr.loc[mask].copy()
    expected = sub["expected_to_dose"].iloc[0]

    vignettes = sorted(sub["vignette_id"].unique())
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for i, vid in enumerate(vignettes):
        vals = sub.loc[sub["vignette_id"] == vid, "observed_to_dose"].values
        jitter = np.random.default_rng(42).normal(0, 0.08, size=len(vals))
        x = np.full_like(vals, i, dtype=float) + jitter
        ax.scatter(x, vals, c=colors[i % len(colors)], s=32, alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=3)
        # Mean bar
        ax.hlines(np.mean(vals), i - 0.25, i + 0.25, colors=colors[i % len(colors)],
                  linewidths=2.0, zorder=4)

    ax.axhline(expected, color=_EXPECTED_COLOR, ls="--", lw=1.4, zorder=2,
               label=f"Expected ({expected:.0f})")

    ax.set_xticks(range(len(vignettes)))
    ax.set_xticklabels([f"Wording {j+1}" for j in range(len(vignettes))],
                       fontsize=7)
    ax.set_title(ex["label"], fontsize=8.5, fontweight="bold", pad=6)
    ax.set_xlabel("Vignette wording", fontsize=8)
    ax.set_ylabel("Observed dose (mg)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.text(-0.12, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


# ── Main build ───────────────────────────────────────────────────────

def build_consistency_figure(
    results_csv: str = "results/experiment_results.csv",
    vignette_csv: str = "results/reworded_vignettes_results.csv",
    output_dir: str = "figures",
):
    """Build the 2×3 consistency illustration figure and save PNG + PDF."""
    raw = pd.read_csv(results_csv)
    vr = pd.read_csv(vignette_csv)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)

    # Row 0: three "good" examples
    for i, ex in enumerate(GOOD_EXAMPLES):
        _plot_good(axes[0, i], raw, ex, chr(ord("A") + i))

    # Shared legend for row 0
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_NAIVE_COLOR,
               markersize=6, label="Naive"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=_NIELSEN_COLOR,
               markersize=6, label="Nielsen"),
        Line2D([0], [0], color=_EXPECTED_COLOR, ls="--", lw=1.4,
               label="Expected coefficient"),
        Line2D([0], [0], color=_NAIVE_COLOR, ls=":", lw=1.0, alpha=0.6,
               label="Naive median"),
    ]
    axes[0, 2].legend(handles=handles, fontsize=7, loc="upper right",
                      framealpha=0.85)

    # Row 1
    _plot_noisy(axes[1, 0], raw, NOISY_EXAMPLE, "D")
    _plot_neg_slope(axes[1, 1], raw, NEG_SLOPE_EXAMPLE, "E")
    _plot_vignette(axes[1, 2], vr, VIGNETTE_EXAMPLE, "F")

    # Row 1 legends
    h_noisy = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_NOISY_COLOR,
               markersize=6, label="Naive"),
        Line2D([0], [0], color=_EXPECTED_COLOR, ls="--", lw=1.4,
               label="Expected coefficient"),
    ]
    axes[1, 0].legend(handles=h_noisy, fontsize=7, loc="upper right",
                      framealpha=0.85)

    h_neg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_NEG_SLOPE_COLOR,
               markersize=6, label="Naive"),
        Line2D([0], [0], color=_NEG_SLOPE_COLOR, ls="-", lw=1.4, alpha=0.7,
               label="Regression"),
        Line2D([0], [0], color=_EXPECTED_COLOR, ls="--", lw=1.4,
               label="Expected coefficient"),
    ]
    axes[1, 1].legend(handles=h_neg, fontsize=7, loc="upper right",
                      framealpha=0.85)

    # Row titles via suptitle annotations
    fig.text(0.5, 1.01, "Consistent but wrong — Nielsen rescues",
             ha="center", fontsize=11, fontweight="bold",
             transform=fig.transFigure)
    fig.text(0.5, 0.50, "High variability / dose-dependent bias / wording sensitivity",
             ha="center", fontsize=11, fontweight="bold",
             transform=fig.transFigure)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "consistency_examples.png")
    pdf_path = os.path.join(output_dir, "consistency_examples.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ {png_path}")
    print(f"✓ {pdf_path}")
    return png_path, pdf_path


if __name__ == "__main__":
    build_consistency_figure()
