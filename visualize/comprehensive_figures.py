"""
Comprehensive figure builder for opioid conversion experiment.

Generates all publication-ready figures with unified model colours
(based on average vignette + random-dose rank), sorted by performance,
and Wilcoxon significance annotations where appropriate.

Standalone figure functions
--------------------------
naive_performance_bars      Horizontal bars of naive accuracy by model
cv_scatter                  Scatter: naive CV vs Nielsen CV
nielsen_effect_bars         Grouped bars: naive vs Nielsen with significance
explain_units_bars          Three-condition bars (mg/day, mg/4h, explained)
full_treatment_bars         Bars: naive vs Nielsen vs Nielsen + explain units
cross_experiment_scatter    Vignette vs random dose scatter
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

from analysis.per_model_wilcoxon import lookup_per_model_p, lookup_pooled_p

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row as bk_row, gridplot
from bokeh.models import (
    ColumnDataSource, LabelSet, HoverTool, Legend, LegendItem,
    FactorRange, Span, Div, Label, FixedTicker,
)
from bokeh.palettes import Greys256
from bokeh.resources import INLINE

from common.models import CONSISTENT_MODELS
from common.config import RESULTS_DIR, FIGURES_DIR
from visualize.visualization_config import (
    MODEL_COLOR_DICT, get_ranked_model_order, get_model_order,
    normalize_model_name, convert_to_pdf_auto_size, convert_to_png_auto_size,
    TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER,
    BROAD_RANGE_LOWER, BROAD_RANGE_UPPER,
)

_CONSISTENT_SET = set(CONSISTENT_MODELS)


# ── helpers ──────────────────────────────────────────────────────────

def _clean(model: str) -> str:
    """Strip provider prefix."""
    return model.split("/")[-1] if "/" in model else model


def _wilcoxon_csv_from_comparison(comparison_csv: str) -> str:
    """Derive the pre-computed Wilcoxon CSV path from a comparison CSV path."""
    d = os.path.dirname(comparison_csv)
    base = os.path.basename(comparison_csv)
    return os.path.join(d, base.replace("comparison_", "wilcoxon_"))


def _per_model_wilcoxon(comparison_csv: str, model: str, range_type: str = "tight"):
    """
    Look up pre-computed Wilcoxon signed-rank result for one model.
    Returns (p_value, median_diff, n_diffs).

    Reads from a companion ``wilcoxon_*.csv`` produced by
    ``analysis.per_model_wilcoxon.compute_per_model_wilcoxon``.
    """
    wilcoxon_csv = _wilcoxon_csv_from_comparison(comparison_csv)
    if not os.path.exists(wilcoxon_csv):
        return None, None, 0
    return lookup_per_model_p(wilcoxon_csv, model, range_type)


def _overall_wilcoxon(comparison_csv: str, range_type: str = "tight"):
    """Look up pre-computed pooled Wilcoxon signed-rank result."""
    wilcoxon_csv = _wilcoxon_csv_from_comparison(comparison_csv)
    if not os.path.exists(wilcoxon_csv):
        return None, None, 0
    return lookup_pooled_p(wilcoxon_csv, range_type)


def _significance_star(p):
    """Return significance annotation: ***, **, *, or ns."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def _star_label(p):
    """Return a star + p-value annotation string."""
    if p is None:
        return ""
    stars = _significance_star(p)
    if not stars:
        return "n.s."
    return f"{stars} (p={p:.1e})"


def _draw_bracket(p, cn, left_label, right_label, bracket_y, stars,
                  cond_labels=None, cap_h=0.015):
    """Draw a U-shaped significance bracket between two bars.

    Parameters
    ----------
    p : figure
        Bokeh figure with FactorRange x-axis.
    cn : str
        Clean model name (first element of the factor tuple).
    left_label, right_label : str
        Second-element factor labels for the left and right bars.
    bracket_y : float
        Y position of the horizontal bracket line.
    stars : str
        Significance text (e.g. "***", "ns").
    cond_labels : list[str] | None
        Full list of condition labels in order.  Used to compute the
        true centre of the bracket for the significance text.
    cap_h : float
        Height of the vertical cap lines.
    """
    x_left = (cn, left_label)
    x_right = (cn, right_label)
    # Horizontal line
    p.line(x=[x_left, x_right], y=[bracket_y, bracket_y],
           line_color="#555", line_width=0.8)
    # Vertical caps
    p.line(x=[x_left, x_left], y=[bracket_y - cap_h, bracket_y],
           line_color="#555", line_width=0.8)
    p.line(x=[x_right, x_right], y=[bracket_y - cap_h, bracket_y],
           line_color="#555", line_width=0.8)
    # Label — use p.text() glyph (not Label annotation) because
    # Label.x does not accept FactorRange tuples in BokehJS.
    # Centre the annotation over the midpoint of the bracket.
    star_color = "#555" if stars == "ns" else "#333"
    star_size = "6pt" if stars == "ns" else "7pt"
    x_off = 0
    if cond_labels:
        li = cond_labels.index(left_label)
        ri = cond_labels.index(right_label)
        mid = (li + ri) / 2.0
        base_idx = int(mid)
        text_x = (cn, cond_labels[base_idx])
        frac = mid - base_idx
        if frac > 0.01:
            n_total = len(p.x_range.factors)
            px_per_factor = (p.width - 120) / max(n_total, 1)
            x_off = int(frac * px_per_factor)
    else:
        text_x = (cn, left_label)
    p.text(
        x=[text_x], y=[bracket_y + 0.005],
        text=[stars], text_font_size=star_size,
        text_align="center", text_baseline="bottom",
        text_color=star_color,
        x_offset=[x_off],
    )


# ── Naive performance horizontal bars ────────────────────────────────

def _get_pairwise_significance(ranking_csv: str):
    """
    Load the pairwise CMH test results that accompany a ranking CSV.
    Returns a dict mapping (model_a, model_b) -> min Bonferroni-corrected p-value.
    """
    pairs_csv = ranking_csv.replace(".csv", "_pairs.csv")
    if not os.path.exists(pairs_csv):
        return {}
    pdf = pd.read_csv(pairs_csv)
    result = {}
    for _, row in pdf.iterrows():
        # Use the minimum of tight and broad corrected p-values
        p_tight = row.get("tight_p_bonf", np.nan)
        p_broad = row.get("broad_p_bonf", np.nan)
        p_min = np.nanmin([p_tight, p_broad])
        result[(row["model_a"], row["model_b"])] = p_min
        result[(row["model_b"], row["model_a"])] = p_min
    return result


def naive_performance_bars(
    ranking_csv: str,
    comparison_csv: Optional[str] = None,
    title: str = "Naive Performance",
    width: int = 900,
    height: int = 500,
):
    """
    Horizontal bar chart of naive performance per model, sorted by rank.
    Bars show tight (solid) + broad (translucent) proportions.
    Adjacent-model significance brackets from CMH pairwise tests are shown.
    """
    df = pd.read_csv(ranking_csv).sort_values("adjusted_rank")
    df = df[df["model"].isin(_CONSISTENT_SET)]
    models = df["model"].tolist()
    ranks = df["adjusted_rank"].tolist()
    clean_names = [f"({int(r)}) {_clean(m)}" for m, r in zip(models, ranks)]
    colors = [MODEL_COLOR_DICT.get(m, "#888888") for m in models]

    tight_col = "mean_tight_prop"
    broad_col = "mean_broad_prop"
    tights = df[tight_col].tolist()
    broads = df[broad_col].tolist()

    # Reverse for horizontal bars (bottom to top, best at top)
    clean_names = list(reversed(clean_names))
    colors = list(reversed(colors))
    tights = list(reversed(tights))
    broads = list(reversed(broads))
    models_rev = list(reversed(models))

    source = ColumnDataSource(dict(
        y=clean_names,
        tight=tights,
        broad=broads,
        color=colors,
    ))

    # Dynamic x_max based on actual bar values, with room for brackets
    bar_max = max(max(broads), max(tights)) if broads else 1.0
    x_max = bar_max + 0.18
    p = figure(
        width=width, height=height,
        y_range=clean_names,
        x_range=(0, x_max),
        toolbar_location=None,
        title=title,
    )

    tight_bars = p.hbar(y="y", right="tight", height=0.7, color="color",
                        alpha=0.9, source=source)
    broad_bars = p.hbar(y="y", right="broad", height=0.7, color="color",
                        alpha=0.25, source=source)

    p.add_tools(HoverTool(
        tooltips=[("Model", "@y"),
                  (f"Tight ({TIGHT_RANGE_LOWER:.0%}–{TIGHT_RANGE_UPPER:.0%})", "@tight{0.0%}"),
                  (f"Broad ({BROAD_RANGE_LOWER:.0%}–{BROAD_RANGE_UPPER:.0%})", "@broad{0.0%}")],
        renderers=[tight_bars, broad_bars],
    ))

    p.xaxis.axis_label = "Mean Proportion within Range"
    p.xaxis.ticker = FixedTicker(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.yaxis.major_label_text_font_size = "8pt"
    p.ygrid.visible = False

    # ── Adjacent-model significance brackets ─────────────────────────
    pairwise = _get_pairwise_significance(ranking_csv)
    n = len(models_rev)
    bracket_base = bar_max + 0.02  # just right of the longest bar
    x_step = 0.03

    for i in range(n - 1):
        model_top = models_rev[i + 1]
        model_bot = models_rev[i]
        p_val = pairwise.get((model_top, model_bot), np.nan)

        if np.isnan(p_val):
            continue

        stars = _significance_star(p_val)
        if not stars:
            continue

        # Position bracket near the longer bar of the pair
        pair_max = max(broads[i], broads[i + 1]) if i + 1 < len(broads) else broads[i]
        bx = max(pair_max + 0.02, bracket_base) + (i % 3) * x_step

        # Use categorical y-axis names for proper positioning
        y_bot_name = clean_names[i]
        y_top_name = clean_names[i + 1]

        # Draw bracket as annotations connected to categorical positions
        # Horizontal cap at bottom bar
        cap_len = 0.01
        p.line(x=[bx - cap_len, bx], y=[y_bot_name, y_bot_name],
               line_color="#333", line_width=1.0)
        # Horizontal cap at top bar
        p.line(x=[bx - cap_len, bx], y=[y_top_name, y_top_name],
               line_color="#333", line_width=1.0)
        # Vertical line connecting them
        p.line(x=[bx, bx], y=[y_bot_name, y_top_name],
               line_color="#333", line_width=1.0)

        # Star label at midpoint between the two bars
        mid_y = (i + i + 1) / 2
        p.add_layout(Label(
            x=bx + 0.005, y=mid_y,
            text=stars, text_font_size="9pt",
            text_baseline="middle", text_align="left",
            text_color="#333",
        ))

    # Legend — outside right, aligned with x-axis
    ls = ColumnDataSource(dict(y=[clean_names[0]], right=[0]))
    d1 = p.hbar(y="y", right="right", height=0.2, color="black", alpha=0.9, source=ls, visible=True)
    d2 = p.hbar(y="y", right="right", height=0.2, color="black", alpha=0.25, source=ls, visible=True)
    legend = Legend(items=[
        LegendItem(label=f"Tight ({TIGHT_RANGE_LOWER:.2f}–{TIGHT_RANGE_UPPER:.2f}x)", renderers=[d1]),
        LegendItem(label=f"Broad ({BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)", renderers=[d2]),
    ])
    p.add_layout(legend, "right")
    p.legend.label_text_font_size = "8pt"

    return p


# ── Error-profile stacked bars ──────────────────────────────────────

def error_profile_bars(
    model_csv: str,
    cv_col: str = "observed_coefficient_cv",
    ratio_num_col: str = "observed_coefficient_mean",
    ratio_den_col: str = "expected_coefficient",
    slope_col: str = "regression_slope",
    slope_p_col: str = "regression_slope_pvalue",
    slope_alpha: float = 0.05,
    naive_filter: Optional[dict] = None,
    width: int = 800,
    height: int = 380,
    cv_threshold: float = 0.2,
    title: str = "Performance Profile by Model",
    profile_csv: str | None = None,
):
    """
    Stacked horizontal bar chart classifying each model's pair-level
    results into five categories (left to right):

      1. Correct  (tight accuracy, any CV)            — full opacity, model colour
      2. Precise but wrong  (CV ≤ threshold, not tight, not dose-correlated) — medium opacity
      3. Other wrong  (not precise, not tight, not dose-correlated) — white space
      4. Dose-correlated (neg.)  (sig. negative slope) — hatched, right-most
      5. Dose-correlated (pos.)  (sig. positive slope) — hatched, lighter, right-most

    Reads pre-computed proportions from ``profile_csv`` (produced by
    ``analysis.performance_profile.compute_random_dose_profile``).
    Falls back to inline computation if CSV is not provided.
    """
    # Read pre-computed profile or compute on the fly
    if profile_csv and os.path.exists(profile_csv):
        pf = pd.read_csv(profile_csv)
        pf = pf[pf["model"].isin(_CONSISTENT_SET)]
    else:
        from analysis.performance_profile import compute_random_dose_profile
        pf = compute_random_dose_profile(
            model_csv, cv_col=cv_col, ratio_num_col=ratio_num_col,
            ratio_den_col=ratio_den_col, slope_col=slope_col,
            slope_p_col=slope_p_col, slope_alpha=slope_alpha,
            cv_threshold=cv_threshold, naive_filter=naive_filter,
        )

    ranked = get_ranked_model_order()
    models = [m for m in ranked if m in pf["model"].unique()]
    if not models:
        models = sorted(pf["model"].unique())

    cat_order = ["correct", "precise_wrong", "other_wrong", "dose_dep_neg", "dose_dep_pos"]

    # Build figure — models on y-axis (best at top)
    y_labels = [_clean(m) for m in models]
    p = figure(
        width=width, height=height,
        y_range=list(reversed(y_labels)),
        x_range=(0, 1),
        toolbar_location=None,
        title=title,
    )

    bar_h = 0.65
    alpha_map = {
        "correct": 0.9,
        "precise_wrong": 0.35,
        "dose_dep_neg": 0.7,
        "dose_dep_pos": 0.35,
    }
    skip_cats = {"other_wrong"}

    for i, m in enumerate(models):
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        row = pf[pf["model"] == m].iloc[0]

        left = 0.0
        for cat in cat_order:
            w = row[cat]
            if w < 0.001:
                left += w
                continue
            right = left + w

            if cat in skip_cats:
                # Leave as white space — inferable from gap
                left = right
                continue

            if cat in ("dose_dep_neg", "dose_dep_pos"):
                # Stack dose-correlated at the RIGHT end
                # Negative slope first from right, positive next
                dd_neg_w = row["dose_dep_neg"]
                dd_pos_w = row["dose_dep_pos"]
                if cat == "dose_dep_neg":
                    dd_right = 1.0
                    dd_left = 1.0 - dd_neg_w
                else:
                    dd_right = 1.0 - dd_neg_w
                    dd_left = dd_right - dd_pos_w
                if dd_left >= dd_right - 0.001:
                    left = right
                    continue
                hatch = "/" if cat == "dose_dep_neg" else "\\"
                p.hbar(
                    y=[cn], left=[dd_left], right=[dd_right], height=bar_h,
                    fill_color=color, fill_alpha=alpha_map[cat],
                    line_color=color, line_alpha=0.6,
                    hatch_pattern=hatch, hatch_color="white",
                    hatch_alpha=0.7, hatch_scale=8, hatch_weight=1.5,
                )
            else:
                p.hbar(
                    y=[cn], left=[left], right=[right], height=bar_h,
                    fill_color=color, fill_alpha=alpha_map[cat],
                    line_color=color, line_alpha=0.3,
                )
            left = right

    # x-axis as percentage
    from bokeh.models import NumeralTickFormatter
    p.xaxis.axis_label = "Proportion of opioid pair conversions"
    p.xaxis.formatter = NumeralTickFormatter(format="0%")
    p.yaxis.axis_label = None

    # Legend — manual items using invisible dummy glyphs
    legend_items = []
    dummy_src = ColumnDataSource(dict(x=[-10], y=[-10]))
    legend_cats = [
        ("correct", "Correct", 0.9, None),
        ("precise_wrong", "Precise but wrong", 0.35, None),
        ("other_wrong", "Imprecise & wrong (white space)", 0.0, None),
        ("dose_dep_neg", "Dose-correlated (neg. slope)", 0.7, "/"),
        ("dose_dep_pos", "Dose-correlated (pos. slope)", 0.35, "\\"),
    ]
    for cat, label, alpha, hatch in legend_cats:
        if cat == "other_wrong":
            kw = dict(
                x="x", y="y", width=0.001, height=0.001,
                fill_color="white", fill_alpha=1.0,
                line_color="#aaaaaa", line_alpha=0.6,
            )
        else:
            kw = dict(
                x="x", y="y", width=0.001, height=0.001,
                fill_color="#666666", fill_alpha=alpha,
                line_color="#666666", line_alpha=0.3,
            )
            if hatch:
                kw["hatch_pattern"] = hatch
                kw["hatch_color"] = "white"
                kw["hatch_alpha"] = 0.7
                kw["hatch_scale"] = 8
                kw["hatch_weight"] = 1.5
        r = p.rect(source=dummy_src, **kw)
        legend_items.append(LegendItem(label=label, renderers=[r]))

    legend = Legend(items=legend_items, location="bottom_center",
                    orientation="horizontal", label_text_font_size="8pt")
    p.add_layout(legend, "below")

    p.title.text_font_size = "11pt"
    p.xgrid.visible = False
    p.ygrid.visible = False

    return p


def vignette_performance_profile(
    model_csv: str,
    cv_col: str = "observed_dose_cv",
    ratio_num_col: str = "observed_dose_mean",
    ratio_den_col: str = "expected_to_dose",
    naive_filter: Optional[dict] = None,
    width: int = 800,
    height: int = 380,
    title: str = "Performance Profile by Model — Vignettes",
    profile_csv: str | None = None,
):
    """
    Stacked horizontal bar chart classifying each model's scenario-level
    vignette results into three categories:

      1. Correct   — tight accuracy (ratio within 0.85–1.15)
      2. Precise but incorrect — CV=0 but NOT tight accuracy
      3. Imprecise — everything else (CV > 0 and not tight)

    Reads pre-computed proportions from ``profile_csv`` (produced by
    ``analysis.performance_profile.compute_vignette_profile``).
    Falls back to inline computation if CSV is not provided.
    """
    # Read pre-computed profile or compute on the fly
    if profile_csv and os.path.exists(profile_csv):
        pf = pd.read_csv(profile_csv)
        pf = pf[pf["model"].isin(_CONSISTENT_SET)]
    else:
        from analysis.performance_profile import compute_vignette_profile
        pf = compute_vignette_profile(
            model_csv, cv_col=cv_col, ratio_num_col=ratio_num_col,
            ratio_den_col=ratio_den_col, naive_filter=naive_filter,
        )

    ranked = get_ranked_model_order()
    models = [m for m in ranked if m in pf["model"].unique()]
    if not models:
        models = sorted(pf["model"].unique())

    cat_order = ["correct", "precise_wrong", "imprecise"]

    y_labels = [_clean(m) for m in models]
    p = figure(
        width=width, height=height,
        y_range=list(reversed(y_labels)),
        x_range=(0, 1),
        toolbar_location=None,
        title=title,
    )

    bar_h = 0.65
    alpha_map = {
        "correct": 0.9,
        "precise_wrong": 0.45,
        "imprecise": 0.15,
    }

    for i, m in enumerate(models):
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        row = pf[pf["model"] == m].iloc[0]

        left = 0.0
        for cat in cat_order:
            w = row[cat]
            if w < 0.001:
                left += w
                continue
            right = left + w
            p.hbar(
                y=[cn], left=[left], right=[right], height=bar_h,
                fill_color=color, fill_alpha=alpha_map[cat],
                line_color=color, line_alpha=0.3,
            )
            left = right

    from bokeh.models import NumeralTickFormatter
    p.xaxis.axis_label = "Proportion of vignette scenarios"
    p.xaxis.formatter = NumeralTickFormatter(format="0%")
    p.yaxis.axis_label = None

    # Legend
    legend_items = []
    dummy_src = ColumnDataSource(dict(x=[-10], y=[-10]))
    legend_descs = [
        ("correct", "Correct (tight accuracy)", 0.9),
        ("precise_wrong", "Precise but incorrect (CV=0, outside tight)", 0.45),
        ("imprecise", "Imprecise (CV>0, not tight)", 0.15),
    ]
    for cat, label, alpha in legend_descs:
        kw = dict(
            x="x", y="y", width=0.001, height=0.001,
            fill_color="#666666", fill_alpha=alpha,
            line_color="#666666", line_alpha=0.3,
        )
        r = p.rect(source=dummy_src, **kw)
        legend_items.append(LegendItem(label=label, renderers=[r]))

    legend = Legend(items=legend_items, location="bottom_center",
                    orientation="horizontal", label_text_font_size="8pt")
    p.add_layout(legend, "below")

    p.title.text_font_size = "11pt"
    p.xgrid.visible = False
    p.ygrid.visible = False

    return p


# ── Coefficient of Variation with mean-ratio scatter ─────────────────

def cv_strip_with_ratio(
    model_csv: str,
    cv_col: str = "observed_coefficient_cv",
    ratio_num_col: str = "observed_coefficient_mean",
    ratio_den_col: str = "expected_coefficient",
    pair_col: str = "opioid_pair",
    naive_filter: Optional[dict] = None,
    title: str = "CV by Model",
    width: int = 800,
    height: int = 1000,
    slope_col: str = "regression_slope",
    slope_p_col: str = "regression_slope_pvalue",
    slope_alpha: float = 0.05,
    tight_col: str = "tight(0.85,1.15)",
    broad_col: str = "broad(0.6,1.7)",
    cv_bins: list | None = None,
    cv_bin_labels: list | None = None,
    chi2_csv: str | None = None,
):
    """
    Three-panel figure:
    Top: strip chart of CV (same as cv_strip_chart).
    Middle: scatter of CV (x) vs mean observed/expected ratio (y) on log scale.
    Bottom: stacked bar mosaic — accuracy bands × CV bins.

    ``chi2_csv`` is the pre-computed per-model chi-square CSV from
    ``analysis.variance_icc_stats.compute_cv_accuracy_chi2``.
    """
    panel_h = height // 3

    # -- top panel: the existing strip chart --
    top = cv_strip_chart(
        model_csv=model_csv, cv_col=cv_col, pair_col=pair_col,
        naive_filter=naive_filter, title=title,
        width=width, height=panel_h,
        slope_col=slope_col, slope_p_col=slope_p_col,
        slope_alpha=slope_alpha,
    )

    # -- load and filter data for middle and bottom panels --
    if naive_filter is None:
        naive_filter = {"include_nielsen": False, "include_examples": False}

    df = pd.read_csv(model_csv)
    for col, val in naive_filter.items():
        if col in df.columns:
            df = df[df[col] == val]
    df = df[df["model"].isin(_CONSISTENT_SET)]

    # -- middle panel: scatter CV vs mean ratio (log y-axis) --
    has_ratio = (cv_col in df.columns and ratio_num_col in df.columns
                 and ratio_den_col in df.columns)
    if has_ratio:
        rdf = df.dropna(subset=[cv_col, ratio_num_col, ratio_den_col]).copy()
        rdf["_ratio"] = rdf[ratio_num_col] / rdf[ratio_den_col]
        rdf = rdf[rdf["_ratio"].between(0.05, 20)]  # filter extreme outliers

        ranked = get_ranked_model_order()
        models = [m for m in ranked if m in rdf["model"].unique()]
        if not models:
            models = sorted(rdf["model"].unique())

        xs, ys, colors, pair_labels, model_labels = [], [], [], [], []
        for m in models:
            mdf = rdf[rdf["model"] == m]
            for _, row in mdf.iterrows():
                xs.append(row[cv_col])
                ys.append(row["_ratio"])
                colors.append(MODEL_COLOR_DICT.get(m, "#888888"))
                pair_labels.append(str(row[pair_col]))
                model_labels.append(_clean(m))

        mid = figure(
            width=width, height=panel_h,
            toolbar_location=None,
            title="Mean Observed / Expected Ratio vs CV",
            y_axis_type="log",
        )
        src = ColumnDataSource(dict(x=xs, y=ys, color=colors,
                                    pair=pair_labels, model=model_labels))
        mid.scatter("x", "y", color="color", alpha=0.5, size=6, source=src)
        mid.add_tools(HoverTool(tooltips=[("Model", "@model"), ("Pair", "@pair"),
                                          ("CV", "@x{0.00}"), ("Ratio", "@y{0.00}")]))

        # Custom log ticks and labels
        mid.yaxis.ticker = FixedTicker(ticks=[0.1, 0.2, 0.5, 1, 2, 5])
        mid.y_range.start = 0.08
        mid.y_range.end = 7

        # Threshold dotted lines at accuracy boundaries
        for thresh, c in [(BROAD_RANGE_LOWER, "#d62728"),
                          (TIGHT_RANGE_LOWER, "#2ca02c"),
                          (TIGHT_RANGE_UPPER, "#2ca02c"),
                          (BROAD_RANGE_UPPER, "#d62728")]:
            mid.add_layout(Span(location=thresh, dimension="width",
                                line_dash="dotted", line_color=c, line_alpha=0.6))

        # Solid reference line at 1.0
        mid.add_layout(Span(location=1.0, dimension="width",
                            line_dash="dashed", line_color="black", line_alpha=0.4))

        mid.xaxis.axis_label = "Coefficient of Variation"
        mid.yaxis.axis_label = "Mean Observed / Expected"
    else:
        mid = figure(width=width, height=panel_h, toolbar_location=None,
                     title="Mean Ratio vs CV (missing columns)")

    # -- bottom panel: accuracy mosaic by CV bin --
    bot = _cv_accuracy_mosaic(df, cv_col, tight_col, broad_col,
                              width=width, height=panel_h,
                              cv_bins=cv_bins, cv_bin_labels=cv_bin_labels,
                              chi2_csv=chi2_csv)

    return column(top, mid, bot)


def _cv_accuracy_mosaic(df, cv_col, tight_col, broad_col,
                        width=800, height=500,
                        cv_bins=None, cv_bin_labels=None,
                        chi2_csv=None):
    """
    Per-model 3×3 circle grid: rows = accuracy categories (tight, broad,
    neither), columns = CV bins.  Circle size ∝ proportion of that model's
    observations in each cell.  Models are laid out in a 5-4-4 grid.

    Each sub-plot shows the χ² p-value in its title.
    An x-axis label ("CV") is placed on the left-most figure of each row.

    Chi-square p-values are read from a pre-computed CSV (``chi2_csv``)
    produced by ``analysis.variance_icc_stats.compute_cv_accuracy_chi2``.
    """
    cols_needed = [c for c in [cv_col, tight_col, broad_col] if c in df.columns]
    if len(cols_needed) < 3:
        return figure(width=width, height=height // 3, toolbar_location=None,
                      title="Accuracy by CV Bin (missing columns)")

    sub = df.dropna(subset=cols_needed).copy()
    sub = sub[sub["model"].isin(_CONSISTENT_SET)]
    if sub.empty:
        return figure(width=width, height=height // 3, toolbar_location=None,
                      title="Accuracy by CV Bin (no data)")

    if cv_bins is None:
        cv_bins = [0, 0.1, 0.5, float("inf")]
    if cv_bin_labels is None:
        cv_bin_labels = ["<0.1", "0.1–0.5", "≥0.5"]

    sub["_cv_bin"] = pd.cut(sub[cv_col], bins=cv_bins, labels=cv_bin_labels, right=False)

    sub["_acc"] = "neither"
    sub.loc[sub[broad_col] >= 0.5, "_acc"] = "broad"
    sub.loc[sub[tight_col] >= 0.5, "_acc"] = "tight"

    acc_labels = ["tight", "broad", "neither"]

    ranked = get_ranked_model_order()
    models = [m for m in ranked if m in sub["model"].unique()]
    if not models:
        models = sorted(sub["model"].unique())

    # Layout: rows of [5, 4, 4] (or adapt to actual count)
    n = len(models)
    if n <= 5:
        row_sizes = [n]
    elif n <= 9:
        row_sizes = [5, n - 5]
    else:
        row_sizes = [5, 4, n - 9] if n <= 13 else [5, 5, n - 10]

    max_circle = 28
    cell_w = width // 5
    cell_h = height // len(row_sizes)

    # Read pre-computed chi-square p-values per model
    model_pvals: dict[str, float] = {}
    if chi2_csv and os.path.exists(chi2_csv):
        chi2_df = pd.read_csv(chi2_csv)
        model_pvals = dict(zip(chi2_df["model"], chi2_df["p_value"]))

    all_rows = []
    idx = 0
    for row_sz in row_sizes:
        row_figs = []
        for fig_in_row in range(row_sz):
            if idx >= n:
                break
            m = models[idx]
            idx += 1
            cn = _clean(m)
            color = MODEL_COLOR_DICT.get(m, "#888888")

            mdf = sub[sub["model"] == m]
            n_total = len(mdf)

            xs, ys, sizes, alphas, pcts = [], [], [], [], []
            for bl in cv_bin_labels:
                bdf = mdf[mdf["_cv_bin"] == bl]
                for al in acc_labels:
                    count = len(bdf[bdf["_acc"] == al])
                    prop = count / n_total if n_total else 0
                    xs.append(bl)
                    ys.append(al)
                    sizes.append(max(3, prop ** 0.5 * max_circle))
                    alphas.append(0.8 if prop > 0 else 0.05)
                    pcts.append(f"{prop:.0%}")

            pval = model_pvals.get(m, np.nan)
            if np.isnan(pval):
                star = " χ² NA"
            elif pval < 0.001:
                star = " χ² ***"
            elif pval < 0.01:
                star = " χ² **"
            elif pval < 0.05:
                star = " χ² *"
            else:
                star = " χ² ns"

            # Shorten long model names for grid cells
            import re as _re
            short = _re.sub(r'[:]\d+$', '', cn)
            short = _re.sub(r'-\d{2}-\d{4}$', '', short)
            short = _re.sub(r'-\d{4}$', '', short)

            p = figure(
                width=cell_w, height=cell_h,
                x_range=cv_bin_labels, y_range=list(reversed(acc_labels)),
                toolbar_location=None,
                title=f"{short}{star}",
            )
            src = ColumnDataSource(dict(
                x=xs, y=ys, size=sizes, alpha=alphas, pct=pcts,
            ))
            p.scatter("x", "y", size="size", color=color, alpha="alpha",
                      source=src)
            p.add_tools(HoverTool(tooltips=[
                ("CV Bin", "@x"), ("Accuracy", "@y"), ("Proportion", "@pct"),
            ]))

            p.title.text_font_size = "6pt"
            p.xaxis.major_label_text_font_size = "6pt"
            p.yaxis.major_label_text_font_size = "6pt"
            p.xaxis.axis_label = None
            p.yaxis.axis_label = None
            p.xgrid.visible = False
            p.ygrid.visible = False
            p.min_border_left = 40
            p.min_border_right = 5
            p.min_border_top = 20
            p.min_border_bottom = 20

            row_figs.append(p)

        # Pad with None to fill to max row width for gridplot
        while len(row_figs) < row_sizes[0]:
            row_figs.append(None)
        all_rows.append(row_figs)

    grid = gridplot(all_rows, merge_tools=False, toolbar_location=None)
    title_div = Div(
        text=("<b>Accuracy Distribution by CV Bin</b>"
              "<br><span style='font-size:9pt; color:#666'>x-axis: coefficient of variation &nbsp;|&nbsp; "
              "circle area ∝ proportion of model's observations in each cell</span>"),
        styles={"font-size": "12pt", "margin-left": "10px"})
    return column(title_div, grid)

def cv_strip_chart(
    model_csv: str,
    cv_col: str = "observed_coefficient_cv",
    pair_col: str = "opioid_pair",
    naive_filter: Optional[dict] = None,
    title: str = "CV by Model",
    width: int = 800,
    height: int = 500,
    slope_col: str = "regression_slope",
    slope_p_col: str = "regression_slope_pvalue",
    slope_alpha: float = 0.05,
):
    """
    Horizontal strip chart: y-axis = model names (sorted by rank),
    x-axis = per-pair CV.  Each dot is one opioid pair's CV for that model.
    Dots for pairs with a statistically significant negative regression
    slope get a black outline.

    Parameters
    ----------
    naive_filter : dict, optional
        Column→value pairs used to select the naive (baseline) condition.
        Default: ``{"include_nielsen": False, "include_examples": False}``.
    slope_col / slope_p_col : str
        Columns for the regression slope and its p-value.  If absent in the
        CSV, slope highlighting is skipped.
    slope_alpha : float
        Significance threshold for outlining slope dots.
    """
    if naive_filter is None:
        naive_filter = {"include_nielsen": False, "include_examples": False}

    df = pd.read_csv(model_csv)

    # Apply naive filter
    for col, val in naive_filter.items():
        if col in df.columns:
            df = df[df[col] == val]
    df = df[df["model"].isin(_CONSISTENT_SET)]

    if cv_col not in df.columns or pair_col not in df.columns:
        return figure(width=width, height=height, title=f"{title} (missing columns)")

    df = df.dropna(subset=[cv_col])
    if df.empty:
        return figure(width=width, height=height, title=f"{title} (no data)")

    # Determine if slope highlighting is possible
    has_slope = slope_col in df.columns and slope_p_col in df.columns

    # Sort models by ranked order (best at top)
    ranked = get_ranked_model_order()
    models = [m for m in ranked if m in df["model"].unique()]
    if not models:
        models = sorted(df["model"].unique())
    models = list(reversed(models))  # best at top in horizontal charts

    clean_map = {m: _clean(m) for m in models}
    clean_names = [clean_map[m] for m in models]

    # Build data source
    xs, ys, colors, outlines, pair_labels = [], [], [], [], []
    for m in models:
        mdf = df[df["model"] == m]
        for _, row in mdf.iterrows():
            xs.append(row[cv_col])
            ys.append(clean_map[m])
            colors.append(MODEL_COLOR_DICT.get(m, "#888888"))
            pair_labels.append(str(row[pair_col]))
            if has_slope:
                slope_val = row.get(slope_col, None)
                p_val = row.get(slope_p_col, None)
                if (slope_val is not None and p_val is not None
                        and not np.isnan(slope_val) and not np.isnan(p_val)
                        and p_val < slope_alpha):
                    if slope_val < 0:
                        outlines.append("black")
                    else:
                        outlines.append("grey")
                else:
                    outlines.append(None)
            else:
                outlines.append(None)

    source = ColumnDataSource(dict(
        x=xs, y=ys, color=colors, outline=outlines, pair=pair_labels,
    ))

    p = figure(
        width=width, height=height,
        y_range=clean_names,
        toolbar_location=None,
        title=title,
    )

    # Dots without outline
    idx_no = [i for i, o in enumerate(outlines) if o is None]
    if idx_no:
        src_no = ColumnDataSource(dict(
            x=[xs[i] for i in idx_no],
            y=[ys[i] for i in idx_no],
            color=[colors[i] for i in idx_no],
            pair=[pair_labels[i] for i in idx_no],
        ))
        r1 = p.scatter("x", "y", color="color", alpha=0.6, size=7, source=src_no)

    # Dots with black outline (significant negative slope)
    idx_neg = [i for i, o in enumerate(outlines) if o == "black"]
    if idx_neg:
        src_neg = ColumnDataSource(dict(
            x=[xs[i] for i in idx_neg],
            y=[ys[i] for i in idx_neg],
            color=[colors[i] for i in idx_neg],
            pair=[pair_labels[i] for i in idx_neg],
        ))
        r2 = p.scatter("x", "y", color="color", alpha=0.8, size=8,
                        line_color="black", line_width=1.5, source=src_neg)

    # Dots with grey outline (significant positive slope)
    idx_pos = [i for i, o in enumerate(outlines) if o == "grey"]
    if idx_pos:
        src_pos = ColumnDataSource(dict(
            x=[xs[i] for i in idx_pos],
            y=[ys[i] for i in idx_pos],
            color=[colors[i] for i in idx_pos],
            pair=[pair_labels[i] for i in idx_pos],
        ))
        r3 = p.scatter("x", "y", color="color", alpha=0.8, size=8,
                        line_color="#888888", line_width=1.5, source=src_pos)

    p.add_tools(HoverTool(tooltips=[("Pair", "@pair"), ("CV", "@x{0.00}")]))

    p.xaxis.axis_label = "Coefficient of Variation"
    p.yaxis.major_label_text_font_size = "8pt"
    p.ygrid.visible = False

    # Median marker per model
    first_median = True
    for m in models:
        mdf = df[df["model"] == m]
        med = mdf[cv_col].median()
        cn = clean_map[m]
        kwargs = dict(x=[med], y=[cn], marker="diamond", size=10,
                      color="black", alpha=0.4)
        if first_median:
            kwargs["legend_label"] = "median"
            first_median = False
        p.scatter(**kwargs)

    # Update axis label to reflect slope highlighting
    slope_parts = []
    if idx_neg:
        slope_parts.append(f"black outline = sig. negative slope")
    if idx_pos:
        slope_parts.append(f"grey outline = sig. positive slope")
    if slope_parts:
        p.below[0].axis_label = (
            f"Coefficient of Variation    "
            f"({', '.join(slope_parts)}, p<{slope_alpha})"
        )

    p.legend.label_text_font_size = "8pt"
    p.add_layout(p.legend[0], "right")

    return p


# ── Nielsen effect comparison bars with significance ────────────────

def nielsen_effect_bars(
    batch_means_csv: str,
    comparison_csv: str,
    naive_batch: str = "all_n0",
    nielsen_batch: str = "all_n1",
    title: str = "Nielsen Effect",
    width: int = 900,
    height: int = 400,
):
    """
    Grouped bar chart: naive vs Nielsen per model, with Wilcoxon
    significance stars above each model's bars.
    """
    df = pd.read_csv(batch_means_csv)
    df = df[df["model"].isin(_CONSISTENT_SET)]
    naive = df[df["batch_name"] == naive_batch].set_index("model")
    nielsen = df[df["batch_name"] == nielsen_batch].set_index("model")

    ranked_models = get_ranked_model_order()
    models = [m for m in ranked_models if m in naive.index and m in nielsen.index]
    if not models:
        models = sorted(set(naive.index) & set(nielsen.index))

    clean_names = [_clean(m) for m in models]
    bar_labels = ["naive", "Nielsen"]

    x_factors = [(cn, bl) for cn in clean_names for bl in bar_labels]

    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [], "model": [], "batch": []}
    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for batch_name, label, batch_df in [(naive_batch, "naive", naive),
                                            (nielsen_batch, "Nielsen", nielsen)]:
            t = batch_df.loc[m, "mean_in_tight"]
            b = batch_df.loc[m, "mean_in_broad"]
            bar_data["x"].append((cn, label))
            bar_data["tight"].append(t)
            bar_data["broad_top"].append(b)
            bar_data["color"].append(color)
            bar_data["model"].append(cn)
            bar_data["batch"].append(label)

    source = ColumnDataSource(bar_data)
    p = figure(
        width=width, height=height,
        x_range=FactorRange(*x_factors),
        y_range=(0, 1.18),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.7, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.7, color="color", alpha=0.25, source=source)

    # Significance annotations (always show stars or ns)
    if os.path.exists(comparison_csv):
        star_x, star_y, star_text = [], [], []
        for m, cn in zip(models, clean_names):
            p_t, _, _ = _per_model_wilcoxon(comparison_csv, m, "tight")
            p_b, _, _ = _per_model_wilcoxon(comparison_csv, m, "broad")
            p_min = min(p for p in [p_t, p_b] if p is not None) if any(p is not None for p in [p_t, p_b]) else None
            s = _significance_star(p_min)
            if s:
                broad_max = max(
                    naive.loc[m, "mean_in_broad"],
                    nielsen.loc[m, "mean_in_broad"],
                )
                star_x.append((cn, "Nielsen"))
                star_y.append(broad_max + 0.03)
                star_text.append(s)

        if star_x:
            star_src = ColumnDataSource(dict(x=star_x, y=star_y, text=star_text))
            p.add_layout(LabelSet(
                x="x", y="y", text="text", source=star_src,
                text_font_size="9pt", text_align="center",
                text_baseline="bottom",
            ))

    _style_grouped_bar(p, clean_names, bar_labels)
    return p


# ── OME vs Non-OME comparison bars ──────────────────────────────────

def ome_comparison_bars(
    batch_means_csv: str,
    ome_batch: str = "ome_n0",
    non_ome_batch: str = "pairs_n0",
    comparison_csv: str | None = None,
    title: str = "OME vs Non-OME",
    width: int = 900,
    height: int = 400,
):
    """
    Grouped bar chart: OME vs non-OME per model for a given condition.
    If *comparison_csv* is provided (BatchComparison output), overall
    Wilcoxon significance annotations are added for tight and broad.
    """
    df = pd.read_csv(batch_means_csv)
    df = df[df["model"].isin(_CONSISTENT_SET)]
    ome_df = df[df["batch_name"] == ome_batch].set_index("model")
    non_ome_df = df[df["batch_name"] == non_ome_batch].set_index("model")

    ranked_models = get_ranked_model_order()
    models = [m for m in ranked_models if m in ome_df.index and m in non_ome_df.index]
    if not models:
        models = sorted(set(ome_df.index) & set(non_ome_df.index))

    clean_names = [_clean(m) for m in models]
    cond_labels = ["OME", "non-OME"]

    x_factors = [(cn, cl) for cn in clean_names for cl in cond_labels]
    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [], "model": [], "batch": []}

    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for label, bdf in [("OME", ome_df), ("non-OME", non_ome_df)]:
            t = bdf.loc[m, "mean_in_tight"]
            b = bdf.loc[m, "mean_in_broad"]
            bar_data["x"].append((cn, label))
            bar_data["tight"].append(t)
            bar_data["broad_top"].append(b)
            bar_data["color"].append(color)
            bar_data["model"].append(cn)
            bar_data["batch"].append(label)

    source = ColumnDataSource(bar_data)
    p = figure(
        width=width, height=height,
        x_range=FactorRange(*x_factors),
        y_range=(0, 1.15),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.7, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.7, color="color", alpha=0.25, source=source)

    # Per-model significance annotation from comparison CSV
    if comparison_csv and os.path.exists(comparison_csv):
        cdf = pd.read_csv(comparison_csv)
        star_x, star_y, star_text = [], [], []
        for m, cn in zip(models, clean_names):
            mrow = cdf[cdf["model"] == m]
            if mrow.empty:
                continue
            p_t = mrow.iloc[0].get("tight_mannwhitney_p", np.nan)
            p_b = mrow.iloc[0].get("broad_mannwhitney_p", np.nan)
            ps = [p for p in [p_t, p_b] if not np.isnan(p)]
            p_min = min(ps) if ps else None
            s = _significance_star(p_min)
            if s:
                ome_key = (cn, "OME")
                nonome_key = (cn, "non-OME")
                tops = []
                for key in (ome_key, nonome_key):
                    if key in bar_data["x"]:
                        pos = bar_data["x"].index(key)
                        tops.append(bar_data["broad_top"][pos])
                top = max(tops) if tops else 0.5
                star_x.append((cn, "non-OME"))
                star_y.append(top + 0.03)
                star_text.append(s)
        if star_x:
            star_src = ColumnDataSource(dict(x=star_x, y=star_y, text=star_text))
            p.add_layout(LabelSet(
                x="x", y="y", text="text", source=star_src,
                text_font_size="9pt", text_align="center",
                text_baseline="bottom",
            ))

    _style_grouped_bar(p, clean_names, cond_labels)
    return p


# ── Explain units effect bars with significance ─────────────────────

def explain_units_bars(
    pair_model_csv: str,
    explain_units_pair_csv: str,
    comparison_csv: str,
    dosing_comparison_csv: str | None = None,
    title: str = "Explain Units Effect",
    width: int = 1000,
    height: int = 450,
):
    """
    Three-condition bar chart computed directly from per-pair CSVs:
      1. mg/day + Nielsen   (pair_model_csv, OME + Nielsen, mg/day)
      2. mg/4h + Nielsen    (explain_units_pair_csv, explain_units=False)
      3. mg/4h + Nielsen + explained  (explain_units_pair_csv, explain_units=True)

    Two significance brackets per model (paired Wilcoxon on broad):
      - mg/day ↔ mg/4h  (dosing_comparison_csv)
      - mg/4h ↔ mg/4h explained  (comparison_csv)
    """
    tight_col = "tight(0.85,1.15)"
    broad_col = "broad(0.6,1.7)"

    # Condition 1: mg/day + Nielsen from main pair model analysis
    df1 = pd.read_csv(pair_model_csv)
    df1 = df1[(df1["ome"] == True) & (df1["include_nielsen"] == True)]
    df1 = df1[df1["model"].isin(_CONSISTENT_SET)]
    c1 = df1.groupby("model").agg(
        mean_in_tight=(tight_col, "mean"),
        mean_in_broad=(broad_col, "mean"),
    )

    # Conditions 2 & 3: mg/4h from explain_units CSV
    df2 = pd.read_csv(explain_units_pair_csv)
    df2 = df2[df2["model"].isin(_CONSISTENT_SET)]
    c2_raw = df2[df2["explain_units"] == False]
    c3_raw = df2[df2["explain_units"] == True]
    c2 = c2_raw.groupby("model").agg(
        mean_in_tight=(tight_col, "mean"),
        mean_in_broad=(broad_col, "mean"),
    )
    c3 = c3_raw.groupby("model").agg(
        mean_in_tight=(tight_col, "mean"),
        mean_in_broad=(broad_col, "mean"),
    )

    ranked_models = get_ranked_model_order()
    models = [m for m in ranked_models
              if m in c1.index and m in c2.index and m in c3.index]

    clean_names = [_clean(m) for m in models]
    cond_labels = ["mg/day", "mg/4h", "mg/4h explained"]

    x_factors = [(cn, cl) for cn in clean_names for cl in cond_labels]
    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [], "model": [], "batch": []}

    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for label, src_df in [("mg/day", c1), ("mg/4h", c2), ("mg/4h explained", c3)]:
            t = src_df.loc[m, "mean_in_tight"]
            b = src_df.loc[m, "mean_in_broad"]
            bar_data["x"].append((cn, label))
            bar_data["tight"].append(t)
            bar_data["broad_top"].append(b)
            bar_data["color"].append(color)
            bar_data["model"].append(cn)
            bar_data["batch"].append(label)

    source = ColumnDataSource(bar_data)
    p = figure(
        width=width, height=height,
        x_range=FactorRange(*x_factors),
        y_range=(0, 1.22),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.6, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.6, color="color", alpha=0.25, source=source)

    # Helper: get broad_top for a bar
    def _bar_top(cn, lbl):
        key = (cn, lbl)
        if key in bar_data["x"]:
            return bar_data["broad_top"][bar_data["x"].index(key)]
        return 0

    BRACKET_PAD = 0.025
    BRACKET_STEP = 0.055

    # Bracket definitions: (csv, left_label, right_label, level)
    # Level 2 for the second bracket so they don't share the mg/4h cap.
    bracket_defs = []
    if dosing_comparison_csv and os.path.exists(dosing_comparison_csv):
        bracket_defs.append((dosing_comparison_csv, "mg/day", "mg/4h", 1))
    if os.path.exists(comparison_csv):
        bracket_defs.append((comparison_csv, "mg/4h", "mg/4h explained", 2))

    for m, cn in zip(models, clean_names):
        model_max = max(_bar_top(cn, cl) for cl in cond_labels)
        for csv_path, left_lbl, right_lbl, level in bracket_defs:
            p_t, _, _ = _per_model_wilcoxon(csv_path, m, "tight")
            p_b, _, _ = _per_model_wilcoxon(csv_path, m, "broad")
            p_min = min(p for p in [p_t, p_b] if p is not None) if any(p is not None for p in [p_t, p_b]) else None
            stars = _significance_star(p_min)
            if stars:
                bracket_y = model_max + BRACKET_PAD + (level - 1) * BRACKET_STEP
                _draw_bracket(p, cn, left_lbl, right_lbl, bracket_y, stars,
                              cond_labels=cond_labels)

    _style_grouped_bar(p, clean_names, cond_labels)
    return p


# ── Full treatment (Nielsen + explain units) ────────────────────────

def full_treatment_bars(
    pair_model_csv: str,
    explain_units_pair_csv: str,
    batch_means_csv: str,
    comparison_nielsen_csv: str,
    comparison_units_csv: str,
    title: str = "Explaining Units Improves Non-Daily Dosing",
    width: int = 1100,
    height: int = 470,
):
    """
    Four-condition bar chart showing the full treatment pipeline for mg/4hr:
      1. naive OME (no reference, mg/day)
      2. mg/day + Nielsen
      3. mg/4h + Nielsen
      4. mg/4h + Nielsen + explained

    Two significance brackets per model (paired Wilcoxon on broad):
      - naive ↔ mg/day+N  (comparison_nielsen_csv)
      - mg/4h+N ↔ mg/4h+N+E  (comparison_units_csv)
    """
    tight_col = "tight(0.85,1.15)"
    broad_col = "broad(0.6,1.7)"

    # Condition 1: naive OME from batch means
    bm = pd.read_csv(batch_means_csv)
    bm = bm[bm["model"].isin(_CONSISTENT_SET)]
    naive_df = bm[bm["batch_name"] == "ome_n0"].set_index("model")

    # Condition 2: mg/day + Nielsen from pair model analysis
    df1 = pd.read_csv(pair_model_csv)
    df1 = df1[(df1["ome"] == True) & (df1["include_nielsen"] == True)]
    df1 = df1[df1["model"].isin(_CONSISTENT_SET)]
    daily_nielsen = df1.groupby("model").agg(
        mean_in_tight=(tight_col, "mean"),
        mean_in_broad=(broad_col, "mean"),
    )

    # Conditions 3 & 4: mg/4h from explain_units CSV
    df2 = pd.read_csv(explain_units_pair_csv)
    df2 = df2[df2["model"].isin(_CONSISTENT_SET)]
    fourh_nielsen = df2[df2["explain_units"] == False].groupby("model").agg(
        mean_in_tight=(tight_col, "mean"),
        mean_in_broad=(broad_col, "mean"),
    )
    fourh_explained = df2[df2["explain_units"] == True].groupby("model").agg(
        mean_in_tight=(tight_col, "mean"),
        mean_in_broad=(broad_col, "mean"),
    )

    ranked_models = get_ranked_model_order()
    available = set(naive_df.index) & set(daily_nielsen.index) & set(fourh_nielsen.index) & set(fourh_explained.index)
    models = [m for m in ranked_models if m in available]
    if not models:
        models = sorted(available)

    clean_names = [_clean(m) for m in models]
    cond_labels = ["naive", "mg/day+N", "mg/4h+N", "mg/4h+N+E"]

    conditions = [
        ("naive", naive_df),
        ("mg/day+N", daily_nielsen),
        ("mg/4h+N", fourh_nielsen),
        ("mg/4h+N+E", fourh_explained),
    ]

    x_factors = [(cn, cl) for cn in clean_names for cl in cond_labels]
    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [], "model": [], "batch": []}

    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for label, src_df in conditions:
            t = src_df.loc[m, "mean_in_tight"]
            b = src_df.loc[m, "mean_in_broad"]
            bar_data["x"].append((cn, label))
            bar_data["tight"].append(t)
            bar_data["broad_top"].append(b)
            bar_data["color"].append(color)
            bar_data["model"].append(cn)
            bar_data["batch"].append(label)

    source = ColumnDataSource(bar_data)
    p = figure(
        width=width, height=height,
        x_range=FactorRange(*x_factors),
        y_range=(0, 1.22),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.6, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.6, color="color", alpha=0.25, source=source)

    # Helper: get broad_top for a bar
    def _bar_top(cn, lbl):
        key = (cn, lbl)
        if key in bar_data["x"]:
            return bar_data["broad_top"][bar_data["x"].index(key)]
        return 0

    BRACKET_PAD = 0.025
    BRACKET_STEP = 0.055

    # Bracket definitions: (csv, left_label, right_label, level)
    bracket_defs = []
    if os.path.exists(comparison_nielsen_csv):
        bracket_defs.append((comparison_nielsen_csv, "naive", "mg/day+N", 1))
    if os.path.exists(comparison_units_csv):
        bracket_defs.append((comparison_units_csv, "mg/4h+N", "mg/4h+N+E", 2))

    for m, cn in zip(models, clean_names):
        model_max = max(_bar_top(cn, cl) for cl in cond_labels)
        for csv_path, left_lbl, right_lbl, level in bracket_defs:
            p_t, _, _ = _per_model_wilcoxon(csv_path, m, "tight")
            p_b, _, _ = _per_model_wilcoxon(csv_path, m, "broad")
            p_min = min(p for p in [p_t, p_b] if p is not None) if any(p is not None for p in [p_t, p_b]) else None
            stars = _significance_star(p_min)
            if stars:
                bracket_y = model_max + BRACKET_PAD + (level - 1) * BRACKET_STEP
                _draw_bracket(p, cn, left_lbl, right_lbl, bracket_y, stars,
                              cond_labels=cond_labels)

    _style_grouped_bar(p, clean_names, cond_labels)
    return p


# ── Cross-experiment: vignette vs random dose ───────────────────────

def cross_experiment_bars(
    matched_means_csv: str,
    matched_wilcoxon_csv: str,
    matched_spearman_csv: str | None = None,
    title: str = "Cross-Experiment Comparison (Matched Pairs)",
    width: int = 1100,
    height: int = 500,
):
    """
    Grouped bar chart: three bars per model — random dose (naive),
    empty vignette (naive), text vignette (naive).
    Each bar shows tight (solid) + broad (translucent).

    Three significance brackets per model (paired Wilcoxon on broad):
        Level 1 (lower):  RD ↔ Empty,  Empty ↔ Text
        Level 2 (higher): RD ↔ Text

    Reads from pre-computed CSVs produced by
    ``analysis.cross_experiment_matched.CrossExperimentMatched``.
    Only opioid pair conversions present in both experiments are included.
    """
    means_df = pd.read_csv(matched_means_csv)
    means_df = means_df[means_df["model"].isin(_CONSISTENT_SET)]
    wilc_df = pd.read_csv(matched_wilcoxon_csv)

    cond_map = {
        "random_dose": "random dose",
        "empty_vignette": "empty vignette",
        "text_vignette": "text vignette",
    }
    cond_labels = ["random dose", "empty vignette", "text vignette"]

    ranked_models = get_ranked_model_order()
    models = [m for m in ranked_models
              if m in set(means_df["model"])]
    clean_names = [_clean(m) for m in models]

    x_factors = [(cn, cl) for cn in clean_names for cl in cond_labels]
    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [],
                "model": [], "batch": []}

    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for cond_key, cond_label in cond_map.items():
            row = means_df[(means_df["model"] == m)
                           & (means_df["condition"] == cond_key)]
            t = row["mean_tight"].iloc[0] if len(row) else 0
            b = row["mean_broad"].iloc[0] if len(row) else 0
            bar_data["x"].append((cn, cond_label))
            bar_data["tight"].append(t)
            bar_data["broad_top"].append(b)
            bar_data["color"].append(color)
            bar_data["model"].append(cn)
            bar_data["batch"].append(cond_label)

    source = ColumnDataSource(bar_data)
    p = figure(
        width=width, height=height,
        x_range=FactorRange(*x_factors),
        y_range=(0, 1.28),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.7, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.7, color="color",
           alpha=0.25, source=source)

    # ── Per-model significance brackets from pre-computed Wilcoxon CSV ──
    # Helper: get broad_top for a (model_clean, cond_label) bar
    def _bar_top(cn, cond_label):
        key = (cn, cond_label)
        if key in bar_data["x"]:
            return bar_data["broad_top"][bar_data["x"].index(key)]
        return 0

    # Bracket definitions: (cond_a_key, cond_b_key, cond_a_label, cond_b_label, level)
    # Level 1 brackets sit just above bars; level 2 sits above level 1
    bracket_defs = [
        ("random_dose", "empty_vignette", "random dose", "empty vignette", 1),
        ("empty_vignette", "text_vignette", "empty vignette", "text vignette", 1),
        ("random_dose", "text_vignette", "random dose", "text vignette", 2),
    ]

    BRACKET_STEP = 0.055  # vertical gap between bracket levels
    BRACKET_PAD = 0.025   # gap above bar top to first bracket
    CAP_FRAC = 0.15       # fractional length of horizontal caps (of bar width)

    for m, cn in zip(models, clean_names):
        # Get the max bar top across all 3 conditions for this model
        model_max = max(_bar_top(cn, cl) for cl in cond_labels)

        for cond_a_key, cond_b_key, cond_a_lbl, cond_b_lbl, level in bracket_defs:
            # Look up p-value
            mrow = wilc_df[
                (wilc_df["model"] == m)
                & (wilc_df["condition_a"] == cond_a_key)
                & (wilc_df["condition_b"] == cond_b_key)
            ]
            if mrow.empty:
                continue
            p_val = mrow.iloc[0]["broad_wilcoxon_p"]
            if pd.isna(p_val):
                continue

            stars = _significance_star(p_val)
            if not stars:
                continue

            # Bracket y position based on level
            bracket_y = model_max + BRACKET_PAD + (level - 1) * BRACKET_STEP

            _draw_bracket(p, cn, cond_a_lbl, cond_b_lbl, bracket_y, stars,
                          cond_labels=cond_labels)

    _style_grouped_bar(p, clean_names, cond_labels)
    return p


# ── styling helper ──────────────────────────────────────────────────

def _style_grouped_bar(p, clean_names, bar_labels):
    """Apply common styling to grouped bar charts."""
    p.xaxis.group_text_font_size = "0pt"
    p.xaxis.major_label_orientation = 1.0
    p.xaxis.major_label_text_font_size = "7pt"
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.xaxis.separator_line_color = None
    p.xgrid.visible = False
    p.yaxis.axis_label = "Mean Proportion within Range"
    p.yaxis.ticker = FixedTicker(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Rotated model name labels
    label_source = ColumnDataSource(dict(
        x=[(name, bar_labels[0]) for name in clean_names],
        y=[0.6] * len(clean_names),
        text=clean_names,
    ))
    model_labels = LabelSet(
        x="x", y="y", text="text", source=label_source,
        text_font_size="7pt", text_align="left", text_baseline="bottom",
        angle=np.pi / 2, x_offset=5,
    )
    p.add_layout(model_labels)


# ── Master builder ──────────────────────────────────────────────────

class ComprehensiveFigureBuilder:
    """
    Generates all publication figures using unified model colours and
    performance-based sorting.
    """

    def __init__(self, results_dir: str = RESULTS_DIR, figures_dir: str = f"{FIGURES_DIR}/pub_figures"):
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)

    def _p(self, *parts):
        return os.path.join(self.results_dir, *parts)

    def _f(self, *parts):
        return os.path.join(self.figures_dir, *parts)

    # ── individual figure methods ────────────────────────────────

    def build_naive_vignette_bars(self):
        """Naive vignette performance horizontal bars."""
        return naive_performance_bars(
            ranking_csv=self._p("ranking_naive_vignettes.csv"),
            title="Naive Vignette Performance by Model",
        )

    def build_naive_random_dose_bars(self):
        """Naive random dose performance horizontal bars + vignette rank comparison scatter."""
        from scipy.stats import spearmanr as _spearmanr

        bar_fig = naive_performance_bars(
            ranking_csv=self._p("ranking_naive_random_doses.csv"),
            title="Naive Random Dose Performance by Model",
        )

        # ── Rank comparison scatter: vignette vs random-dose tight accuracy ──
        rv = pd.read_csv(self._p("ranking_naive_vignettes.csv"))
        rr = pd.read_csv(self._p("ranking_naive_random_doses.csv"))
        rv = rv[rv["model"].isin(_CONSISTENT_SET)]
        rr = rr[rr["model"].isin(_CONSISTENT_SET)]
        merged = rv[["model", "mean_tight_prop"]].merge(
            rr[["model", "mean_tight_prop"]], on="model", suffixes=("_vig", "_rd")
        )
        rho, p_val = _spearmanr(merged["mean_tight_prop_vig"], merged["mean_tight_prop_rd"])
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"

        merged["label"] = merged["model"].apply(_clean)
        merged["color"] = merged["model"].apply(lambda m: MODEL_COLOR_DICT.get(m, "#888888"))

        ax_max = max(merged["mean_tight_prop_vig"].max(), merged["mean_tight_prop_rd"].max()) + 0.05

        scatter = figure(
            width=bar_fig.width, height=450,
            x_range=(0, ax_max), y_range=(0, ax_max),
            toolbar_location=None,
            title="Vignette vs random-dose tight accuracy per model",
        )
        # Diagonal reference line (y = x)
        scatter.line([0, ax_max], [0, ax_max], line_color="#aaaaaa",
                     line_dash="dashed", line_width=1.2)

        src = ColumnDataSource(merged)
        scatter.circle(
            x="mean_tight_prop_vig", y="mean_tight_prop_rd",
            color="color", size=10, alpha=0.85, source=src,
        )
        scatter.add_layout(LabelSet(
            x="mean_tight_prop_vig", y="mean_tight_prop_rd",
            text="label", source=src,
            x_offset=5, y_offset=3,
            text_font_size="8pt", text_color="#333333",
        ))
        scatter.add_layout(Label(
            x=0.02, y=ax_max - 0.06,
            text=f"Spearman ρ = {rho:.2f}, {p_str}  (n = {len(merged)} models)",
            text_font_size="9pt", text_color="#333333",
        ))
        scatter.xaxis.axis_label = "Vignette tight accuracy"
        scatter.yaxis.axis_label = "Random-dose tight accuracy"

        return column(bar_fig, scatter)

    def build_cv_strip_vignettes(self):
        """CV strip chart + ratio scatter — vignettes (naive)."""
        csv = self._p("reworded_vignettes_model_analysis.csv")
        naive_filter = {"include_nielsen": False, "explain_units": False,
                        "has_vignette_text": True}

        # Ensure chi2 CSV exists
        chi2_csv = self._p("cv_accuracy_chi2.csv")
        if not os.path.exists(chi2_csv):
            from analysis.variance_icc_stats import compute_cv_accuracy_chi2
            compute_cv_accuracy_chi2(
                csv,
                cv_col="observed_dose_cv",
                tight_col="tight(0.85,1.15)",
                broad_col="broad(0.6,1.7)",
                cv_bins=[0, 1e-9, 0.2, float("inf")],
                cv_bin_labels=["CV=0", "0<CV≤0.2", "CV>0.2"],
                output_csv=chi2_csv,
                naive_filter=naive_filter,
            )

        return cv_strip_with_ratio(
            model_csv=csv,
            cv_col="observed_dose_cv",
            ratio_num_col="observed_dose_mean",
            ratio_den_col="expected_to_dose",
            pair_col="scenario_key",
            naive_filter=naive_filter,
            title="Naive CV by Model — Vignettes",
            cv_bins=[0, 1e-9, 0.2, float("inf")],
            cv_bin_labels=["CV=0", "0<CV≤0.2", "CV>0.2"],
            chi2_csv=chi2_csv,
        )

    def build_vignette_performance_profile(self):
        """Performance profile + accuracy mosaic — vignettes (naive)."""
        csv = self._p("reworded_vignettes_model_analysis.csv")
        naive_filter = {"include_nielsen": False, "explain_units": False,
                        "has_vignette_text": True}

        # Ensure profile CSV exists
        profile_csv = self._p("performance_profile_vignettes.csv")
        if not os.path.exists(profile_csv):
            from analysis.performance_profile import compute_vignette_profile
            compute_vignette_profile(csv, output_csv=profile_csv,
                                    naive_filter=naive_filter)

        profile = vignette_performance_profile(
            model_csv=csv,
            cv_col="observed_dose_cv",
            ratio_num_col="observed_dose_mean",
            ratio_den_col="expected_to_dose",
            naive_filter=naive_filter,
            title="Performance Profile by Model — Vignettes",
            profile_csv=profile_csv,
        )

        # Re-load data for mosaic
        df = pd.read_csv(csv)
        for col, val in naive_filter.items():
            if col in df.columns:
                df = df[df[col] == val]
        df = df[df["model"].isin(_CONSISTENT_SET)]

        mosaic = _cv_accuracy_mosaic(
            df, cv_col="observed_dose_cv",
            tight_col="tight(0.85,1.15)",
            broad_col="broad(0.6,1.7)",
            width=800, height=500,
            cv_bins=[0, 1e-9, 0.2, float("inf")],
            cv_bin_labels=["CV=0", "0<CV≤0.2", "CV>0.2"],
            chi2_csv=self._p("cv_accuracy_chi2.csv"),
        )

        return column(profile, mosaic)

    def build_cv_strip_random_doses(self):
        """CV strip chart + ratio scatter + error profile — random doses (naive)."""
        naive_filter = {"include_nielsen": False, "include_examples": False}
        csv = self._p("pair_model_analysis.csv")

        # Top two panels from the existing three-panel builder
        top = cv_strip_chart(
            model_csv=csv,
            cv_col="observed_coefficient_cv",
            pair_col="opioid_pair",
            naive_filter=naive_filter,
            title="Naive CV by Model — Random Doses",
            width=800, height=333,
        )

        # Middle: ratio scatter (re-use internals of cv_strip_with_ratio)
        df = pd.read_csv(csv)
        for col, val in naive_filter.items():
            if col in df.columns:
                df = df[df[col] == val]
        df = df[df["model"].isin(_CONSISTENT_SET)]

        rdf = df.dropna(subset=["observed_coefficient_cv",
                                "observed_coefficient_mean",
                                "expected_coefficient"]).copy()
        rdf["_ratio"] = rdf["observed_coefficient_mean"] / rdf["expected_coefficient"]
        rdf = rdf[rdf["_ratio"].between(0.05, 20)]

        ranked = get_ranked_model_order()
        models = [m for m in ranked if m in rdf["model"].unique()]
        if not models:
            models = sorted(rdf["model"].unique())

        xs, ys, colors, pair_labels, model_labels = [], [], [], [], []
        for m in models:
            mdf = rdf[rdf["model"] == m]
            for _, row in mdf.iterrows():
                xs.append(row["observed_coefficient_cv"])
                ys.append(row["_ratio"])
                colors.append(MODEL_COLOR_DICT.get(m, "#888888"))
                pair_labels.append(str(row["opioid_pair"]))
                model_labels.append(_clean(m))

        mid = figure(
            width=800, height=333,
            toolbar_location=None,
            title="Mean Observed / Expected Ratio vs CV",
            y_axis_type="log",
        )
        src = ColumnDataSource(dict(x=xs, y=ys, color=colors,
                                    pair=pair_labels, model=model_labels))
        mid.scatter("x", "y", color="color", alpha=0.5, size=6, source=src)
        mid.add_tools(HoverTool(tooltips=[("Model", "@model"), ("Pair", "@pair"),
                                          ("CV", "@x{0.00}"), ("Ratio", "@y{0.00}")]))
        mid.yaxis.ticker = FixedTicker(ticks=[0.1, 0.2, 0.5, 1, 2, 5])
        mid.y_range.start = 0.08
        mid.y_range.end = 7
        for thresh, c in [(BROAD_RANGE_LOWER, "#d62728"),
                          (TIGHT_RANGE_LOWER, "#2ca02c"),
                          (TIGHT_RANGE_UPPER, "#2ca02c"),
                          (BROAD_RANGE_UPPER, "#d62728")]:
            mid.add_layout(Span(location=thresh, dimension="width",
                                line_dash="dotted", line_color=c, line_alpha=0.6))
        mid.add_layout(Span(location=1.0, dimension="width",
                            line_dash="dashed", line_color="black", line_alpha=0.4))
        mid.xaxis.axis_label = "Coefficient of Variation"
        mid.yaxis.axis_label = "Mean Observed / Expected"

        # Bottom: error profile stacked bars
        profile_csv = self._p("performance_profile_random_doses.csv")
        if not os.path.exists(profile_csv):
            from analysis.performance_profile import compute_random_dose_profile
            compute_random_dose_profile(csv, output_csv=profile_csv,
                                        naive_filter=naive_filter)

        bot = error_profile_bars(
            model_csv=csv,
            naive_filter=naive_filter,
            width=800, height=380,
            title="Performance Profile by Model — Random Doses",
            profile_csv=profile_csv,
        )

        return column(top, mid, bot)

    def build_icc_heatmap(self):
        """ICC heatmap — model × scenario (naive condition)."""
        from visualize.icc_visualizer import icc_heatmap
        return icc_heatmap(self._p("icc_analysis.csv"))

    def build_variance_stacked(self):
        """Stacked variance bars + overall scatter."""
        from visualize.icc_visualizer import variance_stacked_figure
        return variance_stacked_figure(
            icc_csv=self._p("icc_analysis.csv"),
            vignette_model_csv=self._p("reworded_vignettes_model_analysis.csv"),
            spearman_csv=self._p("variance_icc_spearman.csv"),
        )

    def build_icc_bubble_grid(self):
        """ICC bubble grid sorted by accuracy."""
        from visualize.icc_visualizer import bubble_grid_figure
        return bubble_grid_figure(
            icc_csv=self._p("icc_analysis.csv"),
            vignette_model_csv=self._p("reworded_vignettes_model_analysis.csv"),
        )

    def build_spearman_variance_icc(self):
        """Per-model Spearman rho bar chart (variance vs ICC)."""
        from visualize.icc_visualizer import spearman_bars_figure
        return spearman_bars_figure(
            icc_csv=self._p("icc_analysis.csv"),
            vignette_model_csv=self._p("reworded_vignettes_model_analysis.csv"),
            spearman_csv=self._p("variance_icc_spearman.csv"),
        )

    def build_kendall_w_summary(self):
        """Kendall's W vs ICC summary bar chart (naive condition)."""
        from visualize.icc_visualizer import kendall_w_summary
        return kendall_w_summary(
            self._p("kendall_w_vignettes.csv"),
            icc_csv=self._p("icc_analysis.csv"),
        )

    def build_nielsen_effect_vignettes(self):
        """Nielsen effect — vignettes (all)."""
        return nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means_vignettes.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_vignettes.csv"),
            naive_batch="all_n0",
            nielsen_batch="all_n1",
            title="Nielsen Effect — Vignettes (All)",
        )

    def build_nielsen_effect_random_doses(self):
        """Nielsen effect — random doses (all)."""
        return nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_random_dose.csv"),
            naive_batch="all_n0",
            nielsen_batch="all_n1",
            title="Nielsen Effect — Random Doses (All)",
        )

    def build_nielsen_by_ome_vignettes(self):
        """Nielsen by OME — vignettes."""
        ome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means_vignettes.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_vignettes_ome.csv"),
            naive_batch="ome_n0",
            nielsen_batch="ome_n1",
            title="Nielsen — Vignettes (Oral Morphine)",
            width=900, height=350,
        )
        non_ome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means_vignettes.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_vignettes_nonome.csv"),
            naive_batch="pairs_n0",
            nielsen_batch="pairs_n1",
            title="Nielsen — Vignettes (Non–Oral Morphine)",
            width=900, height=350,
        )
        return column(ome, non_ome)

    def build_nielsen_by_ome_random_doses(self):
        """Nielsen by OME — random doses."""
        ome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_random_dose_ome.csv"),
            naive_batch="ome_n0",
            nielsen_batch="ome_n1",
            title="Nielsen — Random Doses (Oral Morphine)",
            width=900, height=350,
        )
        non_ome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_random_dose_nonome.csv"),
            naive_batch="pairs_n0",
            nielsen_batch="pairs_n1",
            title="Nielsen — Random Doses (Non–Oral Morphine)",
            width=900, height=350,
        )
        return column(ome, non_ome)

    def build_ome_vs_nonome(self):
        """OME vs non-OME comparison: naive and Nielsen side by side."""
        naive = ome_comparison_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            ome_batch="ome_n0",
            non_ome_batch="pairs_n0",
            comparison_csv=self._p("comparison_ome_vs_nonome_naive_permodel.csv"),
            title="Naive — OME vs Non-OME (Random Doses)",
            width=900, height=350,
        )
        nielsen = ome_comparison_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            ome_batch="ome_n1",
            non_ome_batch="pairs_n1",
            comparison_csv=self._p("comparison_ome_vs_nonome_nielsen_permodel.csv"),
            title="Nielsen — OME vs Non-OME (Random Doses)",
            width=900, height=350,
        )
        return column(naive, nielsen)

    def build_explain_units(self):
        """Explain units effect."""
        return explain_units_bars(
            pair_model_csv=self._p("pair_model_analysis.csv"),
            explain_units_pair_csv=self._p("explain_units_pair_model_analysis.csv"),
            comparison_csv=self._p("comparison_units_effect_random_dose.csv"),
            dosing_comparison_csv=self._p("comparison_dosing_unit_effect.csv"),
            title="Explain Units Effect — Random Doses (Oral Morphine)",
        )

    def build_full_treatment(self):
        """Full treatment: naive → mg/day+N → mg/4h+N → mg/4h+N+E."""
        return full_treatment_bars(
            pair_model_csv=self._p("pair_model_analysis.csv"),
            explain_units_pair_csv=self._p("explain_units_pair_model_analysis.csv"),
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_nielsen_csv=self._p("comparison_nielsen_effect_random_dose_ome.csv"),
            comparison_units_csv=self._p("comparison_units_effect_random_dose.csv"),
            title="Full Treatment Pipeline — Random Doses (Oral Morphine)",
        )

    def build_nielsen_ranked_random_doses(self):
        """Ranked model performance with Nielsen — random doses."""
        return naive_performance_bars(
            ranking_csv=self._p("ranking_random_doses_nielsen.csv"),
            title="Nielsen Performance by Model — Random Doses",
        )

    def build_nielsen_units_ranked(self):
        """Ranked model performance with Nielsen + explain units."""
        rd = naive_performance_bars(
            ranking_csv=self._p("ranking_random_doses_nielsen_units.csv"),
            title="Nielsen + Explain Units — Random Doses (Oral Morphine)",
        )
        vig = naive_performance_bars(
            ranking_csv=self._p("ranking_vignettes_nielsen_units.csv"),
            title="Nielsen + Explain Units — Vignettes",
        )
        return column(rd, vig)

    def build_cross_experiment(self):
        """Cross-experiment: random dose vs empty vignette vs text vignette (matched pairs only)."""
        return cross_experiment_bars(
            matched_means_csv=self._p("cross_experiment_matched_means.csv"),
            matched_wilcoxon_csv=self._p("cross_experiment_matched_wilcoxon.csv"),
            title="Cross-Experiment: Random Dose vs Empty Vignette vs Text Vignette (Matched Pairs)",
        )

    # ── combined builders for 3-act narrative ────────────────────

    def build_nielsen_effect_combined(self):
        """Nielsen effect — vignettes + random doses on one page."""
        vig = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means_vignettes.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_vignettes.csv"),
            naive_batch="all_n0",
            nielsen_batch="all_n1",
            title="Nielsen Effect — Vignettes",
            width=900, height=370,
        )
        rd = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_random_dose.csv"),
            naive_batch="all_n0",
            nielsen_batch="all_n1",
            title="Nielsen Effect — Random Doses",
            width=900, height=370,
        )
        return column(vig, rd)

    def build_nielsen_by_ome_combined(self):
        """Nielsen by OME — vignettes + random doses on one page."""
        vig_ome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means_vignettes.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_vignettes_ome.csv"),
            naive_batch="ome_n0",
            nielsen_batch="ome_n1",
            title="Vignettes — Oral Morphine",
            width=900, height=300,
        )
        vig_nonome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means_vignettes.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_vignettes_nonome.csv"),
            naive_batch="pairs_n0",
            nielsen_batch="pairs_n1",
            title="Vignettes — Non-Oral-Morphine",
            width=900, height=300,
        )
        rd_ome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_random_dose_ome.csv"),
            naive_batch="ome_n0",
            nielsen_batch="ome_n1",
            title="Random Doses — Oral Morphine",
            width=900, height=300,
        )
        rd_nonome = nielsen_effect_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_csv=self._p("comparison_nielsen_effect_random_dose_nonome.csv"),
            naive_batch="pairs_n0",
            nielsen_batch="pairs_n1",
            title="Random Doses — Non-Oral-Morphine",
            width=900, height=300,
        )
        sep = Div(text="<hr style='border:1px solid #ccc; margin:12px 0'>", width=900)
        return column(vig_ome, vig_nonome, sep, rd_ome, rd_nonome)

    def build_explain_units_full_treatment(self):
        """Explain-units effect + full treatment pipeline on one page."""
        eu = explain_units_bars(
            pair_model_csv=self._p("pair_model_analysis.csv"),
            explain_units_pair_csv=self._p("explain_units_pair_model_analysis.csv"),
            comparison_csv=self._p("comparison_units_effect_random_dose.csv"),
            dosing_comparison_csv=self._p("comparison_dosing_unit_effect.csv"),
            title="Dosing-Unit Effect — OME Random Doses",
            width=1000, height=400,
        )
        ft = full_treatment_bars(
            pair_model_csv=self._p("pair_model_analysis.csv"),
            explain_units_pair_csv=self._p("explain_units_pair_model_analysis.csv"),
            batch_means_csv=self._p("model_batch_means.csv"),
            comparison_nielsen_csv=self._p("comparison_nielsen_effect_random_dose_ome.csv"),
            comparison_units_csv=self._p("comparison_units_effect_random_dose.csv"),
            title="Full Treatment Pipeline — OME Random Doses",
            width=1100, height=420,
        )
        return column(eu, ft)

    # ── save all ─────────────────────────────────────────────────

    FIGURE_METADATA = [
        # ── Clinical Vignettes ───────────────────────────────

        ("naive_vignette_bars", "build_naive_vignette_bars",
         "LLMs are dangerously inaccurate at opioid dose conversion",
         "Horizontal bar chart of each model's accuracy on {n_vignette_prompts} realistic"
         " vignette-based opioid conversion prompts ({n_vignette_scenarios} clinical scenarios,"
         " {n_vignette_wordings} wordings each) without any reference"
         " information. Dark bars show tight-range ({tight_lower_x}\u2013{tight_upper_x}\u00d7) accuracy;"
         " translucent extensions show broad-range ({broad_lower_x}\u2013{broad_upper_x}\u00d7) accuracy."
         " Models are sorted by global rank. Significance brackets compare"
         " adjacent-ranked models with Cochran\u2013Mantel\u2013Haenszel tests."),

        ("cv_strip_vignettes", "build_cv_strip_vignettes",
         "Models are precise: they give the same wrong answer consistently",
         "Three-panel figure (vignettes). Top: strip chart of per-scenario"
         " coefficient-of-variation (CV) under naive conditions — most dots"
         " cluster near zero. Middle: scatter of CV vs mean observed/expected"
         " dose ratio on a log scale, with threshold lines at the tight and"
         " broad accuracy boundaries. Bottom: per-model grid where columns"
         " are CV bins and rows are accuracy categories; circle size is"
         " proportional to the share of observations in each cell."),

        ("vignette_performance_profile", "build_vignette_performance_profile",
         "Most errors are precise: same wrong answer every time",
         "Two-panel figure. Top: performance profile classifying each"
         " model's vignette results into Correct (tight accuracy),"
         " Precise but Incorrect (CV=0 but outside tight range), and"
         " Imprecise (CV>0 and not tight). Bottom: per-model accuracy"
         " distribution by CV bin (circle-area grid, χ² p-values)."),

        ("variance_stacked", "build_variance_stacked",
         "Most response variance comes from prompt wording, not stochastic noise",
         "Stacked horizontal bar chart decomposing mean log-ratio variance"
         " into within-vignette (stochastic noise, dark) and between-vignette"
         " (wording effect, light) components per model, sorted by vignette"
         " accuracy (best at top). Annotations show each model's mean tight"
         " accuracy. Below: scatter of total variance vs ICC across all"
         " {n_model_scenario_cells} model \u00d7 scenario cells."),

        ("spearman_variance_icc", "build_spearman_variance_icc",
         "Wording-driven variance tracks ICC across most models",
         "Horizontal bar chart of per-model Spearman ρ between scenario"
         " log-ratio variance and ICC (n = {n_scenarios_per_model} scenarios per model), sorted"
         " by vignette accuracy. Positive ρ means scenarios with more total"
         " variance have a larger share attributable to wording. Significance"
         " annotations and a sign-test summary are included."),

        ("icc_bubble_grid", "build_icc_bubble_grid",
         "Scenario-level decomposition: wording vs stochastic variance",
         "Concentric-circle grid (model × scenario), sorted by vignette"
         " accuracy (best at top). Outer circle (faint) = total log-ratio"
         " variance; inner circle (bold) = between-vignette (wording)"
         " variance. Where the inner is prominent, most variance"
         " comes from wording differences (high ICC). Annotations show the count"
         " of significant pairwise vignette comparisons out of {icc_max_pairs}"
         " per cell (Bonferroni-corrected Mann–Whitney U; overall"
         " {icc_total_sig_pairs}/{icc_total_pairs}\u2020 comparisons significant)."
         " \u2020 qwen/qwen-2.5-72b-instruct has only 3 wordings (vs 4 for all other models)"
         " in 3 scenarios (elderly-hip-fracture, young-adult-sickle-cell-crisis,"
         " young-man-car-accident), reducing the maximum from 1\u202f560 (13\u202f\u00d7\u202f20\u202f\u00d7\u202f6) to 1\u202f551."),

        ("kendall_w_summary", "build_kendall_w_summary",
         "Wording effect sizes and directions are model-specific",
         "Paired bar chart of two complementary metrics per scenario under"
         " naive prompting, sorted by mean ICC. Blue bars: mean ICC — the"
         " proportion of response variance attributable to wording within"
         " each model. Red bars: Kendall's W — cross-model agreement on"
         " which wording produces higher or lower responses. High ICC with"
         " near-zero W indicates that wording strongly affects output, but"
         " each model is sensitive to different aspects of the phrasing."),

        # ── Random Doses ────────────────────────────────────

        ("naive_random_dose_bars", "build_naive_random_dose_bars",
         "The same accuracy pattern holds on controlled random-dose prompts",
         "Top: horizontal bar chart of model accuracy on {n_opioid_pairs} random-dose opioid"
         " pair conversions without reference information. Significance brackets compare"
         " adjacent-ranked models using Cochran\u2013Mantel\u2013Haenszel tests."
         " Bottom: scatter plot of each model's tight accuracy on vignette prompts"
         " (x-axis) vs random-dose prompts (y-axis). The dashed diagonal is the identity"
         " line (equal performance on both tasks). Spearman \u03c1 = {naive_rank_spearman_rho}"
         " (p = {naive_rank_spearman_p}, n = {naive_rank_n_models} models) confirms that the"
         " model ranking is significantly correlated across the two prompt formats."),

        ("cv_strip_random_doses", "build_cv_strip_random_doses",
         "Precision replicates; dose-dependent bias emerges",
         "Three-panel figure (random doses). Top: strip chart of per-pair"
         " CV under naive conditions; black-outlined dots flag opioid"
         " pairs with a statistically significant negative regression slope,"
         " grey-outlined dots flag significant positive slope"
         " (p < 0.05). Middle:"
         " scatter of CV vs mean observed/expected coefficient ratio on"
         " a log scale. Bottom: performance profile stacked bars."),

        ("cross_experiment_bars", "build_cross_experiment",
         "Performance structure is consistent across prompt formats",
         "Grouped bar chart comparing each model's tight- and broad-range"
         " accuracy across three experimental paradigms — random-dose pair"
         " conversions, empty vignettes, and text-based vignettes — restricted"
         " to the {n_matched_pairs} opioid pairs present in both experiments. All conditions"
         " are naive (no reference data). Three significance brackets per"
         " model show paired Wilcoxon signed-rank tests on broad-range"
         " accuracy: RD vs empty, empty vs text, and RD vs text."),

        # ── Reference & Unit Augmentation ───────────────────

        ("nielsen_effect_combined", "build_nielsen_effect_combined",
         "Adding a reference table improves accuracy on both task formats",
         "Two-panel grouped bar chart. Top: vignette conversions; bottom:"
         " random-dose conversions. Each panel compares naive vs Nielsen-"
         " augmented accuracy per model, with Wilcoxon signed-rank"
         " significance stars. The same pattern of improvement appears"
         " regardless of prompt format."),

        ("nielsen_ranked_random_doses", "build_nielsen_ranked_random_doses",
         "Model ranking after adding the Nielsen reference table",
         "Horizontal bar chart showing each model's accuracy on random-dose"
         " conversions after providing the Nielsen reference table. Models"
         " are re-ranked by adjusted performance, with significance brackets"
         " from Cochran\u2013Mantel\u2013Haenszel pairwise tests. Comparison"
         " with the naive ranking (page 7) reveals which models benefit"
         " most from reference augmentation."),

        ("nielsen_by_ome_combined", "build_nielsen_by_ome_combined",
         "Nielsen improvement concentrates on oral morphine conversions",
         "Four-panel figure splitting the Nielsen effect by conversion type"
         " and experiment. Top two panels: vignettes (oral morphine, then"
         " non-oral-morphine). Bottom two panels: random doses (oral morphine,"
         " then non-oral-morphine). The differential improvement confirms"
         " that the reference table primarily benefits conversions involving"
         " oral morphine, which dominate its coverage."),

        ("explain_units_full_treatment", "build_explain_units_full_treatment",
         "Non-daily dosing confuses models; unit explanation rescues accuracy",
         "Two stacked panels, both from the random-dose OME experiment."
         " Top: three-condition comparison — mg/day + Nielsen, mg/4h + Nielsen,"
         " and mg/4h + Nielsen + explicit unit explanation — isolating the"
         " dosing-unit effect. Bottom: full four-step pipeline from naive"
         " through Nielsen, non-daily dosing drop, and unit-explanation rescue."
         " Together they show that switching from daily to non-daily units"
         " degrades accuracy significantly, and that a brief unit explanation"
         " recovers most of the lost performance."),

        ("nielsen_units_ranked", "build_nielsen_units_ranked",
         "Best achievable accuracy after all prompt augmentations",
         "Two-panel ranked bar chart. Top: random-dose OME conversions with"
         " Nielsen + explicit unit explanation. Bottom: vignette conversions"
         " with Nielsen + explicit unit explanation. Shows the final best-"
         " achievable accuracy per model after all prompt augmentations,"
         " on both task formats."),
    ]

    async def save_all(self):
        """Generate and save all figures as HTML + PDF + PNG."""
        for name, method_name, _title, _desc in self.FIGURE_METADATA:
            try:
                fig = getattr(self, method_name)()
                html_path = self._f(f"{name}.html")
                pdf_path = self._f(f"{name}.pdf")
                png_path = self._f(f"{name}.png")
                output_file(html_path)
                save(fig, resources=INLINE)
                await convert_to_pdf_auto_size(html_path, pdf_path, scale=2.0)
                await convert_to_png_auto_size(html_path, png_path, scale=2.0)
                print(f"✓ {name}")
            except Exception as e:
                print(f"✗ {name}: {e}")
                import traceback; traceback.print_exc()

    def _load_figure_stats(self) -> dict:
        """
        Load figure_stats.json from the figures directory.
        Returns an empty dict (with safe fallback values) if the file is
        missing so that str.format() calls still succeed.
        """
        import json as _json
        stats_path = os.path.join(os.path.dirname(self.figures_dir), "figure_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                return _json.load(f)
        print(f"  [warn] {stats_path} not found — description placeholders will show as '?'")
        # Provide safe defaults so format() never raises KeyError
        return {
            "n_vignette_scenarios": "?", "n_vignette_wordings": "?",
            "n_vignette_prompts": "?", "n_opioid_pairs": "?",
            "n_matched_pairs": "?", "n_model_scenario_cells": "?",
            "n_scenarios_per_model": "?", "icc_max_pairs": "?",
            "icc_total_sig_pairs": "?", "icc_total_pairs": "?",
            "tight_lower_x": "0.85", "tight_upper_x": "1.15",
            "broad_lower_x": "0.6", "broad_upper_x": "1.7",
            "naive_rank_spearman_rho": "?", "naive_rank_spearman_p": "?",
        }

    async def save_annotated_pdf(self, output_path: str = None):
        """
        Generate a single combined PDF with all figures, each preceded
        by a title and descriptive paragraph.

        Description text is formatted against figure_stats.json so that all
        numeric annotations are data-driven rather than hardcoded.
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.figures_dir), "all_comprehensive_figures.pdf")

        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,
        )
        from reportlab.lib.enums import TA_LEFT
        import io
        from PIL import Image as PILImage

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "FigTitle", parent=styles["Heading2"],
            fontSize=16, spaceAfter=8,
        )
        desc_style = ParagraphStyle(
            "FigDesc", parent=styles["BodyText"],
            fontSize=11, leading=15, spaceAfter=12, alignment=TA_LEFT,
        )

        page_w, page_h = letter
        margin = 0.75 * inch
        usable_w = page_w - 2 * margin

        stats = self._load_figure_stats()

        story = []
        for name, method_name, title, desc in self.FIGURE_METADATA:
            png_path = self._f(f"{name}.png")
            if not os.path.exists(png_path):
                print(f"  Skipping {name} — PNG not found")
                continue

            # Fill any {placeholder} fields from figure_stats.json
            try:
                desc = desc.format(**stats)
            except KeyError as e:
                print(f"  [warn] {name}: missing stats key {e} — leaving placeholder")

            # Resize image to fit page width while keeping aspect ratio
            pil = PILImage.open(png_path)
            img_w, img_h = pil.size
            scale = usable_w / img_w
            display_w = usable_w
            display_h = img_h * scale
            # Cap height to leave room for title + description
            max_img_h = page_h - 2 * margin - 1.5 * inch
            if display_h > max_img_h:
                display_h = max_img_h
                display_w = img_w * (max_img_h / img_h)

            story.append(Paragraph(title, title_style))
            story.append(Paragraph(desc, desc_style))
            story.append(RLImage(png_path, width=display_w, height=display_h))
            story.append(PageBreak())

        doc = SimpleDocTemplate(
            output_path, pagesize=letter,
            leftMargin=margin, rightMargin=margin,
            topMargin=margin, bottomMargin=margin,
        )
        doc.build(story)
        print(f"✓ Annotated PDF saved to {output_path}")
