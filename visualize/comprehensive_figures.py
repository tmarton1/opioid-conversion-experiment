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
from scipy.stats import wilcoxon

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row as bk_row
from bokeh.models import (
    ColumnDataSource, LabelSet, HoverTool, Legend, LegendItem,
    FactorRange, Span, Div, Label, FixedTicker,
)
from bokeh.palettes import Greys256

from common.models import CONSISTENT_MODELS
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


def _per_model_wilcoxon(comparison_csv: str, model: str, range_type: str = "tight"):
    """
    Run Wilcoxon signed-rank on tight_diff or broad_diff for one model.
    Returns (p_value, median_diff, n_diffs).
    """
    df = pd.read_csv(comparison_csv)
    mdf = df[df["model"] == model]
    diff_col = f"{range_type}_diff"
    if diff_col not in mdf.columns:
        return None, None, 0
    diffs = mdf[diff_col].dropna()
    if len(diffs) < 2:
        return None, None, len(diffs)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 1:
        return 1.0, 0.0, len(diffs)
    _, p = wilcoxon(nonzero, alternative="two-sided")
    return p, float(diffs.median()), len(diffs)


def _overall_wilcoxon(comparison_csv: str, range_type: str = "tight"):
    """Run Wilcoxon signed-rank on all diffs pooled across models."""
    df = pd.read_csv(comparison_csv)
    diff_col = f"{range_type}_diff"
    if diff_col not in df.columns:
        return None, None, 0
    diffs = df[diff_col].dropna()
    if len(diffs) < 2:
        return None, None, len(diffs)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 1:
        return 1.0, 0.0, len(diffs)
    _, p = wilcoxon(nonzero, alternative="two-sided")
    return p, float(diffs.median()), len(diffs)


def _significance_star(p):
    """Return Unicode star string for p-value."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _star_label(p):
    """Return a star + p-value annotation string."""
    if p is None:
        return ""
    stars = _significance_star(p)
    if not stars:
        return "n.s."
    return f"{stars} (p={p:.1e})"


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
    width: int = 700,
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

        y_bot = i
        y_top = i + 1
        # Vertical line spanning the two bars
        p.line(x=[bx, bx], y=[y_bot + 0.3, y_top - 0.3],
               line_color="#333", line_width=1.0)
        # Horizontal caps touching the bar area
        cap_len = 0.01
        p.line(x=[bx - cap_len, bx], y=[y_bot + 0.3, y_bot + 0.3],
               line_color="#333", line_width=1.0)
        p.line(x=[bx - cap_len, bx], y=[y_top - 0.3, y_top - 0.3],
               line_color="#333", line_width=1.0)

        mid_y = (y_bot + y_top) / 2
        p.add_layout(Label(
            x=bx + 0.005, y=mid_y,
            text=stars, text_font_size="9pt",
            text_baseline="middle", text_align="left",
            text_color="#333",
        ))

    # Legend
    ls = ColumnDataSource(dict(y=[clean_names[0]], right=[0]))
    d1 = p.hbar(y="y", right="right", height=0.2, color="black", alpha=0.9, source=ls, visible=True)
    d2 = p.hbar(y="y", right="right", height=0.2, color="black", alpha=0.25, source=ls, visible=True)
    legend = Legend(items=[
        LegendItem(label=f"Tight ({TIGHT_RANGE_LOWER:.2f}–{TIGHT_RANGE_UPPER:.2f}x)", renderers=[d1]),
        LegendItem(label=f"Broad ({BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)", renderers=[d2]),
    ])
    p.add_layout(legend, "below")
    p.legend.label_text_font_size = "8pt"

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
):
    """
    Three-panel figure:
    Top: strip chart of CV (same as cv_strip_chart).
    Middle: scatter of CV (x) vs mean observed/expected ratio (y) on log scale.
    Bottom: stacked bar mosaic — accuracy bands × CV bins.
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
                              width=width, height=panel_h)

    return column(top, mid, bot)


def _cv_accuracy_mosaic(df, cv_col, tight_col, broad_col,
                        width=800, height=280):
    """Stacked bar chart: accuracy category proportions by CV bin."""
    cols_needed = [c for c in [cv_col, tight_col, broad_col] if c in df.columns]
    if len(cols_needed) < 3:
        return figure(width=width, height=height, toolbar_location=None,
                      title="Accuracy by CV Bin (missing columns)")

    sub = df.dropna(subset=cols_needed).copy()
    if sub.empty:
        return figure(width=width, height=height, toolbar_location=None,
                      title="Accuracy by CV Bin (no data)")

    bins = [0, 0.1, 0.5, float("inf")]
    bin_labels = ["CV < 0.1", "CV 0.1–0.5", "CV ≥ 0.5"]
    sub["_cv_bin"] = pd.cut(sub[cv_col], bins=bins, labels=bin_labels, right=False)

    results = []
    for lbl in bin_labels:
        bdf = sub[sub["_cv_bin"] == lbl]
        n = len(bdf)
        if n == 0:
            results.append({"label": lbl, "n": 0, "tight": 0,
                            "broad_only": 0, "neither": 0})
            continue
        t = bdf[tight_col].mean()
        b = bdf[broad_col].mean()
        bo = max(0, b - t)
        ne = max(0, 1 - b)
        results.append({"label": lbl, "n": n, "tight": t,
                        "broad_only": bo, "neither": ne})

    cats = [f"{r['label']}  (n={r['n']})" for r in results]
    tights = [r["tight"] for r in results]
    broads = [r["broad_only"] for r in results]
    neithers = [r["neither"] for r in results]

    p = figure(
        x_range=cats, y_range=(0, 1.05), width=width, height=height,
        toolbar_location=None, title="Accuracy Distribution by CV Bin",
    )

    src = ColumnDataSource(dict(
        cats=cats, tight=tights, broad_only=broads, neither=neithers,
    ))
    p.vbar_stack(
        ["tight", "broad_only", "neither"],
        x="cats", width=0.6, source=src,
        color=["#2ca02c", "#ff7f0e", "#d62728"],
        alpha=[0.8, 0.7, 0.7],
        legend_label=[
            f"Tight ({TIGHT_RANGE_LOWER:.2f}–{TIGHT_RANGE_UPPER:.2f}x)",
            f"Broad only ({BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)",
            "Neither",
        ],
    )

    # Proportion labels on each segment
    for i, r in enumerate(results):
        y_pos = 0
        for cat_name, val in [("tight", r["tight"]),
                               ("broad_only", r["broad_only"]),
                               ("neither", r["neither"])]:
            if val > 0.04:
                p.add_layout(Label(
                    x=i, y=y_pos + val / 2,
                    text=f"{val:.0%}", text_font_size="9pt",
                    text_align="center", text_baseline="middle",
                    text_color="white",
                ))
            y_pos += val

    p.yaxis.axis_label = "Proportion"
    p.yaxis.ticker = FixedTicker(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.xaxis.major_label_text_font_size = "9pt"
    p.legend.location = "top_right"
    p.legend.label_text_font_size = "8pt"
    p.xgrid.visible = False

    return p

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
                        and slope_val < 0 and p_val < slope_alpha):
                    outlines.append("black")
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
    idx_out = [i for i, o in enumerate(outlines) if o is not None]
    if idx_out:
        src_out = ColumnDataSource(dict(
            x=[xs[i] for i in idx_out],
            y=[ys[i] for i in idx_out],
            color=[colors[i] for i in idx_out],
            pair=[pair_labels[i] for i in idx_out],
        ))
        r2 = p.scatter("x", "y", color="color", alpha=0.8, size=8,
                        line_color="black", line_width=1.5, source=src_out)

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

    if idx_out:
        # Add a note about black outlines
        note = Label(
            x=5, y=5, x_units="screen", y_units="screen",
            text=f"Black outline = sig. negative slope (p<{slope_alpha})",
            text_font_size="8pt", text_color="#555",
        )
        p.add_layout(note)

    p.legend.label_text_font_size = "8pt"
    p.legend.location = "bottom_right"

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
        y_range=(0, 1.08),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.7, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.7, color="color", alpha=0.25, source=source)

    # Significance annotations
    if os.path.exists(comparison_csv):
        star_x, star_y, star_text = [], [], []
        for m, cn in zip(models, clean_names):
            p_t, _, _ = _per_model_wilcoxon(comparison_csv, m, "tight")
            p_b, _, _ = _per_model_wilcoxon(comparison_csv, m, "broad")
            # Use the most significant (lowest) p-value across tight and broad
            p_min = min(p for p in [p_t, p_b] if p is not None) if any(p is not None for p in [p_t, p_b]) else None
            s = _significance_star(p_min)
            if s:
                broad_max = max(
                    naive.loc[m, "mean_in_broad"],
                    nielsen.loc[m, "mean_in_broad"],
                )
                star_x.append((cn, "naive"))
                star_y.append(broad_max + 0.03)
                star_text.append(s)

        if star_x:
            star_src = ColumnDataSource(dict(x=star_x, y=star_y, text=star_text))
            p.add_layout(LabelSet(
                x="x", y="y", text="text", source=star_src,
                text_font_size="9pt", text_align="center",
                text_baseline="bottom", x_offset=15,
            ))

    _style_grouped_bar(p, clean_names, bar_labels)
    return p


# ── Explain units effect bars with significance ─────────────────────

def explain_units_bars(
    batch_means_csv: str,
    batch_means_explain_csv: str,
    comparison_csv: str,
    title: str = "Explain Units Effect",
    width: int = 1000,
    height: int = 400,
):
    """
    Three-condition bar chart: mg/day + Nielsen, non-daily + Nielsen,
    non-daily + Nielsen + explain units.
    Wilcoxon significance stars for the explain_units effect.
    """
    df_main = pd.read_csv(batch_means_csv)
    df_main = df_main[df_main["model"].isin(_CONSISTENT_SET)]
    df_units = pd.read_csv(batch_means_explain_csv)
    df_units = df_units[df_units["model"].isin(_CONSISTENT_SET)]

    ome_n1 = df_main[df_main["batch_name"] == "ome_n1"].set_index("model")
    units = df_units[df_units["batch_name"] == "nielsen_only"].set_index("model")
    units_exp = df_units[df_units["batch_name"] == "nielsen_units_explained"].set_index("model")

    ranked_models = get_ranked_model_order()
    models = [m for m in ranked_models
              if m in ome_n1.index and m in units.index and m in units_exp.index]

    clean_names = [_clean(m) for m in models]
    cond_labels = ["mg/day", "mg/4h", "explained"]

    x_factors = [(cn, cl) for cn in clean_names for cl in cond_labels]
    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [], "model": [], "batch": []}

    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for label, batch_df in [("mg/day", ome_n1), ("mg/4h", units), ("explained", units_exp)]:
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
        y_range=(0, 1.08),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.6, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.6, color="color", alpha=0.25, source=source)

    # Significance annotations for explain_units effect
    if os.path.exists(comparison_csv):
        star_x, star_y, star_text = [], [], []
        for m, cn in zip(models, clean_names):
            p_t, _, _ = _per_model_wilcoxon(comparison_csv, m, "tight")
            p_b, _, _ = _per_model_wilcoxon(comparison_csv, m, "broad")
            p_min = min(p for p in [p_t, p_b] if p is not None) if any(p is not None for p in [p_t, p_b]) else None
            s = _significance_star(p_min)
            if s:
                broad_max = max(
                    ome_n1.loc[m, "mean_in_broad"] if m in ome_n1.index else 0,
                    units.loc[m, "mean_in_broad"] if m in units.index else 0,
                    units_exp.loc[m, "mean_in_broad"] if m in units_exp.index else 0,
                )
                star_x.append((cn, "mg/4h"))
                star_y.append(broad_max + 0.03)
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


# ── Full treatment (Nielsen + explain units) ────────────────────────

def full_treatment_bars(
    batch_means_csv: str,
    comparison_nielsen_csv: str,
    comparison_units_csv: Optional[str] = None,
    naive_batch: str = "all_n0",
    nielsen_batch: str = "all_n1",
    full_batch: Optional[str] = None,
    title: str = "Nielsen + Explain Units Performance",
    width: int = 900,
    height: int = 400,
):
    """
    Bar chart: naive vs nielsen vs Nielsen + explain units per model with significance.
    """
    df = pd.read_csv(batch_means_csv)
    df = df[df["model"].isin(_CONSISTENT_SET)]
    naive_df = df[df["batch_name"] == naive_batch].set_index("model")
    nielsen_df = df[df["batch_name"] == nielsen_batch].set_index("model")

    # Full treatment batch is optional
    full_df = None
    if full_batch and full_batch in df["batch_name"].values:
        full_df = df[df["batch_name"] == full_batch].set_index("model")

    ranked_models = get_ranked_model_order()
    available = set(naive_df.index) & set(nielsen_df.index)
    if full_df is not None:
        available &= set(full_df.index)
    models = [m for m in ranked_models if m in available]
    if not models:
        models = sorted(available)

    clean_names = [_clean(m) for m in models]

    if full_df is not None:
        cond_labels = ["naive", "Nielsen", "Nielsen + explain units"]
        batches_data = [("naive", naive_df), ("Nielsen", nielsen_df), ("Nielsen + explain units", full_df)]
    else:
        cond_labels = ["naive", "Nielsen"]
        batches_data = [("naive", naive_df), ("Nielsen", nielsen_df)]

    x_factors = [(cn, cl) for cn in clean_names for cl in cond_labels]
    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [], "model": [], "batch": []}

    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for label, bdf in batches_data:
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
        y_range=(0, 1.08),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.6, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.6, color="color", alpha=0.25, source=source)

    # Significance annotations for Nielsen effect
    if os.path.exists(comparison_nielsen_csv):
        star_x, star_y, star_text = [], [], []
        for m, cn in zip(models, clean_names):
            p_t, _, _ = _per_model_wilcoxon(comparison_nielsen_csv, m, "tight")
            p_b, _, _ = _per_model_wilcoxon(comparison_nielsen_csv, m, "broad")
            p_min = min(p for p in [p_t, p_b] if p is not None) if any(p is not None for p in [p_t, p_b]) else None
            s = _significance_star(p_min)
            if s:
                broad_vals = [naive_df.loc[m, "mean_in_broad"], nielsen_df.loc[m, "mean_in_broad"]]
                if full_df is not None and m in full_df.index:
                    broad_vals.append(full_df.loc[m, "mean_in_broad"])
                star_x.append((cn, "naive"))
                star_y.append(max(broad_vals) + 0.03)
                star_text.append(s)
        if star_x:
            star_src = ColumnDataSource(dict(x=star_x, y=star_y, text=star_text))
            p.add_layout(LabelSet(
                x="x", y="y", text="text", source=star_src,
                text_font_size="9pt", text_align="center",
                text_baseline="bottom", x_offset=15,
            ))

    _style_grouped_bar(p, clean_names, cond_labels)
    return p


# ── Cross-experiment: vignette vs random dose ───────────────────────

def cross_experiment_bars(
    random_dose_batch_means_csv: str,
    vignette_batch_means_csv: str,
    title: str = "Cross-Experiment Comparison",
    width: int = 1000,
    height: int = 450,
):
    """
    Grouped bar chart: three bars per model — random dose (naive),
    empty vignette (naive), text vignette (naive).
    Each bar shows tight (solid) + broad (translucent).
    """
    rd_df = pd.read_csv(random_dose_batch_means_csv)
    rd_df = rd_df[rd_df["model"].isin(_CONSISTENT_SET)]
    vt_df = pd.read_csv(vignette_batch_means_csv)
    vt_df = vt_df[vt_df["model"].isin(_CONSISTENT_SET)]

    rd_naive = rd_df[rd_df["batch_name"] == "all_n0"].set_index("model")
    vt_text = vt_df[vt_df["batch_name"] == "text_all_naive"].set_index("model")
    vt_empty = vt_df[vt_df["batch_name"] == "empty_all_naive"].set_index("model")

    ranked_models = get_ranked_model_order()
    available = set(rd_naive.index) & set(vt_text.index) & set(vt_empty.index)
    models = [m for m in ranked_models if m in available]
    if not models:
        models = sorted(available)

    clean_names = [_clean(m) for m in models]
    cond_labels = ["random dose", "empty vignette", "text vignette"]

    x_factors = [(cn, cl) for cn in clean_names for cl in cond_labels]
    bar_data = {"x": [], "tight": [], "broad_top": [], "color": [], "model": [], "batch": []}

    for m in models:
        cn = _clean(m)
        color = MODEL_COLOR_DICT.get(m, "#888888")
        for label, bdf in [("random dose", rd_naive),
                           ("empty vignette", vt_empty),
                           ("text vignette", vt_text)]:
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
        y_range=(0, 1.08),
        toolbar_location=None,
        title=title,
    )
    p.vbar(x="x", top="tight", width=0.7, color="color", alpha=0.9, source=source)
    p.vbar(x="x", top="broad_top", bottom="tight", width=0.7, color="color",
           alpha=0.25, source=source)

    # ── Pairwise significance tests (model-level paired Wilcoxon) ──
    cond_pairs = [("random dose", "empty vignette", rd_naive, vt_empty),
                  ("empty vignette", "text vignette", vt_empty, vt_text),
                  ("random dose", "text vignette", rd_naive, vt_text)]
    sig_lines = []
    for label_a, label_b, df_a, df_b in cond_pairs:
        common = sorted(set(df_a.index) & set(df_b.index) & set(models))
        if len(common) < 4:
            continue
        diffs_t = [df_b.loc[m, "mean_in_tight"] - df_a.loc[m, "mean_in_tight"]
                   for m in common]
        nonzero = [d for d in diffs_t if d != 0]
        if len(nonzero) < 2:
            continue
        _, p_val = wilcoxon(nonzero, alternative="two-sided")
        stars = _significance_star(p_val)
        if stars:
            sig_lines.append((label_a, label_b, stars, p_val))

    if sig_lines:
        # Add overall title annotation with significance results
        sig_parts = []
        for la, lb, stars, pv in sig_lines:
            sig_parts.append(f"{la} vs {lb}: {stars} (p={pv:.1e})")
        sig_text = "  |  ".join(sig_parts)
        p.add_layout(Label(
            x=5, y=5, x_units="screen", y_units="screen",
            text=sig_text, text_font_size="8pt", text_color="#555",
        ))

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

    def __init__(self, results_dir: str = "results", figures_dir: str = "figures"):
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
        """Naive random dose performance horizontal bars."""
        return naive_performance_bars(
            ranking_csv=self._p("ranking_naive_random_doses.csv"),
            title="Naive Random Dose Performance by Model",
        )

    def build_cv_strip_vignettes(self):
        """CV strip chart + ratio scatter — vignettes (naive)."""
        return cv_strip_with_ratio(
            model_csv=self._p("reworded_vignettes_model_analysis.csv"),
            cv_col="observed_dose_cv",
            ratio_num_col="observed_dose_mean",
            ratio_den_col="expected_to_dose",
            pair_col="scenario_key",
            naive_filter={"include_nielsen": False, "explain_units": False,
                          "has_vignette_text": True},
            title="Naive CV by Model — Vignettes",
        )

    def build_cv_strip_random_doses(self):
        """CV strip chart + ratio scatter — random doses (naive)."""
        return cv_strip_with_ratio(
            model_csv=self._p("pair_model_analysis.csv"),
            cv_col="observed_coefficient_cv",
            ratio_num_col="observed_coefficient_mean",
            ratio_den_col="expected_coefficient",
            pair_col="opioid_pair",
            naive_filter={"include_nielsen": False, "include_examples": False},
            title="Naive CV by Model — Random Doses",
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

    def build_explain_units(self):
        """Explain units effect."""
        return explain_units_bars(
            batch_means_csv=self._p("model_batch_means.csv"),
            batch_means_explain_csv=self._p("model_batch_means_explain_units.csv"),
            comparison_csv=self._p("comparison_units_effect_random_dose.csv"),
            title="Explain Units Effect (Oral Morphine)",
        )

    def build_full_treatment(self):
        """Full treatment: naive vs Nielsen vs Nielsen + explain units."""
        # Try vignette batch means first (has ome_n1_units)
        csv = self._p("model_batch_means_vignettes.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            full_batch = "ome_n1_units" if "ome_n1_units" in df["batch_name"].values else None
        else:
            csv = self._p("model_batch_means.csv")
            full_batch = None

        return full_treatment_bars(
            batch_means_csv=csv,
            comparison_nielsen_csv=self._p("comparison_nielsen_effect_vignettes.csv"),
            naive_batch="all_n0",
            nielsen_batch="all_n1",
            full_batch=full_batch,
            title="Nielsen + Explain Units Performance",
        )

    def build_nielsen_ranked_random_doses(self):
        """Ranked model performance with Nielsen — random doses."""
        return naive_performance_bars(
            ranking_csv=self._p("ranking_random_doses_nielsen.csv"),
            title="Nielsen Random Dose Performance by Model",
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
        """Cross-experiment: random dose vs empty vignette vs text vignette."""
        return cross_experiment_bars(
            random_dose_batch_means_csv=self._p("model_batch_means.csv"),
            vignette_batch_means_csv=self._p("model_batch_means_text_vs_empty.csv"),
            title="Cross-Experiment: Random Dose vs Empty Vignette vs Text Vignette",
        )

    # ── save all ─────────────────────────────────────────────────

    FIGURE_METADATA = [
        ("naive_vignette_bars", "build_naive_vignette_bars",
         "Naive performance is dangerous",
         "Horizontal bar chart showing the proportion of model responses falling within"
         " the tight (0.85–1.15×) and broad (0.6–1.7×) accuracy ranges for vignette-based"
         " opioid conversion prompts, without any reference information. Models are"
         " ordered by global rank. Significance brackets compare adjacent-ranked models"
         " using Cochran–Mantel–Haenszel tests."),

        ("naive_random_dose_bars", "build_naive_random_dose_bars",
         "Naive performance is dangerous",
         "Horizontal bar chart of model accuracy on random-dose opioid pair"
         " conversion prompts without reference information. Dark bars represent"
         " tight-range accuracy; translucent extensions show broad-range accuracy."
         " Models are ranked by global performance. Adjacent-model significance"
         " brackets are from CMH pairwise tests."),

        ("cv_strip_vignettes", "build_cv_strip_vignettes",
         "Naive performance reflects a consistent ratio, even when inaccurate",
         "Three-panel figure (vignettes). Top: strip chart of per-scenario CVs under"
         " naive conditions; black-outlined dots indicate significant negative"
         " Nielsen regression slopes. Middle: scatter of CV vs mean observed/expected"
         " dose ratio on a log scale, with dotted threshold lines at the tight and"
         " broad accuracy boundaries. Bottom: stacked bar mosaic showing how accuracy"
         " (tight / broad-only / neither) varies across CV bins."),

        ("cv_strip_random_doses", "build_cv_strip_random_doses",
         "Naive performance reflects a consistent ratio, even when inaccurate",
         "Three-panel figure (random doses). Top: strip chart of per-pair CVs under"
         " naive conditions. Middle: scatter of CV vs mean observed/expected"
         " coefficient ratio on a log scale. Bottom: accuracy distribution by CV bin."),

        ("nielsen_effect_vignettes", "build_nielsen_effect_vignettes",
         "Including Nielsen reference improves accuracy",
         "Grouped bar chart comparing naive versus Nielsen-augmented accuracy"
         " for all vignette conversions. Each model has paired bars showing"
         " performance with and without Nielsen reference data. Wilcoxon"
         " signed-rank significance stars indicate per-model improvement."),

        ("nielsen_effect_random_doses", "build_nielsen_effect_random_doses",
         "Including Nielsen reference improves accuracy",
         "Grouped bar chart comparing naive versus Nielsen-augmented accuracy"
         " for all random-dose pair conversions. Wilcoxon signed-rank"
         " significance stars are shown per model."),

        ("nielsen_by_ome_vignettes", "build_nielsen_by_ome_vignettes",
         "The Nielsen reference selectively improves oral morphine conversions",
         "Two-panel figure splitting the Nielsen effect for vignettes into"
         " oral-morphine-equivalent (OME) conversions (top) and non-OME"
         " conversions (bottom). The differential improvement reveals whether"
         " Nielsen reference data primarily benefits oral morphine conversions."),

        ("nielsen_by_ome_random_doses", "build_nielsen_by_ome_random_doses",
         "The Nielsen reference selectively improves oral morphine conversions",
         "Two-panel figure splitting the Nielsen effect for random-dose"
         " conversions into OME (top) and non-OME (bottom) subsets."
         " Significance annotations are computed independently for each subset."),

        ("explain_units", "build_explain_units",
         "Models struggle with non-daily dosing",
         "Three-condition bar chart comparing model accuracy on oral-morphine"
         " conversions: standard daily dosing with Nielsen, non-daily dosing with"
         " Nielsen, and non-daily dosing with Nielsen plus explicit unit explanation."
         " The drop from daily to non-daily reveals sensitivity to dosing intervals."),

        ("full_treatment", "build_full_treatment",
         "Explaining units improves non-daily dosing",
         "Bar chart comparing prompting conditions: naive (no reference),"
         " Nielsen-augmented, and Nielsen + explain units. Shows the cumulative"
         " benefit of adding reference data and explicit unit clarification to"
         " opioid conversion prompts."),

        ("cross_experiment_bars", "build_cross_experiment",
         "Performance structure is consistent across task formats",
         "Grouped bar chart comparing each model's tight- and broad-range"
         " accuracy across three experimental paradigms: random-dose pair"
         " conversions, empty vignettes, and text-based vignettes. All"
         " conditions are naive (no reference data). Wilcoxon signed-rank"
         " tests compare conditions across models."),

        ("nielsen_ranked_random_doses", "build_nielsen_ranked_random_doses",
         "Including Nielsen reference improves accuracy",
         "Horizontal bar chart showing model accuracy after providing"
         " Nielsen reference data for random-dose conversions. Models are"
         " ranked by adjusted performance across conditions."),

        ("nielsen_units_ranked", "build_nielsen_units_ranked",
         "Prompt improvements enable safe accuracy",
         "Two-panel ranked bar chart. Top: random-dose OME conversions with"
         " Nielsen + explain units. Bottom: vignette conversions with"
         " Nielsen + explain units. Shows final best-achievable accuracy"
         " per model after all prompt augmentations."),
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
                save(fig)
                await convert_to_pdf_auto_size(html_path, pdf_path, scale=2.0)
                await convert_to_png_auto_size(html_path, png_path, scale=2.0)
                print(f"✓ {name}")
            except Exception as e:
                print(f"✗ {name}: {e}")
                import traceback; traceback.print_exc()

    async def save_annotated_pdf(self, output_path: str = None):
        """
        Generate a single combined PDF with all figures, each preceded
        by a title and descriptive paragraph.
        """
        if output_path is None:
            output_path = self._f("all_comprehensive_figures.pdf")

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

        story = []
        for name, method_name, title, desc in self.FIGURE_METADATA:
            png_path = self._f(f"{name}.png")
            if not os.path.exists(png_path):
                print(f"  Skipping {name} — PNG not found")
                continue

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
