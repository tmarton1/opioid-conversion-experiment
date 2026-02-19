"""
ICC heatmap and bubble-grid visualizations.

Two figures from results/icc_analysis.csv:

1. **ICC Heatmap** — model × scenario, colour-mapped by ICC value,
   sorted by mean ICC on both axes.

2. **Bubble Grid** — model × scenario.  Each cell has two concentric circles:
   - Outer (opaque): area ∝ total variance  (σ²_total)
   - Inner (lighter/transparent): area ∝ between-vignette variance (σ²_between)
   The inner/outer ratio *is* the ICC.
   Circles inherit the model colour. Models sorted by lowest mean total
   variance at top; scenarios sorted by lowest mean total variance at left.
"""

import os
import math
import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource, HoverTool, Div, BasicTicker, ColorBar,
    LinearColorMapper, FixedTicker, LabelSet, Label,
)
from bokeh.palettes import RdYlGn11
from bokeh.transform import transform

from common.models import CONSISTENT_MODELS
from visualize.visualization_config import (
    MODEL_COLOR_DICT, get_ranked_model_order, normalize_model_name,
    convert_to_pdf_auto_size, convert_to_png_auto_size,
)


def _short_scenario(s: str) -> str:
    """Shorten a scenario key for axis labels."""
    return s.replace("-", " ").title()


def _short_model(m: str) -> str:
    """Shorten a model name for axis labels."""
    return m.split("/")[-1]


def _get_vignette_accuracy_order(vignette_model_csv: str):
    """Return models sorted by mean tight accuracy on naive vignettes (best first).

    Uses the same CSV as the analysis, filtered identically to
    ``_prepare_vignette_data``.  This gives an accuracy sort that
    matches the ranking figure on page 1.
    """
    tight_col = "tight(0.85,1.15)"
    vma = pd.read_csv(vignette_model_csv)
    vma = vma[
        (vma["include_nielsen"] == False) &
        (vma["explain_units"] == False) &
        (vma["has_vignette_text"] == True)
    ]
    vma = vma[vma["model"].isin(set(CONSISTENT_MODELS))]
    model_acc = vma.groupby("model")[tight_col].mean().sort_values(ascending=False)
    return list(model_acc.index)


# ═══════════════════════════════════════════════════════════════════
# 1. ICC heatmap
# ═══════════════════════════════════════════════════════════════════

def icc_heatmap(icc_csv: str, width: int = 1100, height: int = 600):
    """
    Bokeh heatmap of ICC values: model (y-axis) × scenario (x-axis).
    Sorted by mean ICC on both axes (highest mean ICC at top / left).
    """
    df = pd.read_csv(icc_csv)

    # Sort models by mean ICC descending (highest at top)
    model_mean = df.groupby("model")["icc"].mean().sort_values(ascending=False)
    models = list(model_mean.index)

    # Sort scenarios by mean ICC descending (highest at left)
    scenario_mean = df.groupby("scenario_key")["icc"].mean().sort_values(ascending=False)
    scenarios = list(scenario_mean.index)

    # Build data source
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model": _short_model(r["model"]),
            "scenario": _short_scenario(r["scenario_key"]),
            "icc": r["icc"],
            "var_between": r["var_between"],
            "var_within": r["var_within"],
            "var_total": r["var_total"],
            "f_pvalue": r["f_pvalue"],
        })
    src = ColumnDataSource(pd.DataFrame(rows))

    short_models = [_short_model(m) for m in models]
    short_scenarios = [_short_scenario(s) for s in scenarios]

    # Colour mapper: green = high ICC, red = low
    mapper = LinearColorMapper(
        palette=list(reversed(RdYlGn11)),
        low=0, high=1,
    )

    p = figure(
        width=width, height=height,
        x_range=short_scenarios, y_range=list(reversed(short_models)),
        toolbar_location=None,
        tools="hover",
        x_axis_location="above",
    )
    p.toolbar.logo = None

    p.rect(
        x="scenario", y="model", width=1, height=1,
        source=src,
        fill_color=transform("icc", mapper),
        line_color="white", line_width=1,
    )

    color_bar = ColorBar(
        color_mapper=mapper, location=(0, 0),
        title="ICC(1,1)", title_text_font_size="10pt",
        ticker=FixedTicker(ticks=[0, 0.25, 0.5, 0.75, 1.0]),
    )
    p.add_layout(color_bar, "right")

    p.hover.tooltips = [
        ("Model", "@model"),
        ("Scenario", "@scenario"),
        ("ICC", "@icc{0.000}"),
        ("σ²_between", "@var_between{0.0000}"),
        ("σ²_within", "@var_within{0.0000}"),
        ("F p-value", "@f_pvalue{0.0000}"),
    ]

    p.xaxis.major_label_orientation = math.pi / 3
    p.xaxis.axis_label_text_font_size = "9pt"
    p.yaxis.axis_label_text_font_size = "9pt"
    p.xaxis.major_label_text_font_size = "8pt"
    p.yaxis.major_label_text_font_size = "9pt"

    title_div = Div(
        text=("<b>Vignette wording explains most response variability (ICC heatmap)</b>"
              "<br><span style='font-size:10pt; color:#555'>ICC(1,1) per model × scenario — "
              "naive condition, 4 text vignettes per scenario. "
              "Green = wording dominates; Red = stochastic noise dominates.</span>"),
        styles={"font-size": "13pt", "margin-left": "10px", "margin-bottom": "6px"},
    )

    return column(title_div, p)


# ═══════════════════════════════════════════════════════════════════
# 2. Bubble grid (concentric circles)
# ═══════════════════════════════════════════════════════════════════

def icc_bubble_grid(icc_csv: str, width: int = 1200, height: int = 650,
                    model_order=None):
    """
    Model (y) × Scenario (x) bubble grid.

    Outer circle (high transparency): area ∝ total variance
    Inner circle (low transparency): area ∝ between-vignette variance (= ICC × total)

    If *model_order* is provided, models are displayed in that order
    (best at top).  Otherwise sorted by lowest mean total variance at top.
    Scenarios sorted by lowest mean total variance at left.
    """
    df = pd.read_csv(icc_csv)

    # Model order: use provided order or fall back to variance sort
    # Reverse so index 0 = bottom of plot, best model = highest index = top
    if model_order is not None:
        models = list(reversed([m for m in model_order if m in df["model"].unique()]))
    else:
        model_mean_var = df.groupby("model")["var_total"].mean().sort_values()
        models = list(model_mean_var.index)

    scenario_mean_var = df.groupby("scenario_key")["var_total"].mean().sort_values()
    scenarios = list(scenario_mean_var.index)

    # Max radius for scaling (in data coords — cell is 1×1)
    max_var = df["var_total"].max()
    MAX_RADIUS = 0.42  # leave a little padding in each cell

    def _radius(var_val):
        """Circle radius such that area ∝ var_val."""
        if var_val <= 0 or max_var <= 0:
            return 0
        # area = π r², so r = sqrt(area / π);  normalise so max -> MAX_RADIUS
        return MAX_RADIUS * math.sqrt(var_val / max_var)

    # Build data
    outer_rows = []
    inner_rows = []
    for _, r in df.iterrows():
        m_idx = models.index(r["model"])
        s_idx = scenarios.index(r["scenario_key"])
        color = MODEL_COLOR_DICT.get(r["model"], "#888888")

        sig_pairs = int(r["sig_pairs"]) if "sig_pairs" in r.index else 0
        total_pairs = int(r["total_pairs"]) if "total_pairs" in r.index else 6

        base = {
            "x": s_idx,
            "y": m_idx,
            "model": _short_model(r["model"]),
            "scenario": _short_scenario(r["scenario_key"]),
            "icc": r["icc"],
            "var_total": r["var_total"],
            "var_between": r["var_between"],
            "var_within": r["var_within"],
            "color": color,
            "sig_pairs_label": f"{sig_pairs}/{total_pairs}",
        }

        outer_rows.append({
            **base,
            "radius": _radius(r["var_total"]),
            "alpha": 0.85,
        })
        inner_rows.append({
            **base,
            "radius": _radius(r["var_between"]),
            "alpha": 0.3,
        })

    outer_src = ColumnDataSource(pd.DataFrame(outer_rows))
    inner_src = ColumnDataSource(pd.DataFrame(inner_rows))

    short_models = [_short_model(m) for m in models]
    short_scenarios = [_short_scenario(s) for s in scenarios]

    p = figure(
        width=width, height=height,
        x_range=(-0.5, len(scenarios) - 0.5),
        y_range=(-0.5, len(models) - 0.5),
        toolbar_location=None,
    )
    p.toolbar.logo = None

    # Grid lines at cell boundaries
    for i in range(len(scenarios) + 1):
        p.line([i - 0.5, i - 0.5], [-0.5, len(models) - 0.5],
               line_color="#e0e0e0", line_width=0.5)
    for i in range(len(models) + 1):
        p.line([-0.5, len(scenarios) - 0.5], [i - 0.5, i - 0.5],
               line_color="#e0e0e0", line_width=0.5)

    # Outer circles (total variance)
    p.circle(
        x="x", y="y", radius="radius", source=outer_src,
        fill_color="color", fill_alpha="alpha",
        line_color="color", line_alpha=0.4, line_width=0.5,
    )

    # Inner circles (between-vignette variance)
    inner_r = p.circle(
        x="x", y="y", radius="radius", source=inner_src,
        fill_color="color", fill_alpha="alpha",
        line_color="color", line_alpha=0.8, line_width=0.5,
    )

    # Hover on inner circles
    hover = HoverTool(renderers=[inner_r], tooltips=[
        ("Model", "@model"),
        ("Scenario", "@scenario"),
        ("ICC", "@icc{0.000}"),
        ("σ²_total", "@var_total{0.0000}"),
        ("σ²_between", "@var_between{0.0000}"),
        ("σ²_within", "@var_within{0.0000}"),
        ("Sig pairs", "@sig_pairs_label"),
    ])
    p.add_tools(hover)

    # Text overlay: X/6 significant pairwise comparisons
    if "sig_pairs" in df.columns:
        label_rows = []
        for _, r in df.iterrows():
            m_idx = models.index(r["model"])
            s_idx = scenarios.index(r["scenario_key"])
            sp = int(r["sig_pairs"])
            tp = int(r["total_pairs"])
            label_rows.append({
                "x": s_idx,
                "y": m_idx,
                "text": f"{sp}/{tp}",
                "text_color": "#000000" if sp > 0 else "#aaaaaa",
            })
        label_src = ColumnDataSource(pd.DataFrame(label_rows))
        labels = LabelSet(
            x="x", y="y", text="text", source=label_src,
            text_font_size="7pt", text_align="center",
            text_baseline="middle", text_color="text_color",
        )
        p.add_layout(labels)

    # Axis ticks
    p.xaxis.ticker = FixedTicker(ticks=list(range(len(scenarios))))
    p.yaxis.ticker = FixedTicker(ticks=list(range(len(models))))
    p.xaxis.major_label_overrides = {i: s for i, s in enumerate(short_scenarios)}
    p.yaxis.major_label_overrides = {i: m for i, m in enumerate(short_models)}
    p.xaxis.major_label_orientation = math.pi / 3
    p.xaxis.major_label_text_font_size = "8pt"
    p.yaxis.major_label_text_font_size = "9pt"
    p.xaxis.axis_label = "Scenario (sorted by lowest mean total variance →)"
    if model_order is not None:
        p.yaxis.axis_label = "Model (sorted by accuracy ↑)"
    else:
        p.yaxis.axis_label = "Model (sorted by lowest mean total variance ↑)"

    p.background_fill_color = "#fafafa"
    p.grid.visible = False

    title_div = Div(
        text=("<b>Variance decomposition: wording vs stochastic noise</b>"
              "<br><span style='font-size:10pt; color:#555'>"
              "Outer circle = total log-ratio variance; "
              "inner circle = between-vignette (wording) variance. "
              "Larger inner fill = more variance from vignette wording (high ICC). "
              "Text = X/6 pairwise vignette comparisons significant "
              "(Bonferroni-corrected Mann–Whitney U, α=0.05).</span>"),
        styles={"font-size": "13pt", "margin-left": "10px", "margin-bottom": "6px"},
    )

    return column(title_div, p)


# ═══════════════════════════════════════════════════════════════════
# 2b. Vignette variance panels (4-panel combined figure)
# ═══════════════════════════════════════════════════════════════════

# ── Shared helpers ──────────────────────────────────────────────────

_CV_LABELS_VIGNETTE = ["CV=0", "0<CV≤0.2", "CV>0.2"]
_ACC_LABELS = ["tight", "broad", "neither"]


def _cv_bin_vignette(cv):
    """Assign a vignette CV value to one of three bins (zero / low / high)."""
    if cv <= 0:
        return _CV_LABELS_VIGNETTE[0]
    elif cv <= 0.2:
        return _CV_LABELS_VIGNETTE[1]
    return _CV_LABELS_VIGNETTE[2]


def _prepare_vignette_data(icc_csv, vignette_model_csv):
    """Load and classify the filtered vignette data.

    Returns (vma, icc_df, var_labels, t1, t2, ranked_models, icc_labels, icc_t1, icc_t2).
    """
    icc_df = pd.read_csv(icc_csv)
    vma = pd.read_csv(vignette_model_csv)

    # Filter to naive, text (include multi-from to keep all 20 scenarios)
    vma = vma[
        (vma["include_nielsen"] == False) &
        (vma["explain_units"] == False) &
        (vma["has_vignette_text"] == True)
    ]
    vma = vma[vma["model"].isin(set(CONSISTENT_MODELS))]

    tight_col = "tight(0.85,1.15)"
    broad_col = "broad(0.6,1.7)"

    # Accuracy classification
    vma["_acc"] = "neither"
    vma.loc[vma[broad_col] >= 0.5, "_acc"] = "broad"
    vma.loc[vma[tight_col] >= 0.5, "_acc"] = "tight"

    # CV bins (vignette-specific: zero / low / high)
    vma["_cv_bin"] = vma["observed_dose_cv"].apply(_cv_bin_vignette)

    # Scenario variance terciles
    scenario_var = icc_df.groupby("scenario_key")["var_total"].mean()
    t1, t2 = np.percentile(scenario_var.values, [33.3, 66.7])
    var_labels = [f"Low (<{t1:.2f})", f"Mid ({t1:.2f}–{t2:.2f})", f"High (≥{t2:.2f})"]

    def _var_bin(scenario):
        v = scenario_var.get(scenario, 0)
        if v < t1:
            return var_labels[0]
        elif v < t2:
            return var_labels[1]
        return var_labels[2]

    vma["_var_bin"] = vma["scenario_key"].apply(_var_bin)

    # ICC terciles (per model×scenario from icc_df)
    icc_vals = icc_df[icc_df["model"].isin(set(CONSISTENT_MODELS))]["icc"]
    icc_t1, icc_t2 = np.percentile(icc_vals.dropna().values, [33.3, 66.7])
    icc_labels = [f"Low (<{icc_t1:.2f})", f"Mid ({icc_t1:.2f}–{icc_t2:.2f})", f"High (≥{icc_t2:.2f})"]

    ranked = get_ranked_model_order()
    models = [m for m in ranked if m in vma["model"].unique()]
    if not models:
        models = sorted(vma["model"].unique())

    return vma, icc_df, var_labels, t1, t2, models, icc_labels, icc_t1, icc_t2


def _build_circle_grid(
    vma, models, x_col, y_col, x_labels, y_labels,
    title_html, x_axis_note=None,
    width=900, height=500,
    precomputed_pvals=None,
):
    """Generic per-model circle-area grid with chi-square p-values.

    Returns a Bokeh ``column`` with a title Div and a gridplot.
    ``x_col`` and ``y_col`` are column names in *vma* containing
    the categorical bin assignments.

    Each sub-plot title includes the model name and the χ² p-value.
    The title Div includes an x-axis annotation.

    If ``precomputed_pvals`` is a dict mapping model names to p-values,
    those are used directly.  Otherwise p-values are omitted.
    """
    from bokeh.layouts import gridplot

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

    # Use pre-computed chi-square p-values if provided
    model_pvals = precomputed_pvals or {}

    all_rows = []
    idx = 0
    for row_sz in row_sizes:
        row_figs = []
        for _ in range(row_sz):
            if idx >= n:
                break
            m = models[idx]
            idx += 1
            cn = _short_model(m)
            color = MODEL_COLOR_DICT.get(m, "#888888")

            mdf = vma[vma["model"] == m]
            n_total = len(mdf)

            xs, ys, sizes, alphas, pcts = [], [], [], [], []
            for xl in x_labels:
                for yl in y_labels:
                    count = len(mdf[(mdf[x_col] == xl) & (mdf[y_col] == yl)])
                    prop = count / n_total if n_total else 0
                    xs.append(xl)
                    ys.append(yl)
                    sizes.append(max(3, prop ** 0.5 * max_circle))
                    alphas.append(0.8 if prop > 0 else 0.05)
                    pcts.append(f"{prop:.0%}")

            pval = model_pvals.get(m, np.nan)
            if np.isnan(pval):
                p_str = "χ²: N/A"
            elif pval < 0.001:
                p_str = "χ² p<.001"
            elif pval < 0.05:
                p_str = f"χ² p={pval:.3f}"
            else:
                p_str = f"χ² p={pval:.2f}"

            p = figure(
                width=cell_w, height=cell_h,
                x_range=x_labels, y_range=list(reversed(y_labels)),
                toolbar_location=None,
                title=f"{cn}  ({p_str})",
            )
            src = ColumnDataSource(dict(
                x=xs, y=ys, size=sizes, alpha=alphas, pct=pcts,
            ))
            p.scatter("x", "y", size="size", color=color, alpha="alpha",
                      source=src)
            p.add_tools(HoverTool(tooltips=[
                ("Column", "@x"), ("Row", "@y"), ("Proportion", "@pct"),
            ]))

            p.title.text_font_size = "7pt"
            p.xaxis.major_label_text_font_size = "6pt"
            p.yaxis.major_label_text_font_size = "6pt"
            p.xaxis.major_label_orientation = math.pi / 6
            p.xaxis.axis_label = None
            p.yaxis.axis_label = None
            p.xgrid.visible = False
            p.ygrid.visible = False
            p.min_border_left = 50
            p.min_border_right = 5
            p.min_border_top = 20
            p.min_border_bottom = 30

            row_figs.append(p)

        while len(row_figs) < row_sizes[0]:
            row_figs.append(None)
        all_rows.append(row_figs)

    grid = gridplot(all_rows, merge_tools=False, toolbar_location=None)
    x_ann = x_axis_note or ""
    title_div = Div(
        text=(title_html +
              f"<br><span style='font-size:9pt; color:#666'>"
              f"x-axis: {x_ann} &nbsp;|&nbsp; "
              f"circle area ∝ proportion of model's observations (all 9 cells sum to 100%)</span>"),
        styles={"font-size": "12pt", "margin-left": "10px"},
    )
    return column(title_div, grid)


# ── Panel 1: Accuracy × Scenario Variance ──────────────────────────

def _accuracy_by_variance_grid(vma, models, var_labels, width=900, height=500,
                               precomputed_pvals=None):
    """Per-model grid: accuracy category (y) × scenario variance tercile (x)."""
    return _build_circle_grid(
        vma, models,
        x_col="_var_bin", y_col="_acc",
        x_labels=var_labels, y_labels=_ACC_LABELS,
        title_html=(
            "<b>Distribution of Accuracy × Scenario Log-ratio Variance</b>"
            "<br><span style='font-size:10pt; color:#555'>"
            "Columns = scenario variance terciles (mean total log-ratio variance "
            "across models). Rows = accuracy categories.</span>"
        ),
        x_axis_note="scenario log-ratio variance tercile",
        width=width, height=height,
        precomputed_pvals=precomputed_pvals,
    )


# ── Panel 3: Scenario Variance × within-Vignette CV ────────────────

def _variance_by_cv_grid(vma, models, var_labels, width=900, height=500,
                         precomputed_pvals=None):
    """Per-model grid: scenario variance tercile (y) × vignette CV bin (x)."""
    return _build_circle_grid(
        vma, models,
        x_col="_cv_bin", y_col="_var_bin",
        x_labels=_CV_LABELS_VIGNETTE, y_labels=var_labels,
        title_html=(
            "<b>Scenario Log-ratio Variance × within-Vignette CV</b>"
            "<br><span style='font-size:10pt; color:#555'>"
            "Columns = within-vignette CV bins. "
            "Rows = scenario log-ratio variance terciles.</span>"
        ),
        x_axis_note="within-vignette CV",
        width=width, height=height,
        precomputed_pvals=precomputed_pvals,
    )


# ── Panel 4: Total variance vs ICC scatter ──────────────────────────

def _variance_vs_icc_scatter(icc_csv, width=900, height=400,
                             spearman_csv=None):
    """Scatter of total log-ratio variance (x) vs ICC (y), coloured by model.

    Reads Spearman ρ from pre-computed ``spearman_csv`` (produced by
    ``analysis.variance_icc_stats``) and annotates the plot.
    """
    df = pd.read_csv(icc_csv)
    df = df[df["model"].isin(set(CONSISTENT_MODELS))]

    ranked = get_ranked_model_order()
    models = [m for m in ranked if m in df["model"].unique()]

    xs, ys, colors, mlabels, slabels = [], [], [], [], []
    for m in models:
        mdf = df[df["model"] == m]
        for _, r in mdf.iterrows():
            xs.append(r["var_total"])
            ys.append(r["icc"])
            colors.append(MODEL_COLOR_DICT.get(m, "#888888"))
            mlabels.append(_short_model(m))
            slabels.append(_short_scenario(r["scenario_key"]))

    # Read pre-computed Spearman correlation
    rho, sp_pval = np.nan, np.nan
    if spearman_csv and os.path.exists(spearman_csv):
        sp_df = pd.read_csv(spearman_csv)
        overall = sp_df[
            (sp_df["stat_type"] == "variance_vs_icc") &
            (sp_df["model"] == "OVERALL")
        ]
        if not overall.empty:
            rho = float(overall.iloc[0]["rho"])
            sp_pval = float(overall.iloc[0]["p_value"])

    src = ColumnDataSource(dict(
        x=xs, y=ys, color=colors, model=mlabels, scenario=slabels,
    ))

    p = figure(
        width=width, height=height,
        toolbar_location=None,
        y_range=(-0.05, 1.05),
    )
    p.scatter("x", "y", color="color", alpha=0.55, size=7, source=src)
    p.add_tools(HoverTool(tooltips=[
        ("Model", "@model"),
        ("Scenario", "@scenario"),
        ("σ²_total", "@x{0.0000}"),
        ("ICC", "@y{0.000}"),
    ]))

    p.xaxis.axis_label = "Total log-ratio variance (σ²_total)"
    p.yaxis.axis_label = "ICC  (0 = no wording effect → 1 = all variance from wording)"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.axis_label_text_font_size = "10pt"

    # Spearman annotation
    sp_str = f"Spearman ρ = {rho:.3f}, p {'< 0.001' if sp_pval < 0.001 else f'= {sp_pval:.3f}'}"
    p.add_layout(Label(
        x=5, y=5, text=sp_str,
        text_font_size="9pt", text_color="#333",
        x_units="screen", y_units="screen",
        text_align="left", text_baseline="bottom",
    ))

    title_div = Div(
        text=("<b>Total scenario variance vs ICC (naive condition)</b>"
              "<br><span style='font-size:10pt; color:#555'>"
              f"Each dot is one model × scenario cell. Colour = model. "
              f"{sp_str}. "
              "Scenarios with higher total variance tend to show higher ICC, "
              "meaning vignette wording drives most of the spread.</span>"),
        styles={"font-size": "12pt", "margin-left": "10px", "margin-bottom": "6px"},
    )
    return column(title_div, p), rho, sp_pval


# ── Panel 5: Variance tercile × ICC tercile grid ───────────────────

def _variance_by_icc_grid(icc_df, models, var_labels, t1, t2,
                          icc_labels, icc_t1, icc_t2,
                          width=900, height=500,
                          precomputed_pvals=None):
    """Per-model grid: scenario variance tercile (y) × ICC tercile (x).

    Operates on icc_df (one row per model×scenario) rather than vma
    (one row per model×vignette).

    If ``precomputed_pvals`` is provided (dict of model→p-value), those
    are used for the chi-square annotations; otherwise they are omitted.
    """
    from bokeh.layouts import gridplot

    df = icc_df[icc_df["model"].isin(set(CONSISTENT_MODELS))].copy()

    # Variance bin per row
    scenario_var = df.groupby("scenario_key")["var_total"].mean()

    def _v_bin(row):
        v = scenario_var.get(row["scenario_key"], 0)
        if v < t1:
            return var_labels[0]
        elif v < t2:
            return var_labels[1]
        return var_labels[2]

    def _i_bin(row):
        icc = row["icc"]
        if np.isnan(icc):
            return icc_labels[0]
        if icc < icc_t1:
            return icc_labels[0]
        elif icc < icc_t2:
            return icc_labels[1]
        return icc_labels[2]

    df["_var_bin"] = df.apply(_v_bin, axis=1)
    df["_icc_bin"] = df.apply(_i_bin, axis=1)

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

    # Use pre-computed chi-square p-values if provided
    model_pvals = precomputed_pvals or {}

    all_rows_figs = []
    idx = 0
    for row_sz in row_sizes:
        row_figs = []
        for _ in range(row_sz):
            if idx >= n:
                break
            m = models[idx]
            idx += 1
            cn = _short_model(m)
            color = MODEL_COLOR_DICT.get(m, "#888888")

            mdf = df[df["model"] == m]
            n_total = len(mdf)

            xs, ys, sizes, alphas, pcts = [], [], [], [], []
            for xl in icc_labels:
                for yl in var_labels:
                    count = int(((mdf["_icc_bin"] == xl) & (mdf["_var_bin"] == yl)).sum())
                    prop = count / n_total if n_total else 0
                    xs.append(xl)
                    ys.append(yl)
                    sizes.append(max(3, prop ** 0.5 * max_circle))
                    alphas.append(0.8 if prop > 0 else 0.05)
                    pcts.append(f"{prop:.0%}")

            pval = model_pvals.get(m, np.nan)
            if np.isnan(pval):
                p_str = "χ²: N/A"
            elif pval < 0.001:
                p_str = "χ² p<.001"
            elif pval < 0.05:
                p_str = f"χ² p={pval:.3f}"
            else:
                p_str = f"χ² p={pval:.2f}"

            p = figure(
                width=cell_w, height=cell_h,
                x_range=icc_labels, y_range=list(reversed(var_labels)),
                toolbar_location=None,
                title=f"{cn}  ({p_str})",
            )
            src = ColumnDataSource(dict(
                x=xs, y=ys, size=sizes, alpha=alphas, pct=pcts,
            ))
            p.scatter("x", "y", size="size", color=color, alpha="alpha",
                      source=src)
            p.add_tools(HoverTool(tooltips=[
                ("ICC Bin", "@x"), ("Variance Bin", "@y"), ("Proportion", "@pct"),
            ]))

            p.title.text_font_size = "7pt"
            p.xaxis.major_label_text_font_size = "6pt"
            p.yaxis.major_label_text_font_size = "6pt"
            p.xaxis.major_label_orientation = math.pi / 6
            p.xgrid.visible = False
            p.ygrid.visible = False
            p.min_border_left = 50
            p.min_border_right = 5
            p.min_border_top = 20
            p.min_border_bottom = 30

            row_figs.append(p)

        while len(row_figs) < row_sizes[0]:
            row_figs.append(None)
        all_rows_figs.append(row_figs)

    grid = gridplot(all_rows_figs, merge_tools=False, toolbar_location=None)
    title_div = Div(
        text=(
            "<b>Scenario Log-ratio Variance × ICC Tercile</b>"
            "<br><span style='font-size:10pt; color:#555'>"
            "Columns = ICC terciles per model×scenario cell. "
            "Rows = scenario log-ratio variance terciles.</span>"
            "<br><span style='font-size:9pt; color:#666'>"
            "x-axis: ICC tercile &nbsp;|&nbsp; "
            "circle area ∝ proportion of model's 20 scenario cells (all 9 bins sum to 100%)</span>"
        ),
        styles={"font-size": "12pt", "margin-left": "10px"},
    )
    return column(title_div, grid)


# ── Stacked variance bar chart ──────────────────────────────────────

def _prepare_scenario_level_data(vma, icc_df):
    """Aggregate vignette-level *vma* to model×scenario, merge with ICC.

    Returns a DataFrame with columns:
      model, scenario_key, tight_prop, mean_cv, var_total, var_between,
      var_within, icc
    """
    tight_col = "tight(0.85,1.15)"
    scen = vma.groupby(["model", "scenario_key"]).agg(
        tight_prop=(tight_col, "mean"),
        mean_cv=("observed_dose_cv", "mean"),
    ).reset_index()
    icc_sub = icc_df[icc_df["model"].isin(set(CONSISTENT_MODELS))][
        ["model", "scenario_key", "var_total", "var_between", "var_within", "icc"]
    ].copy()
    return scen.merge(icc_sub, on=["model", "scenario_key"], how="inner")


def stacked_variance_bars(
    icc_csv: str,
    vignette_model_csv: str,
    width: int = 800,
    height: int = 500,
    spearman_csv: str | None = None,
):
    """Stacked horizontal bar: within-vignette vs between-vignette variance
    per model, sorted by vignette accuracy (best at top).

    The headline: models are precise (small within-vignette noise) but
    wording-sensitive (large between-vignette variance), unless they are
    accurate — in which case both components collapse.

    Reads Spearman ρ and between-ratio from a pre-computed
    ``spearman_csv`` produced by ``analysis.variance_icc_stats``.
    """
    icc_df = pd.read_csv(icc_csv)
    icc_df = icc_df[icc_df["model"].isin(set(CONSISTENT_MODELS))]

    # Accuracy-sorted order
    acc_order = _get_vignette_accuracy_order(vignette_model_csv)

    # Per-model means
    model_var = icc_df.groupby("model").agg(
        mean_within=("var_within", "mean"),
        mean_between=("var_between", "mean"),
        mean_total=("var_total", "mean"),
        mean_icc=("icc", "mean"),
    ).reindex(acc_order)

    # Get accuracy for annotation
    tight_col = "tight(0.85,1.15)"
    vma = pd.read_csv(vignette_model_csv)
    vma = vma[
        (vma["include_nielsen"] == False) &
        (vma["explain_units"] == False) &
        (vma["has_vignette_text"] == True)
    ]
    vma = vma[vma["model"].isin(set(CONSISTENT_MODELS))]
    model_acc = vma.groupby("model")[tight_col].mean().reindex(acc_order)

    # Build labels (reversed for bottom→top)
    clean_names = [_short_model(m) for m in acc_order]
    clean_names_rev = list(reversed(clean_names))
    models_rev = list(reversed(acc_order))

    within_vals = list(reversed(model_var["mean_within"].tolist()))
    between_vals = list(reversed(model_var["mean_between"].tolist()))
    total_vals = list(reversed(model_var["mean_total"].tolist()))
    icc_vals = list(reversed(model_var["mean_icc"].tolist()))
    acc_vals = list(reversed(model_acc.tolist()))
    colors = [MODEL_COLOR_DICT.get(m, "#888888") for m in models_rev]

    src = ColumnDataSource(dict(
        y=clean_names_rev,
        within=within_vals,
        between=between_vals,
        total=total_vals,
        icc=icc_vals,
        acc=acc_vals,
        color=colors,
        left_between=within_vals,  # between bar starts where within ends
    ))

    x_max = max(total_vals) * 1.3
    p = figure(
        width=width, height=height,
        y_range=clean_names_rev,
        x_range=(0, x_max),
        toolbar_location=None,
    )

    # Within-vignette (stochastic noise) — darker, from 0
    p.hbar(y="y", right="within", height=0.6, color="color",
           alpha=0.9, source=src,
           legend_label="Within-vignette (stochastic noise)")

    # Between-vignette (wording effect) — lighter, stacked
    p.hbar(y="y", left="within", right="total", height=0.6,
           color="color", alpha=0.3, source=src,
           legend_label="Between-vignette (wording effect)")

    p.add_tools(HoverTool(tooltips=[
        ("Model", "@y"),
        ("σ²_within (noise)", "@within{0.0000}"),
        ("σ²_between (wording)", "@between{0.0000}"),
        ("σ²_total", "@total{0.0000}"),
        ("Mean ICC", "@icc{0.000}"),
        ("Tight accuracy", "@acc{0.0%}"),
    ]))

    # Accuracy annotation on right side
    for i, (name, acc_v) in enumerate(zip(clean_names_rev, acc_vals)):
        total_v = total_vals[i]
        p.text(
            x=[total_v + x_max * 0.01], y=[name],
            text=[f"{acc_v:.0%} tight acc"],
            text_font_size="8pt", text_color="#555",
            text_baseline="middle", text_align="left",
        )

    p.xaxis.axis_label = "Mean log-ratio variance"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "9pt"
    p.ygrid.visible = False

    p.legend.location = "bottom_right"
    p.legend.label_text_font_size = "9pt"
    p.legend.background_fill_alpha = 0.8

    # Overall stats for subtitle (read from pre-computed CSV)
    overall_ratio = icc_df["var_between"].mean() / icc_df["var_total"].mean()
    rho, sp_p = np.nan, np.nan
    if spearman_csv and os.path.exists(spearman_csv):
        sp_df = pd.read_csv(spearman_csv)
        acc_v_row = sp_df[
            (sp_df["stat_type"] == "accuracy_vs_variance") &
            (sp_df["model"] == "ALL")
        ]
        if not acc_v_row.empty:
            rho = float(acc_v_row.iloc[0]["rho"])
            sp_p = float(acc_v_row.iloc[0]["p_value"])

    spearman_str = ""
    if not np.isnan(rho):
        spearman_str = (
            f"Spearman ρ (accuracy vs total variance) = {rho:.2f}, "
            f"p {'< 0.001' if sp_p < 0.001 else f'= {sp_p:.3f}'}."
        )

    title_div = Div(
        text=(
            "<b>Models are precise but wording-sensitive</b>"
            "<br><span style='font-size:10pt; color:#555'>"
            "Stacked bar: within-vignette variance (stochastic noise, dark) + "
            "between-vignette variance (wording effect, light). "
            f"Overall, {overall_ratio:.0%} of variance is between-vignette. "
            "Both components collapse for accurate models — they converge on "
            "the right answer regardless of phrasing. "
            f"{spearman_str}"
            "</span>"
        ),
        styles={"font-size": "13pt", "margin-left": "10px", "margin-bottom": "6px"},
    )
    return column(title_div, p)


# ── Spearman ρ bar chart (variance ↔ ICC per model) ────────────────

def spearman_variance_icc_bars(
    icc_csv: str,
    vignette_model_csv: str,
    width: int = 800,
    height: int = 500,
    spearman_csv: str | None = None,
):
    """Horizontal bar chart of per-model Spearman ρ (variance ↔ ICC),
    sorted by vignette accuracy (best at top), with p-value annotations.

    Reads per-model Spearman ρ and the binomial sign-test summary
    from a pre-computed ``spearman_csv`` produced by
    ``analysis.variance_icc_stats``.
    """
    icc_df = pd.read_csv(icc_csv)
    icc_df = icc_df[icc_df["model"].isin(set(CONSISTENT_MODELS))]
    acc_order = _get_vignette_accuracy_order(vignette_model_csv)

    # Read pre-computed per-model Spearman stats
    model_stats = {}
    if spearman_csv and os.path.exists(spearman_csv):
        sp_df = pd.read_csv(spearman_csv)
        per_model = sp_df[sp_df["stat_type"] == "variance_vs_icc"]
        for _, r in per_model.iterrows():
            if r["model"] != "OVERALL":
                model_stats[r["model"]] = (float(r["rho"]), float(r["p_value"]))

    # Fill in any models without precomputed data
    for m in acc_order:
        if m not in model_stats:
            model_stats[m] = (np.nan, np.nan)

    clean_names = [_short_model(m) for m in acc_order]
    clean_names_rev = list(reversed(clean_names))
    models_rev = list(reversed(acc_order))

    rhos = [model_stats[m][0] for m in models_rev]
    pvals = [model_stats[m][1] for m in models_rev]
    colors = [MODEL_COLOR_DICT.get(m, "#888888") for m in models_rev]

    src = ColumnDataSource(dict(
        y=clean_names_rev,
        rho=rhos,
        pval=pvals,
        color=colors,
    ))

    rho_max = max(abs(r) for r in rhos) * 1.3
    p = figure(
        width=width, height=height,
        y_range=clean_names_rev,
        x_range=(-rho_max, rho_max),
        toolbar_location=None,
    )

    # Bars
    p.hbar(y="y", right="rho", height=0.6, color="color",
           alpha=0.8, source=src)

    # Zero line
    p.line([0, 0], [-0.5, len(models_rev) - 0.5 + 1],
           line_color="#333", line_width=1, line_dash="dashed")

    p.add_tools(HoverTool(tooltips=[
        ("Model", "@y"),
        ("Spearman ρ", "@rho{0.000}"),
        ("p-value", "@pval{0.0000}"),
    ]))

    # Significance annotations
    for i, (name, rho_v, pv) in enumerate(zip(clean_names_rev, rhos, pvals)):
        if pv < 0.001:
            star = "***"
        elif pv < 0.01:
            star = "**"
        elif pv < 0.05:
            star = "*"
        else:
            star = ""

        p_str = f"p<.001" if pv < 0.001 else f"p={pv:.3f}"
        label = f"ρ={rho_v:.2f} {p_str}{star}"
        x_pos = rho_v + (0.02 if rho_v >= 0 else -0.02)
        align = "left" if rho_v >= 0 else "right"

        p.text(
            x=[x_pos], y=[name], text=[label],
            text_font_size="7pt", text_color="#333",
            text_baseline="middle", text_align=align,
        )

    p.xaxis.axis_label = "Spearman ρ  (variance ↔ ICC)"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "9pt"
    p.ygrid.visible = False

    # Sign test summary (read from pre-computed CSV)
    sign_str = ""
    if spearman_csv and os.path.exists(spearman_csv):
        sp_df = pd.read_csv(spearman_csv)
        sign_row = sp_df[sp_df["stat_type"] == "sign_test"]
        if not sign_row.empty:
            sr = sign_row.iloc[0]
            n_sig_models = int(sr.get("n_individually_sig", 0))
            n_pos = int(sr.get("n_positive", 0))
            n_neg = int(sr.get("n_negative", 0))
            sign_p = float(sr["p_value"])
            sign_str = (
                f"{n_sig_models}/{len(acc_order)} models individually significant (p<0.05). "
                f"{n_pos} positive, {n_neg} negative. "
                f"Sign test p={'<0.001' if sign_p < 0.001 else f'= {sign_p:.3f}'}."
            )

    title_div = Div(
        text=(
            "<b>Wording-driven variance tracks ICC for many models</b>"
            "<br><span style='font-size:10pt; color:#555'>"
            "Per-model Spearman ρ between scenario log-ratio variance and ICC "
            "(n=20 scenarios per model). "
            "Positive ρ = scenarios with more total variance have a larger "
            "proportion attributable to vignette wording. "
            f"{sign_str}"
            "</span>"
        ),
        styles={"font-size": "13pt", "margin-left": "10px", "margin-bottom": "6px"},
    )
    return column(title_div, p)


# ── Page-level figure builders ──────────────────────────────────────

def variance_stacked_figure(
    icc_csv: str,
    vignette_model_csv: str,
    width: int = 800,
    spearman_csv: str | None = None,
):
    """Page 6a: Stacked variance bars + overall scatter.

    Stacked bar shows within vs between variance per model.
    Scatter shows total variance vs ICC across all 260 model×scenario cells.
    """
    bar_panel = stacked_variance_bars(icc_csv, vignette_model_csv, width=width,
                                      spearman_csv=spearman_csv)
    scatter_panel, _rho, _sp_p = _variance_vs_icc_scatter(icc_csv, width=width,
                                                           spearman_csv=spearman_csv)
    return column(bar_panel, scatter_panel)


def bubble_grid_figure(
    icc_csv: str,
    vignette_model_csv: str,
    width: int = 1200,
    height: int = 700,
):
    """Page 6b: ICC bubble grid sorted by accuracy."""
    acc_order = _get_vignette_accuracy_order(vignette_model_csv)
    return icc_bubble_grid(icc_csv, width=width, height=height,
                           model_order=acc_order)


def spearman_bars_figure(
    icc_csv: str,
    vignette_model_csv: str,
    width: int = 800,
    spearman_csv: str | None = None,
):
    """Page 6c: Per-model Spearman ρ bar chart (variance ↔ ICC)."""
    return spearman_variance_icc_bars(icc_csv, vignette_model_csv, width=width,
                                      spearman_csv=spearman_csv)


# ── Chi-square independence tests for 3×3 grids ────────────────────

def compute_grid_independence_tests(icc_csv, vignette_model_csv, output_csv=None):
    """Delegate to ``analysis.grid_independence_tests``.

    Kept as a thin wrapper for backward compatibility.  All statistical
    computation now lives in the analysis module.
    """
    from analysis.grid_independence_tests import (
        compute_grid_independence_tests as _compute,
    )
    return _compute(icc_csv, vignette_model_csv, output_csv=output_csv)


# ── Combined multi-panel figure (legacy wrapper) ────────────────────

def vignette_variance_panels(
    icc_csv: str,
    vignette_model_csv: str,
    results_csv: str = None,
    width: int = 900,
    height_grid: int = 500,
    height_mid: int = 700,
    height_scatter: int = 400,
):
    """Legacy wrapper — returns the stacked bar + scatter as a single column.

    Kept for backward compatibility but the three separate page-level
    functions (``variance_stacked_figure``, ``bubble_grid_figure``,
    ``spearman_bars_figure``) are preferred.
    """
    bar_panel = stacked_variance_bars(icc_csv, vignette_model_csv,
                                      width=width)
    scatter_panel, _rho, _sp_p = _variance_vs_icc_scatter(
        icc_csv, width=width, height=height_scatter)

    if results_csv:
        compute_grid_independence_tests(icc_csv, vignette_model_csv,
                                        output_csv=results_csv)

    return column(bar_panel, scatter_panel)


# ═══════════════════════════════════════════════════════════════════
# 3. Kendall's W summary bar chart
# ═══════════════════════════════════════════════════════════════════

def kendall_w_summary(
    kendall_csv: str,
    icc_csv: str = None,
    width: int = 900,
    height: int = 500,
):
    """
    Horizontal bar chart of Mean ICC and Kendall's W per scenario,
    sorted by mean ICC (highest at top).

    ICC (intraclass correlation coefficient) measures the proportion of
    response-log-ratio variance attributable to vignette wording within
    each model–scenario cell.  Kendall's W measures cross-model agreement
    on which wording produces higher or lower responses.  Both range 0–1
    and are displayed on the same axis.
    """
    w_df = pd.read_csv(kendall_csv)

    # Build ICC means per scenario if available
    icc_means = None
    if icc_csv and os.path.exists(icc_csv):
        icc_df = pd.read_csv(icc_csv)
        icc_means = icc_df.groupby("scenario_key")["icc"].mean()

    # Sort by mean ICC (descending → highest at top after reversal)
    if icc_means is not None:
        w_df["_icc_mean"] = w_df["scenario_key"].map(icc_means).fillna(0)
        w_df = w_df.sort_values("_icc_mean", ascending=True)  # lowest at bottom → highest at top
    else:
        w_df = w_df.sort_values("kendall_w", ascending=True)

    scenarios = list(w_df["scenario_key"])
    short_scenarios = [_short_scenario(s) for s in scenarios]

    rows = []
    for _, r in w_df.iterrows():
        rows.append({
            "scenario": _short_scenario(r["scenario_key"]),
            "w": r["kendall_w"],
            "p": r["kendall_p"],
            "sig": "p<0.05" if r["kendall_p"] < 0.05 else "ns",
        })
    w_data = pd.DataFrame(rows)

    p = figure(
        width=width, height=height,
        y_range=short_scenarios,
        x_range=(0, 1.05),
        toolbar_location=None,
        tools="hover",
    )
    p.toolbar.logo = None

    # ICC bars (background, wider, lighter)
    if icc_means is not None:
        icc_rows = []
        for s in scenarios:
            icc_val = icc_means.get(s, 0)
            icc_rows.append({
                "scenario": _short_scenario(s),
                "icc": icc_val,
            })
        icc_src = ColumnDataSource(pd.DataFrame(icc_rows))
        p.hbar(
            y="scenario", right="icc", source=icc_src,
            height=0.6, fill_color="#4a90d9", fill_alpha=0.3,
            line_color="#4a90d9", line_alpha=0.5, line_width=0.5,
            legend_label="Mean ICC (vignette effect size)",
        )

    # Kendall's W bars (foreground, narrower, solid)
    w_src = ColumnDataSource(w_data)
    p.hbar(
        y="scenario", right="w", source=w_src,
        height=0.35, fill_color="#d94a4a", fill_alpha=0.7,
        line_color="#d94a4a", line_width=0.5,
        legend_label="Kendall's W (cross-model ranking agreement)",
    )

    p.hover.tooltips = [
        ("Scenario", "@scenario"),
        ("Kendall's W", "@w{0.000}"),
        ("p-value", "@p{0.0000}"),
        ("Significance", "@sig"),
    ]

    p.xaxis.axis_label = "ICC / Kendall W  (0 = no wording effect  →  1 = all variance from vignette wording)"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "8pt"

    p.legend.location = "bottom_right"
    p.legend.label_text_font_size = "9pt"
    p.legend.background_fill_alpha = 0.8

    title_div = Div(
        text=("<b>Wording effect sizes and directions are model-specific (naive condition)</b>"
              "<br><span style='font-size:10pt; color:#555'>"
              "Blue bars: mean ICC (intraclass correlation) — proportion of log-ratio "
              "variance explained by vignette wording within each model. "
              "Red bars: Kendall's W — agreement across models on which vignette wording "
              "produces higher/lower responses. Both range 0–1. "
              "High ICC + low W = wording matters, but "
              "models are sensitive to different aspects of the phrasing.</span>"),
        styles={"font-size": "13pt", "margin-left": "10px", "margin-bottom": "6px"},
    )

    return column(title_div, p)


# ═══════════════════════════════════════════════════════════════════
# Convenience: generate all ICC figures
# ═══════════════════════════════════════════════════════════════════

async def save_icc_figures(
    icc_csv: str = "results/icc_analysis.csv",
    kendall_csv: str = "results/kendall_w_vignettes.csv",
    figures_dir: str = "figures",
):
    """Save all ICC figures as HTML + PDF + PNG."""
    os.makedirs(figures_dir, exist_ok=True)

    for name, builder in [
        ("icc_heatmap", lambda: icc_heatmap(icc_csv)),
        ("icc_bubble_grid", lambda: icc_bubble_grid(icc_csv)),
        ("kendall_w_summary", lambda: kendall_w_summary(kendall_csv, icc_csv)),
    ]:
        fig = builder()
        html_path = os.path.join(figures_dir, f"{name}.html")
        pdf_path = os.path.join(figures_dir, f"{name}.pdf")
        png_path = os.path.join(figures_dir, f"{name}.png")
        output_file(html_path)
        save(fig)
        await convert_to_pdf_auto_size(html_path, pdf_path, scale=2.0)
        await convert_to_png_auto_size(html_path, png_path, scale=2.0)
        print(f"✓ {name}")
