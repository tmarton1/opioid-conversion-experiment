"""
ICC heatmap and bubble-grid visualizations.

Two figures from results/icc_analysis.csv:

1. **ICC Heatmap** — model × scenario, colour-mapped by ICC value,
   sorted by mean ICC on both axes.

2. **Bubble Grid** — model × scenario.  Each cell has two concentric circles:
   - Outer (high transparency): area ∝ total variance  (σ²_total)
   - Inner (low transparency): area ∝ between-vignette variance (σ²_between)
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
    LinearColorMapper, FixedTicker, LabelSet,
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
        toolbar_location="above",
        tools="hover,save",
        x_axis_location="above",
    )

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

def icc_bubble_grid(icc_csv: str, width: int = 1200, height: int = 650):
    """
    Model (y) × Scenario (x) bubble grid.

    Outer circle (high transparency): area ∝ total variance
    Inner circle (low transparency): area ∝ between-vignette variance (= ICC × total)

    Models sorted by lowest mean total variance at top.
    Scenarios sorted by lowest mean total variance at left.
    """
    df = pd.read_csv(icc_csv)

    # Sort by mean total variance — lowest at top / left
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
            "alpha": 0.25,
        })
        inner_rows.append({
            **base,
            "radius": _radius(r["var_between"]),
            "alpha": 0.75,
        })

    outer_src = ColumnDataSource(pd.DataFrame(outer_rows))
    inner_src = ColumnDataSource(pd.DataFrame(inner_rows))

    short_models = [_short_model(m) for m in models]
    short_scenarios = [_short_scenario(s) for s in scenarios]

    p = figure(
        width=width, height=height,
        x_range=(-0.5, len(scenarios) - 0.5),
        y_range=(-0.5, len(models) - 0.5),
        toolbar_location="above",
        tools="save",
    )

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
# 3. Kendall's W summary bar chart
# ═══════════════════════════════════════════════════════════════════

def kendall_w_summary(
    kendall_csv: str,
    icc_csv: str = None,
    width: int = 900,
    height: int = 500,
):
    """
    Horizontal bar chart of Kendall's W per scenario.

    If icc_csv is provided, overlays mean ICC per scenario as a second bar
    for comparison (ICC = within-scenario vignette effect size,
    W = cross-model ranking agreement).
    """
    w_df = pd.read_csv(kendall_csv).sort_values("kendall_w", ascending=True)

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

    # Optionally load ICC means per scenario
    icc_means = None
    if icc_csv and os.path.exists(icc_csv):
        icc_df = pd.read_csv(icc_csv)
        icc_means = icc_df.groupby("scenario_key")["icc"].mean()

    p = figure(
        width=width, height=height,
        y_range=short_scenarios,
        x_range=(0, 1.05),
        toolbar_location="above",
        tools="hover,save",
    )

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

    p.xaxis.axis_label = "Coefficient value (0–1)"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "8pt"

    p.legend.location = "bottom_right"
    p.legend.label_text_font_size = "9pt"
    p.legend.background_fill_alpha = 0.8

    title_div = Div(
        text=("<b>Vignette wording matters, but differently for each model (naive condition)</b>"
              "<br><span style='font-size:10pt; color:#555'>"
              "Blue bars: mean ICC — proportion of variance explained by vignette wording. "
              "Red bars: Kendall's W — agreement across models on which vignette wording "
              "produces higher/lower responses. High ICC + low W = wording matters, but "
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
