"""
Per-scenario vignette strip plots with Kendall's W annotation.

For each scenario, produces a figure where:
  - Y-axis: models (rows)
  - X-axis: mean log(observed / expected) for each vignette
  - Each vignette is a colored circle (labeled with vignette ID)
    with horizontal error bars (±1 SD)
  - Kendall's W annotation shows cross-model ranking agreement
  - Vignette colors come from the reworded_vignettes.json

All 20 scenario figures are collected into a single combined PDF.
"""

import os
import json
import re
import math
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import rankdata

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource, LabelSet, HoverTool, Div, Whisker, Span,
    FixedTicker, Range1d,
)

from common.models import CONSISTENT_MODELS
from analysis.kendall_w import kendall_w as _kendall_w
from visualize.visualization_config import (
    get_ranked_model_order, convert_to_pdf_auto_size, convert_to_png_auto_size,
)

_CONSISTENT_SET = set(CONSISTENT_MODELS)


def _short_model(m: str) -> str:
    return m.split("/")[-1]


def _load_vignette_colors(json_path: str = "resources/reworded_vignettes.json") -> dict:
    """Return {vignette_id: hex_color} from the JSON."""
    with open(json_path) as f:
        vigs = json.load(f)
    return {vid: v.get("vignette_color", "#888888") for vid, v in vigs.items()}


def scenario_vignette_strip(
    results_csv: str,
    scenario_key: str,
    vignette_colors: dict,
    filters: dict = None,
    width: int = 900,
    height: int = 550,
):
    """
    Build a single scenario's strip plot.

    Returns (bokeh_figure, kendall_w, kendall_p)
    """
    df = pd.read_csv(results_csv)

    # Filter
    df = df[df["model"].isin(_CONSISTENT_SET)]
    df = df[df["scenario_key"] == scenario_key]
    df = df[df["has_vignette_text"] == True]
    if filters:
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]

    df = df[df["observed_to_dose"].notna() & (df["observed_to_dose"] > 0)]
    df = df[df["expected_to_dose"].notna() & (df["expected_to_dose"] > 0)]
    df["log_ratio"] = np.log(df["observed_to_dose"] / df["expected_to_dose"])

    # Get models in ranked order
    ranked = get_ranked_model_order()
    available = set(df["model"].unique())
    models = [m for m in ranked if m in available]
    if not models:
        models = sorted(available)

    # Get vignettes sorted by ID
    vignettes = sorted(
        df["vignette_id"].unique(),
        key=lambda x: int(re.search(r'\d+', x).group()),
    )

    # Compute stats per (model, vignette)
    stats = (
        df.groupby(["model", "vignette_id"])["log_ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Build data for plotting
    rows = []
    for _, s in stats.iterrows():
        m_idx = models.index(s["model"]) if s["model"] in models else -1
        if m_idx < 0:
            continue
        color = vignette_colors.get(s["vignette_id"], "#888888")
        mean_val = s["mean"]
        std_val = s["std"] if not np.isnan(s["std"]) else 0
        rows.append({
            "model_idx": m_idx,
            "model": _short_model(s["model"]),
            "vignette_id": s["vignette_id"],
            "mean": mean_val,
            "std": std_val,
            "n": int(s["count"]),
            "upper": mean_val + std_val,
            "lower": mean_val - std_val,
            "color": color,
        })
    plot_df = pd.DataFrame(rows)
    src = ColumnDataSource(plot_df)

    # Compute Kendall's W
    # Rankings matrix: each model ranks the 4 vignettes by mean log_ratio
    rankings = []
    for m in models:
        m_stats = stats[stats["model"] == m].set_index("vignette_id")["mean"]
        if len(m_stats) == len(vignettes):
            # Rank by mean value (1 = lowest)
            vals = [m_stats.get(v, np.nan) for v in vignettes]
            rankings.append(vals)
    rankings_matrix = np.array(rankings)
    W, chi2, p_W = _kendall_w(rankings_matrix)

    # Determine x range
    all_means = plot_df["mean"]
    all_upper = plot_df["upper"]
    all_lower = plot_df["lower"]
    x_min = min(all_lower.min(), all_means.min()) - 0.15
    x_max = max(all_upper.max(), all_means.max()) + 0.15

    short_models = [_short_model(m) for m in models]

    p = figure(
        width=width, height=height,
        x_range=(x_min, x_max),
        y_range=(-0.5, len(models) - 0.5),
        toolbar_location="above",
        tools="hover,save",
    )

    # Zero line (log_ratio = 0 means perfect conversion)
    p.line([0, 0], [-0.5, len(models) - 0.5],
           line_color="#cccccc", line_width=1, line_dash="dashed")

    # Error bars (whiskers)
    p.segment(
        x0="lower", x1="upper", y0="model_idx", y1="model_idx",
        source=src, line_color="color", line_width=1.5, line_alpha=0.5,
    )
    # Whisker caps
    cap_h = 0.08
    for end_col in ["lower", "upper"]:
        cap_rows = []
        for _, r in plot_df.iterrows():
            cap_rows.append({
                "x": r[end_col],
                "y_top": r["model_idx"] + cap_h,
                "y_bot": r["model_idx"] - cap_h,
                "color": r["color"],
            })
        cap_src = ColumnDataSource(pd.DataFrame(cap_rows))
        p.segment(
            x0="x", x1="x", y0="y_bot", y1="y_top",
            source=cap_src, line_color="color", line_width=1.5, line_alpha=0.5,
        )

    # Circles for means
    p.circle(
        x="mean", y="model_idx", source=src,
        size=14, fill_color="color", fill_alpha=0.7,
        line_color="color", line_width=1,
    )

    # Vignette ID labels on circles
    labels = LabelSet(
        x="mean", y="model_idx", text="vignette_id", source=src,
        text_font_size="6pt", text_align="center", text_baseline="middle",
        text_color="white",
    )
    p.add_layout(labels)

    # Hover
    p.hover.tooltips = [
        ("Model", "@model"),
        ("Vignette", "@vignette_id"),
        ("Mean log(obs/exp)", "@mean{0.000}"),
        ("SD", "@std{0.000}"),
        ("n trials", "@n"),
    ]

    # Y axis
    p.yaxis.ticker = FixedTicker(ticks=list(range(len(models))))
    p.yaxis.major_label_overrides = {i: m for i, m in enumerate(short_models)}
    p.yaxis.major_label_text_font_size = "9pt"
    p.xaxis.axis_label = "Mean log(observed / expected)"
    p.xaxis.axis_label_text_font_size = "10pt"

    p.grid.grid_line_alpha = 0.3

    # Title with Kendall's W
    scenario_title = scenario_key.replace("-", " ").title()
    w_str = f"W={W:.2f}" if not np.isnan(W) else "W=N/A"
    p_str = f"p={p_W:.3f}" if not np.isnan(p_W) else ""
    sig_label = " *" if (not np.isnan(p_W) and p_W < 0.05) else ""

    title_div = Div(
        text=(f"<b>{scenario_title}</b>"
              f"<br><span style='font-size:10pt; color:#555'>"
              f"Kendall's W = {W:.3f} ({p_str}{sig_label}) — "
              f"cross-model ranking agreement of vignette means. "
              f"Circles = vignette mean ± SD of log(observed/expected).</span>"),
        styles={"font-size": "13pt", "margin-left": "10px", "margin-bottom": "6px"},
    )

    # Legend for vignette colors
    legend_parts = []
    for vid in vignettes:
        c = vignette_colors.get(vid, "#888888")
        legend_parts.append(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{c};border-radius:50%;margin-right:3px;'
            f'vertical-align:middle;"></span><span style="vertical-align:middle">{vid}</span>'
        )
    legend_div = Div(
        text="<span style='font-size:9pt; color:#666; margin-left:10px'>"
             + "&nbsp;&nbsp;".join(legend_parts) + "</span>",
        styles={"margin-left": "10px", "margin-bottom": "4px"},
    )

    return column(title_div, legend_div, p), W, p_W


async def save_all_scenario_strips(
    results_csv: str = "results/reworded_vignettes_results.csv",
    figures_dir: str = "figures",
    filters: dict = None,
):
    """
    Generate per-scenario strip plots, save as individual HTML + combined PDF.
    """
    os.makedirs(figures_dir, exist_ok=True)
    vignette_colors = _load_vignette_colors()

    # Determine scenario order (by mean ICC from icc_analysis.csv if available)
    icc_path = os.path.join("results", "icc_analysis.csv")
    if os.path.exists(icc_path):
        icc_df = pd.read_csv(icc_path)
        scenario_order = (
            icc_df.groupby("scenario_key")["icc"]
            .mean().sort_values(ascending=False).index.tolist()
        )
    else:
        df_tmp = pd.read_csv(results_csv)
        scenario_order = sorted(df_tmp["scenario_key"].unique())

    png_paths = []
    w_results = []

    for scenario_key in scenario_order:
        fig, W, p_W = scenario_vignette_strip(
            results_csv, scenario_key, vignette_colors, filters=filters,
        )
        slug = scenario_key.replace("-", "_")
        html_path = os.path.join(figures_dir, f"scenario_strip_{slug}.html")
        png_path = os.path.join(figures_dir, f"scenario_strip_{slug}.png")

        output_file(html_path)
        save(fig)
        await convert_to_png_auto_size(html_path, png_path, scale=2.0)
        png_paths.append((scenario_key, png_path, W, p_W))
        w_results.append({"scenario_key": scenario_key, "kendall_w": W, "kendall_p": p_W})
        print(f"  ✓ {scenario_key} (W={W:.3f})")

    # Save Kendall's W results
    w_df = pd.DataFrame(w_results)
    w_csv = os.path.join("results", "kendall_w_vignettes.csv")
    w_df.to_csv(w_csv, index=False)
    print(f"\nSaved Kendall's W results to {w_csv}")

    # Combined PDF using reportlab
    pdf_path = os.path.join(figures_dir, "all_scenario_strips.pdf")

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,
    )
    from reportlab.lib.enums import TA_LEFT
    from PIL import Image as PILImage

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FigTitle", parent=styles["Heading2"],
        fontSize=14, spaceAfter=6,
    )
    desc_style = ParagraphStyle(
        "FigDesc", parent=styles["BodyText"],
        fontSize=10, leading=13, spaceAfter=10, alignment=TA_LEFT,
    )

    page_w, page_h = letter
    margin = 0.75 * inch
    usable_w = page_w - 2 * margin

    story = []

    # Title page
    story.append(Paragraph(
        "Vignette Wording Effects by Scenario",
        ParagraphStyle("BigTitle", parent=styles["Title"], fontSize=20),
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "For each scenario, the 4 text vignettes are shown per model. "
        "Each circle is the mean log(observed/expected) for that vignette, "
        "with ±1 SD error bars. Vignette colors are consistent across scenarios. "
        "Kendall's W measures cross-model agreement on vignette rankings.",
        desc_style,
    ))
    story.append(Spacer(1, 12))

    # Summary table of W values
    w_sorted = w_df.sort_values("kendall_w", ascending=False)
    sig_count = (w_sorted["kendall_p"] < 0.05).sum()
    story.append(Paragraph(
        f"<b>Summary:</b> {sig_count}/{len(w_sorted)} scenarios have "
        f"significant ranking agreement (Kendall's W, p&lt;0.05). "
        f"Mean W = {w_sorted['kendall_w'].mean():.3f}.",
        desc_style,
    ))
    story.append(PageBreak())

    for scenario_key, png_path, W, p_W in png_paths:
        if not os.path.exists(png_path):
            continue

        pil = PILImage.open(png_path)
        img_w, img_h = pil.size
        scale = usable_w / img_w
        display_w = usable_w
        display_h = img_h * scale
        max_img_h = page_h - 2 * margin - 1.0 * inch
        if display_h > max_img_h:
            display_h = max_img_h
            display_w = img_w * (max_img_h / img_h)

        story.append(RLImage(png_path, width=display_w, height=display_h))
        story.append(PageBreak())

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
    )
    doc.build(story)
    print(f"\n✓ Combined PDF saved to {pdf_path}")
