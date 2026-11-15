"""
Shared configuration for opioid conversion visualizations.
Ensures consistent model ordering and coloring across all visualizations.
"""

from typing import List, Dict
import colorsys
import os
import pandas as pd
from bokeh.palettes import Spectral11
from playwright.async_api import async_playwright
from common.models import CONSISTENT_MODELS
from common.config import RESULTS_DIR


# Model name aliases (maps non-prefixed names to canonical names)
MODEL_ALIASES = {
    'gpt-4.1-mini': 'openai/gpt-4.1-mini',
    'gpt-3.5-turbo': 'openai/gpt-3.5-turbo',
    'claude-3-5-haiku-20241022': 'anthropic/claude-3-5-haiku-20241022',
    'claude-3.5-haiku': 'anthropic/claude-3-5-haiku-20241022',
    'llama-3.3-70b-versatile': 'meta-llama/llama-3.3-70b-versatile',
    'gemini-2.5-flash': 'google/gemini-2.5-flash',
}


# Canonical model order (by company/family)
MODEL_ORDER = [
    'openai/gpt-4.1',
    'openai/gpt-4.1-mini',
    'openai/gpt-3.5-turbo',
    'anthropic/claude-3.5-sonnet:20241022',
    'anthropic/claude-3-5-haiku-20241022',
    'mistralai/mistral-large-2407',
    'mistralai/mixtral-8x22b-instruct',
    'google/gemini-2.5-pro',
    'google/gemini-2.5-flash',
    'cohere/command-r-plus-08-2024',
    'ai21/jamba-large-1.7',
    'ai21/jamba-mini-1.7',
    'qwen/qwen-2.5-72b-instruct',
    'meta-llama/llama-3.3-70b-instruct',
    'meta-llama/llama-3.3-70b-versatile',
    'deepseek/deepseek-chat'
]

# Ranking CSVs used to determine the global model order (colors are a fixed map).
_RANKING_CSV_NAMES = [
    "ranking_vignettes_nielsen_units.csv",
    "ranking_random_doses_nielsen.csv",
    "ranking_random_doses_nielsen_units.csv",
]


def _collect_ranking_ranks() -> Dict[str, List[float]]:
    """Read adjusted_rank from the canonical ranking CSVs and return per-model lists."""
    # MODEL_ORDER_RESULTS_DIR (environment) fixes the global model order to one
    # results directory whatever RESULTS_DIR is; the revision framework sets it
    # to results_nielsen so every score's figures share one model order.
    order_dir = os.environ.get("MODEL_ORDER_RESULTS_DIR", RESULTS_DIR)
    results_dir = os.path.join(os.path.dirname(__file__), "..", order_dir)
    ranks: Dict[str, List[float]] = {}
    for name in _RANKING_CSV_NAMES:
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m = str(row["model"])
            ranks.setdefault(m, []).append(float(row["adjusted_rank"]))
    return ranks


def _build_ranked_color_dict() -> Dict[str, str]:
    """
    Not called at import any more: MODEL_COLOR_DICT is a fixed map (below),
    computed once with this function from the results_all/ rankings. Kept to
    document how those colors were derived.

    Assign rainbow colors based on the average of each model's adjusted
    rank across three conditions:
      1. Vignettes + Nielsen + explain-units
      2. Random doses + Nielsen
      3. Random doses + Nielsen + explain-units

    Models that appear in only a subset of rankings use those ranks.
    The best-performing model gets the first hue (red-ish), the worst
    gets the last hue.  Falls back to a static rainbow if the ranking
    CSVs are not yet available.
    """
    ranks = _collect_ranking_ranks()

    if not ranks:
        # Fallback: return colors in MODEL_ORDER
        consistent = [m for m in MODEL_ORDER if m in CONSISTENT_MODELS]
        n = len(consistent)
        return {m: _hue_color(i, n) for i, m in enumerate(consistent)}

    # Average rank per model, then sort (lower = better = earlier hue)
    avg_ranks = {m: sum(rs) / len(rs) for m, rs in ranks.items()}
    # Filter to consistent models only
    consistent_set = set(CONSISTENT_MODELS)
    avg_ranks = {m: r for m, r in avg_ranks.items() if m in consistent_set}
    sorted_models = sorted(avg_ranks, key=lambda m: avg_ranks[m])
    n = len(sorted_models)
    return {m: _hue_color(i, n) for i, m in enumerate(sorted_models)}


def _hue_color(index: int, total: int) -> str:
    """Return an evenly-spaced rainbow hex color."""
    hue = index / max(total, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.90)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# Fixed model-to-color map, used by every figure under every score and view.
# Colors follow the model, never its rank, and are never recomputed from
# another score's rankings. Computed once (2026-11-14) with the rank-based
# rainbow of _build_ranked_color_dict() from the all-standards rankings in
# results_all/ (ranking_vignettes_nielsen_units with text-only vignettes,
# ranking_random_doses_nielsen, ranking_random_doses_nielsen_units). These are
# reference-grounded conditions, where ranking against Nielsen alone would be
# circular. Ties in average rank were broken by input order. Models not listed
# fall back to the callers' default color.
MODEL_COLOR_DICT: Dict[str, str] = {
    "anthropic/claude-3.5-sonnet:20241022":  "#e54040",
    "google/gemini-2.5-pro":                 "#e58c40",
    "openai/gpt-4.1":                        "#e5d840",
    "deepseek/deepseek-chat":                "#a5e540",
    "mistralai/mistral-large-2407":          "#59e540",
    "openai/gpt-4.1-mini":                   "#40e573",
    "google/gemini-2.5-flash":               "#40e5bf",
    "qwen/qwen-2.5-72b-instruct":            "#40bfe5",
    "mistralai/mixtral-8x22b-instruct":      "#4073e5",
    "meta-llama/llama-3.3-70b-instruct":     "#5940e5",
    "ai21/jamba-large-1.7":                  "#a540e5",
    "cohere/command-r-plus-08-2024":         "#e540d8",
    "openai/gpt-3.5-turbo":                  "#e5408c",
}

# Range thresholds and weighting epsilon — canonical definitions live in
# common/config.py; imported here for backward compatibility.
from common.config import (
    TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER,
    BROAD_RANGE_LOWER, BROAD_RANGE_UPPER,
    EPSILON_FOR_WEIGHT,
)

def get_ranked_model_order() -> List[str]:
    """
    Return models sorted by average adjusted rank (best first).
    Averages across vignettes+nielsen+units, random_doses+nielsen,
    and random_doses+nielsen+units.
    Falls back to MODEL_ORDER if ranking CSVs are absent.
    """
    ranks = _collect_ranking_ranks()

    if not ranks:
        return [m for m in MODEL_ORDER if m in CONSISTENT_MODELS]

    avg_ranks = {m: sum(rs) / len(rs) for m, rs in ranks.items()}
    consistent_set = set(CONSISTENT_MODELS)
    avg_ranks = {m: r for m, r in avg_ranks.items() if m in consistent_set}
    return sorted(avg_ranks, key=lambda m: avg_ranks[m])


def normalize_model_name(model: str) -> str:
    """
    Normalize a model name to its canonical form.
    
    Args:
        model: Model name (potentially without prefix)
    
    Returns:
        Canonical model name with prefix
    """
    # Return as-is if already in MODEL_ORDER
    if model in MODEL_ORDER:
        return model
    
    # Check aliases
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    
    # Return original if no alias found
    return model


def get_model_order(available_models: List[str] = None) -> List[str]:
    """
    Get canonical model order, optionally filtered to available models.
    
    Args:
        available_models: Optional list to filter by. If None, returns full order.
    
    Returns:
        Ordered list of models (in their original form from available_models)
    """
    if available_models is None:
        return MODEL_ORDER.copy()
    
    # Normalize available models and create a reverse mapping
    normalized_to_original = {}
    for original in available_models:
        normalized = normalize_model_name(original)
        normalized_to_original[normalized] = original
    
    # Return models in canonical order that exist in available_models
    ordered = []
    for canonical in MODEL_ORDER:
        if canonical in normalized_to_original:
            ordered.append(normalized_to_original[canonical])
    
    # Add any models not in canonical order at the end
    normalized_set = set(normalize_model_name(m) for m in MODEL_ORDER)
    extras = [m for m in available_models 
              if normalize_model_name(m) not in normalized_set]
    
    return ordered + sorted(extras)


def assign_model_colors(models):
    return {m: MODEL_COLOR_DICT.get(m, "#1f77b4") for m in models}

def get_model_color(model_name, model_order=None):
    """
    Return the color for a given model name, using the assigned colors.
    If model_order is not provided, use Spectral11 and normalized name.
    """
    if model_order is None:
        model_order = [model_name]
    model_colors = assign_model_colors(model_order)
    return model_colors.get(model_name, "#000000")

async def convert_to_pdf_auto_size(html_file: str, pdf_file: str,
                                   scale: float = 2.0,
                                   max_width: int = 2400):
    """
    Convert HTML to PDF with automatic width and height calculation.
    
    Args:
        html_file: Path to input HTML
        pdf_file: Path to output PDF
        scale: Resolution scale
        max_width: Maximum width in pixels (to prevent excessively wide PDFs)
    """
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=scale)
        await page.goto(f"file://{os.path.abspath(html_file)}")
        await page.wait_for_load_state("networkidle")

        # Wait a moment to ensure all rendering is complete
        await page.wait_for_timeout(6000)
        
        # Get the actual content dimensions
        dimensions = await page.evaluate("""
            () => ({
                width: document.documentElement.scrollWidth,
                height: document.documentElement.scrollHeight
            })
        """)
        
        content_width = min(dimensions['width'], max_width)
        content_height = dimensions['height']
        
        # Add some padding
        total_width = content_width + 20
        total_height = content_height + 20
        
        await page.pdf(
            path=pdf_file,
            width=f"{total_width}px",
            height=f"{total_height}px",
            print_background=True,
            margin={"top": "10px", "right": "10px", "bottom": "10px", "left": "10px"}
        )
        await browser.close()
        
        print(f"✓ Generated PDF: {total_width}px x {total_height}px")


async def convert_to_png_auto_size(html_file: str, png_file: str,
                                   scale: float = 2.0,
                                   max_width: int = 2400):
    """
    Convert HTML to PNG with automatic sizing via full-page screenshot.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=scale)
        await page.goto(f"file://{os.path.abspath(html_file)}")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(6000)

        dimensions = await page.evaluate("""
            () => ({
                width: document.documentElement.scrollWidth,
                height: document.documentElement.scrollHeight
            })
        """)

        content_width = min(dimensions['width'], max_width)
        await page.set_viewport_size({"width": content_width + 20,
                                       "height": dimensions['height'] + 20})
        await page.wait_for_timeout(1000)

        await page.screenshot(path=png_file, full_page=True)
        await browser.close()
        print(f"✓ Generated PNG: {png_file}")