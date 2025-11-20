"""
Shared configuration for opioid conversion visualizations.
Ensures consistent model ordering and coloring across all visualizations.
"""

from typing import List, Dict
import colorsys
import os
from bokeh.palettes import Spectral11
from playwright.async_api import async_playwright


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

MODEL_COLOR_DICT = {
    'openai/gpt-4.1': '#e54444',
    'openai/gpt-4.1-mini': '#e58144',
    'openai/gpt-3.5-turbo': '#e5bd44',
    'anthropic/claude-3.5-sonnet:20241022': '#d1e544',
    'anthropic/claude-3-5-haiku-20241022': '#95e544',
    'mistralai/mistral-large-2407': '#58e544',
    'mistralai/mixtral-8x22b-instruct': '#44e56d',
    'google/gemini-2.5-pro': '#44e5a9',
    'google/gemini-2.5-flash': '#44e5e5',
    'cohere/command-r-plus-08-2024': '#44a9e5',
    'ai21/jamba-large-1.7': '#446de5',
    'ai21/jamba-mini-1.7': '#5844e5',
    'qwen/qwen-2.5-72b-instruct': '#9544e5',
    'meta-llama/llama-3.3-70b-instruct': '#d144e5',
    'meta-llama/llama-3.3-70b-versatile': '#e544bd',
    'deepseek/deepseek-chat': '#e54481'
}

# Range thresholds for opioid conversion visualizations
TIGHT_RANGE_LOWER = 0.85
TIGHT_RANGE_UPPER = 1.15
BROAD_RANGE_LOWER = 0.6
BROAD_RANGE_UPPER = 1.7

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