"""
Shared configuration for opioid conversion visualizations.
Ensures consistent model ordering and coloring across all visualizations.
"""

from typing import List, Dict
import colorsys
from bokeh.palettes import Spectral11


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
    'models/gemini-2.5-flash',  # Alternative path
    'cohere/command-r-plus-08-2024',
    'ai21/jamba-large-1.7',
    'ai21/jamba-mini-1.7',
    'qwen/qwen-2.5-72b-instruct',
    'meta-llama/llama-3.3-70b-instruct',
    'meta-llama/llama-3.3-70b-versatile',
    'deepseek/deepseek-chat'
]


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


def assign_model_colors(models: List[str]) -> Dict[str, str]:
    """
    Assign consistent colors to models based on their normalized names.
    Models with the same normalized name get the same color.
    
    Args:
        models: List of model names to assign colors to
    
    Returns:
        Dictionary mapping model names to hex color strings
    """
    # Normalize models for color assignment
    normalized_models = [normalize_model_name(m) for m in models]
    unique_normalized = list(dict.fromkeys(normalized_models))  # Preserve order, remove dupes
    
    n_models = len(unique_normalized)
    
    # Use Spectral palette or generate colors if needed
    if n_models <= 11:
        colors = Spectral11[:n_models]
    else:
        # Generate colors using HSV space for more colors
        colors = []
        for i in range(n_models):
            hue = i / n_models
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
            colors.append(hex_color)
    
    # Create color mapping for normalized names
    normalized_color_map = dict(zip(unique_normalized, colors))
    
    # Map original model names to colors via their normalized forms
    return {model: normalized_color_map[normalize_model_name(model)] 
            for model in models}