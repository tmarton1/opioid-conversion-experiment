import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, gridplot
from bokeh.models import HoverTool, Label, ColumnDataSource, Legend
from bokeh.palettes import Spectral11
from bokeh.io import output_file, save
from beeswarm import BeeswarmPlacer
from typing import Optional, Dict, List
import colorsys

class OpioidPairVisualizer:
    """
    Creates a multi-panel visualization for opioid pair conversion analysis.
    """
    
    def __init__(self,
                 experiment_results_csv: str,
                 pair_model_csv: str,
                 consensus_csv: str,
                 from_opioid: str,
                 to_opioid: str,
                 prompt_conditions: Dict[str, bool]):
        """
        Initialize visualizer for a specific opioid pair and prompt condition.
        
        Args:
            experiment_results_csv: Path to experiment_results_by_ome.csv (raw results)
            pair_model_csv: Path to pair_model_analysis.csv
            consensus_csv: Path to opioid_pair_consensus.csv
            from_opioid: Source opioid (e.g., "iv_fentanyl")
            to_opioid: Target opioid (e.g., "oral_morphine")
            prompt_conditions: Dict with 'include_nielson' and 'include_examples' keys
        """
        self.from_opioid = from_opioid
        self.to_opioid = to_opioid
        self.prompt_conditions = prompt_conditions
        
        # Load raw experiment results for first plot
        experiment_df = pd.read_csv(experiment_results_csv)
        self.raw_data = experiment_df[
            (experiment_df['from_opioid'] == from_opioid) &
            (experiment_df['to_opioid'] == to_opioid) &
            (experiment_df['include_nielson'] == prompt_conditions.get('include_nielson', False)) &
            (experiment_df['include_examples'] == prompt_conditions.get('include_examples', False))
        ].copy()
        
        # Calculate observed coefficient for raw data: observed_to_dose / from_dose
        self.raw_data['observed_coefficient'] = self.raw_data['observed_to_dose'] / self.raw_data['from_dose']
        
        # Get units from first row
        self.from_unit = self.raw_data['from_opioid_unit'].iloc[0]
        self.to_unit = self.raw_data['to_opioid_unit'].iloc[0]
        
        # Load aggregated data for second plot
        pair_model_df = pd.read_csv(pair_model_csv)
        self.pair_data = pair_model_df[
            (pair_model_df['from_opioid'] == from_opioid) &
            (pair_model_df['to_opioid'] == to_opioid) &
            (pair_model_df['include_nielson'] == prompt_conditions.get('include_nielson', False)) &
            (pair_model_df['include_examples'] == prompt_conditions.get('include_examples', False))
        ].copy()
        
        # Load consensus data for third plot
        consensus_df = pd.read_csv(consensus_csv)
        consensus_filtered = consensus_df[
            (consensus_df['from_opioid'] == from_opioid) &
            (consensus_df['to_opioid'] == to_opioid) &
            (consensus_df['include_nielson'] == prompt_conditions.get('include_nielson', False)) &
            (consensus_df['include_examples'] == prompt_conditions.get('include_examples', False))
        ]
        self.consensus_data = consensus_filtered.iloc[0] if len(consensus_filtered) > 0 else None
        
        if len(self.raw_data) == 0:
            raise ValueError(f"No raw data found for {from_opioid} -> {to_opioid} with conditions {prompt_conditions}")
        
        if len(self.pair_data) == 0:
            raise ValueError(f"No aggregated data found for {from_opioid} -> {to_opioid} with conditions {prompt_conditions}")
        
        # Get literature coefficient from pair_data
        self.literature_coeff = self.pair_data['expected_coefficient'].iloc[0]
        
        # Define consistent model order (by company/family)
        self.model_order = self._get_model_order()
        
        # Assign colors to models
        self.model_colors = self._assign_colors()
        
    def _get_model_order(self) -> List[str]:
        """Define a consistent, meaningful order for models."""
        # Group by company/family
        model_groups = {
            'OpenAI': ['openai/gpt-4.1', 'gpt-4.1-mini', 'gpt-3.5-turbo'],
            'Anthropic': ['anthropic/claude-3.5-sonnet:20241022', 'claude-3-5-haiku-20241022'],
            'Google': ['google/gemini-2.5-pro', 'models/gemini-2.5-flash'],
            'Meta': ['meta-llama/llama-3.3-70b-instruct', 'llama-3.3-70b-versatile'],
            'Mistral': ['mistralai/mistral-large-2407', 'mistralai/mixtral-8x22b-instruct'],
            'AI21': ['ai21/jamba-large-1.7', 'ai21/jamba-mini-1.7'],
            'Cohere': ['cohere/command-r-plus-08-2024'],
            'Qwen': ['qwen/qwen-2.5-72b-instruct'],
            'DeepSeek': ['deepseek/deepseek-chat']
        }
        
        # Flatten to ordered list
        ordered_models = []
        for group in model_groups.values():
            ordered_models.extend(group)
        
        # Add any models not in the predefined order
        unique_models = self.raw_data['model'].unique()
        for model in unique_models:
            if model not in ordered_models:
                ordered_models.append(model)
        
        return ordered_models
    
    def _assign_colors(self) -> Dict[str, str]:
        """Assign colors to models using a readable gradient."""
        unique_models = [m for m in self.model_order if m in self.raw_data['model'].values]
        n_models = len(unique_models)
        
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
        
        return dict(zip(unique_models, colors))
    
    def _calculate_alpha(self, weight: float) -> float:
        max_weight = 10.0
        min_alpha = 0.05
        max_alpha = 0.9
        alpha = min_alpha + (min(weight, max_weight) / max_weight) * (max_alpha - min_alpha)
        return alpha
    
    def _p_to_stars(self, p: float) -> str:
        """Convert p-value to significance stars."""
        if pd.isna(p):
            return ''
        elif p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'
    
    def _get_log_ticks(self, include_literature: bool = True) -> List[float]:
        """Generate log scale tick values including literature coefficient."""
        base_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
        
        if include_literature and self.literature_coeff not in base_ticks:
            ticks = base_ticks + [self.literature_coeff]
            ticks.sort()
            return ticks
        
        return base_ticks
    
    def create_plot(self, width: int = 500, height: int = 300, 
                   output_html: Optional[str] = None) -> row:
        """
        Create the three-panel visualization with legend and a weight bar to the right.
        """
        # Calculate subplot widths
        w2 = int(width * 0.20)  # jitter panel width
        w3 = int(w2 * 0.5)      # consensus panel
        w1 = width - w2 - w3    # main panel

        # Create subplots — returns p1, p2, legend_items
        p1, p2_renderers_dict, legend_items = self._create_coefficient_and_jitter_plots(
            w1, w2, height
        )
        p2 = p2_renderers_dict['figure']
        # we just computed normalized_weight in the CSV; no extra info needed here

        p3 = self._create_consensus_plot(w3, height)

        # Link y-ranges so they're aligned vertically
        p2.y_range = p1.y_range
        p3.y_range = p1.y_range

        # Legend (tall)
        legend_fig = self._create_legend(height, legend_items)

        # NEW: stacked vertical weight bar to the right of the legend
        weight_fig = self._create_weight_bar(height, legend_items)

        # Create layout: plots + legend + weight bar
        layout = row(p1, p2, p3, weight_fig, legend_fig)

        if output_html:
            output_file(output_html)
            save(layout)

        return layout

    
    def _create_coefficient_and_jitter_plots(self, width1: int, width2: int, height: int) -> tuple:
        """
        Create both subplot 1 and subplot 2, collecting all renderers for the legend.
        
        Returns:
            (p1, p2_dict, legend_items) where:
            - p1 is the coefficient plot figure
            - p2_dict contains the jitter plot figure
            - legend_items is list of (model_name, [renderers]) for legend
        """
        # --- Subplot 1: raw points per model ---
        p1 = figure(
            width=width1,
            height=height,
            title=f"{self.from_opioid} \u2192 {self.to_opioid}",
            x_axis_label=f"From Dose ({self.from_unit})",
            y_axis_label=f"Observed Coefficient ({self.to_unit}/{self.from_unit})",
            x_axis_type="log",
            y_axis_type="log",
            toolbar_location=None,
        )

        # Y-axis ticks in raw space
        y_ticks = self._get_log_ticks(include_literature=True)
        p1.yaxis.ticker = y_ticks
        p1.xaxis.minor_tick_line_color = None

        # Literature coefficient line
        x_min = self.raw_data["from_dose"].min()
        x_max = self.raw_data["from_dose"].max()
        p1.line(
            [x_min, x_max],
            [self.literature_coeff, self.literature_coeff],
            line_dash="dashed",
            line_color="gray",
            line_width=2,
        )

        # --- Subplot 2: model-level coefficients with beeswarm ---
        p2 = figure(
            width=width2,
            height=height,
            title="Model Coefficients",
            x_axis_label="",
            y_axis_label="",
            y_axis_type="log",
            toolbar_location=None,
        )

        p2.yaxis.ticker = y_ticks
        p2.line(
            [-0.5, 1.0],
            [self.literature_coeff, self.literature_coeff],
            line_dash="dashed",
            line_color="gray",
            line_width=2,
        )

        p2.xaxis.visible = False
        p2.xgrid.grid_line_color = None
        p2.x_range.start = -0.4
        p2.x_range.end = 0.4

        # Data holders for subplot 2
        x_positions_p2: List[float] = []
        y_values_p2: List[float] = []
        colors_p2: List[str] = []
        alphas_p2: List[float] = []
        models_list_p2: List[str] = []
        weights_list_p2: List[float] = []

        # legend_items: list of (model_name_clean, [renderers])
        legend_items: List[tuple] = []

        # --- Build per-model renderers for subplot 1 and gather data for subplot 2 ---
        for model in self.model_order:
            # Raw points (subplot 1)
            model_data_p1 = self.raw_data[self.raw_data["model"] == model]
            if len(model_data_p1) == 0:
                continue

            x_p1 = model_data_p1["from_dose"].values
            y_p1 = model_data_p1["observed_coefficient"].values

            color = self.model_colors.get(model, "#000000")

            renderer_p1 = p1.scatter(
                x_p1,
                y_p1,
                size=6,
                color=color,
                alpha=0.5,
                marker="circle",
            )

            # Start legend entry with subplot 1 renderer
            model_name_clean = model.split("/")[-1] if "/" in model else model
            legend_items.append((model_name_clean, [renderer_p1]))

            # Aggregated model-level data (subplot 2)
            model_data_p2 = self.pair_data[self.pair_data["model"] == model]
            if len(model_data_p2) == 0:
                continue

            # Use log mean → exp to get raw coefficient
            log_mean = model_data_p2["observed_log_coefficient_mean"].iloc[0]
            y_raw = float(np.exp(log_mean))
            y_values_p2.append(y_raw)

            colors_p2.append(color)

            weight = float(model_data_p2["weight"].iloc[0])
            alpha = self._calculate_alpha(weight)
            alphas_p2.append(alpha)

            models_list_p2.append(model_name_clean)
            weights_list_p2.append(weight)

        # --- Use BeeswarmPlacer to compute x positions for subplot 2 ---
        if len(y_values_p2) > 0:
            y_array = np.array(y_values_p2, dtype=float)
            placer = BeeswarmPlacer(x_max=0.35, r=0.06, use_log_y=True)
            x_positions_p2 = list(placer.compute(y_array))

            # Normalized weights (v_j) for hover
            total_w = float(sum(weights_list_p2))
            if total_w > 0:
                norm_weights_list_p2 = [w / total_w for w in weights_list_p2]
            else:
                norm_weights_list_p2 = [np.nan for _ in weights_list_p2]

            # Create one CDS per model so hover shows its fields
            for model in self.model_order:
                model_data_p2 = self.pair_data[self.pair_data["model"] == model]
                if len(model_data_p2) == 0:
                    continue

                model_name_clean = model.split("/")[-1] if "/" in model else model
                if model_name_clean not in models_list_p2:
                    continue

                idx = models_list_p2.index(model_name_clean)

                raw_w = float(model_data_p2["weight"].iloc[0])
                norm_w = float(model_data_p2["normalized_weight"].iloc[0])

                source_model = ColumnDataSource(
                    data=dict(
                        x=[x_positions_p2[idx]],
                        y=[y_values_p2[idx]],
                        model=[models_list_p2[idx]],
                        weight_norm=[norm_w],
                        weight_raw=[raw_w],
                    )
                )

                renderer_p2 = p2.scatter(
                    "x",
                    "y",
                    source=source_model,
                    size=10,
                    color=colors_p2[idx],
                    alpha=alphas_p2[idx],
                    line_color=colors_p2[idx],
                    line_alpha=alphas_p2[idx],
                    marker="circle",
                )

                # Attach subplot-2 renderer to legend
                for j, (name, renderers) in enumerate(legend_items):
                    if name == model_name_clean:
                        legend_items[j] = (name, renderers + [renderer_p2])
                        break

            hover = HoverTool(
                tooltips=[
                    ("Model", "@model"),
                    ("Coefficient", "@y{0.000}"),
                    ("Normalized weight", "@weight_norm{0.00}"),
                    ("Raw weight", "@weight_raw{0.00}"),  # optional
                ]
            )
            p2.add_tools(hover)

        p2_dict = {"figure": p2}
        return p1, p2_dict, legend_items
    
    def _create_consensus_plot(self, width: int, height: int) -> figure:
        """
        Third subplot: Consensus coefficient with error bars (vertical lines only).
        No y-axis labels, directly abuts subplot 2.

        Consensus and SE are computed in log space; we exponentiate μ_log ± SE_log to
        get multiplicative error bars on the raw (log-scaled) y-axis.
        """
        p = figure(
            width=width,
            height=height,
            title="Consensus",
            x_axis_label="",
            y_axis_label="",
            y_axis_type="log",
            toolbar_location=None
        )
        
        # Set y-axis ticks to align (same as subplot 1)
        y_ticks = self._get_log_ticks(include_literature=True)
        p.yaxis.ticker = y_ticks
        
        # Hide y-axis labels (but keep the axis for alignment)
        p.yaxis.major_label_text_font_size = '0pt'
        
        # Reference line at literature coefficient
        p.line([0, 1], [self.literature_coeff, self.literature_coeff], 
               line_dash='dashed', line_color='gray', line_width=2)
        
        if self.consensus_data is not None:
            # Use raw consensus coefficient (NOT normalized)
            consensus_log = self.consensus_data['consensus_log_coefficient']
            se_log = self.consensus_data['se_consensus_log']

            if pd.isna(consensus_log):
                return p
            
            consensus_raw = float(np.exp(consensus_log))

            if pd.notna(se_log) and se_log > 0:
                lower_raw = float(np.exp(consensus_log - se_log))
                upper_raw = float(np.exp(consensus_log + se_log))
            else:
                lower_raw = upper_raw = consensus_raw
            
            # Plot consensus point using scatter
            p.scatter([0.5], [consensus_raw], size=10, color='black', alpha=0.8, marker='circle')
            
            # Add vertical error bars (if SE available)
            if lower_raw != upper_raw:
                err_xs = [[0.5, 0.5]]
                err_ys = [[lower_raw, upper_raw]]
                p.multi_line(err_xs, err_ys, color='black', line_width=2)
            
            # Add significance label
            p_value = self.consensus_data['p_value_log']
            sig_label = self._p_to_stars(p_value)
            
            if sig_label:
                label_y = upper_raw * 1.3            
                label = Label(
                    x=0.5, y=label_y,
                    text=sig_label,
                    text_font_size=p.title.text_font_size,
                    text_font=p.title.text_font,
                    text_align='center',
                    text_baseline='bottom'
                )
                p.add_layout(label)
        
        # Hide x-axis
        p.xaxis.visible = False
        p.xgrid.grid_line_color = None
        p.x_range.start = 0
        p.x_range.end = 1
        
        return p
    
    def _create_legend(self, height: int, legend_items: List[tuple]) -> figure:
        """
        Create a separate legend figure to the right of the plots.
        Legend controls both subplot 1 and subplot 2.

        This version makes the legend wider and splits items into two columns
        so it doesn't get too tall.
        """
        # Wider, normal height figure just for the legend
        legend_fig = figure(
            width=400,          # twice-ish the previous width
            height=height,      # same vertical extent as main plots
            toolbar_location=None,
            outline_line_color=None,
        )

        # Hide axes and grids
        legend_fig.xaxis.visible = False
        legend_fig.yaxis.visible = False
        legend_fig.xgrid.visible = False
        legend_fig.ygrid.visible = False

        # Create dummy glyphs for the legend with proper colors
        legend_items_with_glyphs = []
        for model_name, renderers in legend_items:
            # Find full model name from the cleaned name
            full_model = None
            for m in self.model_order:
                clean = m.split("/")[-1] if "/" in m else m
                if clean == model_name:
                    full_model = m
                    break

            color = self.model_colors.get(full_model, "#000000")

            dummy_glyph = legend_fig.scatter(
                [0], [0],
                size=10,
                color=color,
                alpha=0.7,
                marker="circle",
                visible=True,
            )

            legend_items_with_glyphs.append((model_name, [dummy_glyph] + renderers))

        # Legend with two columns
        legend = Legend(
            items=legend_items_with_glyphs,
            location="top_left",
            orientation="vertical",
            spacing=3,
            label_text_font_size="9pt",
            label_text_color="black",
            click_policy="hide",
            ncols=2,                     # <-- key change: two columns of items
        )

        legend_fig.add_layout(legend, "center")
        return legend_fig

    def _create_weight_bar(self, height: int, legend_items: List[tuple]) -> figure:
        """
        Create a vertical stacked bar showing the normalized consensus weights (v_j)
        for each model, in the same order and color as the legend.

        Uses `normalized_weight` from self.pair_data.
        """
        # Narrow, tall figure (same height scale as legend)
        p = figure(
            width=25,
            height=height,
            toolbar_location=None,
            outline_line_color=None,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # No axes or grids
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        # Build segment list in the same order as the legend
        models_for_bar = []
        weights_for_bar = []
        colors_for_bar = []

        # We have full model names in self.model_order and
        # cleaned names (no prefix) in legend_items.
        for model_name_clean, _renderers in legend_items:
            full_model = None
            for m in self.model_order:
                clean = m.split('/')[-1] if '/' in m else m
                if clean == model_name_clean:
                    full_model = m
                    break

            if full_model is None:
                continue

            model_rows = self.pair_data[self.pair_data['model'] == full_model]
            if len(model_rows) == 0:
                continue

            norm_w = float(model_rows['normalized_weight'].iloc[0])
            if not np.isfinite(norm_w) or norm_w <= 0:
                continue

            color = self.model_colors.get(full_model, '#000000')

            models_for_bar.append(model_name_clean)
            weights_for_bar.append(norm_w)
            colors_for_bar.append(color)

        if not models_for_bar:
            return p  # nothing to show

        # Normalize again so segments fill the bar exactly
        total = float(sum(weights_for_bar))
        if total <= 0:
            return p

        weights_for_bar = [w / total for w in weights_for_bar]

        # Reverse order so top of bar is first in legend
        models_for_bar = models_for_bar[::-1]
        weights_for_bar = weights_for_bar[::-1]
        colors_for_bar = colors_for_bar[::-1]

        # Compute bottoms / tops as cumulative sums in [0,1]
        bottoms = [0.0]
        for w in weights_for_bar[:-1]:
            bottoms.append(bottoms[-1] + w)
        tops = [b + w for b, w in zip(bottoms, weights_for_bar)]

        source = ColumnDataSource(
            data=dict(
                left=[0.0] * len(models_for_bar),
                right=[1.0] * len(models_for_bar),
                bottom=bottoms,
                top=tops,
                model=models_for_bar,
                weight_norm=weights_for_bar,
                color=colors_for_bar,
            )
        )

        p.quad(
            left="left",
            right="right",
            bottom="bottom",
            top="top",
            color="color",
            source=source,
        )

        hover = HoverTool(
            tooltips=[
                ("Model", "@model"),
                ("Normalized weight", "@weight_norm{0.00}")
            ]
        )
        p.add_tools(hover)

        return p
