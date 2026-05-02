from typing import Optional, List
import numpy as np
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import row, column
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
from bokeh.transform import transform
from bokeh.palettes import Greys256
from analysis.batch import OpioidPairBatch
from visualize.visualization_config import get_model_order, get_ranked_model_order, assign_model_colors, normalize_model_name


class OpioidPairBatchHeatmapVisualizer:
    """
    Creates heatmap visualizations across all model-pair combinations in a batch.
    
    Three subplots:
    1. Weight heatmap (colored by raw weight, labeled with CV)
    2. Proportion in broad range 
    3. Proportion in tight range 
    
    Models and opioid pairs are sorted by mean raw weight (best first),
    independently per heatmap.
    """
    
    def __init__(self, batch: OpioidPairBatch, cell_size: int = 30,
                 width_padding: int = 50, height_padding: int = 50,
                 cell_text_size: str = "6pt", label_text_size: str = "7pt"):
        self.batch = batch
        self.cell_size = cell_size
        self.width_padding = width_padding
        self.height_padding = height_padding
        self.cell_text_size = cell_text_size
        self.label_text_size = label_text_size
        
        # Get data from batch (already filtered to CONSISTENT_MODELS)
        all_models = batch.get_models()
        self.models = get_model_order(all_models)
        self.model_colors = assign_model_colors(self.models)
        self.pairs = batch.get_items()
        
        print(f"Found {len(self.models)} models and {len(self.pairs)} pairs")
        
        # Compute matrices using batch methods (reordered for model order)
        self.weight_matrix = self._compute_ordered_matrix(batch.compute_weight_matrix(), all_models)
        self.cv_matrix = self._compute_ordered_matrix(batch.compute_cv_matrix(), all_models)
        self.broad_range_matrix = self._compute_ordered_matrix(batch.compute_range_matrix("broad"), all_models)
        self.tight_range_matrix = self._compute_ordered_matrix(batch.compute_range_matrix("tight"), all_models)
        
        print(f"Weight matrix shape: {self.weight_matrix.shape}")
        print(f"Non-NaN values in weight matrix: {np.sum(~np.isnan(self.weight_matrix))}")
    
    
    def _compute_ordered_matrix(self, matrix: np.ndarray, original_models: List[str]) -> np.ndarray:
        """Reorder matrix rows to match self.models order."""
        if len(original_models) != matrix.shape[0]:
            return matrix
        
        # Create mapping from original to new order
        new_order = [original_models.index(m) for m in self.models if m in original_models]
        return matrix[new_order, :]

    def _sort_by_weight(self, models, weight_matrix, cv_matrix, broad_matrix, tight_matrix, pairs):
        """Sort models (rows) and pairs (cols) by mean raw weight, best first.
        
        Returns new copies of all arrays/lists, sorted independently.
        """
        # Sort models by mean weight (descending — higher weight = better)
        model_means = np.nanmean(weight_matrix, axis=1)
        model_order = np.argsort(-model_means)
        models = [models[i] for i in model_order]
        weight_matrix = weight_matrix[model_order, :]
        cv_matrix = cv_matrix[model_order, :]
        broad_matrix = broad_matrix[model_order, :]
        tight_matrix = tight_matrix[model_order, :]

        # Sort pairs by mean weight (descending)
        pair_means = np.nanmean(weight_matrix, axis=0)
        pair_order = np.argsort(-pair_means)
        pairs = [pairs[i] for i in pair_order]
        weight_matrix = weight_matrix[:, pair_order]
        cv_matrix = cv_matrix[:, pair_order]
        broad_matrix = broad_matrix[:, pair_order]
        tight_matrix = tight_matrix[:, pair_order]

        return models, weight_matrix, cv_matrix, broad_matrix, tight_matrix, pairs

    @staticmethod
    def _sort_cols_by_mean(matrix: np.ndarray, pairs: List[str], descending: bool = True):
        """Sort columns by their mean value, returning reordered matrix and pair labels."""
        col_means = np.nanmean(matrix, axis=0)
        order = np.argsort(-col_means) if descending else np.argsort(col_means)
        return matrix[:, order], [pairs[i] for i in order]

    def _create_heatmap(self, color_matrix: np.ndarray, title: str,
                       models: List[str], pairs: List[str],
                       text_matrix: np.ndarray = None,
                       colormap_range: tuple = None) -> figure:
        """
        Create a single heatmap figure.
        
        Args:
            color_matrix: Data matrix (models × pairs) for cell colors, including summary row/col
            title: Title for the heatmap
            models: Model labels (including 'Mean')
            pairs: Pair labels (including 'Mean')
            text_matrix: Optional separate matrix for cell text labels. If None, uses color_matrix.
            colormap_range: (min, max) for color scale, or None for auto
        
        Returns:
            Bokeh figure
        """
        if text_matrix is None:
            text_matrix = color_matrix
            
        # Clean model names
        model_labels = []
        for m in models:
            normalized = normalize_model_name(m)
            clean = normalized.split('/')[-1] if '/' in normalized else normalized
            model_labels.append(clean)
        
        # Prepare data for heatmap
        model_list = []
        pair_list = []
        color_value_list = []
        text_list = []
        is_summary_list = []
        
        for i, model in enumerate(model_labels):
            for j, pair in enumerate(pairs):
                color_val = color_matrix[i, j]
                text_val = text_matrix[i, j]
                model_list.append(model)
                pair_list.append(pair)
                color_value_list.append(color_val)
                
                is_summary = (model == 'Mean' or pair == 'Mean')
                is_summary_list.append(is_summary)
                
                if np.isnan(text_val):
                    text_list.append("")
                else:
                    text_list.append(f"{text_val:.2f}")
        
        source = ColumnDataSource(data=dict(
            model=model_list,
            pair=pair_list,
            value=color_value_list,
            text=text_list,
            is_summary=is_summary_list
        ))
        
        # Determine color range (excluding summary row/col)
        if colormap_range is None:
            non_summary_values = [v for v, is_sum in zip(color_value_list, is_summary_list) 
                                 if not is_sum and not np.isnan(v)]
            if non_summary_values:
                vmin, vmax = min(non_summary_values), max(non_summary_values)
            else:
                vmin, vmax = 0, 1
        else:
            vmin, vmax = colormap_range
        
        mapper = LinearColorMapper(
            palette=Greys256,
            low=vmin,
            high=vmax,
            nan_color='lightgray'
        )
        
        width = len(pairs) * self.cell_size + self.width_padding
        height = len(models) * self.cell_size + self.height_padding
        
        p = figure(
            width=width,
            height=height,
            x_range=pairs,
            y_range=list(reversed(model_labels)),
            toolbar_location=None,
            title=title
        )
        
        # Draw rectangles
        rects = p.rect(
            x='pair',
            y='model',
            width=0.9,
            height=0.9,
            source=source,
            fill_color=transform('value', mapper),
            line_color='white',
            line_width=1
        )
        
        # Add text annotations (smaller font)
        p.text(
            x='pair',
            y='model',
            text='text',
            source=source,
            text_align='center',
            text_baseline='middle',
            text_font_size=self.cell_text_size,
            text_color='black'
        )
        
        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ('Model', '@model'),
                ('Pair', '@pair'),
                ('Value', '@value{0.00}')
            ],
            renderers=[rects]
        )
        p.add_tools(hover)
        
        # Styling
        p.xaxis.major_label_orientation = 1.2
        p.xaxis.major_label_text_font_size = self.label_text_size
        p.yaxis.major_label_text_font_size = self.label_text_size
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.grid.grid_line_color = None
        
        # Add color bar
        color_bar = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=6),
            formatter=PrintfTickFormatter(format="%.2f"),
            label_standoff=10,
            border_line_color=None,
            location=(0, 0),
            width=12
        )
        p.add_layout(color_bar, 'right')
        
        return p
    
    def _create_mean_accuracy_bar_chart(self, width=1600, height=500):
        """Create bar chart sorted by global rank order (from rankings CSVs)."""
        # Use global rank order, filtered to models in this batch
        global_ranked = get_ranked_model_order()
        batch_models = self.models  # already filtered to CONSISTENT_MODELS
        model_order = [m for m in global_ranked if m in batch_models]
        model_colors = self.model_colors

        tight_means, caution_means = self.batch.get_mean_accuracy_by_model(model_order)
        clean_names = [m.split('/')[-1] if '/' in m else m for m in model_order]
        colors = [model_colors.get(m, "#000000") for m in model_order]

        tops_caution = [t + c for t, c in zip(tight_means, caution_means)]
        source_tight = ColumnDataSource(data=dict(
            x=list(range(len(clean_names))),
            top=tight_means,
            color=colors,
            model=clean_names
        ))
        source_caution = ColumnDataSource(data=dict(
            x=list(range(len(clean_names))),
            top=tops_caution,
            bottom=tight_means,
            color=colors,
            caution=caution_means,
            model=clean_names
        ))

        p = figure(
            width=width,
            height=height,
            x_range=(-0.5, len(clean_names)-0.5),
            y_range=(0, 1.05),
            toolbar_location=None,
            title="Mean Model Accuracy Across All Pairs"
        )

        tight_bars = p.vbar(
            x='x',
            top='top',
            width=0.8,
            color='color',
            alpha=0.9,
            source=source_tight
        )
        caution_bars = p.vbar(
            x='x',
            top='top',
            bottom='bottom',
            width=0.8,
            color='color',
            alpha=0.3,
            source=source_caution
        )

        hover_tight = HoverTool(
            tooltips=[
                ('Model', '@model'),
                ('Tight (±15%)', '@top{0.0%}')
            ],
            renderers=[tight_bars]
        )
        hover_caution = HoverTool(
            tooltips=[
                    ('Model', '@model'),
                    ('Caution', '@caution{0.0%}')
            ],
            renderers=[caution_bars]
        )
        p.add_tools(hover_tight)
        p.add_tools(hover_caution)

        p.xaxis.visible = False
        p.xgrid.visible = False
        p.yaxis.axis_label = "Mean Proportion"

        return p

    def _create_model_legend(self, width=50, height=250):
        """Create legend sorted by global rank order."""
        global_ranked = get_ranked_model_order()
        batch_models = self.models
        model_order = [m for m in global_ranked if m in batch_models]
        model_colors = self.model_colors
        clean_names = [m.split('/')[-1] if '/' in m else m for m in model_order]
        colors = [model_colors.get(m, "#000000") for m in model_order]

        half = (len(clean_names) + 1) // 2

        def legend_column(names, colors, width, height):
            n = len(names)
            y_gap = height / (n + 1)
            y_positions = [height - (i + 1) * y_gap for i in range(n)]
            fig = figure(
                width=width,
                height=height,
                x_range=(0, 40),
                y_range=(0, height),
                toolbar_location=None,
                outline_line_color=None,
            )
            fig.xaxis.visible = False
            fig.yaxis.visible = False
            fig.xgrid.visible = False
            fig.ygrid.visible = False
            fig.scatter([8]*n, y_positions, color=colors, marker="circle", size=8, alpha=0.6)
            fig.text([11]*n, y_positions, text=names, text_font_size="10pt", text_baseline="middle", text_align="left")
            return fig

        legend_fig1 = legend_column(clean_names[:half], colors[:half], width, height)
        legend_fig2 = legend_column(clean_names[half:], colors[half:], width, height)

        return row(legend_fig1, legend_fig2)

    def create_heatmaps(self, output_html: Optional[str] = None) -> column:
        """
        Create all three heatmaps, each sorted independently by mean raw weight.
        
        1. Weight heatmap (colored by weight, labeled with CV)
        2. Broad range heatmap
        3. Tight range heatmap
        
        Args:
            output_html: Optional path to save HTML file
        
        Returns:
            Bokeh column layout with three heatmaps
        """
        # Sort models by mean weight (rows — shared across all heatmaps)
        sorted_models, sorted_weight, sorted_cv, sorted_broad, sorted_tight, sorted_pairs = \
            self._sort_by_weight(
                self.models, self.weight_matrix.copy(), self.cv_matrix.copy(),
                self.broad_range_matrix.copy(), self.tight_range_matrix.copy(), self.pairs
            )
        sorted_models_s = sorted_models + ['Mean']

        # --- Heatmap 1: weight-colored, CV-labeled, pairs sorted by weight ---
        w_sorted, w_pairs = self._sort_cols_by_mean(sorted_weight, sorted_pairs, descending=True)
        cv_for_w = sorted_cv[:, [sorted_pairs.index(p) for p in w_pairs]]
        w_sorted_s = self.batch.add_mean_row_and_col(w_sorted)
        cv_for_w_s = self.batch.add_mean_row_and_col(cv_for_w)
        heatmap1 = self._create_heatmap(
            w_sorted_s,
            title="Weight (labeled with CV)",
            models=sorted_models_s,
            pairs=w_pairs + ['Mean'],
            text_matrix=cv_for_w_s,
            colormap_range=None
        )

        # --- Heatmap 2: broad range, pairs sorted by broad mean ---
        b_sorted, b_pairs = self._sort_cols_by_mean(sorted_broad, sorted_pairs, descending=True)
        b_sorted_s = self.batch.add_mean_row_and_col(b_sorted)
        heatmap2 = self._create_heatmap(
            b_sorted_s,
            title="Proportion in Broad Range",
            models=sorted_models_s,
            pairs=b_pairs + ['Mean'],
            colormap_range=(0, 1)
        )

        # --- Heatmap 3: tight range, pairs sorted by tight mean ---
        t_sorted, t_pairs = self._sort_cols_by_mean(sorted_tight, sorted_pairs, descending=True)
        t_sorted_s = self.batch.add_mean_row_and_col(t_sorted)
        heatmap3 = self._create_heatmap(
            t_sorted_s,
            title="Proportion in Tight Range",
            models=sorted_models_s,
            pairs=t_pairs + ['Mean'],
            colormap_range=(0, 1)
        )
        
        # Create bar chart and legend (uses global rank order)
        mean_accuracy_bar = self._create_mean_accuracy_bar_chart(width=350, height=200)
        model_legend = self._create_model_legend(width=180, height=250)

        bar_and_legend = row(mean_accuracy_bar, model_legend)
        layout = column(heatmap1, heatmap2, heatmap3, bar_and_legend)

        # Store individual subplots for per-subplot export
        self._subplots = {
            "weight_cv": heatmap1,
            "broad_range": heatmap2,
            "tight_range": heatmap3,
            "bar_and_legend": bar_and_legend,
        }

        if output_html:
            output_file(output_html)
            save(layout)
            show(layout)
            print(f"Heatmaps saved to {output_html}")
        
        return layout

    def save_subplot_pngs(self, prefix: str):
        """
        Save each heatmap subplot as an individual HTML + PNG.
        Must be called after create_heatmaps().

        Args:
            prefix: File path prefix, e.g. 'raw_figures/all_n0' →
                    raw_figures/all_n0_weight_cv.html / .png, etc.
        """
        if not hasattr(self, '_subplots') or not self._subplots:
            raise RuntimeError("Call create_heatmaps() before save_subplot_pngs()")

        html_paths = {}
        for name, fig in self._subplots.items():
            html_path = f"{prefix}_{name}.html"
            output_file(html_path)
            save(fig)
            html_paths[name] = html_path

        return html_paths