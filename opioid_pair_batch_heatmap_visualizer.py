from typing import Optional, List
import numpy as np
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import row, column
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
from bokeh.transform import transform
from bokeh.palettes import Greys256
from opioid_pair_batch import OpioidPairBatch
from visualization_config import get_model_order, assign_model_colors, normalize_model_name
from visualization_config import (
    TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER,
    BROAD_RANGE_LOWER, BROAD_RANGE_UPPER
)

class OpioidPairBatchHeatmapVisualizer:
    """
    Creates heatmap visualizations across all model-pair combinations in a batch.
    
    Three subplots:
    1. Raw weight heatmap
    2. Proportion in broad range 
    3. Proportion in tight range 
    """
    
    def __init__(self, batch: OpioidPairBatch, cell_size: int = 30,
                 width_padding: int = 50, height_padding: int = 50,
                 cell_text_size: str = "6pt", label_text_size: str = "7pt"):
        """
        Initialize the heatmap visualizer.
        
        Args:
            batch: OpioidPairBatch containing the keys to visualize
            cell_size: Size of each cell in pixels (default: 10)
        """
        self.batch = batch
        self.cell_size = cell_size
        self.width_padding = width_padding
        self.height_padding = height_padding
        self.cell_text_size = cell_text_size
        self.label_text_size = label_text_size
        
        # Get data from batch and order models
        all_models = batch.get_models()
        self.models = get_model_order(all_models)
        self.model_colors = assign_model_colors(self.models)
        self.pairs = batch.get_pairs()
        
        print(f"Found {len(self.models)} models and {len(self.pairs)} pairs")
        
        # Compute matrices using batch methods (reordered for model order)
        self.weight_matrix = self._compute_ordered_matrix(batch.compute_weight_matrix(), all_models)
        self.broad_range_matrix = self._compute_ordered_matrix(batch.compute_range_matrix(BROAD_RANGE_LOWER, BROAD_RANGE_UPPER), all_models)
        self.tight_range_matrix = self._compute_ordered_matrix(batch.compute_range_matrix(TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER), all_models)
        
        # Add summary statistics
        self._add_summaries()
        
        print(f"Weight matrix shape: {self.weight_matrix.shape}")
        print(f"Non-NaN values in weight matrix: {np.sum(~np.isnan(self.weight_matrix))}")
    
    
    def _compute_ordered_matrix(self, matrix: np.ndarray, original_models: List[str]) -> np.ndarray:
        """Reorder matrix rows to match self.models order."""
        if len(original_models) != matrix.shape[0]:
            return matrix
        
        # Create mapping from original to new order
        new_order = [original_models.index(m) for m in self.models if m in original_models]
        return matrix[new_order, :]
    
    def _add_summaries(self):
        """Add mean row and mean column to each matrix, and update labels."""
        self.weight_matrix = self.batch.get_weight_matrix_with_summary()
        self.broad_range_matrix = self.batch.get_range_matrix_with_summary(BROAD_RANGE_LOWER, BROAD_RANGE_UPPER)
        self.tight_range_matrix = self.batch.get_range_matrix_with_summary(TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER)
        self.pairs = self.pairs + ['Mean']
        self.models = self.models + ['Mean']
    
    def _create_heatmap(self, matrix: np.ndarray, title: str, 
                       colormap_range: tuple = None) -> figure:
        """
        Create a single heatmap figure.
        
        Args:
            matrix: Data matrix (models × pairs), including summary row/col
            title: Title for the heatmap
            colormap_range: (min, max) for color scale, or None for auto
        
        Returns:
            Bokeh figure
        """
        # Clean model names - use normalization for consistency
        model_labels = []
        for m in self.models:
            # Normalize first, then extract clean name
            normalized = normalize_model_name(m)
            clean = normalized.split('/')[-1] if '/' in normalized else normalized
            model_labels.append(clean)
        
        # Prepare data for heatmap
        model_list = []
        pair_list = []
        value_list = []
        text_list = []
        is_summary_list = []
        
        for i, model in enumerate(model_labels):
            for j, pair in enumerate(self.pairs):
                val = matrix[i, j]
                model_list.append(model)
                pair_list.append(pair)
                value_list.append(val)
                
                # Check if this is a summary cell
                is_summary = (model == 'Mean' or pair == 'Mean')
                is_summary_list.append(is_summary)
                
                # Format text (2 sig figs)
                if np.isnan(val):
                    text_list.append("")
                else:
                    text_list.append(f"{val:.2g}")
        
        # Create ColumnDataSource
        source = ColumnDataSource(data=dict(
            model=model_list,
            pair=pair_list,
            value=value_list,
            text=text_list,
            is_summary=is_summary_list
        ))
        
        # Determine color range (excluding summary row/col for better scaling)
        if colormap_range is None:
            non_summary_values = [v for v, is_sum in zip(value_list, is_summary_list) 
                                 if not is_sum and not np.isnan(v)]
            if non_summary_values:
                vmin, vmax = min(non_summary_values), max(non_summary_values)
            else:
                vmin, vmax = 0, 1
        else:
            vmin, vmax = colormap_range
        
        # Use Greys palette (light gray to dark)
        mapper = LinearColorMapper(
            palette=Greys256,
            low=vmin,
            high=vmax,
            nan_color='lightgray'
        )
        
        # Calculate figure size (half of previous)
        width = len(self.pairs) * self.cell_size + self.width_padding
        height = len(self.models) * self.cell_size + self.height_padding
        
        # Create figure
        p = figure(
            width=width,
            height=height,
            x_range=self.pairs,
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
        model_order = self.models[:-1]
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
            y_range=(0, max(tops_caution) * 1.05 if tops_caution else 1),
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
        model_order = self.models[:-1]
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
        Create all three heatmaps.
        
        Args:
            output_html: Optional path to save HTML file
        
        Returns:
            Bokeh column layout with three heatmaps
        """
        # Create three heatmaps
        heatmap1 = self._create_heatmap(
            self.weight_matrix,
            title="Raw Weights",
            colormap_range=None
        )
        
        heatmap2 = self._create_heatmap(
            self.broad_range_matrix,
            title="Proportion in Broad Range",
            colormap_range=(0, 1)
        )
        
        heatmap3 = self._create_heatmap(
            self.tight_range_matrix,
            title="Proportion in Tight Range",
            colormap_range=(0, 1)
        )
        # Create bar chart and legend
        mean_accuracy_bar = self._create_mean_accuracy_bar_chart(width=350, height=200)
        model_legend = self._create_model_legend(width=180, height=250)

        # Place bar chart and legend side-by-side
        bar_and_legend = row(mean_accuracy_bar, model_legend)

        # Arrange everything vertically
        layout = column(heatmap1, heatmap2, heatmap3, bar_and_legend)
        
        # Save and show if requested
        if output_html:
            output_file(output_html)
            save(layout)
            show(layout)
            print(f"Heatmaps saved to {output_html}")
        
        return layout