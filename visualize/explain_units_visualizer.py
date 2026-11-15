from common.config import RESULTS_DIR
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, FactorRange, Legend, LegendItem
from visualize.visualization_config import MODEL_COLOR_DICT, get_model_order, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER, BROAD_RANGE_LOWER, BROAD_RANGE_UPPER
from visualize.visualization_config import convert_to_pdf_auto_size

EXCLUDE_MODEL = "anthropic/claude-3-haiku-20240307"

class ExplainUnitsVisualizer:
    def __init__(self,
                 batch_means_csv=f"{RESULTS_DIR}/model_batch_means.csv",
                 batch_means_explain_units_csv=f"{RESULTS_DIR}/model_batch_means_explain_units.csv",
                 models=None):
        # Load data
        self.df_ome = pd.read_csv(batch_means_csv)
        self.df_units = pd.read_csv(batch_means_explain_units_csv)

        # Filter for batch_name and exclude Claude Haiku
        self.ome_n1 = self.df_ome[(self.df_ome["batch_name"] == "ome_n1") & (self.df_ome["model"] != EXCLUDE_MODEL)]
        self.units = self.df_units[(self.df_units["batch_name"] == "4hours_units") & (self.df_units["model"] != EXCLUDE_MODEL)]
        self.units_explained = self.df_units[(self.df_units["batch_name"] == "4hours_units_explained") & (self.df_units["model"] != EXCLUDE_MODEL)]

        # Get model order
        self.models = get_model_order([m for m in models if m in self.ome_n1["model"].tolist()] if models is not None else self.ome_n1["model"].tolist())
        self.model_colors = {m: MODEL_COLOR_DICT.get(m, "#1f77b4") for m in self.models}
        self.clean_names = [m.split('/')[-1] if '/' in m else m for m in self.models]

    def plot(self, title="Explain Units Comparison", width=1100, height=400, bar_width=0.7):
        # Prepare data for each bar
        batch_labels = ["mg/day", "mg/4 hours", "units explained"]
        bars = [self.ome_n1, self.units, self.units_explained]
        bar_data = {
            "x": [],
            "top": [],
            "bottom": [],
            "broad": [],
            "broad_top": [],
            "model": [],
            "batch": [],
            "color": []
        }
        for i, (batch_df, label) in enumerate(zip(bars, batch_labels)):
            batch_df_ordered = batch_df.set_index("model").reindex(self.models).reset_index()
            for _, row in batch_df_ordered.iterrows():
                model_full = row["model"]
                clean_name = model_full.split('/')[-1] if '/' in model_full else model_full
                tight = row["mean_in_tight"]
                broad = row["mean_in_broad"]
                bar_data["x"].append((clean_name, label))
                bar_data["top"].append(tight)
                bar_data["bottom"].append(0)
                bar_data["broad"].append(broad-tight)
                bar_data["broad_top"].append(broad)
                bar_data["model"].append(clean_name)
                bar_data["batch"].append(label)
                bar_data["color"].append(self.model_colors.get(model_full, "#1f77b4"))

        source = ColumnDataSource(bar_data)
        x_factors = [(name, label) for name in self.clean_names for label in batch_labels]
        p = figure(
            width=width,
            height=height,
            x_range=FactorRange(*x_factors),
            y_range=(0, 1.05),
            toolbar_location=None,
            title=title
        )

        # Bars
        tight_bars = p.vbar(
            x='x',
            top='top',
            width=bar_width,
            color='color',
            alpha=0.9,
            source=source
        )
        broad_bars = p.vbar(
            x='x',
            top='broad_top',
            bottom='top',
            width=bar_width,
            color='color',
            alpha=0.3,
            source=source
        )

        # Dummy bars for legend (black and gray squares)
        dummy_x = x_factors[0]

        legend_tight = p.vbar(
            x=[dummy_x], top=[None], width=bar_width, color='black', alpha=0.9, visible=True
        )
        legend_broad = p.vbar(
            x=[dummy_x], top=[None], bottom=[None], width=bar_width, color='gray', alpha=0.3, visible=True
        )

        # Legend uses dummy bars for squares
        legend_items = [
            LegendItem(label=f"Tight Range ({TIGHT_RANGE_LOWER:.2f}–{TIGHT_RANGE_UPPER:.2f}x)", renderers=[legend_tight]),
            LegendItem(label=f"Broad Range ({BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)", renderers=[legend_broad])
        ]
        legend = Legend(items=legend_items)
        p.add_layout(legend, 'right')
        p.legend.label_text_font_size = "9pt"

        # Hover
        hover_tight = HoverTool(
            tooltips=[
                ('Model', '@model'),
                ('Batch', '@batch'),
                (f'Tight ({TIGHT_RANGE_LOWER:.0%}–{TIGHT_RANGE_UPPER:.0%})', '@top{0.0%}')
            ],
            renderers=[tight_bars]
        )
        hover_broad = HoverTool(
            tooltips=[
                ('Model', '@model'),
                ('Batch', '@batch'),
                (f'Broad ({BROAD_RANGE_LOWER:.0%}–{BROAD_RANGE_UPPER:.0%})', '@broad{0.0%}')
            ],
            renderers=[broad_bars]
        )
        p.add_tools(hover_tight)
        p.add_tools(hover_broad)

        # Axis and labels
        p.xaxis.group_text_font_size = "0pt"
        p.xaxis.major_label_orientation = 1.0
        p.xaxis.major_label_text_font_size = "7pt"
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.xaxis.separator_line_color = None
        p.xgrid.visible = False
        p.xgrid.grid_line_color = None
        p.yaxis.axis_label = "Mean Proportion within Range"

        # Model labels above the second bar
        label_source = ColumnDataSource(dict(
            x=[(name, batch_labels[1]) for name in self.clean_names],
            y=[.6] * len(self.clean_names),
            text=self.clean_names
        ))
        model_labels = LabelSet(
            x='x', y='y', text='text', source=label_source,
            text_font_size="9pt",
            text_align="left",
            text_baseline="bottom",
            angle=1.5708,
            x_offset=5,
        )
        p.add_layout(model_labels)

        return p

    async def save(self, output_html, output_pdf):
        from bokeh.io import output_file, save
        plot = self.plot()
        output_file(output_html)
        save(plot)
        print(f"✓ Saved HTML to {output_html}")
        await convert_to_pdf_auto_size(output_html, output_pdf, scale=2.0)
        print(f"✓ Saved PDF to {output_pdf}")