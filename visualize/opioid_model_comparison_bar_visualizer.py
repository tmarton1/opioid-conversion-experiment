import pandas as pd
from bokeh.layouts import column
from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.models import LabelSet, ColumnDataSource, HoverTool, FactorRange, Legend, LegendItem, Div
from visualize.visualization_config import MODEL_COLOR_DICT, convert_to_pdf_auto_size, get_model_order, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER, BROAD_RANGE_LOWER, BROAD_RANGE_UPPER, assign_model_colors
import matplotlib.pyplot as plt
from adjustText import adjust_text
import io
from PIL import Image
import numpy as np

class OpioidModelComparisonBarVisualizer:
    def __init__(self, batch_names, batch_means_csv="model_batch_means.csv", batch_label_dict=None):
        self.batch_names = batch_names
        self.batch_means_csv = batch_means_csv
        self.df = pd.read_csv(batch_means_csv)
        self.data = [self.df[self.df["batch_name"] == name] for name in batch_names]
        self.batch_label_dict = batch_label_dict or {}

    @classmethod
    def from_combinations(cls, batch_names, batch_means_csv="model_batch_means.csv", batch_label_dict=None):
        return cls(batch_names, batch_means_csv, batch_label_dict)

    def plot(self, title="Model Comparison", width=900, height=400, bar_group_spacing=1.0, bar_width_ratio=0.7):
        # Get models from your data
        models = self.data[0]["model"].tolist()
        # Order models canonically
        ordered_models = get_model_order(models)
        model_colors = {m: MODEL_COLOR_DICT.get(m, "#1f77b4") for m in ordered_models}

        # Use ordered_models for x_factors and all plotting
        clean_names = [m.split('/')[-1] if '/' in m else m for m in ordered_models]
        batch_labels = [self.batch_label_dict.get(name, name) for name in self.batch_names]
        x_factors = [(clean_name, label) for clean_name in clean_names for label in batch_labels]
        p = figure(
            width=width,
            height=height,
            x_range=FactorRange(*x_factors),
            y_range=(0, 1.05),
            toolbar_location=None,
            title=title
        )

        clean_map = {m: m.split('/')[-1] if '/' in m else m for m in ordered_models}

        # Prepare data for all bars
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
        for batch_idx, batch_df in enumerate(self.data):
            tight_means = batch_df["mean_in_tight"].tolist()
            broad_means = batch_df["mean_in_broad"].tolist()
            batch_label = batch_labels[batch_idx]

            batch_df_ordered = (
                batch_df.set_index("model")
                .reindex(ordered_models)
                .reset_index()
            )

            for _, row in batch_df_ordered.iterrows():
                model_full = row["model"]
                clean_name = clean_map[model_full]

                tight = row["mean_in_tight"]
                broad = row["mean_in_broad"]
                
                bar_data["x"].append((clean_name, batch_label))
                bar_data["top"].append(tight)
                bar_data["bottom"].append(0)
                bar_data["broad"].append(broad-tight)
                bar_data["broad_top"].append(broad)
                bar_data["model"].append(clean_name)
                bar_data["batch"].append(batch_label)
                bar_data["color"].append(model_colors.get(model_full, "#1f77b4"))

        source = ColumnDataSource(bar_data)
        bar_width = bar_width_ratio / len(self.batch_names)
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

        p.xaxis.group_text_font_size = "0pt"
        p.xaxis.major_label_orientation = 1.0
        p.xaxis.major_label_text_font_size = "7pt"
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.xaxis.separator_line_color = None
        p.xgrid.visible = False
        p.xgrid.grid_line_color = None
        p.yaxis.axis_label = "Mean Proportion within Range"

        # Add invisible black bars for legend only
        legend_source = ColumnDataSource(dict(x=[(clean_names[0], batch_labels[0])], top=[0]))
        tight_dummy = p.vbar(x='x', top='top', width=0.2, color='black', alpha=0.9, source=legend_source, visible=True)
        broad_dummy = p.vbar(x='x', top='top', width=0.2, color='black', alpha=0.3, source=legend_source, visible=True)

        legend_items = [
            LegendItem(label=f"Tight Range ({TIGHT_RANGE_LOWER:.2f}–{TIGHT_RANGE_UPPER:.2f}x)", renderers=[tight_dummy]),
            LegendItem(label=f"Broad Range ({BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)", renderers=[broad_dummy])
        ]
        legend = Legend(items=legend_items)
        p.add_layout(legend, 'right')
        p.legend.label_text_font_size = "7pt"

        # One label per model, positioned above the *first* batch in that group
        label_source = ColumnDataSource(dict(
            x=[(name, batch_labels[0]) for name in clean_names],
            y=[.6] * len(clean_names),   # just below the top of y_range
            text=clean_names               # or an abbreviated version if you want
        ))

        model_labels = LabelSet(
            x='x', y='y', text='text', source=label_source,
            text_font_size="7pt",
            text_align="left",
            text_baseline="bottom",
            angle=np.pi/2,
            x_offset=5,
        )

        p.add_layout(model_labels)

        return p


class ModelComparisonFigureBuilder:
    def __init__(self, batch_means_csv="model_batch_means.csv", batch_label_dict=None):
        self.batch_means_csv = batch_means_csv
        self.batch_label_dict = batch_label_dict or {
            'ome_n0': 'naive',
            'ome_n1': 'with Nielson',
            'pairs_n0': 'naive',
            'pairs_n1': 'with Nielson',
            'all_n0': 'naive',
            'all_n1': 'with Nielson'
        }

    def matplotlib_scatter_to_bokeh(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically for Bokeh
        arr = np.array(img)
        # Convert RGBA to uint32 for Bokeh
        view = arr.view(dtype=np.uint32).reshape(arr.shape[0], arr.shape[1])
        height, width = arr.shape[:2]
        p = figure(width=width, height=height, x_range=(0, width), y_range=(0, height), toolbar_location=None)
        p.image_rgba(image=[view], x=0, y=0, dw=width, dh=height)
        p.axis.visible = False
        p.grid.visible = False
        return p

    def build_figures(self):
        combinations_list = [
            ["all_n0", "all_n1"],
            ["ome_n0", "ome_n1"],
            ["pairs_n0", "pairs_n1"]
        ]
        titles = [
            "All Opioid Pairs",
            "Oral Morphine Conversions",
            "Conversions w/o Oral Morphine"
        ]
        figures = []

        # Bar charts
        for combos, title in zip(combinations_list, titles):
            visualizer = OpioidModelComparisonBarVisualizer.from_combinations(
                combos,
                batch_means_csv=self.batch_means_csv,
                batch_label_dict=self.batch_label_dict
            )
            fig = visualizer.plot(title=title, width=800, height=300, bar_group_spacing=1.0, bar_width_ratio=1.5)
            figures.append(fig)

        # Scatter plot 1: broad range, oral morphine naive vs. pairs naive
        df = pd.read_csv(self.batch_means_csv)
        # Filter for broad range, naive
        ome_naive = df[(df["batch_name"] == "ome_n0")].sort_values("model")
        pairs_naive = df[(df["batch_name"] == "pairs_n0")].sort_values("model")
        models = ome_naive["model"].tolist()
        colors = [assign_model_colors(models)[m] for m in models]
        clean_names = [m.split('/')[-1] if '/' in m else m for m in models]

        source1 = ColumnDataSource(data=dict(
            x=pairs_naive["mean_in_broad"],
            y=ome_naive["mean_in_broad"],
            label=clean_names
        ))


        p1 = figure(
            width=600, height=400,
            title="Naive Knowledge of Opioid Conversions",
            x_range=(0, 1.05), y_range=(0, 1.05),
            toolbar_location=None
        )
        p1.scatter(
            x=pairs_naive["mean_in_broad"],
            y=ome_naive["mean_in_broad"],
            color=colors,
            alpha=0.6,
            size=12
        )
        labels1 = LabelSet(x='x', y='y', text='label', level='glyph',
                           x_offset=5, y_offset=5, source=source1, text_font_size="8pt")
        p1.add_layout(labels1)
        p1.line([0, 1.05], [0, 1.05], color="black", line_dash="dashed", line_width=1)
        p1.xaxis.axis_label = f"Conversions w/o Oral Morphine (Proportion {BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)"
        p1.yaxis.axis_label = f"Oral Morphine Conversions (Proportion i{BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)"

        return figures
    
    async def save(self, html_path="model_comparison_bar.html", pdf_path="model_comparison_bar.pdf"):
        grand_title = Div(text="<h2>Model Comparison Figures</h2>", styles={"text-align": "center"})
        figs = self.build_figures()
        output_file(html_path)
        save(column(grand_title, *figs))
        await convert_to_pdf_auto_size(html_path, pdf_path, scale=2.0)

    def save_scatter_figures_pdf(self, pdf_path="model_comparison_scatter.pdf"):
        df = pd.read_csv(self.batch_means_csv)

        # Scatter 1: Naive Knowledge of Opioid Conversions
        ome_naive = df[df["batch_name"] == "ome_n0"].sort_values("model")
        pairs_naive = df[df["batch_name"] == "pairs_n0"].sort_values("model")
        models = ome_naive["model"].tolist()
        clean_names = [m.split('/')[-1] if '/' in m else m for m in models]

        # Assign colors by model name for each point
        colors = [MODEL_COLOR_DICT.get(m, "#1f77b4") for m in ome_naive["model"]]

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(
            pairs_naive["mean_in_broad"], ome_naive["mean_in_broad"],
            c=colors, alpha=0.9, s=60
        )
        ax1.plot([0, 1.05], [0, 1.05], color="black", linestyle="dashed", linewidth=1)
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Naive Knowledge of Opioid Conversions")
        ax1.set_xlabel(f"Conversions w/o Oral Morphine (Proportion {BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)", fontsize=8, fontstyle='italic')
        ax1.set_ylabel(f"Oral Morphine Conversions (Proportion {BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)", fontsize=8, fontstyle='italic')
        texts = []
        for x, y, label in zip(pairs_naive["mean_in_broad"], ome_naive["mean_in_broad"], clean_names):
            texts.append(ax1.text(x, y, label, fontsize=6, ha='left', va='bottom'))
        adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle='-', color='gray'))
        plt.tight_layout()

        # Scatter 2: Nielson Effect
        all_nielson = df[df["batch_name"] == "all_n1"].sort_values("model")
        colors2 = [MODEL_COLOR_DICT.get(m, "#1f77b4") for m in all_nielson["model"]]
        clean_names2 = [m.split('/')[-1] if '/' in m else m for m in all_nielson["model"]]


        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(
            ome_naive["mean_in_broad"], all_nielson["mean_in_tight"],
            c=colors2, alpha=0.9, s=60
        )
        ax2.plot([0, 1.05], [0, 1.05], color="black", linestyle="dashed", linewidth=1)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Nielson Effect")
        ax2.set_xlabel(f"Oral Morphine Conversions (Proportion {BROAD_RANGE_LOWER:.2f}–{BROAD_RANGE_UPPER:.2f}x)", fontsize=8, fontstyle='italic')
        ax2.set_ylabel(f"All Opioid Pair Conversions with Nielson (Proportion {TIGHT_RANGE_LOWER:.2f}–{TIGHT_RANGE_UPPER:.2f}x)", fontsize=8, fontstyle='italic')
        texts2 = []
        for x, y, label in zip(ome_naive["mean_in_broad"], all_nielson["mean_in_tight"], clean_names2):
            texts2.append(ax2.text(x, y, label, fontsize=6, ha='left', va='bottom'))
        adjust_text(texts2, ax=ax2, arrowprops=dict(arrowstyle='-', color='gray'))
        plt.tight_layout()

        # Save both figures to one PDF
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
        plt.close(fig1)
        plt.close(fig2)