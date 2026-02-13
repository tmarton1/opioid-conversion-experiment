import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.io import output_file, save
from visualize.visualization_config import get_model_color, MODEL_ORDER, convert_to_pdf_auto_size

class VignetteScatterVisualizer:
    def __init__(self, csv_path, scenario_keys=None, tight_col=None):
        self.df = pd.read_csv(csv_path)
        self.scenario_keys = scenario_keys or self.df['scenario_key'].unique().tolist()
        self.tight_col = tight_col or [c for c in self.df.columns if c.startswith("tight(")][0]
        self.cond_base = (
            (self.df['include_nielsen'] == True) &
            (self.df['include_examples'] == False) &
            (self.df['ask_for_units'] == False) &
            (self.df['explain_units'] == False)
        )
        self.cond_units = (
            (self.df['include_nielsen'] == True) &
            (self.df['include_examples'] == False) &
            (self.df['ask_for_units'] == True) &
            (self.df['explain_units'] == True)
        )

    def get_merged(self, sub_df):
        # Use vignette_id instead of scenario_key
        base_df = sub_df[self.cond_base][['vignette_id', 'model', self.tight_col]]
        units_df = sub_df[self.cond_units][['vignette_id', 'model', self.tight_col]]

        # Aggregate to one row per vignette_id/model
        base_df = base_df.groupby(['vignette_id', 'model'])[self.tight_col].mean().reset_index()
        base_df = base_df.rename(columns={self.tight_col: 'tight_base'})
        units_df = units_df.groupby(['vignette_id', 'model'])[self.tight_col].mean().reset_index()
        units_df = units_df.rename(columns={self.tight_col: 'tight_units'})

        merged = pd.merge(base_df, units_df, on=['vignette_id', 'model'])
        return merged

    def make_plot(self, merged, title):
        merged['color'] = merged['model'].apply(get_model_color)
        jitter_strength = 0.05 * (merged['tight_base'].max() - merged['tight_base'].min())
        merged['tight_base_jitter'] = merged['tight_base'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(merged))
        merged['tight_units_jitter'] = merged['tight_units'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(merged))
        source = ColumnDataSource(merged)
        n_points = len(merged)
        # Count points per model
        model_counts = merged.groupby('model').size()
        min_per_model = model_counts.min() if not model_counts.empty else 0
        max_per_model = model_counts.max() if not model_counts.empty else 0
        plot_title = f"{title} (n = {n_points}, {min_per_model}-{max_per_model}/model)"
        p = figure(
            width=600, height=300,
            x_axis_label="Tight (no units prompt)",
            y_axis_label="Tight (units prompt)",
            title=plot_title,
            toolbar_location=None
        )
        for model in MODEL_ORDER:
            sub = merged[merged['model'] == model]
            if len(sub) == 0:
                continue
            p.scatter(
                x=sub['tight_base_jitter'],
                y=sub['tight_units_jitter'],
                color=sub['color'].iloc[0],
                size=4, alpha=0.5,
                legend_label=model
            )
        p.legend.orientation = "vertical"
        p.legend.click_policy = "hide"
        p.add_layout(p.legend[0], 'right')
        return p

    def create_layout(self):
        plots = []
        merged_all = self.get_merged(self.df)
        plots.append(self.make_plot(merged_all, "All Scenarios"))
        for key in self.scenario_keys:
            merged = self.get_merged(self.df[self.df['scenario_key'] == key])
            plots.append(self.make_plot(merged, key.replace("-", " ").title()))
        return column(*plots)

    def show(self):
        show(self.create_layout())

    def save_pdf(self, pdf_path):
        output_html = pdf_path.replace(".pdf", ".html")
        layout = self.create_layout()
        output_file(output_html)
        save(layout)
        convert_to_pdf_auto_size(output_html, pdf_path, scale=2.0)
