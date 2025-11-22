from visualizer import Visualizer
from bokeh.plotting import figure
from bokeh.layouts import row
from visualization_config import MODEL_COLOR_DICT
from beeswarm import BeeswarmPlacer
from bokeh.models import ColumnDataSource, HoverTool
import numpy as np

class VignetteVisualizer(Visualizer):
    def __init__(
        self,
        experiment_results_csv: str,
        model_csv: str,
        consensus_csv: str,
        vignette_id: str,
        prompt_conditions: dict,
    ):
        self.vignette_id = vignette_id
        self.prompt_conditions = prompt_conditions
        self.consensus_plot_type = "dose"

        self.model_data = self._load_model_data(model_csv, vignette_id, prompt_conditions)
        self.consensus_data = self._load_consensus_data(consensus_csv, vignette_id, prompt_conditions)
        self.model_order = self.model_data['model'].unique().tolist()
        self.model_colors = MODEL_COLOR_DICT
        self.from_unit = self.model_data['from_opioid_unit'].iloc[0] if 'from_opioid_unit' in self.model_data.columns else None
        self.to_unit = self.model_data['to_opioid_unit'].iloc[0] if 'to_opioid_unit' in self.model_data.columns else None

    def _create_beeswarm_subplot(self, width, height):
        """
        Create a beeswarm plot for model-level doses.
        """
        # Prepare figure
        p = figure(
            width=width,
            height=height,
            title="Model Doses (Beeswarm)",
            x_axis_label="",
            y_axis_label="Dose",
            y_axis_type="log",
            toolbar_location=None,
        )

        # Collect model-level dose data
        y_values = []
        colors = []
        alphas = []
        models_list = []
        weights_list = []

        for model in self.model_order:
            model_data = self.model_data[self.model_data["model"] == model]
            if len(model_data) == 0:
                continue

            # Use log mean → exp to get raw dose
            log_mean = model_data["observed_log_dose_mean"].iloc[0]
            y_raw = float(np.exp(log_mean))
            y_values.append(y_raw)

            color = self.model_colors.get(model, "#000000")
            colors.append(color)

            weight = float(model_data["weight"].iloc[0])
            alpha = 0.5 + 0.4 * min(weight, 10.0) / 10.0
            alphas.append(alpha)

            models_list.append(model.split("/")[-1] if "/" in model else model)
            weights_list.append(weight)

        # Beeswarm placement
        if len(y_values) > 0:
            y_array = np.array(y_values, dtype=float)
            placer = BeeswarmPlacer(x_max=0.35, r=0.06, use_log_y=True)
            x_positions = list(placer.compute(y_array))

            for idx, model_name in enumerate(models_list):
                source_model = ColumnDataSource(
                    data=dict(
                        x=[x_positions[idx]],
                        y=[y_values[idx]],
                        model=[model_name],
                        weight=[weights_list[idx]],
                    )
                )
                p.scatter(
                    "x",
                    "y",
                    source=source_model,
                    size=10,
                    color=colors[idx],
                    alpha=alphas[idx],
                    marker="circle",
                )

            hover = HoverTool(
                tooltips=[
                    ("Model", "@model"),
                    ("Dose", "@y{0.000}"),
                    ("Weight", "@weight{0.00}"),
                ]
            )
            p.add_tools(hover)

        p.xaxis.visible = False
        p.xgrid.grid_line_color = None
        p.x_range.start = -0.4
        p.x_range.end = 0.4

        return p

    def create_plot(self, width=900, height=350):
        # Subplot 1: empty
        subplot1 = figure(width=width, height=height, toolbar_location=None)
        subplot1.xaxis.visible = False
        subplot1.yaxis.visible = False
        subplot1.grid.visible = False

        # Subplot 2: model scatter (dose y-axis)
        subplot2 = self._create_beeswarm_subplot(width=180, height=height)

        # Subplot 3: consensus plot (dose y-axis)
        subplot3 = self._create_consensus_plot(width, height)

        # Subplot 4: accuracy subplot
        legend_items = self._get_legend_items()
        subplot4 = self._create_model_accuracy_subplot(height, legend_items)

        # Legend
        legend = self._create_legend(height, legend_items)

        # Subplot 5: normalized weight bar
        subplot5 = self._create_weight_bar(height, legend_items)

        # Subplot 6: raw weight bar
        subplot6 = self._create_weight_raw_bar(height, legend_items)

        # Layout
        layout = row(subplot1, subplot2, subplot3, subplot4, legend, subplot5, subplot6)
        return layout

    def _load_model_data(self, model_csv, vignette_id, prompt_conditions):
        import pandas as pd
        df = pd.read_csv(model_csv)
        mask = (df['vignette_id'] == vignette_id)
        for k, v in prompt_conditions.items():
            mask &= (df[k] == v)
        return df[mask].copy()

    def _load_consensus_data(self, consensus_csv, vignette_id, prompt_conditions):
        import pandas as pd
        df = pd.read_csv(consensus_csv)
        mask = (df['vignette_id'] == vignette_id)
        for k, v in prompt_conditions.items():
            mask &= (df[k] == v)
        if mask.any():
            return df[mask].iloc[0].to_dict()
        return {}

    def _get_legend_items(self):
        return [(m.split('/')[-1] if '/' in m else m, []) for m in self.model_order]