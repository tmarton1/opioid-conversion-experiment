from visualizer import Visualizer
from bokeh.plotting import figure
from bokeh.layouts import row
from visualization_config import MODEL_COLOR_DICT, MODEL_ORDER
from visualization_config import TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER, BROAD_RANGE_LOWER, BROAD_RANGE_UPPER
from beeswarm import BeeswarmPlacer
from bokeh.models import ColumnDataSource, HoverTool, FixedTicker, Span, Legend, LegendItem
import numpy as np
import json

class VignetteVisualizer(Visualizer):
    def __init__(
        self,
        experiment_results_csv: str,
        vignette_model_csv: str,
        consensus_csv: str,
        vignette_id: str,
        prompt_conditions: dict,
    ):
        self.vignette_id = vignette_id
        self.prompt_conditions = prompt_conditions
        self.consensus_plot_type = "dose"

        self.model_data = self._load_model_data(vignette_model_csv, vignette_id, prompt_conditions)
        self.consensus_data = self._load_consensus_data(consensus_csv, vignette_id, prompt_conditions)
        self.model_order = [m for m in MODEL_ORDER if m in self.model_data['model'].unique()]        
        self.model_colors = MODEL_COLOR_DICT
        self.from_unit = self.model_data['from_opioid_unit'].iloc[0] if 'from_opioid_unit' in self.model_data.columns else None
        self.to_unit = self.model_data['to_opioid_unit'].iloc[0] if 'to_opioid_unit' in self.model_data.columns else None

        self.tight_col = next((col for col in self.model_data.columns if col.startswith("tight")), None)
        self.broad_col = next((col for col in self.model_data.columns if col.startswith("broad")), None)
        if not self.tight_col or not self.broad_col:
            raise ValueError("Could not find tight or broad columns in analysis CSV.")
    
    def _get_scenario_text(self, vignette_id, num_chars=70):
        # Load vignettes.json once (consider caching if called repeatedly)
        with open("vignettes.json", "r") as f:
            vignettes = json.load(f)
        text = vignettes[vignette_id]["text"]
        title = text[:num_chars]
        if len(text) > num_chars:
            title += "..."
        return title
    
    def _prompt_conditions_label(self) -> str:
        inc_n = self.prompt_conditions.get("include_nielson", False)
        inc_ex = self.prompt_conditions.get("include_examples", False)
        return (
            f"(Nielson: {'yes' if inc_n else 'no'}, "
            f"Examples: {'yes' if inc_ex else 'no'})"
        )
    
    def _create_beeswarm_subplot(self, width, height):
        """
        Create a beeswarm plot for model-level doses.
        """
        y_axis_label = f"Dose ({self.to_unit})" if self.to_unit else "Dose"
        p = figure(
            width=width,
            height=height,
            title="Model Doses (Beeswarm)",
            x_axis_label="",
            y_axis_label=y_axis_label,
            y_axis_type="log",
            toolbar_location=None,
            y_range=(0.18, 210)
        )

        p.yaxis.ticker = FixedTicker(ticks=[0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200])
        
        # Collect model-level dose data
        y_values = []
        colors = []
        alphas = []
        models_list = []
        weights_list = []

        renderers = []
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
                renderer = p.scatter(
                    "x",
                    "y",
                    source=source_model,
                    size=10,
                    color=colors[idx],
                    alpha=alphas[idx],
                    marker="circle",
                )
                renderers.append((model_name, renderer))

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

        expected_to_dose = self.consensus_data.get('expected_to_dose')
        if expected_to_dose is not None:
            # Add expected_to_dose as horizontal dashed line
            expected_span = Span(
                location=float(expected_to_dose),
                dimension='width',
                line_color='gray',
                line_dash='dashed',
                line_width=2
            )
            p.add_layout(expected_span)

            # Add expected_to_dose tick and label
            ticks = list(p.yaxis.ticker.ticks) if hasattr(p.yaxis.ticker, "ticks") else []
            ticks.append(expected_to_dose)
            p.yaxis.ticker = FixedTicker(ticks=ticks)
            p.yaxis.major_label_overrides = {expected_to_dose: f"{expected_to_dose:.2g}"}

        return p, renderers
    
    def _create_cdf_subplot(self, width, height):
        """
        Create a CDF plot for model-level doses.
        """
        # Prepare figure
        title_main = self._get_scenario_text(self.vignette_id)
        title_sub = self._prompt_conditions_label()

        x_axis_label = f"Dose ({self.to_unit})" if self.to_unit else "Dose"
        p = figure(
            width=width,
            height=height,
            x_axis_label=x_axis_label,
            y_axis_label="CDF",
            x_axis_type="log",
            toolbar_location=None,
            x_range=(0.18, 210),
            title=f"{title_main}\n{title_sub}",
        )
        p.xaxis.ticker = FixedTicker(ticks=[0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200])
        renderers = []

        expected_to_dose = self.consensus_data.get('expected_to_dose')
        if expected_to_dose is not None:
            expected_to_dose = float(expected_to_dose)
            # Tight range lines
            tight_lower = expected_to_dose * TIGHT_RANGE_LOWER
            tight_upper = expected_to_dose * TIGHT_RANGE_UPPER
            p.add_layout(Span(location=tight_lower, dimension='height', line_color='gray', line_dash='dashed', line_width=2, line_alpha=0.7))
            p.add_layout(Span(location=tight_upper, dimension='height', line_color='gray', line_dash='dashed', line_width=2, line_alpha=0.7))
            # Broad range lines
            broad_lower = expected_to_dose * BROAD_RANGE_LOWER
            broad_upper = expected_to_dose * BROAD_RANGE_UPPER
            p.add_layout(Span(location=broad_lower, dimension='height', line_color='gray', line_dash='dashed', line_width=2, line_alpha=0.3))
            p.add_layout(Span(location=broad_upper, dimension='height', line_color='gray', line_dash='dashed', line_width=2, line_alpha=0.3))

        # Plot CDF for each model
        for model in self.model_order:
            model_data = self.model_data[self.model_data["model"] == model]
            if len(model_data) == 0:
                continue
            # If you have multiple observations, use them; otherwise, use mean
            dose_mean = float(np.exp(model_data["observed_log_dose_mean"].iloc[0]))
            dose_std = float(model_data["observed_log_dose_std"].iloc[0])
            n_obs = int(model_data["n_observations"].iloc[0])
            # Simulate samples if only mean/std/n_obs are available
            if n_obs > 1:
                samples = np.random.normal(loc=np.log(dose_mean), scale=dose_std, size=n_obs)
                doses = np.exp(samples)
            else:
                doses = np.array([dose_mean])
            doses = np.sort(doses)
            cdf = np.arange(1, len(doses) + 1) / len(doses)
            color = self.model_colors.get(model, "#000000")
            line_renderer = p.line(doses, cdf, line_width=2, color=color, legend_label=model.split("/")[-1] if "/" in model else model)
            renderers.append((model, line_renderer))
            circle_renderer = p.scatter(
                doses, cdf,
                size=4,
                color=color,
                alpha=0.8,
                marker="circle",
            )
            renderers.append((model, circle_renderer))

        # Add expected_to_dose as vertical dashed line
        expected_to_dose = self.consensus_data.get('expected_to_dose')
        #if expected_to_dose is not None:
            #expected_span = Span(location=float(expected_to_dose), dimension='height',
                                #line_color='gray', line_dash='dashed', line_width=2)
            #p.add_layout(expected_span)

        p.y_range.start = 0
        p.y_range.end = 1.05
        p.ygrid.grid_line_alpha = 0.3
        p.legend.visible = False

        hover = HoverTool(
            tooltips=[
                ("Dose", "$x{0.000}"),
                ("CDF", "$y{0.00}")
            ]
        )
        p.add_tools(hover)

        if expected_to_dose is not None:
            ticks = list(p.xaxis.ticker.ticks) if hasattr(p.xaxis.ticker, "ticks") else []
            ticks.append(expected_to_dose)
            p.xaxis.ticker = FixedTicker(ticks=ticks)
            p.xaxis.major_label_overrides = {expected_to_dose: f"{expected_to_dose:.2g}"}

        return p, renderers
    
    def _create_shared_legend(self, cdf_renderers, beeswarm_renderers, height):
        # Collect all renderers for each model
        renderer_dict = {}
        for model, renderer in cdf_renderers:
            renderer_dict.setdefault(model, []).append(renderer)
        for model, renderer in beeswarm_renderers:
            renderer_dict.setdefault(model, []).append(renderer)

        legend_items = [
            LegendItem(label=model, renderers=renderers)
            for model, renderers in renderer_dict.items()
        ]

        legend = Legend(items=legend_items, location="center")
        legend.click_policy = "hide"

        legend_fig = figure(height=height, width=300, toolbar_location=None)
        legend_fig.add_layout(legend, 'center')
        legend_fig.outline_line_alpha = 0
        legend_fig.grid.visible = False
        legend_fig.axis.visible = False
        return legend_fig

    def create_plot(self, width=900, height=350):
       # Subplot 1: CDF plot
        subplot1, cdf_renderers = self._create_cdf_subplot(width=int(0.7*width), height=height)

        # Subplot 2: model scatter (dose y-axis)
        subplot2, beeswarm_renderers = self._create_beeswarm_subplot(width=int(0.2*width), height=height)

        # Subplot 3: consensus plot (dose y-axis)
        subplot3 = self._create_consensus_plot(int(0.1*width), height)

        # Combine renderers for legend: map cleaned model names to all renderers
        renderer_dict = {}
        for model, renderer in cdf_renderers + beeswarm_renderers:
            clean_name = model.split("/")[-1] if "/" in model else model
            renderer_dict.setdefault(clean_name, []).append(renderer)

        # Prepare legend_items in model order
        legend_items = []
        for model in self.model_order:
            clean_name = model.split("/")[-1] if "/" in model else model
            renderers = renderer_dict.get(clean_name, [])
            legend_items.append((clean_name, renderers))

        # Use Visualizer._create_legend for consistent legend
        legend_fig = self._create_legend(height, legend_items)

        subplot4 = self._create_weight_bar(height, legend_items)
        subplot5 = self._create_weight_raw_bar(height, legend_items)
        subplot6 = self._create_model_accuracy_subplot(height, legend_items)


        # Layout
        layout = row(subplot1, subplot2, subplot3, subplot4, legend_fig, subplot5, subplot6)
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