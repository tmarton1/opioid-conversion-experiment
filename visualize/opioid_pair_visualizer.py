import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import row
from bokeh.models import HoverTool, Label, ColumnDataSource
from visualize.visualization_config import get_model_order, assign_model_colors, EPSILON_FOR_WEIGHT
from visualize.visualizer import Visualizer
from visualize.beeswarm import BeeswarmPlacer
from typing import Optional, Dict, List

class OpioidPairVisualizer(Visualizer):
    """
    Creates a multi-panel visualization for opioid pair conversion analysis.
    Inherits shared visualization logic from Visualizer.
    """

    MAX_WEIGHT = 1.0 / EPSILON_FOR_WEIGHT

    def __init__(self,
                 *,
                 experiment_results_csv: str,
                 pair_model_csv: str,
                 consensus_csv: str,
                 from_opioid: str,
                 from_opioid_unit: str,
                 to_opioid: str,
                 to_opioid_unit: str,
                 prompt_conditions: Dict[str, bool]):
        self.from_opioid = from_opioid
        self.from_opioid_unit = from_opioid_unit
        self.to_opioid = to_opioid
        self.to_opioid_unit = to_opioid_unit
        self.prompt_conditions = prompt_conditions
        self.consensus_plot_type = "coefficient"

        # Load raw experiment results for first plot
        experiment_df = pd.read_csv(experiment_results_csv)
        self.raw_data = experiment_df[
            (experiment_df['from_opioid'] == from_opioid) &
            (experiment_df['from_opioid_unit'] == from_opioid_unit) &
            (experiment_df['to_opioid'] == to_opioid) &
            (experiment_df['to_opioid_unit'] == to_opioid_unit) &
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
            (pair_model_df['from_opioid_unit'] == from_opioid_unit) &
            (pair_model_df['to_opioid'] == to_opioid) &
            (pair_model_df['to_opioid_unit'] == to_opioid_unit) &
            (pair_model_df['include_nielson'] == prompt_conditions.get('include_nielson', False)) &
            (pair_model_df['include_examples'] == prompt_conditions.get('include_examples', False))
        ].copy()

        # Load consensus data for third plot
        consensus_df = pd.read_csv(consensus_csv)
        consensus_filtered = consensus_df[
            (consensus_df['from_opioid'] == from_opioid) &
            (consensus_df['from_opioid_unit'] == from_opioid_unit) &
            (consensus_df['to_opioid'] == to_opioid) &
            (consensus_df['to_opioid_unit'] == to_opioid_unit) &
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

        # Define consistent model order using shared config
        available_models = self.raw_data['model'].unique().tolist()
        self.model_order = get_model_order(available_models)

        # Assign colors to models using shared config
        self.model_colors = assign_model_colors(self.model_order)

        # Call parent constructor for shared logic
        super().__init__(self.pair_data, self.consensus_data, prompt_conditions)

    def _get_log_ticks(self, include_literature: bool = True) -> List[float]:
        base_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
        if include_literature and self.literature_coeff not in base_ticks:
            ticks = base_ticks + [self.literature_coeff]
            ticks.sort()
            return ticks
        return base_ticks

    def _force_y_range(self, p):
        L = float(self.literature_coeff)
        p.y_range.start = 0.04 * L
        p.y_range.end = 25.0 * L

    def _prompt_conditions_label(self) -> str:
        inc_n = self.prompt_conditions.get("include_nielson", False)
        inc_ex = self.prompt_conditions.get("include_examples", False)
        return (
            f"(Nielson: {'yes' if inc_n else 'no'}, "
            f"Examples: {'yes' if inc_ex else 'no'})"
        )

    def _create_coefficient_and_beeswarm_plots(self, width1: int, width2: int, height: int) -> tuple:
        title_main = f"{self.from_opioid}_{self.from_opioid_unit} \u2192 {self.to_opioid}_{self.to_opioid_unit}"
        title_sub = self._prompt_conditions_label()

        p1 = figure(
            width=width1,
            height=height,
            title=f"{title_main}\n{title_sub}",
            x_axis_label=f"From Dose ({self.from_opioid_unit})",
            y_axis_label=f"Observed Coefficient ({self.to_opioid_unit}/{self.from_opioid_unit})",
            x_axis_type="log",
            y_axis_type="log",
            toolbar_location=None,
        )

        y_ticks = self._get_log_ticks(include_literature=True)
        p1.yaxis.ticker = y_ticks
        p1.xaxis.minor_tick_line_color = None

        x_min = self.raw_data["from_dose"].min()
        x_max = self.raw_data["from_dose"].max()
        p1.line(
            [x_min, x_max],
            [self.literature_coeff, self.literature_coeff],
            line_dash="dashed",
            line_color="gray",
            line_width=2,
        )

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

        x_positions_p2: List[float] = []
        y_values_p2: List[float] = []
        colors_p2: List[str] = []
        alphas_p2: List[float] = []
        models_list_p2: List[str] = []
        weights_list_p2: List[float] = []
        legend_items: List[tuple] = []

        for model in self.model_order:
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

            model_name_clean = model.split("/")[-1] if "/" in model else model
            legend_items.append((model_name_clean, [renderer_p1]))

            model_data_p2 = self.pair_data[self.pair_data["model"] == model]
            if len(model_data_p2) == 0:
                continue

            log_mean = model_data_p2["observed_log_coefficient_mean"].iloc[0]
            y_raw = float(np.exp(log_mean))
            y_values_p2.append(y_raw)

            colors_p2.append(color)

            weight = float(model_data_p2["weight"].iloc[0])
            alpha = self._calculate_alpha(weight)
            alphas_p2.append(alpha)

            models_list_p2.append(model_name_clean)
            weights_list_p2.append(weight)

        if len(y_values_p2) > 0:
            y_array = np.array(y_values_p2, dtype=float)
            placer = BeeswarmPlacer(x_max=0.35, r=0.06, use_log_y=True)
            x_positions_p2 = list(placer.compute(y_array))

            total_w = float(sum(weights_list_p2))
            if total_w > 0:
                norm_weights_list_p2 = [w / total_w for w in weights_list_p2]
            else:
                norm_weights_list_p2 = [np.nan for _ in weights_list_p2]

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

                for j, (name, renderers) in enumerate(legend_items):
                    if name == model_name_clean:
                        legend_items[j] = (name, renderers + [renderer_p2])
                        break

            hover = HoverTool(
                tooltips=[
                    ("Model", "@model"),
                    ("Coefficient", "@y{0.000}"),
                    ("Normalized weight", "@weight_norm{0.00}"),
                    ("Raw weight", "@weight_raw{0.00}"),
                ]
            )
            p2.add_tools(hover)

        p2_dict = {"figure": p2}
        return p1, p2_dict, legend_items

    def create_plot(self, width: int = 500, height: int = 300, output_html: Optional[str] = None) -> row:
        w2 = int(width * 0.20)
        w3 = int(w2 * 0.5)
        w1 = width - w2 - w3

        p1, p2_renderers_dict, legend_items = self._create_coefficient_and_beeswarm_plots(w1, w2, height)
        p2 = p2_renderers_dict['figure']
        p3 = self._create_consensus_plot(w3, height)

        self._force_y_range(p1)
        self._force_y_range(p2)
        self._force_y_range(p3)

        p2.y_range = p1.y_range
        p3.y_range = p1.y_range

        legend_fig = self._create_legend(height, legend_items)
        weight_fig = self._create_weight_bar(height, legend_items)
        raw_weight_fig = self._create_weight_raw_bar(height, legend_items)
        accuracy_fig = self._create_model_accuracy_subplot(height, legend_items)

        layout = row(p1, p2, p3, weight_fig, legend_fig, raw_weight_fig, accuracy_fig)

        if output_html:
            from bokeh.io import output_file, save
            output_file(output_html)
            save(layout)

        return layout