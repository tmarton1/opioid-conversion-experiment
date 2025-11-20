import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, FactorRange, Label, LabelSet
from visualization_config import get_model_order, assign_model_colors, normalize_model_name
from visualization_config import (
    TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER,
    BROAD_RANGE_LOWER, BROAD_RANGE_UPPER
)
from opioid_pair_batch import OpioidPairBatch

class OpioidModelComparisonBarVisualizer:
    def __init__(self, batch_list, combinations):
        self.batch_list = batch_list
        self.combinations = combinations

        # Infer models from the first batch, using the same logic as the heatmap visualizer
        all_models = batch_list[0].get_models()
        self.models = get_model_order(all_models)
        self.model_colors = assign_model_colors(self.models)

    @classmethod
    def from_combinations(cls, combinations, experiment_results_csv, pair_model_csv, consensus_csv):
        batch_list = []
        for combo in combinations:
            filter_spec = {
                "include_nielson": combo["include_nielson"],
                "include_examples": False,
                "exclude_self_pairs": True,
            }
            if combo.get("ome") is not None:
                filter_spec["ome"] = combo["ome"]
            batch = OpioidPairBatch.from_filter_specs(
                experiment_results_csv=experiment_results_csv,
                pair_model_csv=pair_model_csv,
                consensus_csv=consensus_csv,
                filter_specs=[filter_spec],
            )
            batch_list.append(batch)
        return cls(batch_list, combinations)

    def plot(self, width=500, height=300, title="Model Comparison Across Conditions"):
        # Exclude summary column/row ('Mean') for plotting
        model_order = self.models[:-1]
        n_models = len(model_order)
        n_combos = len(self.combinations)

        # Prepare data for grouped bar chart
        x = []
        tight_means = []
        caution_means = []
        tops_caution = []
        colors = []
        combo_labels = []
        model_labels = []

        for i, model in enumerate(model_order):
            for j, combo in enumerate(self.combinations):
                batch = self.batch_list[j]
                batch_tight_means, batch_caution_means = batch.get_mean_accuracy_by_model(model_order)
                tight = batch_tight_means[i]
                caution = batch_caution_means[i]
                tops_caution.append(tight + caution)
                tight_means.append(tight)
                caution_means.append(caution)
                clean = model.split('/')[-1] if '/' in model else model
                model_labels.append(clean)
                combo_labels.append(combo["name"])
                x.append((clean, combo["name"]))
                colors.append(self.model_colors.get(model, "#000000"))


        source_tight = ColumnDataSource(data=dict(
            x=x,
            top=tight_means,
            color=colors,
            model=model_labels,
            combo=combo_labels
        ))
        source_caution = ColumnDataSource(data=dict(
            x=x,
            top=tops_caution,
            bottom=tight_means,
            color=colors,
            caution=caution_means,
            model=model_labels,
            combo=combo_labels
        ))

        p = figure(
            width=width,
            height=height,
            x_range=FactorRange(*x),
            y_range=(0, max(tops_caution) * 1.05 if tops_caution else 1),
            toolbar_location=None,
            title=title
        )

        tight_bars = p.vbar(
            x='x',
            top='top',
            width=0.8,
            color='color',
            alpha=0.9,
            source=source_tight,
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
                ('Condition', '@combo'),
                ('Tight (±15%)', '@top{0.0%}')
            ],
            renderers=[tight_bars]
        )
        hover_caution = HoverTool(
            tooltips=[
                ('Model', '@model'),
                ('Condition', '@combo'),
                ('Caution', '@caution{0.0%}')
            ],
            renderers=[caution_bars]
        )
        p.add_tools(hover_tight)
        p.add_tools(hover_caution)

        p.xaxis.major_label_orientation = 1.0
        p.xaxis.visible = False
        p.xgrid.visible = False

        # Add custom bar labels (condition labels for each bar)
        bar_labels = LabelSet(
            x='x',
            y=0,
            text='combo',  # Show the condition name below each bar
            level='glyph',
            x_offset=0,
            y_offset=-15,
            source=source_tight,
            text_font_size="8pt",
            angle=1.0
        )
        p.add_layout(bar_labels)


        for i, model in enumerate(model_order):
            # Use the first combo for each model group
            x_val = (model.split('/')[-1] if '/' in model else model, self.combinations[0]["name"])
            group_label = Label(
                x=x_val,
                y=-0.08,  # Adjust as needed
                text=model.split('/')[-1] if '/' in model else model,
                x_offset=0,
                y_offset=-30,
                text_font_size="10pt",
                text_align="center"
            )
            p.add_layout(group_label)

        return p