import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import HoverTool, Label, ColumnDataSource, Legend
from visualization_config import get_model_order, assign_model_colors, EPSILON_FOR_WEIGHT, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER, BROAD_RANGE_LOWER, BROAD_RANGE_UPPER

class Visualizer:
    MAX_WEIGHT = 1.0 / EPSILON_FOR_WEIGHT

    def __init__(self, model_data, consensus_data, prompt_conditions, from_unit=None, to_unit=None):
        self.prompt_conditions = prompt_conditions
        self.model_data = model_data
        self.consensus_data = consensus_data
        self.model_order = get_model_order(self.model_data['model'].unique().tolist())
        self.model_colors = assign_model_colors(self.model_order)
        self.from_unit = from_unit
        self.to_unit = to_unit

    def _calculate_alpha(self, weight: float) -> float:
        max_weight = 10.0
        min_alpha = 0.05
        max_alpha = 0.9
        alpha = min_alpha + (min(weight, max_weight) / max_weight) * (max_alpha - min_alpha)
        return alpha

    def _p_to_stars(self, p: float) -> str:
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

    def _create_legend(self, height, legend_items):
        legend_fig = figure(
            width=400,
            height=height,
            toolbar_location=None,
            outline_line_color=None,
        )
        legend_fig.xaxis.visible = False
        legend_fig.yaxis.visible = False
        legend_fig.xgrid.visible = False
        legend_fig.ygrid.visible = False

        legend_items_with_glyphs = []
        for model_name, renderers in legend_items:
            # Find full model name from cleaned name
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

        legend = Legend(
            items=legend_items_with_glyphs,
            location="top_left",
            orientation="vertical",
            spacing=3,
            label_text_font_size="9pt",
            label_text_color="black",
            click_policy="hide",
            ncols=2,
        )
        legend_fig.add_layout(legend, "center")
        return legend_fig

    def _create_weight_bar(self, height, legend_items):
        p = figure(
            width=25,
            height=height,
            toolbar_location=None,
            outline_line_color=None,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        models_for_bar = []
        weights_for_bar = []
        colors_for_bar = []

        # Try to use normalized_weight if present, else fallback to weight
        weight_col = 'normalized_weight' if 'normalized_weight' in self.model_data.columns else 'weight'

        for model_name_clean, _ in legend_items:
            full_model = None
            for m in self.model_order:
                clean = m.split('/')[-1] if '/' in m else m
                if clean == model_name_clean:
                    full_model = m
                    break
            if full_model is None:
                continue
            model_rows = self.model_data[self.model_data['model'] == full_model]
            if len(model_rows) == 0:
                continue
            norm_w = float(model_rows[weight_col].iloc[0])
            if not np.isfinite(norm_w) or norm_w <= 0:
                continue
            color = self.model_colors.get(full_model, '#000000')
            models_for_bar.append(model_name_clean)
            weights_for_bar.append(norm_w)
            colors_for_bar.append(color)

        if not models_for_bar:
            return p
        total = float(sum(weights_for_bar))
        if total <= 0:
            return p
        weights_for_bar = [w / total for w in weights_for_bar]
        models_for_bar = models_for_bar[::-1]
        weights_for_bar = weights_for_bar[::-1]
        colors_for_bar = colors_for_bar[::-1]
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

    def _create_weight_raw_bar(self, height, legend_items):
        model_names_ordered = []
        raw_weights = []
        colors = []
        alphas = []
        for model_name_clean, _ in legend_items:
            # Find full model name from cleaned name
            full_model = None
            for m in self.model_order:
                clean = m.split('/')[-1] if '/' in m else m
                if clean == model_name_clean:
                    full_model = m
                    break
            if full_model is None:
                continue
            model_rows = self.model_data[self.model_data["model"] == full_model]
            if len(model_rows) == 0:
                continue
            w_raw = float(model_rows["weight"].iloc[0])
            if not np.isfinite(w_raw):
                continue
            model_names_ordered.append(model_name_clean)
            raw_weights.append(w_raw)
            color = self.model_colors.get(full_model, "#000000")
            colors.append(color)
            alpha = self._calculate_alpha(w_raw)
            alphas.append(alpha)
        if not model_names_ordered:
            p = figure(
                width=200,
                height=height,
                toolbar_location=None,
                x_range=[],
                y_range=(0, self.MAX_WEIGHT),
                outline_line_color=None,
            )
            p.xaxis.visible = False
            p.yaxis.visible = False
            p.grid.visible = False
            return p
        p = figure(
            width=200,
            height=height,
            x_range=model_names_ordered,
            y_range=(0, self.MAX_WEIGHT),
            toolbar_location=None,
            title="Raw weight by model",
        )
        p.xaxis.visible = False
        p.xgrid.visible = False
        source = ColumnDataSource(
            data=dict(
                model=model_names_ordered,
                weight_raw=raw_weights,
                color=colors,
                alpha=alphas,
            )
        )
        p.vbar(
            x="model",
            top="weight_raw",
            bottom=0,
            width=0.8,
            color="color",
            alpha="alpha",
            line_color="color",
            line_alpha="alpha",
            source=source,
        )
        hover = HoverTool(
            tooltips=[
                ("Model", "@model"),
                ("Raw weight", "@weight_raw{0.00}"),
            ]
        )
        p.add_tools(hover)
        return p
    
    def _create_model_accuracy_subplot(self, height, legend_items):
        model_names_ordered = []
        tight_props = []
        caution_props = []
        colors_list = []
        # Use self.raw_data if available, else fallback to self.model_data
        data_source = getattr(self, "raw_data", self.model_data)
        for model_name_clean, _ in legend_items:
            # Find full model name
            full_model = None
            for m in self.model_order:
                clean = m.split('/')[-1] if '/' in m else m
                if clean == model_name_clean:
                    full_model = m
                    break
            if full_model is None:
                continue
            model_data = data_source[data_source['model'] == full_model]
            if len(model_data) == 0:
                continue
            ratios = model_data['ratio_observed_expected'].values
            ratios = ratios[~np.isnan(ratios)]
            if len(ratios) == 0:
                continue
            tight = np.sum((ratios >= TIGHT_RANGE_LOWER) & (ratios <= TIGHT_RANGE_UPPER)) / len(ratios)
            broad = np.sum((ratios >= BROAD_RANGE_LOWER) & (ratios <= BROAD_RANGE_UPPER)) / len(ratios)
            caution = broad - tight
            model_names_ordered.append(model_name_clean)
            tight_props.append(tight)
            caution_props.append(caution)
            colors_list.append(self.model_colors.get(full_model, '#000000'))
        if not model_names_ordered:
            p = figure(
                width=200,
                height=height,
                toolbar_location=None,
                outline_line_color=None
            )
            p.xaxis.visible = False
            p.yaxis.visible = False
            return p
        p = figure(
            width=200,
            height=height,
            x_range=model_names_ordered,
            y_range=(0, 1),
            toolbar_location=None,
            title="Accuracy Ranges"
        )
        source_tight = ColumnDataSource(data=dict(
            x=model_names_ordered,
            top=tight_props,
            color=colors_list,
            model=model_names_ordered
        ))
        tops_caution = [t + c for t, c in zip(tight_props, caution_props)]
        source_caution = ColumnDataSource(data=dict(
            x=model_names_ordered,
            top=tops_caution,
            bottom=tight_props,
            color=colors_list,
            caution=caution_props,
            model=model_names_ordered
        ))
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
        p.add_tools(hover_tight)
        hover_caution = HoverTool(
            tooltips=[
                ('Model', '@model'),
                ('Caution', '@caution{0.0%}')
            ],
            renderers=[caution_bars]
        )
        p.add_tools(hover_caution)
        p.xaxis.visible = False
        p.xgrid.grid_line_color = None
        p.yaxis.axis_label = "Proportion"
        return p
    
    def _create_consensus_plot(self, width, height):
        plot_type = getattr(self, "consensus_plot_type", "dose")
        if plot_type == "coefficient":
            y_label = ""  # Hide label for alignment
            y_value = self.consensus_data.get('consensus_coefficient')
            se = self.consensus_data.get('se_consensus_log')
            log_value = self.consensus_data.get('consensus_log_coefficient')
            literature_coeff = getattr(self, "literature_coeff", None)
        else:
            y_label = f"Dose ({self.to_unit})"
            y_value = self.consensus_data.get('consensus_dose')
            se = self.consensus_data.get('se_consensus_log')
            log_value = self.consensus_data.get('consensus_log_dose')
            literature_coeff = None

        p = figure(
            width=width,
            height=height,
            title="Consensus",
            x_axis_label="",
            y_axis_label=y_label,
            y_axis_type="log",
            toolbar_location=None
        )

        # Hide x-axis and x-grid
        p.xaxis.visible = False
        p.xgrid.grid_line_color = None
        p.x_range.start = 0
        p.x_range.end = 1

        # Hide y-axis labels (but keep axis for alignment)
        p.yaxis.major_label_text_font_size = '0pt'

        # Set y-ticks if you have a method for it
        if hasattr(self, "_get_log_ticks"):
            p.yaxis.ticker = self._get_log_ticks(include_literature=True)

        # Reference line at literature coefficient
        if literature_coeff is not None:
            p.line([0, 1], [literature_coeff, literature_coeff],
                line_dash='dashed', line_color='gray', line_width=2)

        if y_value is not None and log_value is not None:
            consensus_raw = float(np.exp(log_value)) if plot_type == "coefficient" else y_value
            if pd.notna(se) and se > 0:
                lower_raw = float(np.exp(log_value - se)) if plot_type == "coefficient" else consensus_raw - se
                upper_raw = float(np.exp(log_value + se)) if plot_type == "coefficient" else consensus_raw + se
            else:
                lower_raw = upper_raw = consensus_raw

            # Plot consensus point
            p.scatter([0.5], [consensus_raw], size=10, color='black', alpha=0.8, marker='circle')

            # Error bars
            if lower_raw != upper_raw:
                p.multi_line([[0.5, 0.5]], [[lower_raw, upper_raw]], color='black', line_width=2)

            # Significance stars
            p_value = self.consensus_data.get('p_value_log')
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

        return p