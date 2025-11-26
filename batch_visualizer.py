from opioid_pair_visualizer import OpioidPairVisualizer
from vignette_visualizer import VignetteVisualizer
from bokeh.layouts import column
from bokeh.io import output_file, save, show

from opioid_pair_visualizer import OpioidPairVisualizer
from opioid_pair_batch import OpioidPairBatch
from typing import Optional

class BatchVisualizer:
    def __init__(self, batch, row_width=600, row_height=300, VisualizerClass=None, model_csv_attr=None):
        self.batch = batch
        self.VisualizerClass = VisualizerClass
        self.model_csv_attr = model_csv_attr
        self.row_width = row_width
        self.row_height = row_height

    def create_grid(
        self,
        output_html: Optional[str] = None,
        show_inline: bool = True,
    ):
        rows = []

        for key in self.batch.keys:
            try:
                prompt_conditions = {
                    "include_nielson": getattr(key, "include_nielson", None),
                    "include_examples": getattr(key, "include_examples", None),
                }
                # Prepare kwargs for the visualizer
                if not isinstance(self.model_csv_attr, str):
                    raise TypeError(f"model_csv_attr must be a string, got {type(self.model_csv_attr)}: {self.model_csv_attr}")
                model_csv_value = getattr(self.batch, self.model_csv_attr, None)
                vis_kwargs = dict(
                    experiment_results_csv=getattr(self.batch, "experiment_results_csv", None),
                    prompt_conditions=prompt_conditions,
                    consensus_csv=getattr(self.batch, "consensus_csv", None),
                )
                # Add key-specific fields
                vis_kwargs[self.model_csv_attr] = model_csv_value

                # Only add expected key fields based on the visualizer class
                if self.VisualizerClass is OpioidPairVisualizer:
                    if hasattr(key, "from_opioid"):
                        vis_kwargs["from_opioid"] = key.from_opioid
                    if hasattr(key, "from_opioid_unit"):
                        vis_kwargs["from_opioid_unit"] = key.from_opioid_unit
                    if hasattr(key, "to_opioid"):
                        vis_kwargs["to_opioid"] = key.to_opioid
                    if hasattr(key, "to_opioid_unit"):
                        vis_kwargs["to_opioid_unit"] = key.to_opioid_unit

                elif self.VisualizerClass is VignetteVisualizer:
                    if hasattr(key, "vignette_id"):
                        vis_kwargs["vignette_id"] = key.vignette_id

                try:
                    vis = self.VisualizerClass(**vis_kwargs)
                    row_layout = vis.create_plot(
                        width=self.row_width,
                        height=self.row_height,
                    )
                    rows.append(row_layout)
                except ValueError as e:
                    print(f"Skipping {key}: {e}")
            
            except Exception as e:
                print(f"Error for key {key}: {e}")

        if not rows:
            raise ValueError("No visualizations could be created for this batch.")

        layout = column(*rows, sizing_mode="stretch_width")

        if output_html:
            output_file(output_html)
            save(layout)

        if show_inline:
            show(layout)

        return layout

class OpioidPairBatchVisualizer(BatchVisualizer):
    def __init__(self, batch, pair_model_csv="pair_model_csv", row_width=600, row_height=300):
        super().__init__(
            batch,
            row_width=row_width,
            row_height=row_height,
            VisualizerClass=OpioidPairVisualizer,
            model_csv_attr=pair_model_csv
        )

class VignetteBatchVisualizer(BatchVisualizer):
    def __init__(self, batch, vignette_model_csv="vignette_model_csv", row_width=600, row_height=300):
        super().__init__(
            batch,
            row_width=row_width,
            row_height=row_height,
            VisualizerClass=VignetteVisualizer,
            model_csv_attr=vignette_model_csv
        )