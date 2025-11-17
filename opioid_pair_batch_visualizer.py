from typing import Optional

from bokeh.layouts import column
from bokeh.io import output_file, save, show

from opioid_pair_visualizer import OpioidPairVisualizer
from opioid_pair_batch import OpioidPairBatch, PairPromptKey


class OpioidPairBatchVisualizer:
    """
    Builds a tall, scrollable grid of OpioidPairVisualizer layouts,
    one row per PairPromptKey in an OpioidPairBatch.
    """

    def __init__(
        self,
        batch: OpioidPairBatch,
        row_width: int = 900,
        row_height: int = 300,
    ):
        """
        Args:
            batch: An OpioidPairBatch instance defining which rows to visualize.
            row_width: Total width (pixels) for each row (passed to OpioidPairVisualizer).
            row_height: Height (pixels) for each per-pair visualization.
        """
        self.batch = batch
        self.row_width = row_width
        self.row_height = row_height

    def create_grid(
        self,
        output_html: Optional[str] = None,
        show_inline: bool = True,
    ):
        """
        Create a vertical stack (grid) of per-pair visualizations.

        The browser window will be scrollable if this is tall, which is what we want.

        Args:
            output_html: If provided, write the layout to this HTML file.
            show_inline: If True, call bokeh.io.show(layout).

        Returns:
            A Bokeh layout (column of rows).
        """
        rows = []

        for key in self.batch.keys:
            prompt_conditions = {
                "include_nielson": key.include_nielson,
                "include_examples": key.include_examples,
            }

            try:
                vis = OpioidPairVisualizer(
                    experiment_results_csv=self.batch.experiment_results_csv,
                    pair_model_csv=self.batch.pair_model_csv,
                    consensus_csv=self.batch.consensus_csv,
                    from_opioid=key.from_opioid,
                    to_opioid=key.to_opioid,
                    prompt_conditions=prompt_conditions,
                )

                row_layout = vis.create_plot(
                    width=self.row_width,
                    height=self.row_height,
                )
                rows.append(row_layout)
            except ValueError as e:
                # e.g. if there is some mismatch across CSVs for this key
                print(
                    f"Skipping {key.from_opioid} -> {key.to_opioid} "
                    f"(Nielson={key.include_nielson}, Examples={key.include_examples}): {e}"
                )

        if not rows:
            raise ValueError("No per-pair visualizations could be created for this batch.")

        layout = column(*rows, sizing_mode="stretch_width")

        if output_html:
            output_file(output_html)
            save(layout)

        if show_inline:
            show(layout)

        return layout
