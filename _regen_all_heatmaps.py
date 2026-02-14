"""Regenerate just the all_n0 and all_n1 heatmaps (ome=None)."""
import asyncio
from analysis.batch import OpioidPairBatch
from visualize.opioid_pair_batch_heatmap_visualizer import OpioidPairBatchHeatmapVisualizer
from visualize.visualization_config import convert_to_pdf_auto_size, convert_to_png_auto_size

COMBOS = [
    {"include_nielsen": False, "include_examples": False, "name": "all_n0"},
    {"include_nielsen": True,  "include_examples": False, "name": "all_n1"},
]

async def main():
    for combo in COMBOS:
        name = combo.pop("name")
        filter_spec = {**combo, "exclude_self_pairs": True}
        batch = OpioidPairBatch.from_filter_specs(
            experiment_results_csv="results/experiment_results_by_ome.csv",
            variable_model_csv="results/pair_model_analysis.csv",
            consensus_csv="results/opioid_pair_consensus.csv",
            filter_specs=[filter_spec],
        )
        vis = OpioidPairBatchHeatmapVisualizer(
            batch, cell_size=35,
            width_padding=0, height_padding=0,
            cell_text_size="6pt", label_text_size="7pt",
        )
        output_html = f"figures/{name}_heatmaps.html"
        vis.create_heatmaps(output_html=output_html)
        print(f"✓ HTML: {output_html}")

        output_pdf = f"figures/{name}_heatmaps.pdf"
        await convert_to_pdf_auto_size(output_html, output_pdf, scale=3.0)
        print(f"✓ PDF: {output_pdf}")

        subplot_htmls = vis.save_subplot_pngs(f"figures/{name}")
        for subplot_name, html_path in subplot_htmls.items():
            png_path = html_path.replace(".html", ".png")
            await convert_to_png_auto_size(html_path, png_path, scale=3.0)
        print(f"✓ {len(subplot_htmls)} subplot PNGs for {name}")

asyncio.run(main())
