"""
Regenerate all heatmaps (random dose + vignette) and export each subplot
as an individual PNG.
"""
import asyncio
from analysis.batch import OpioidPairBatch, VignetteBatch
from visualize.opioid_pair_batch_heatmap_visualizer import OpioidPairBatchHeatmapVisualizer
from visualize.visualization_config import convert_to_pdf_auto_size, convert_to_png_auto_size


# ── Random-dose heatmaps ────────────────────────────────────────────
RANDOM_DOSE_COMBOS = [
    {"ome": None,  "include_nielsen": False, "include_examples": False, "name": "all_n0"},
    {"ome": None,  "include_nielsen": True,  "include_examples": False, "name": "all_n1"},
    {"ome": True,  "include_nielsen": False, "include_examples": False, "name": "ome_n0"},
    {"ome": True,  "include_nielsen": True,  "include_examples": False, "name": "ome_n1"},
    {"ome": False, "include_nielsen": False, "include_examples": False, "name": "pairs_n0"},
    {"ome": False, "include_nielsen": True,  "include_examples": False, "name": "pairs_n1"},
]

# ── Vignette heatmaps ───────────────────────────────────────────────
VIGNETTE_COMBOS = [
    {"has_vignette_text": True,  "include_nielsen": False, "name": "vignette_all_n0"},
    {"has_vignette_text": True,  "include_nielsen": True,  "name": "vignette_all_n1"},
    {"has_vignette_text": True,  "include_nielsen": False, "ome": True,  "name": "vignette_ome_n0"},
    {"has_vignette_text": True,  "include_nielsen": True,  "ome": True,  "name": "vignette_ome_n1"},
    {"has_vignette_text": True,  "include_nielsen": False, "ome": False, "name": "vignette_pairs_n0"},
    {"has_vignette_text": True,  "include_nielsen": True,  "ome": False, "name": "vignette_pairs_n1"},
]


async def build_heatmaps(batch, name: str):
    """Build heatmaps, save combined PDF + individual subplot PNGs."""
    vis = OpioidPairBatchHeatmapVisualizer(
        batch, cell_size=35,
        width_padding=0, height_padding=0,
        cell_text_size="6pt", label_text_size="7pt",
    )
    output_html = f"figures/{name}_heatmaps.html"
    vis.create_heatmaps(output_html=output_html)
    print(f"✓ Saved HTML: {output_html}")

    output_pdf = f"figures/{name}_heatmaps.pdf"
    await convert_to_pdf_auto_size(output_html, output_pdf, scale=3.0)
    print(f"✓ Saved PDF: {output_pdf}")

    # Save each subplot as individual HTML → PNG
    subplot_htmls = vis.save_subplot_pngs(f"figures/{name}")
    for subplot_name, html_path in subplot_htmls.items():
        png_path = html_path.replace(".html", ".png")
        await convert_to_png_auto_size(html_path, png_path, scale=3.0)
    print(f"✓ Saved {len(subplot_htmls)} subplot PNGs for {name}")


async def main():
    # ── Random dose ──
    for combo in RANDOM_DOSE_COMBOS:
        name = combo.pop("name")
        filter_spec = {"exclude_self_pairs": True}
        for k, v in combo.items():
            if v is not None:
                filter_spec[k] = v
        try:
            batch = OpioidPairBatch.from_filter_specs(
                experiment_results_csv="results/experiment_results_by_ome.csv",
                variable_model_csv="results/pair_model_analysis.csv",
                consensus_csv="results/opioid_pair_consensus.csv",
                filter_specs=[filter_spec],
            )
            await build_heatmaps(batch, name)
        except Exception as e:
            print(f"✗ Error ({name}): {e}")
            import traceback; traceback.print_exc()

    # ── Vignettes ──
    for combo in VIGNETTE_COMBOS:
        name = combo.pop("name")
        filter_spec = {k: v for k, v in combo.items()}
        try:
            batch = VignetteBatch.from_filter_specs(
                experiment_results_csv="results/reworded_vignettes_results_by_ome.csv",
                variable_model_csv="results/reworded_vignettes_model_analysis.csv",
                consensus_csv="results/reworded_vignettes_model_consensus.csv",
                filter_specs=[filter_spec],
            )
            await build_heatmaps(batch, name)
        except Exception as e:
            print(f"✗ Error ({name}): {e}")
            import traceback; traceback.print_exc()

    print("\n=== ALL HEATMAPS DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
