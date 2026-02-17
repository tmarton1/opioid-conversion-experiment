import asyncio
from visualize.visualization_config import convert_to_png_auto_size, convert_to_pdf_auto_size

async def main():
    for name in ["explain_units", "full_treatment", "cross_experiment_bars"]:
        html = f"figures/{name}.html"
        png = f"figures/{name}_test.png"
        pdf = f"figures/{name}_test.pdf"
        try:
            await convert_to_png_auto_size(html, png, scale=2.0)
        except Exception as e:
            print(f"PNG FAIL {name}: {e}")
        try:
            await convert_to_pdf_auto_size(html, pdf, scale=2.0)
        except Exception as e:
            print(f"PDF FAIL {name}: {e}")

asyncio.run(main())
