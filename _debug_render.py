import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    for name in ["explain_units", "cross_experiment_bars", "nielsen_effect_vignettes"]:
        html = f"figures/{name}.html"
        png = f"figures/{name}_debug.png"
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page(device_scale_factor=2)
            await page.goto(f"file://{os.path.abspath(html)}")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(8000)
            
            dims = await page.evaluate("""
                () => ({
                    width: document.documentElement.scrollWidth,
                    height: document.documentElement.scrollHeight,
                    bodyHTML: document.body.innerHTML.substring(0, 500),
                    scripts: document.querySelectorAll('script').length,
                    canvases: document.querySelectorAll('canvas').length,
                })
            """)
            print(f"\n{name}:")
            print(f"  dims: {dims['width']}x{dims['height']}")
            print(f"  scripts: {dims['scripts']}, canvases: {dims['canvases']}")
            print(f"  body preview: {dims['bodyHTML'][:200]}")
            
            await page.set_viewport_size({"width": dims['width'] + 20, "height": dims['height'] + 20})
            await page.wait_for_timeout(2000)
            await page.screenshot(path=png, full_page=True)
            size = os.path.getsize(png)
            print(f"  PNG size: {size} bytes")
            
            await browser.close()

asyncio.run(main())
