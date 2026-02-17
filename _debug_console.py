import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    for name in ["explain_units", "cross_experiment_bars", "full_treatment"]:
        html = f"figures/{name}.html"
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            
            errors = []
            page.on("console", lambda msg: errors.append(f"[{msg.type}] {msg.text}"))
            page.on("pageerror", lambda err: errors.append(f"[PAGE ERROR] {err}"))
            
            await page.goto(f"file://{os.path.abspath(html)}")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(10000)
            
            print(f"\n=== {name} ===")
            for e in errors:
                print(f"  {e}")
            if not errors:
                print("  (no console messages)")
            
            await browser.close()

asyncio.run(main())
