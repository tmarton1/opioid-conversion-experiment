"""Regenerate all figures and the annotated PDF."""
import asyncio
from visualize.comprehensive_figures import ComprehensiveFigureBuilder

async def main():
    fb = ComprehensiveFigureBuilder('results')
    await fb.save_all()
    await fb.save_annotated_pdf()
    print('=== ALL DONE ===')

asyncio.run(main())
