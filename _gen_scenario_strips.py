#!/usr/bin/env python3
"""Generate all scenario strip plots + combined PDF."""
import asyncio
from visualize.scenario_strip_visualizer import save_all_scenario_strips

asyncio.run(save_all_scenario_strips(
    filters={"include_nielsen": False, "explain_units": False},
))
