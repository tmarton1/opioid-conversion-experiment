"""Task 0.3: output directories come from common.config and follow env vars."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r"""
import inspect, json
import orchestration.run_analysis as ra
import orchestration.extract_figure_stats as efs
import orchestration.generate_raw_figures as grf
import analysis.statistical_methods as sm
from visualize.comprehensive_figures import ComprehensiveFigureBuilder
sig = inspect.signature(ComprehensiveFigureBuilder.__init__).parameters
print(json.dumps({
    "pair_csv": ra.PAIR_CSV,
    "stats_json": efs.OUTPUT_PATH,
    "efs_results": efs.RESULTS_DIR,
    "raw_figs": grf.RAW_FIGURES_DIR,
    "sm_results": sm.RESULTS_DIR.name,
    "sm_figs": sm.FIGURES_DIR.name,
    "cfb_results": sig["results_dir"].default,
    "cfb_figs": sig["figures_dir"].default,
}))
"""


def _probe(env_overrides):
    env = {k: v for k, v in os.environ.items() if k not in ("RESULTS_DIR", "FIGURES_DIR", "RAW_FIGURES_DIR")}
    env.update(env_overrides)
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_defaults_match_original_layout():
    d = _probe({})
    assert d == {
        "pair_csv": "results/pair_model_analysis.csv",
        "stats_json": "summary_w_figures/figure_stats.json",
        "efs_results": "results",
        "raw_figs": "raw_figures",
        "sm_results": "results",
        "sm_figs": "raw_figures",
        "cfb_results": "results",
        "cfb_figs": "summary_w_figures/pub_figures",
    }


def test_env_vars_redirect_every_output():
    d = _probe({"RESULTS_DIR": "results_x", "FIGURES_DIR": "figs_x", "RAW_FIGURES_DIR": "raw_x"})
    assert d == {
        "pair_csv": "results_x/pair_model_analysis.csv",
        "stats_json": "figs_x/figure_stats.json",
        "efs_results": "results_x",
        "raw_figs": "raw_x",
        "sm_results": "results_x",
        "sm_figs": "raw_x",
        "cfb_results": "results_x",
        "cfb_figs": "figs_x/pub_figures",
    }
