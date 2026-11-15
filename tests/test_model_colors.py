"""Model colours are a fixed map: they follow the model, never its rank."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.models import CONSISTENT_MODELS  # noqa: E402


def _colors_under(results_dir):
    code = ("import json, visualize.visualization_config as v; "
            "print(json.dumps(v.MODEL_COLOR_DICT))")
    env = dict(os.environ, RESULTS_DIR=results_dir)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    import json
    return json.loads(out.strip().splitlines()[-1])


def test_fixed_map_covers_models_with_distinct_colors():
    from visualize.visualization_config import MODEL_COLOR_DICT
    assert set(MODEL_COLOR_DICT) == set(CONSISTENT_MODELS)
    assert len(set(MODEL_COLOR_DICT.values())) == len(MODEL_COLOR_DICT)


def test_colors_do_not_depend_on_results_dir_or_ranking():
    a = _colors_under("results")
    b = _colors_under("results_nielsen")
    c = _colors_under("results_does_not_exist")
    assert a == b == c
    # published colours (summary_w_figures/pub_figures)
    assert a["qwen/qwen-2.5-72b-instruct"] == "#40bfe5"
    assert a["mistralai/mixtral-8x22b-instruct"] == "#4073e5"
