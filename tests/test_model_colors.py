"""Model colors are a fixed map: they follow the model, never its rank."""
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


def test_fixed_map_equals_rainbow_from_all_standards_rankings():
    """The frozen map is the rank-based rainbow computed from results_all/
    (decision 2026-11-14); it must not be recomputed from another score."""
    code = ("import json, visualize.visualization_config as v; "
            "print(json.dumps(v._build_ranked_color_dict()))")
    env = dict(os.environ, RESULTS_DIR="results_all")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    import json
    assert json.loads(out.strip().splitlines()[-1]) == _colors_under("results_nielsen")


def test_global_order_follows_the_color_map_and_results_all_ranking():
    """Model order = color-map order = average adjusted rank in results_all/,
    whatever RESULTS_DIR is, so the color gradient runs in sequence."""
    code = ("import json, visualize.visualization_config as v; "
            "r = v._collect_ranking_ranks(); "
            "a = {m: sum(x) / len(x) for m, x in r.items() if m in v.MODEL_COLOR_DICT}; "
            "print(json.dumps([v.get_ranked_model_order(), list(v.MODEL_COLOR_DICT), "
            "sorted(a, key=lambda m: a[m])]))")
    import json
    for d in ("results_all", "results_nielsen"):
        env = dict(os.environ, RESULTS_DIR=d)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout
        order, colors, computed = json.loads(out.strip().splitlines()[-1])
        assert order == colors
        if d == "results_all":
            assert order == computed
