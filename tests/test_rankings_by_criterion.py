"""Rankings by criterion (analysis/revision/rankings_by_criterion.py): hand-checked cases."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.revision import rankings_by_criterion as R  # noqa: E402


def test_tiers_follow_the_chain_rule():
    order = ["a", "b", "c", "d"]
    sig = {(0, 2), (0, 3), (1, 3)}                 # a>c, a>d, b>d
    t = R.tiers_from(order, lambda i, j: (i, j) in sig)
    assert t == {"a": 1, "b": 1, "c": 2, "d": 2}   # d: deepest superior tier (b: 1) + 1 = 2, a is tier 1 too
    sig |= {(2, 3)}
    assert R.tiers_from(order, lambda i, j: (i, j) in sig)["d"] == 3


def test_friedman_mean_rank_ties_and_tiers():
    # 6 items x 3 models: a best in every item, b and c identical
    w = pd.DataFrame({"a": [0.1] * 6, "b": [0.5] * 6, "c": [0.5] * 6})
    f = R.friedman_tiers(w)
    assert f["mean_rank"] == {"a": 1.0, "b": 2.5, "c": 2.5}      # ties get the average rank
    cd = R.nemenyi_cd(3, 6)
    assert np.isclose(cd, 3.314 / np.sqrt(2) * np.sqrt(3 * 4 / (6 * 6)), atol=2e-3)   # q(0.05; 3, inf) = 3.314
    assert f["tiers"]["b"] == f["tiers"]["c"]                     # equal mean ranks stay tied
    assert f["tiers"]["a"] == 1 and f["tiers"]["b"] == 2          # 1.5 > CD (about 0.96)
    assert f["n"] == 6


def test_finish_orders_and_ties_and_correlations():
    rows = pd.DataFrame({"score": "s", "condition": "c", "criterion": ["median_fold_error"] * 3 + ["broad"] * 3,
                         "model": list("abc") * 2, "value": [1.0, 2.5, 2.5, 90.0, 80.0, 80.0], "tier": [1, 2, 2, 1, 2, 2],
                         "n_items": 5, "n_items_used": 5, "n_responses": 10, "tier_rule": "x",
                         "median_fold_error_pooled": 1.0})
    f = R.finish(rows)
    mf = f[f.criterion == "median_fold_error"].set_index("model")
    assert list(mf["order"]) == [1, 2, 2] and list(mf["tied"]) == [False, True, True]
    br = f[f.criterion == "broad"].set_index("model")
    assert list(br["order"]) == [1, 2, 2]
    a = R._oriented(f[f.criterion == "median_fold_error"])
    b = R._oriented(f[f.criterion == "broad"])
    c = R._corr(a, b)
    assert np.isclose(c["spearman_value"], 1.0) and np.isclose(c["kendall_tau_b_tiers"], 1.0)   # mean rank enters negated
    flat = b.assign(tier=1)
    assert np.isnan(R._corr(flat, b)["kendall_tau_b_tiers"])


def test_build_outputs_structure_and_determinism():
    r1, c1 = R.build()
    r2, c2 = R.build()
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(c1, c2)
    assert set(r1["criterion"]) == set(R.CRITERIA) and set(r1["score"]) == {"nielsen", "all", "non_nielsen", "geomean"}
    assert (r1.groupby(["score", "condition", "criterion"]).size() == 13).all()
    top = r1[r1["order"] == 1]
    assert (top.groupby(["score", "condition", "criterion"])["tier"].min() == 1).all()
    assert c1["spearman_value"].dropna().between(-1, 1).all() and c1["kendall_tau_b_tiers"].dropna().between(-1, 1).all()
    # broad ranking equals the CMH ranking of stats.rank_models on the one band
    from analysis.revision import stats
    res = stats.rank_models(stats.load_long(), "nielsen", "rd_naive", bands=("broad",)).result_df.set_index("model")
    g = r1[(r1.score == "nielsen") & (r1.condition == "rd_naive") & (r1.criterion == "broad")].set_index("model")
    assert (g.loc[res.index, "tier"] == res["adjusted_rank"]).all()
