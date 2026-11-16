"""stats.rank_models: one band, freedom from a tail, and the unchanged two-band form."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.model_ranking import ModelRanking  # noqa: E402
from analysis.revision import stats  # noqa: E402
from common.models import CONSISTENT_MODELS  # noqa: E402


def _toy():
    """3 models x 4 units x 10 responses; shares 1.0, 0.5, 0.0 on one column."""
    rows = [dict(model=m, unit=u, in_x=p, _dup=p, n_observations=10)
            for m, p in zip(CONSISTENT_MODELS[:3], [1.0, 0.5, 0.0]) for u in "wxyz"]
    r = ModelRanking(None, {}, "unit", df=pd.DataFrame(rows), range_cols=("in_x", "_dup"))
    r.build()
    return r


def test_single_band_bonferroni_counts_pairs_once():
    r = _toy()
    p = r.pairs_df["tight_p"].astype(float).to_numpy()
    assert np.allclose(r.pairs_df["tight_p_bonf"], np.minimum(1, p * 2 * len(p)))   # duplicated column: 2 x pairs
    stats._single_band_bonferroni(r)
    assert len(p) == 3
    assert np.allclose(r.pairs_df["tight_p_bonf"], np.minimum(1, p * 3))          # one band: 3 pairs
    assert list(r.result_df["adjusted_rank"]) == [1, 2, 3]
    assert r.n_total_tests == 3


def test_single_band_matches_mean_share_and_rank_rules():
    L = stats.load_long()
    r = stats.rank_models(L, "nielsen", "rd_naive", bands=("broad",))
    res = r.result_df
    assert r.bands == ("broad",) and r.criterion == "broad" and len(res) == 13
    ref = stats.model_summary(L, "nielsen", "rd_naive", bands=["broad"]).set_index("model")["pct_broad"]
    assert np.allclose(res["mean_broad_prop"].to_numpy() * 100, ref.loc[res["model"]].to_numpy())
    assert (res["mean_tight_prop"] == res["mean_broad_prop"]).all()
    assert res["adjusted_rank"].iloc[0] == 1 and (res["adjusted_rank"] >= 1).all()
    pr = r.pairs_df
    n = int(pr["tight_p"].notna().sum())
    assert np.allclose(pr["tight_p_bonf"], np.minimum(1, pr["tight_p"] * n))
    assert set(res["model"]) == set(CONSISTENT_MODELS)


def test_freedom_from_tail_ranks_on_the_complement():
    L = stats.load_long()
    r = stats.rank_models(L, "nielsen", "rd_naive", bands=("overdose",))
    res = r.result_df.set_index("model")
    assert r.criterion == "free_overdose"
    over = stats.model_summary(L, "nielsen", "rd_naive", bands=["overdose"]).set_index("model")["pct_overdose"]
    assert np.allclose(res["mean_broad_prop"] * 100, 100 - over.loc[res.index].to_numpy())
    # the model with the lowest overdose share is ranked first
    assert res.index[0] == over.idxmin()
    # a tail passed with an accuracy band: both enter as shares inside (complement for the tail)
    r2 = stats.rank_models(L, "nielsen", "rd_naive", bands=("broad", "overdose"))
    assert len(r2.result_df) == 13 and r2.criterion == "broad+free_overdose"


def test_two_band_form_unchanged_against_stored_rankings():
    L = stats.load_long()
    for cond, f in [("rd_naive", "ranking_naive_random_doses"), ("vignette_naive", "ranking_naive_vignettes"),
                    ("rd_nielsen", "ranking_random_doses_nielsen")]:
        r = stats.rank_models(L, "nielsen", cond)
        ref = pd.read_csv(f"results_nielsen/{f}.csv")
        assert list(r.result_df["model"]) == list(ref["model"])
        assert (r.result_df["adjusted_rank"].to_numpy() == ref["adjusted_rank"].to_numpy()).all()
        assert np.allclose(r.result_df["mean_tight_prop"], ref["mean_tight_prop"])
        assert np.allclose(r.result_df["mean_broad_prop"], ref["mean_broad_prop"])
        pairs = pd.read_csv(f"results_nielsen/{f}_pairs.csv")
        assert np.allclose(r.pairs_df["broad_p_bonf"].astype(float), pairs["broad_p_bonf"].astype(float), equal_nan=True)
