"""Phase 2 framework: stats reproduce the pipeline under Nielsen; figure
registry, output paths and adapters behave as documented."""
import os
import shutil

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.conditions import CONDITIONS, SUBSETS, select


def _long():
    return stats.load_long()


def test_model_summary_reproduces_rankings_and_batch_means():
    L = _long()
    for cond, f in [("vignette_naive", "ranking_naive_vignettes"),
                    ("rd_naive", "ranking_naive_random_doses"),
                    ("eu_4h_nielsen_units", "ranking_random_doses_nielsen_units")]:
        m = stats.model_summary(L, "nielsen", cond).merge(pd.read_csv(f"results_nielsen/{f}.csv"), on="model")
        assert len(m) == 13
        assert np.allclose(m.pct_tight / 100, m.mean_tight_prop, atol=1e-12)
        assert np.allclose(m.pct_broad / 100, m.mean_broad_prop, atol=1e-12)
    for cond, sub, batch, f in [("vignette_naive", "ome", "ome_n0", "_vignettes"),
                                ("vignette_naive", "nonome", "pairs_n0", "_vignettes"),
                                ("rd_nielsen", "all", "all_n1", "")]:
        bm = pd.read_csv(f"results_nielsen/model_batch_means{f}.csv")
        bm = bm[bm.batch_name == batch]
        m = stats.model_summary(L, "nielsen", cond, sub, average="cell").merge(bm, on="model")
        assert len(m) == 13 and np.allclose(m.pct_tight / 100, m.mean_in_tight, atol=1e-12)


def test_cmh_ranking_and_paired_wilcoxon_reproduce_pipeline():
    L = _long()
    r = stats.rank_models(L, "nielsen", "rd_naive").result_df
    ref = pd.read_csv("results_nielsen/ranking_naive_random_doses.csv")
    m = r.merge(ref, on="model", suffixes=("", "_ref"))
    assert (m.adjusted_rank == m.adjusted_rank_ref).all() and (m.n_total == m.n_total_ref).all()
    pe = stats.paired_effect(L, "nielsen", "rd_naive", "rd_nielsen", outcome="tight")
    w = pd.read_csv("results_nielsen/wilcoxon_nielsen_effect_random_dose.csv")
    m = pe.merge(w, on="model")
    assert len(m) == 14 and (m.n_pairs == m.tight_n).all()
    assert np.allclose(m.p, m.tight_p, rtol=0, atol=1e-9)


def test_threshold_free_outcome_and_bands_for_any_score():
    L = _long()
    pe = stats.paired_effect(L, "non_nielsen", "rd_naive", "rd_nielsen", outcome=stats.THRESHOLD_FREE)
    pooled = pe[pe.model == "POOLED"].iloc[0]
    assert pooled.n_pairs > 0 and pooled.mean_diff > 0   # grounding reduces |log ratio|
    ms = stats.model_summary(L, "all", "vignette_naive", bands=["tight", "reference_envelope"])
    assert {"pct_tight", "pct_reference_envelope"} <= set(ms.columns)
    # `all` is the nearest of a set that includes Nielsen: never worse than Nielsen
    nz = stats.model_summary(L, "nielsen", "vignette_naive").set_index("model")
    assert (ms.set_index("model").median_fold_error <= nz.median_fold_error + 1e-12).all()


def test_conditions_and_subsets():
    L = _long()
    v = select(L, "vignette_naive")
    assert v.has_vignette_text.all() and v.scenario_key.nunique() == 20
    assert select(L, "vignette_naive", "ome").scenario_key.nunique() == 11     # pipeline `ome` flag
    assert select(L, "rd_naive", "matched").opioid_pair.nunique() == 11        # figure 10, all scenarios
    assert select(L, "rd_naive", "matched_same_conversion").opioid_pair.nunique() == 10
    assert select(L, "vignette_empty_naive", "reduction").scenario_key.nunique() == 4
    assert {"all", "ome", "nonome", "matched", "matched_same_conversion", "same_conversion",
            "reduction", "di2", "no_di2"} <= set(SUBSETS)
    assert L.model.nunique() == 13


def test_figure_registry_and_output_paths():
    from analysis.revision.tolerances import DEFAULT_BANDS
    from visualize.revision.compose import engine_for, out_stem
    from visualize.revision.registry import FIGURES, parse_selection
    assert sorted(FIGURES) == list(range(1, 16))
    assert [n for n, f in FIGURES.items() if f.rescorable] == [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    assert parse_selection("1-3,11") == [1, 2, 3, 11]
    f1 = FIGURES[1]
    assert engine_for(f1, ["nielsen"], "bands", DEFAULT_BANDS, False) == "legacy"
    assert engine_for(f1, ["nielsen", "all"], "bands", DEFAULT_BANDS, True) == "grid"
    assert engine_for(f1, ["all"], "distribution", DEFAULT_BANDS, False) == "grid"
    assert engine_for(f1, ["all"], "bands", ["tight", "reference_envelope"], False) == "grid"
    a = out_stem("o", f1, ["nielsen"], "bands", DEFAULT_BANDS)
    b = out_stem("o", f1, ["nielsen", "all"], "distribution", DEFAULT_BANDS)
    c = out_stem("o", f1, ["nielsen"], "bands", ["tight", "reference_envelope"])
    assert len({a, b, c}) == 3 and a.endswith("fig01_naive_vignette_bars__nielsen__bands")


def test_precision_adapter_is_identity_under_nielsen_and_cv_matches():
    from visualize.revision.compose import adapt_inputs
    from visualize.revision.panels import precision_cells
    L = _long()
    d = adapt_inputs("nielsen", L)
    try:
        for f in ["reworded_vignettes_model_analysis.csv", "pair_model_analysis.csv"]:
            a = pd.read_csv(f"results_nielsen/{f}")
            b = pd.read_csv(os.path.join(d, f))
            assert a.shape == b.shape
            for c in a.columns:
                if a[c].dtype.kind in "fc":
                    assert np.allclose(a[c], b[c], rtol=1e-12, equal_nan=True), (f, c)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    p = pd.read_csv("results_nielsen/pair_model_analysis.csv")
    m = precision_cells(L, "nielsen", "rd_naive").merge(
        p[(p.include_nielsen == False) & (p.include_examples == False)], on=["model", "opioid_pair"])  # noqa: E712
    assert np.allclose(m.cv, m.observed_coefficient_cv, atol=1e-12, equal_nan=True)
