"""Bonferroni within each family (analysis/revision/multiple_comparisons.py): arithmetic, family sizes, sources."""
import numpy as np
import pandas as pd

RR = "results_revision"


def _tests():
    return pd.read_csv(f"{RR}/multiple_comparisons_tests.csv")


def test_bonferroni_arithmetic_within_family():
    t = _tests()
    v = t[t.p.notna() & (t.family_scope == "all tests listed")]
    m = v.groupby("family").p.transform("size")
    assert (v.m == m).all()
    assert np.allclose(v.p_bonf, np.minimum(1.0, v.p * v.m))
    assert (v.sig_bonf <= v.sig_unadj).all()          # Bonferroni never adds a significant test


def test_group_families_use_the_group_size():
    t = _tests()
    g = t[(t.family_scope == "within group") & t.p.notna()]
    size = g.groupby(["family", "group"]).p.transform("size")
    assert (g.m == size).all()


def test_pooled_rows_are_not_listed_and_only_13_models():
    from common.models import CONSISTENT_MODELS
    t = _tests()
    units = set(t.unit.astype(str))
    assert not {"POOLED", "ALL_MODELS", "OVERALL"} & units
    pm = t[t.unit.isin(CONSISTENT_MODELS)]
    assert pm.groupby("family").unit.nunique().max() <= 13


def test_unadjusted_counts_reproduce_the_published_counts():
    """The 'before' counts equal the counts the paper already reports (figure_stats.json)."""
    import json
    fs = json.load(open("figures_nielsen/figure_stats.json"))
    a = pd.read_csv(f"{RR}/multiple_comparisons_audit.csv").set_index("family")
    assert a.loc["results_nielsen/wilcoxon_nielsen_effect_vignettes:tight", "n_sig_unadj"] == fs["nielsen_n_sig_tight_vignettes"]
    assert a.loc["results_nielsen/wilcoxon_nielsen_effect_random_dose:tight", "n_sig_unadj"] == fs["nielsen_n_sig_tight_rd"]
    assert a.loc["results_nielsen/wilcoxon_dosing_unit_effect:tight", "n_sig_unadj"] == fs["dosing_n_sig_drop_tight"]
    assert a.loc["results_nielsen/cv_accuracy_chi2:Figure 2", "n_sig_unadj"] == fs["n_sig_cv_chi2_vignettes"]
    assert a.loc["results_nielsen/variance_icc_spearman:variance_vs_icc", "n_sig_unadj"] == fs["n_sig_rho_models"]
    assert a.loc["figure9:cv_chi2", "n_sig_unadj"] == fs["n_sig_cv_chi2_rd"]
    assert round(a.loc["figure9:slope_panel", "dose_dep_neg_pct_unadj"], 1) == fs["rd_mean_dose_dep_neg_pct"]
    assert a.loc["results_nielsen/icc_analysis:f_test", "n_groups_sig_unadj"] == fs["n_models_with_sig_icc"]


def test_broad_only_rankings():
    r = pd.read_csv(f"{RR}/rankings_fig12_fig15_broad.csv")
    assert set(r.figure) == {"Figure 1", "Figure 8", "Figure 12", "Figure 15 (top)", "Figure 15 (bottom)"}
    for fig, g in r.groupby("figure"):
        assert len(g) == 13
        # sorted by group, then by % within 0.6-1.7x
        assert (g.sort_values("row_order").group_broad.diff().dropna() >= 0).all()
        for _, h in g.groupby("group_broad"):
            assert (h.sort_values("row_order").pct_broad.diff().dropna() <= 1e-12).all()
        assert g.group_broad.min() == 1
    # one-band ranking = rank_models(..., bands=("broad",)): recompute one panel
    from analysis.revision import stats
    x = stats.rank_models(stats.load_long(), "nielsen", "rd_nielsen", "all", bands=("broad",)).result_df.set_index("model")
    y = r[r.figure == "Figure 12"].set_index("model")
    assert (x.adjusted_rank.reindex(y.index) == y.group_broad).all()
