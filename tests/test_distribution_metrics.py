"""Task 3A/3B outputs: internal consistency and agreement with the framework."""
import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.distribution_metrics import _paired


def test_paired_matches_framework_for_same_subset():
    L = stats.load_long()
    a = _paired(L, "nielsen", "rd_naive", "all", "rd_nielsen", "all", "tight").set_index("model")
    b = stats.paired_effect(L, "nielsen", "rd_naive", "rd_nielsen", outcome="tight").set_index("model")
    assert np.allclose(a.p, b.p, equal_nan=True) and (a.n_pairs == b.n_pairs).all()


def test_outputs_consistent():
    s = pd.read_csv("results_revision/distribution_summary.csv")
    assert set(s.role) == {"main", "supplementary_empty_prompts"}
    assert s[s.model != "POOLED"].model.nunique() == 13
    e = pd.read_csv("results_revision/extreme_errors.csv")
    assert set(e.score) == {"nielsen"}
    c = pd.read_csv("results_revision/circularity.csv")
    g = c[(c.score == "nielsen") & (c.outcome == "tight") & (c.model == "ALL_MODELS") & (c.subset == "all")]
    assert (g.gain > 20).all()   # grounding gain under Nielsen is large
