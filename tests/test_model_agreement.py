import numpy as np
import pandas as pd

from analysis.revision import model_agreement as ma
from analysis.revision import standard_matches as sm


def test_group_values_ten_percent():
    v = np.log([1.0, 1.05, 1.09, 1.3, 2.0, 2.1])
    assert ma.group_values(v) == [3, 1, 2]


def test_is_wrong_uses_tight_band():
    assert list(ma.is_wrong(np.log([0.9, 1.1, 1.2, 0.8]))) == [False, False, True, True]


def test_match_matrix_and_nan_standards():
    rel = np.log(np.array([[1.5, np.nan], [np.nan, np.nan]]))
    m = sm._match_matrix(np.log(np.array([1.45, 1.45])), rel)
    assert list(m) == [True, False]


def test_consensus_weights_and_p():
    g = pd.DataFrame({"mean_log": [0.0, 0.0, 0.0], "sd_log": [0.1, 0.2, 0.3]})
    c = ma.consensus(g)
    assert abs(c["consensus_ratio"] - 1) < 1e-12 and c["effective_n_models"] > 2


def test_build_reproducible_and_seeded():
    a1, p1 = ma.build()
    a2, p2 = ma.build()
    pd.testing.assert_frame_equal(a1, a2)
    s = a1[a1.row_type == "summary"].set_index("experiment")
    assert s.loc["random_dose", "n_models"] == 13
    m1, _ = sm.build(n_perm=50)
    m2, _ = sm.build(n_perm=50)
    pd.testing.assert_frame_equal(m1, m2)
