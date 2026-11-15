"""Task 1.2: the score registry is pluggable (a new score needs one register() call)."""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from analysis.revision import scores as S
from analysis.revision.rescore import rescore


def test_registry_has_initial_scores_in_order():
    assert list(S.SCORES)[:4] == ["nielsen", "all", "non_nielsen", "geomean"]
    assert S.SCORES["geomean"].variance_valid and S.SCORES["nielsen"].variance_valid
    assert not S.SCORES["all"].variance_valid and not S.SCORES["non_nielsen"].variance_valid


def test_nearest_and_geometric_mean_rules():
    e = {"nielsen": 100.0, "cdc_2022": 80.0, "nih_heal": None, "mcpherson_2018": 125.0}
    assert S.SCORES["all"].fn(82.0, e) == (80.0, "cdc_2022")
    assert S.SCORES["all"].fn(120.0, e) == (125.0, "mcpherson_2018")
    assert S.SCORES["non_nielsen"].fn(99.0, e)[1] in ("cdc_2022", "mcpherson_2018")
    g, _ = S.SCORES["geomean"].fn(None, e)
    assert abs(g - (100 * 80 * 125) ** (1 / 3)) < 1e-9
    # no alternative standard -> excluded under non_nielsen
    assert S.SCORES["non_nielsen"].fn(50.0, {"nielsen": 100.0}) == (None, None)
    # missing response keeps a reference so it still counts as incorrect
    assert S.SCORES["all"].fn(None, e)[0] == 100.0


def test_dummy_score_flows_through_to_csv():
    dummy = S.Score(name="dummy_double", label="Dummy", title_suffix=" — dummy",
                    fn=lambda obs, exp: ((2 * exp["nielsen"], "nielsen") if exp.get("nielsen") else (None, None)),
                    variance_valid=True)
    S.register(dummy)
    tmp = tempfile.mkdtemp()
    try:
        res = rescore(raw_dir="results", out_root=tmp, revision_dir=os.path.join(tmp, "rev"),
                      scores=["dummy_double"])
        out = pd.read_csv(os.path.join(tmp, "results_dummy_double", "experiment_results.csv"))
        assert np.allclose(out["expected_to_dose"], 2 * out["expected_nielsen"])
        L = res["responses_long"]
        assert "ratio_dummy_double" in L.columns
        m = L["ratio_nielsen"].notna()
        assert np.allclose(L.loc[m, "ratio_dummy_double"], L.loc[m, "ratio_nielsen"] / 2)
    finally:
        S.unregister("dummy_double")
        shutil.rmtree(tmp, ignore_errors=True)
    assert "dummy_double" not in S.SCORES
