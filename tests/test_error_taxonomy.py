"""Error taxonomy (analysis/revision/error_taxonomy.py): hand-checked cases."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.revision import error_taxonomy as T  # noqa: E402
from analysis.revision.extreme_breakdown import classify, match_lists  # noqa: E402
from analysis.revision.rescore import _factor  # noqa: E402


def _row(observed, to_unit="mg/day", from_unit="mg/day"):
    f = _factor("nielsen", "oral_morphine") / _factor("nielsen", "oral_hydromorphone")
    expected = 60 * f
    return {"from_opioid": "oral_morphine", "from_dose": 60, "to_opioid": "oral_hydromorphone",
            "observed_to_dose": observed, "from_opioid_unit": from_unit, "to_opioid_unit": to_unit,
            "expected_nielsen": expected, "expected_cdc_2016": None, "expected_cdc_2022": None,
            "expected_nih_heal": None, "expected_mcpherson_2018": None}, expected, f


def test_round_number_rule_by_magnitude():
    d = [5, 7.5, 7.3, 12, 15, 100, 150, 120, 250, 1000, 1500, 1200, 0.5, 0.25, 0.23, 0, -50, np.nan]
    assert list(T.is_round(d)) == [True, True, False, False, True, True, True, False, True, True, True,
                                   False, True, True, False, False, False, False]


def test_precedence_inversion_then_unit_then_standard_then_round():
    flags = dict(factor_inversion=True, unit_error=True, standard_match=True, round_number=True)
    assert T.primary_class(True, flags) == "factor_inversion"
    flags["factor_inversion"] = False
    assert T.primary_class(True, flags) == "unit_error"
    flags["unit_error"] = False
    assert T.primary_class(True, flags) == "standard_match"
    flags["standard_match"] = False
    assert T.primary_class(True, flags) == "round_number"
    flags["round_number"] = False
    assert T.primary_class(True, flags) == "other"
    assert T.primary_class(False, flags) == "no_usable_answer"


def test_standard_match_needs_a_standard_distinct_from_nielsen():
    row, e, _ = _row(None)
    row["expected_cdc_2016"] = e * 1.5          # differs from Nielsen by 50%
    row["observed_to_dose"] = e * 1.5 * 1.05
    assert T.standard_matches(row) == ["cdc_2016"]
    row["observed_to_dose"] = e * 1.5 * 1.2     # more than 10% away
    assert T.standard_matches(row) == []
    row["expected_cdc_2016"] = e * 1.05         # within 10% of Nielsen: a match to Nielsen, not a distinct standard
    row["observed_to_dose"] = e * 1.05
    assert T.standard_matches(row) == []
    row["observed_to_dose"] = None
    assert T.standard_matches(row) == []


def test_unit_error_both_directions_and_extreme_breakdown_unchanged():
    row, e, _ = _row(None, to_unit="mg/4 hours")
    row["observed_to_dose"] = e * 6
    inv, unit = match_lists(row)
    assert unit and not inv                      # "mg/4 hours" targets allow the per-4h confusion
    row["observed_to_dose"] = e / 6
    assert match_lists(row)[1] == []             # default: only the too-large side, as in extreme_breakdown
    assert match_lists(row, unit_both_directions=True)[1] == [u for u in match_lists(row, True)[1] if "inverse" in u]
    assert match_lists(row, True)[1]
    row2, e2, _ = _row(None)                     # mg/day target: no per-4h confusion possible
    row2["observed_to_dose"] = e2 / 6
    assert match_lists(row2, True)[1] == []
    assert classify(row)["error_class"] in ("other", "factor_inversion", "unit_error")


def test_summary_row_counts_and_overlaps():
    g = pd.DataFrame({"model": ["a", "a", "b"], "condition": "c", "item": ["x", "y", "x"],
                      "primary_class": ["factor_inversion", "round_number", "round_number"],
                      "factor_inversion": [True, True, False], "unit_error": [False] * 3,
                      "standard_match": [False] * 3, "round_number": [False, True, True],
                      "n_classes_matched": [1, 2, 1]})
    tot = pd.DataFrame({"model": list("aaabbb"), "condition": "c", "item": list("xyzxyz"),
                        "outside_nielsen": [True, True, False, True, False, False],
                        "is_round": [False, True, True, True, False, False]})
    r = T._summ(g, tot, "nielsen", "c", "ALL", "ALL", "both")
    assert (r["n_outside_broad"], r["n_models"], r["n_cells"], r["n_cells_total"]) == (3, 2, 3, 6)
    assert r["n_factor_inversion"] == 1 and r["n_round_number"] == 2 and np.isclose(r["pct_round_number"], 200 / 3)
    assert r["n_any_factor_inversion"] == 2 and r["n_multi"] == 1 and r["n_factor_inversion_and_round_number"] == 1
    assert r["n_within_broad"] == 3 and np.isclose(r["pct_round_within_broad"], 100 / 3)


def test_build_is_consistent_with_extreme_breakdown_and_deterministic():
    out, sc_df = T.build()
    out2, _ = T.build()
    pd.testing.assert_frame_equal(out, out2)
    assert out["primary_class"].isin(T.CLASSES).all()
    assert (out.loc[out["primary_class"] == "other", T.FLAGS].sum(axis=1) == 0).all()
    assert (out[[f"outside_{s}" for s in ("nielsen", "all", "non_nielsen", "geomean")]].any(axis=1)).all()
    # the severe-overdose tail (> 4x highest standard) is outside broad under Nielsen; its factor-inversion and
    # unit-error classes agree with extreme_errors_breakdown.csv
    b = pd.read_csv(os.path.join("results_revision", "extreme_errors_breakdown.csv"))
    m = out.reset_index().merge(b[["condition", "model", "observed_to_dose", "from_opioid", "from_dose", "to_opioid",
                                   "error_class"]], on=["condition", "model", "from_opioid", "to_opioid"], how="inner")
    def key(v):
        try:
            return str(float(v))
        except ValueError:
            return str(v)
    same = m["from_dose_x"].map(key) == m["from_dose_y"].map(key)
    sub = m[np.isclose(m["observed_to_dose_x"], m["observed_to_dose_y"]) & same]
    assert len(sub) > 100
    assert ((sub["error_class"] == "factor_inversion") == sub["factor_inversion"]).all()
    assert ((sub["error_class"] == "unit_error") <= sub["unit_error"]).all()
