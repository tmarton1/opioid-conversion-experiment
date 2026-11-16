"""Classifier of extreme overdoses (analysis/revision/extreme_breakdown.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.revision.extreme_breakdown import classify  # noqa: E402
from analysis.revision.rescore import _factor  # noqa: E402


def _row(observed, from_opioid="oral_morphine", to_opioid="oral_hydromorphone", from_unit="mg/day", to_unit="mg/day"):
    f = _factor("nielsen", from_opioid) / _factor("nielsen", to_opioid)
    expected = 60 * f
    return {"from_opioid": from_opioid, "from_dose": 60, "to_opioid": to_opioid, "observed_to_dose": observed,
            "from_opioid_unit": from_unit, "to_opioid_unit": to_unit,
            "expected_nielsen": expected, "expected_cdc_2022": None, "expected_nih_heal": None,
            "expected_mcpherson_2018": None}, expected, f


def test_inverted_factor_is_classified_as_inversion():
    row, expected, f = _row(None)
    row["observed_to_dose"] = 60 / f          # multiplied where it should divide
    assert classify(row)["error_class"] == "factor_inversion"


def test_mg_mcg_error_only_where_mcg_is_in_the_prompt():
    row, expected, _ = _row(None)
    row["observed_to_dose"] = expected * 1000
    assert classify(row)["error_class"] == "other"            # mg/day -> mg/day: no mcg in the prompt
    row["to_opioid_unit"] = "mcg/day"
    assert classify(row)["error_class"] == "unit_error"


def test_per_4h_only_for_4_hourly_targets():
    row, expected, _ = _row(None)
    row["observed_to_dose"] = expected * 6
    assert classify(row)["error_class"] != "unit_error"
    row["to_opioid_unit"] = "mg/4hr"
    assert classify(row)["error_class"] == "unit_error"
