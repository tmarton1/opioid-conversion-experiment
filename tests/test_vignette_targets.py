"""Empty-vignette prompts are scored against the conversion they ask for
(author decision 2026-11-14)."""
import hashlib

import numpy as np
import pandas as pd

from analysis.revision import vignette_targets as V

RAW = "results/reworded_vignettes_results.csv"


def test_reduction_scenarios_read_from_definitions():
    assert V.reduction_scenarios() == {
        "elderly-hip-fracture": 0.75, "elderly-hip-replacement": 0.8,
        "middle-aged-ankle-surgery": 0.5, "middle-aged-carpenter-amputation": 0.75}
    f = V.scenario_factors().set_index("scenario_key")
    assert len(f) == 20
    # unit factor (mg/4 h) is kept for both prompts
    assert np.isclose(f.loc["pediatric-burn-patient", "empty_unit"], 1 / 6)
    assert f.loc["pediatric-burn-patient", "same_conversion"]
    assert len(V.same_conversion_scenarios()) == 16


def test_only_empty_prompts_in_reduction_scenarios_change():
    before = hashlib.sha256(open(RAW, "rb").read()).hexdigest()
    raw = pd.read_csv(RAW, low_memory=False)
    c = V.correct_vignettes(raw)
    changed = ~np.isclose(c["expected_to_dose"], raw["expected_to_dose"])
    sub = c[changed]
    assert set(sub["scenario_key"]) == set(V.reduction_scenarios())
    assert not sub["has_vignette_text"].astype(bool).any()
    k = sub["scenario_key"].map(V.reduction_scenarios())
    assert np.allclose(sub["expected_to_dose"] * k, raw.loc[changed, "expected_to_dose"])
    # hip fracture: 120 mg/day oxycodone -> oral hydromorphone = 36 mg/day (text: 27 with 25% reduction)
    hf = c[(c.scenario_key == "elderly-hip-fracture") & ~c.has_vignette_text.astype(bool)]
    assert np.allclose(hf["expected_to_dose"], 36.0)
    assert np.allclose(c["ratio_observed_expected"], c["observed_to_dose"] / c["expected_to_dose"], equal_nan=True)
    assert hashlib.sha256(open(RAW, "rb").read()).hexdigest() == before   # raw file untouched
