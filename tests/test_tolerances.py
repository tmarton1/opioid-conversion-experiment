"""Task 1.7: band registry and band justification."""
import numpy as np
import pandas as pd

from analysis.revision import tolerances as T


def test_registry_and_fixed_bands():
    assert list(T.BANDS)[:4] == ["tight", "broad", "reference_agreement", "reference_envelope"]
    assert T.band_bounds("tight", "oral_codeine", "1", "oral_morphine") == (0.85, 1.15)
    assert T.band_bounds("broad", "oral_codeine", "1", "oral_morphine") == (0.6, 1.7)
    assert T.MAIN_REFERENCE_BAND == "reference_envelope"


def test_reference_agreement_k_and_envelope():
    k = T.reference_agreement_k()
    assert abs(k - 1.5) < 1e-9                       # set by codeine (Nielsen 0.1 vs 0.15)
    lo, hi = T.band_bounds("reference_envelope", "oral_hydrocodone", "1", "oral_morphine")
    assert abs(hi - 1.2) < 1e-9 and abs(lo - 1 / 1.2) < 1e-9
    assert T.band_bounds("reference_envelope", "iv_methadone", "1", "oral_morphine") is None


def test_within_marks_missing_band_as_nan():
    f = pd.DataFrame({"from_opioid": ["oral_hydrocodone", "iv_methadone"], "from_dose": ["1", "1"],
                      "to_opioid": ["oral_morphine", "oral_morphine"]})
    w = T.within("reference_envelope", pd.Series([1.1, 1.0]), f)
    assert w.iloc[0] == 1.0 and np.isnan(w.iloc[1])


def test_band_justification_matches_planning():
    df, info = T.band_justification()
    assert abs(info["disagreement_median"] - 1.2) < 1e-9
    assert abs(info["disagreement_p95"] - 1.4) < 1e-9
    assert abs(info["disagreement_max"] - 1.5) < 1e-9
    assert abs(info["k_without_codeine"] - 1.4) < 1e-9
    assert "oral_codeine -> oral_morphine" in info["k_set_by"]
    t = df[(df.band == "tight") & (df.side == "upper")].iloc[0]
    assert t.relation == "narrower than the median disagreement"
