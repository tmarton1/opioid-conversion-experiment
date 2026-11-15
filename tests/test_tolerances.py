"""Task 1.7: band registry and band justification."""
import numpy as np
import pandas as pd

from analysis.revision import tolerances as T


def test_registry_and_fixed_bands():
    assert list(T.BANDS) == ["tight", "broad", "overdose", "severe_overdose", "underdose"]  # 2026-11-15
    assert T.TAILS == ["overdose", "severe_overdose", "underdose"]
    assert T.MAIN_BANDS == ["tight", "broad"]
    assert T.band_bounds("tight", "oral_codeine", "1", "oral_morphine") == (0.85, 1.15)
    assert T.band_bounds("broad", "oral_codeine", "1", "oral_morphine") == (0.6, 1.7)


def test_reference_agreement_k_still_computed():
    k = T.reference_agreement_k()
    assert abs(k - 1.5) < 1e-9                       # set by codeine (Nielsen 0.1 vs 0.15)
    assert abs(T.conversion_spread("oral_hydrocodone", "1", "oral_morphine") - 1.2) < 1e-9
    assert T.conversion_spread("iv_methadone", "1", "oral_morphine") is None


def test_within_fixed_band():
    f = pd.DataFrame({"from_opioid": ["oral_hydrocodone", "iv_methadone"], "from_dose": ["1", "1"],
                      "to_opioid": ["oral_morphine", "oral_morphine"]})
    w = T.within("broad", pd.Series([1.65, 1.75]), f)
    assert w.iloc[0] == 1.0 and w.iloc[1] == 0.0


def test_band_justification_matches_planning():
    df, info = T.band_justification()
    assert abs(info["disagreement_median"] - 1.2) < 1e-9
    assert abs(info["disagreement_p95"] - 1.4) < 1e-9
    assert abs(info["disagreement_max"] - 1.5) < 1e-9
    assert abs(info["k_without_codeine"] - 1.4) < 1e-9
    assert "oral_codeine -> oral_morphine" in info["k_set_by"]
    t = df[(df.band == "tight") & (df.side == "upper")].iloc[0]
    assert t.relation == "narrower than the median disagreement"
    assert "just wider than the largest disagreement" in info["broad_interpretation"]
    assert info["broad_covers_any_standard"]
    assert set(df.band) == {"tight", "broad", "reference_agreement (superseded)"}


def test_open_ended_standard_anchored_tail():
    """overdose: > 2x the highest current standard (open upper bound, strict lower
    bound, anchored on the standards, lower share is better)."""
    b = T.BANDS["overdose"]
    assert b.anchor == "highest_standard" and not b.higher_is_better and b.kind == "tail"
    assert b.bounds("x", "1", "y") == (2.0, np.inf)
    # oral hydrocodone -> oral morphine: Nielsen expects 120, the others 100: highest = 120
    f = pd.DataFrame({"from_opioid": ["oral_hydrocodone"] * 4, "from_dose": ["1"] * 4,
                      "to_opioid": ["oral_morphine"] * 4,
                      "observed_to_dose": [240.0, 241.0, 1e6, 0.0],
                      "expected_nielsen": [120.0] * 4, "expected_cdc_2022": [100.0] * 4,
                      "expected_nih_heal": [100.0] * 4, "expected_mcpherson_2018": [100.0] * 4})
    # the score's ratio column is ignored for a standard-anchored entry
    w = T.within("overdose", pd.Series([9.9, 9.9, 9.9, 9.9]), f)
    assert list(w) == [0.0, 1.0, 1.0, 0.0]          # exactly 2x is not > 2x; unusable answer not in the tail
    u = T.within("underdose", pd.Series([1.0] * 4), f.assign(observed_to_dose=[49.0, 50.0, 60.0, np.nan]))
    assert list(u) == [1.0, 0.0, 0.0, 0.0]          # < 0.5x the lowest standard (100)


def test_tail_polarity_in_paired_tests():
    from analysis.revision import stats
    L = stats.load_long()
    pe = stats.paired_effect(L, "nielsen", "rd_naive", "rd_nielsen", outcome="overdose").set_index("model")
    # grounding lowers the overdose share, and a positive diff means "better"
    assert pe.loc["POOLED", "mean_b"] < pe.loc["POOLED", "mean_a"] and pe.loc["POOLED", "mean_diff"] > 0
