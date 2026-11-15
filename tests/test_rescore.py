"""Task 1.4: hand-checked rescoring examples (REVISION_PLAN.md 1.4). Stop the run if any fails."""
import math

from analysis.revision.rescore import standard_ratio
from analysis.revision.scores import SCORES

TOL = 1e-3


def close(a, b, tol=TOL):
    return a is not None and abs(a - b) < tol


def test_hydrocodone_to_morphine_all_alternatives_0833():
    for s in ("cdc_2022", "nih_heal", "mcpherson_2018"):
        assert close(standard_ratio(s, ["oral_hydrocodone"], [1.0], "oral_morphine"), 1 / 1.2), s


def test_tramadol_to_oxycodone():
    for s in ("cdc_2022", "nih_heal"):
        assert close(standard_ratio(s, ["oral_tramadol"], [1.0], "oral_oxycodone"), 1.0), s
    assert standard_ratio("mcpherson_2018", ["oral_tramadol"], [1.0], "oral_oxycodone") is None


def test_iv_hydromorphone_to_morphine_only_mcpherson():
    assert close(standard_ratio("mcpherson_2018", ["iv_hydromorphone"], [1.0], "oral_morphine"), 0.714)
    for s in ("cdc_2022", "nih_heal"):
        assert standard_ratio(s, ["iv_hydromorphone"], [1.0], "oral_morphine") is None


def test_elderly_female_renal_colic_multidrug():
    froms, doses = ["iv_morphine", "iv_hydromorphone"], [6.0, 8.0]
    assert close(standard_ratio("mcpherson_2018", froms, doses, "oral_morphine"), 0.728)
    for s in ("cdc_2022", "nih_heal"):
        assert standard_ratio(s, froms, doses, "oral_morphine") is None


def test_middle_aged_female_cancer_patch():
    for s in ("cdc_2022", "nih_heal"):
        assert close(standard_ratio(s, ["oral_hydromorphone"], [1.0], "transdermal_fentanyl"), 1.125), s


def test_response_equal_to_nielsen_for_hydrocodone():
    n = 100.0
    expected = {"nielsen": n, "cdc_2022": n / 1.2, "nih_heal": n / 1.2, "mcpherson_2018": n / 1.2}
    obs = n
    ratios = {}
    for name in ("nielsen", "all", "non_nielsen", "consensus"):
        ref, _ = SCORES[name].fn(obs, expected)
        r = obs / ref
        ratios[name] = max(r, 1 / r)  # fold error
    assert close(ratios["nielsen"], 1.0)
    assert close(ratios["all"], 1.0)
    assert close(ratios["non_nielsen"], 1.2)
    gm = math.exp((math.log(1) + 3 * math.log(1 / 1.2)) / 4)
    assert close(gm, 0.872)
    assert close(ratios["consensus"], 1 / gm)
    assert close(ratios["consensus"], 1.147)
