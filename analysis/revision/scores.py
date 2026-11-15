"""
Score registry: how a response's reference (expected) dose is chosen.

Every score is a name plus a function that receives the response's observed
dose and the expected dose under every available scoring standard, and
returns the reference dose to score against (or None) and the standard it
came from. Everything downstream (rescoring, per-score pipeline runs,
statistics, figures, the report, orchestration/run_revision.py) loops over
SCORES; nothing else hard-codes score names. Adding a score = one
``register(...)`` call plus a rerun of orchestration/run_revision.py.

Expected doses are passed as {standard_name: dose or None}, using the names in
resources/reference_standards_index.json. CDC 2016 is never passed to a score
(it is recall-check only); see SCORING_INPUTS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

# Standards a score may use, in tie-break order (first wins on an exact tie).
SCORING_INPUTS = ["nielsen", "cdc_2022", "nih_heal", "mcpherson_2018"]
NON_NIELSEN = ["cdc_2022", "nih_heal", "mcpherson_2018"]

Expected = dict  # {standard: float | None}
ScoreFn = Callable[[Optional[float], Expected], tuple[Optional[float], Optional[str]]]


@dataclass(frozen=True)
class Score:
    name: str
    label: str             # short name for legends / panel labels
    title_suffix: str      # appended to figure titles ("" for the original score)
    fn: ScoreFn
    variance_valid: bool   # True if reference is a fixed multiple of Nielsen per conversion
    description: str = ""


SCORES: dict[str, Score] = {}


def register(score: Score) -> Score:
    if score.name in SCORES:
        raise ValueError(f"score {score.name!r} already registered")
    SCORES[score.name] = score
    return score


def unregister(name: str) -> None:
    SCORES.pop(name, None)


def _valid(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and x > 0


def _available(expected: Expected, names: list[str]) -> list[tuple[str, float]]:
    return [(n, float(expected[n])) for n in names if _valid(expected.get(n))]


def nearest_of(names: list[str]) -> ScoreFn:
    """Reference = the available standard nearest the response on the log scale.

    If the response is missing or not positive, no 'nearest' exists; the first
    available standard (in ``names`` order) is returned so the row keeps a
    reference dose and still counts as an incorrect answer, as it does under
    the original score.
    """
    def fn(observed, expected):
        avail = _available(expected, names)
        if not avail:
            return None, None
        if not _valid(observed):
            return avail[0][1], avail[0][0]
        lo = math.log(observed)
        best = min(avail, key=lambda kv: abs(lo - math.log(kv[1])))  # stable: first on ties
        return best[1], best[0]
    return fn


def geometric_mean_of(names: list[str]) -> ScoreFn:
    def fn(observed, expected):
        avail = _available(expected, names)
        if not avail:
            return None, None
        g = math.exp(sum(math.log(v) for _, v in avail) / len(avail))
        return g, "geometric_mean(" + ",".join(n for n, _ in avail) + ")"
    return fn


def _nielsen(observed, expected):
    v = expected.get("nielsen")
    return (float(v), "nielsen") if _valid(v) else (None, None)


register(Score(
    name="nielsen", label="Nielsen", title_suffix="",
    fn=_nielsen, variance_valid=True,
    description="Original expected dose (Nielsen et al. 2016).",
))
register(Score(
    name="all", label="All standards", title_suffix=" — scored against all standards",
    fn=nearest_of(SCORING_INPUTS), variance_valid=False,
    description="Nearest (log scale) of Nielsen, CDC 2022, NIH HEAL, McPherson 2018; "
                "matching any standard scores 1x.",
))
register(Score(
    name="non_nielsen", label="All except Nielsen",
    title_suffix=" — scored against all standards except Nielsen",
    fn=nearest_of(NON_NIELSEN), variance_valid=False,
    description="Nearest of CDC 2022, NIH HEAL, McPherson 2018; conversions with no "
                "non-Nielsen standard are excluded.",
))
register(Score(
    name="consensus", label="Consensus (geometric mean)",
    title_suffix=" — scored against the consensus of all standards",
    fn=geometric_mean_of(SCORING_INPUTS), variance_valid=True,
    description="Geometric mean of the available standards' expected doses "
                "(a fixed multiple of Nielsen per conversion).",
))


def rescored_names() -> list[str]:
    """Registered scores other than the original, in registration order."""
    return [n for n in SCORES if n != "nielsen"]
