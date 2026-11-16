"""
Accuracy-band registry (task 1.7).

Each band is a name plus a lower and upper bound on observed/expected, either
fixed or computed per conversion. Statistics, figures and the report loop over
BANDS; a new band is one ``register(...)`` call plus a rerun.

    from analysis.revision.tolerances import BANDS, band_bounds
    lo, hi = band_bounds("broad", "oral_codeine", "1", "oral_morphine")

Bands (author decision 2026-11-15: tight and broad only):
  tight   0.85-1.15x (original)
  broad   0.6-1.7x   (original): just wider than the largest disagreement between
          current standards (1.5-fold, computed below), so an answer copying any
          current standard, Nielsen included, is always within broad of every
          other one. Broad under non_nielsen is therefore a cross-reference measure of
          circularity; tight under that score fails by construction.
Safety tails (registry entries, author decision 2026-11-15), anchored on the
current standards and so the same under every score; lower share is better:
  overdose         > 2x the highest current standard for the conversion
  severe_overdose  > 4x the highest current standard
  underdose        < 0.5x the lowest current standard
Every entry is an interval on a ratio with an anchor (the score's expected dose,
the highest or the lowest current standard), open bounds allowed, and a
polarity flag; 3A, 3B, the tables and the report treat all entries alike.
Superseded (definitions kept below, commented): reference_agreement,
reference_envelope.

    python -m analysis.revision.tolerances   # writes results_revision/references/band_justification.csv
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pandas as pd

from analysis.revision.rescore import _parse_sources, standard_ratio
from analysis.revision.scores import SCORING_INPUTS
from common.config import BROAD_RANGE_LOWER, BROAD_RANGE_UPPER, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER

Bounds = Optional[tuple[float, float]]
BoundsFn = Callable[[str, str, str], Bounds]   # (from_opioid, from_dose, to_opioid) -> (lo, hi)


@dataclass(frozen=True)
class Band:
    """An interval on a ratio (observed / anchor dose).

    bounds          (lo, hi); lo may be 0 and hi may be inf (open-ended)
    anchor          "expected" (the score's reference dose: the original bands),
                    "highest_standard" or "lowest_standard" (the highest / lowest
                    expected dose among the current standards for the conversion:
                    the same under every score)
    higher_is_better True for accuracy bands (more responses inside is better);
                    False for tails (overdose, underdose: fewer inside is better).
                    Paired tests, gains and verdicts use it so that a positive
                    difference always means "better".
    closed          whether each finite bound is included (bands: both; tails:
                    strict, e.g. > 2x)
    kind            "band" or "tail" (for labels and default selections)
    """
    name: str
    label: str
    bounds: BoundsFn
    per_conversion: bool
    description: str = ""
    anchor: str = "expected"
    higher_is_better: bool = True
    closed: tuple = (True, True)
    kind: str = "band"


BANDS: dict[str, Band] = {}


def register(band: Band) -> Band:
    if band.name in BANDS:
        raise ValueError(f"band {band.name!r} already registered")
    BANDS[band.name] = band
    return band


def unregister(name: str) -> None:
    BANDS.pop(name, None)


def conversion_spread(from_opioid: str, from_dose: str, to_opioid: str) -> Optional[float]:
    """Fold spread (max/min) of the current standards' expected doses for one
    conversion; None when no standard other than Nielsen covers it."""
    froms, doses = _parse_sources(from_opioid, from_dose)
    vals = [standard_ratio(s, froms, doses, to_opioid) for s in SCORING_INPUTS]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    return max(vals) / min(vals)


@lru_cache(maxsize=None)
def _spread_cached(f, d, t):
    return conversion_spread(f, d, t)


@lru_cache(maxsize=1)
def scored_spreads() -> pd.DataFrame:
    """Spread for each of the 69 scored conversions (from the reference comparison)."""
    from analysis.revision.reference_comparison import scored_conversions
    conv = scored_conversions("results")
    conv["spread"] = [conversion_spread(r.from_opioid, r.from_dose, r.to_opioid)
                      for r in conv.itertuples(index=False)]
    return conv


def reference_agreement_k() -> float:
    return float(scored_spreads()["spread"].max())


def _fixed(lo, hi) -> BoundsFn:
    return lambda f, d, t: (lo, hi)


# ── Superseded bands (author decision 2026-11-15) ─────────────────────────
# reference_agreement (1/k-k, k = largest disagreement between any two current
# standards, 1.5) nearly duplicated broad; reference_envelope (per conversion,
# the spread of that conversion's own standards) has zero width where the
# standards agree exactly, so it rewarded only exact copies there. Both are no
# longer registered (3A, 3B, the figures and the report use tight and broad);
# their definitions are kept here, commented out, for the record.
# reference_agreement_k() and conversion_spread() above remain in use for the
# band justification.
#
# def _agreement(f, d, t):
#     k = reference_agreement_k()
#     return (1 / k, k)
#
# def _envelope(f, d, t):
#     k = _spread_cached(str(f), str(d), str(t))
#     return None if k is None else (1 / k, k)
#
# register(Band("reference_agreement", "Reference agreement (1/k–k×)", _agreement, False,
#               "Narrowest symmetric band inside which every pair of current standards agrees."))
# register(Band("reference_envelope", "Reference envelope (per conversion)", _envelope, True,
#               "Per conversion, the fold spread between its most extreme current standards."))

register(Band("tight", f"Tight ({TIGHT_RANGE_LOWER}–{TIGHT_RANGE_UPPER}×)",
              _fixed(TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER), False, "Original tight band."))
register(Band("broad", f"Broad ({BROAD_RANGE_LOWER}–{BROAD_RANGE_UPPER}×)",
              _fixed(BROAD_RANGE_LOWER, BROAD_RANGE_UPPER), False,
              "Original broad band; just wider than the largest disagreement between current "
              "standards, so an answer copying any current standard is within broad of every other."))

# Safety tails (author decision 2026-11-15): one-sided intervals anchored on the
# current standards (Nielsen, CDC 2022, NIH HEAL, McPherson 2018), so they are
# the same under every score; a higher share is worse.
register(Band("overdose", "Overdose (> 2× highest standard)", _fixed(2.0, np.inf), False,
              "Response more than twice the highest current standard for the conversion.",
              anchor="highest_standard", higher_is_better=False, closed=(False, False), kind="tail"))
register(Band("severe_overdose", "Severe overdose (> 4× highest standard)", _fixed(4.0, np.inf), False,
              "Response more than four times the highest current standard for the conversion.",
              anchor="highest_standard", higher_is_better=False, closed=(False, False), kind="tail"))
register(Band("underdose", "Underdose (< 0.5× lowest standard)", _fixed(0.0, 0.5), False,
              "Response less than half the lowest current standard for the conversion.",
              anchor="lowest_standard", higher_is_better=False, closed=(False, False), kind="tail"))

DEFAULT_BANDS = ["tight", "broad"]
MAIN_BANDS = ["tight", "broad"]                       # accuracy bands in the report and the example figures
TAILS = [b for b in BANDS if BANDS[b].kind == "tail"]  # safety tails (lower is better)
ANCHOR_COLUMNS = [f"expected_{s}" for s in SCORING_INPUTS]   # current standards' expected doses


def band_bounds(name: str, from_opioid: str, from_dose: str, to_opioid: str) -> Bounds:
    return BANDS[name].bounds(str(from_opioid), str(from_dose), str(to_opioid))


def anchored_ratios(name: str, ratios: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """The ratio a registry entry is evaluated on: the score's ratio for
    anchor "expected"; observed / highest (or lowest) current standard
    otherwise (frame needs observed_to_dose and the expected_<standard> columns)."""
    anchor = BANDS[name].anchor
    if anchor == "expected":
        return pd.to_numeric(ratios, errors="coerce")
    e = frame[ANCHOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ref = e.max(axis=1) if anchor == "highest_standard" else e.min(axis=1)
    obs = pd.to_numeric(frame["observed_to_dose"], errors="coerce")
    return (obs.where(obs > 0) / ref).astype(float)


def within(name: str, ratios: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """1/0 per row: the (anchored) ratio inside the entry's interval; NaN where
    the interval is undefined. A row whose ratio is missing or not positive
    counts as outside (for bands: incorrect, as in the original pipeline; for
    tails: not in the tail).

    `frame` needs from_opioid, from_dose and to_opioid columns aligned with
    `ratios`, plus observed_to_dose and expected_<standard> for entries anchored
    on a standard."""
    band = BANDS[name]
    keys = list(zip(frame["from_opioid"].astype(str), frame["from_dose"].astype(str),
                    frame["to_opioid"].astype(str)))
    if not band.per_conversion:
        lo, hi = band.bounds(*keys[0]) if keys else (np.nan, np.nan)
        lo = np.full(len(keys), lo, dtype=float); hi = np.full(len(keys), hi, dtype=float)
    else:
        b = {k: band_bounds(name, *k) for k in set(keys)}
        lo = np.array([np.nan if b[k] is None else b[k][0] for k in keys], dtype=float)
        hi = np.array([np.nan if b[k] is None else b[k][1] for k in keys], dtype=float)
    r = anchored_ratios(name, ratios, frame).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        above = (r >= lo - 1e-12) if band.closed[0] else (r > lo + 1e-12)
        below = (r <= hi + 1e-12) if band.closed[1] else (r < hi - 1e-12)
    inside = above & below & np.isfinite(r)
    out = pd.Series(inside.astype(float), index=ratios.index)
    out[np.isnan(lo)] = np.nan
    return out


# ── Band justification (REVISION_PLAN_REANALYSIS.md 1.7) ─────────────────────────────

def band_justification() -> tuple[pd.DataFrame, dict]:
    conv = scored_spreads()
    s = conv["spread"].dropna().to_numpy()
    med, p95, mx = float(np.median(s)), float(np.percentile(s, 95)), float(s.max())
    no_codeine = conv[~conv.from_opioid.str.contains("codeine") & ~conv.to_opioid.str.contains("codeine")]
    setters = conv[np.isclose(conv["spread"], mx)]
    info = {
        "n_conversions_with_alternative": int(len(s)),
        "disagreement_median": med, "disagreement_p95": p95, "disagreement_max": mx,
        "k_reference_agreement": mx,
        "k_without_codeine": float(no_codeine["spread"].max()),
        "k_set_by": ", ".join(setters["conversion"]),
    }
    blo, bhi = BANDS["broad"].bounds("x", "1", "y")
    bfold = min(bhi, 1 / blo)
    info["broad_fold_half_width"] = float(bfold)
    info["broad_covers_any_standard"] = bool(bfold >= mx)
    info["broad_interpretation"] = (
        f"broad ({blo:g}–{bhi:g}×, at least {bfold:.2f}-fold either way) is just wider than the largest "
        f"disagreement between current standards ({mx:.2f}-fold), so an answer copying any current standard, "
        f"Nielsen included, is always within broad of every other one: broad accuracy means within the range "
        f"the published standards disagree over, plus a margin")
    rows = []
    for name, (lo, hi) in [(b, BANDS[b].bounds("oral_morphine", "1", "oral_morphine")) for b in MAIN_BANDS] + \
            [("reference_agreement (superseded)", (1 / mx, mx))]:
        for side, fold in [("upper", hi), ("lower", 1 / lo)]:
            rows.append({
                "band": name, "side": side, "bound": hi if side == "upper" else lo,
                "fold_half_width": fold,
                "disagreement_median": med, "disagreement_p95": p95, "disagreement_max": mx,
                "percentile_of_disagreement": float(100 * np.mean(s <= fold + 1e-12)),
                "relation": ("narrower than the median disagreement" if fold < med - 1e-12 else
                             "between the median and the 95th percentile" if fold < p95 - 1e-12 else
                             "between the 95th percentile and the maximum" if fold < mx - 1e-12 else
                             "at or wider than the maximum disagreement"),
            })
    df = pd.DataFrame(rows)
    return df, info


def main():
    df, info = band_justification()
    out = os.path.join("results_revision", "references")
    os.makedirs(out, exist_ok=True)
    for name, d in [("band_justification.csv", df),
                    ("band_justification_summary.csv", pd.DataFrame([info]))]:
        p = os.path.join(out, name)
        d.to_csv(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
    print(df.to_string(index=False))
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
