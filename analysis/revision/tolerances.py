"""
Accuracy-band registry (task 1.7).

Each band is a name plus a lower and upper bound on observed/expected, either
fixed or computed per conversion. Statistics, figures and the report loop over
BANDS; a new band is one ``register(...)`` call plus a rerun.

    from analysis.revision.tolerances import BANDS, band_bounds
    lo, hi = band_bounds("reference_envelope", "oral_codeine", "1", "oral_morphine")

Initial bands:
  tight                0.85-1.15x (original)
  broad                0.6-1.7x   (original)
  reference_agreement  1/k-k, k = largest fold difference between any two current
                       standards on any scored conversion (computed, never hard-coded)
  reference_envelope   per conversion, 1/k_c-k_c, k_c = fold spread between that
                       conversion's most extreme current standards; blank when only
                       Nielsen covers it. The main reference-based band.

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
    name: str
    label: str
    bounds: BoundsFn
    per_conversion: bool
    description: str = ""


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


def _agreement(f, d, t):
    k = reference_agreement_k()
    return (1 / k, k)


def _envelope(f, d, t):
    k = _spread_cached(str(f), str(d), str(t))
    return None if k is None else (1 / k, k)


register(Band("tight", f"Tight ({TIGHT_RANGE_LOWER}–{TIGHT_RANGE_UPPER}×)",
              _fixed(TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER), False, "Original tight band."))
register(Band("broad", f"Broad ({BROAD_RANGE_LOWER}–{BROAD_RANGE_UPPER}×)",
              _fixed(BROAD_RANGE_LOWER, BROAD_RANGE_UPPER), False, "Original broad band."))
register(Band("reference_agreement", "Reference agreement (1/k–k×)", _agreement, False,
              "Narrowest symmetric band inside which every pair of current standards agrees on every scored conversion."))
register(Band("reference_envelope", "Reference envelope (per conversion)", _envelope, True,
              "Per conversion, the fold spread between its most extreme current standards. Main reference-based band."))

DEFAULT_BANDS = ["tight", "broad"]
MAIN_REFERENCE_BAND = "reference_envelope"


def band_bounds(name: str, from_opioid: str, from_dose: str, to_opioid: str) -> Bounds:
    return BANDS[name].bounds(str(from_opioid), str(from_dose), str(to_opioid))


def within(name: str, ratios: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """Boolean Series: ratio inside the band for each row (NaN where the band is undefined).

    `frame` needs from_opioid, from_dose and to_opioid columns aligned with `ratios`.
    A row whose ratio is missing counts as outside (as in the original pipeline).
    """
    keys = list(zip(frame["from_opioid"].astype(str), frame["from_dose"].astype(str),
                    frame["to_opioid"].astype(str)))
    if not BANDS[name].per_conversion:
        lo, hi = BANDS[name].bounds(*keys[0]) if keys else (np.nan, np.nan)
        lo = np.full(len(keys), lo); hi = np.full(len(keys), hi)
    else:
        b = {k: band_bounds(name, *k) for k in set(keys)}
        lo = np.array([np.nan if b[k] is None else b[k][0] for k in keys])
        hi = np.array([np.nan if b[k] is None else b[k][1] for k in keys])
    r = pd.to_numeric(ratios, errors="coerce").to_numpy(dtype=float)
    inside = (r >= lo - 1e-12) & (r <= hi + 1e-12)
    out = pd.Series(inside.astype(float), index=ratios.index)
    out[np.isnan(lo)] = np.nan
    return out


# ── Band justification (REVISION_PLAN.md 1.7) ─────────────────────────────

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
    rows = []
    for name in ["tight", "broad", "reference_agreement"]:
        lo, hi = BANDS[name].bounds("oral_morphine", "1", "oral_morphine")
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
