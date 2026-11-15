"""
Experimental conditions and conversion subsets for the revision framework.

A *condition* selects the responses of one experiment arm from
``results_revision/responses_long.csv`` (experiment + flag filters) and names
the *item* (the pairing unit of the paired tests and the stratum of the CMH
rankings: the vignette scenario, or the random-dose opioid pair) and the
*cell* (the unit the original pipeline averages first: one vignette wording,
or one opioid pair).

A *subset* restricts the conversions (all; to oral morphine; opioid-to-opioid;
the conversions matched across experiments in figure 10; the DI 2 items).
Both are registries, so a new condition or subset is one entry.

    from analysis.revision.conditions import CONDITIONS, SUBSETS
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    experiment: str                 # value of responses_long.experiment
    filters: dict = field(default_factory=dict)  # column -> required value
    item: str = "scenario_key"      # pairing unit / CMH stratum
    cell: str = "vignette_id"       # unit averaged first (original pipeline)
    description: str = ""


CONDITIONS: dict[str, Condition] = {}


def register(c: Condition) -> Condition:
    if c.name in CONDITIONS:
        raise ValueError(f"condition {c.name!r} already registered")
    CONDITIONS[c.name] = c
    return c


_V = dict(experiment="vignette", item="scenario_key", cell="vignette_id")
_R = dict(experiment="random_dose", item="opioid_pair", cell="opioid_pair")
_E = dict(experiment="explain_units", item="opioid_pair", cell="opioid_pair")

# Vignettes: text vignettes (the experiment as described in the manuscript)
register(Condition("vignette_naive", "Vignettes, naive", filters=dict(
    include_nielsen=False, explain_units=False, ask_for_units=False, has_vignette_text=True), **_V))
register(Condition("vignette_nielsen", "Vignettes, Nielsen", filters=dict(
    include_nielsen=True, explain_units=False, ask_for_units=False, has_vignette_text=True), **_V))
register(Condition("vignette_nielsen_units", "Vignettes, Nielsen + units", filters=dict(
    include_nielsen=True, explain_units=True, has_vignette_text=True), **_V))
# Empty-vignette controls (analysed separately: text vs empty, figure 10)
register(Condition("vignette_empty_naive", "Empty vignettes, naive", filters=dict(
    include_nielsen=False, explain_units=False, ask_for_units=False, has_vignette_text=False), **_V))
register(Condition("vignette_empty_nielsen", "Empty vignettes, Nielsen", filters=dict(
    include_nielsen=True, explain_units=False, ask_for_units=False, has_vignette_text=False), **_V))
register(Condition("vignette_empty_nielsen_units", "Empty vignettes, Nielsen + units", filters=dict(
    include_nielsen=True, explain_units=True, has_vignette_text=False), **_V))
# Random doses (mg/day)
register(Condition("rd_naive", "Random doses, naive", filters=dict(
    include_nielsen=False, include_examples=False), **_R))
register(Condition("rd_nielsen", "Random doses, Nielsen", filters=dict(
    include_nielsen=True, include_examples=False), **_R))
register(Condition("rd_nielsen_examples", "Random doses, Nielsen + examples", filters=dict(
    include_nielsen=True, include_examples=True), **_R))
# Dosing-unit experiment (target in mg/4 hours, Nielsen supplied)
register(Condition("eu_4h_nielsen", "mg/4 h, Nielsen", filters=dict(explain_units=False), **_E))
register(Condition("eu_4h_nielsen_units", "mg/4 h, Nielsen + units explained",
                   filters=dict(explain_units=True), **_E))


@dataclass(frozen=True)
class Subset:
    name: str
    label: str
    keep: Callable[[pd.DataFrame], pd.Series]   # frame -> boolean mask
    description: str = ""


SUBSETS: dict[str, Subset] = {}


def register_subset(s: Subset) -> Subset:
    if s.name in SUBSETS:
        raise ValueError(f"subset {s.name!r} already registered")
    SUBSETS[s.name] = s
    return s


def _is_ome(df: pd.DataFrame) -> pd.Series:
    """Oral morphine is the source or the target: the pipeline's `ome` flag
    (analysis/analysis_utils.add_ome_column), used by figure 13 and the OME
    comparisons. For random doses every such pair is *to* oral morphine; three
    vignettes convert *from* oral morphine (oral to IV morphine, to oxycodone,
    to hydrocodone)."""
    return df["from_opioid"].eq("oral_morphine") | df["to_opioid"].eq("oral_morphine")


def _matched_keys(df: pd.DataFrame, same_conversion_only: bool = False) -> set:
    """(from, to, from unit, to unit) shared by naive random doses and the
    single-drug vignettes: the matched conversions of figure 10
    (analysis/cross_experiment_matched.py). With ``same_conversion_only`` the
    vignettes are limited to scenarios whose text and empty prompts ask for the
    same conversion (robustness check, author decision 2026-11-14)."""
    from analysis.revision.vignette_targets import same_conversion_scenarios
    key = ["from_opioid", "to_opioid", "from_opioid_unit", "to_opioid_unit"]
    rd = df[df["experiment"].eq("random_dose")][key].drop_duplicates()
    vg = df[df["experiment"].eq("vignette") & ~df["from_opioid"].astype(str).str.contains(",")]
    if same_conversion_only:
        vg = vg[vg["scenario_key"].isin(same_conversion_scenarios())]
    vg = vg[key].drop_duplicates()
    return set(map(tuple, rd.merge(vg, on=key)[key].to_numpy()))


def _matched_fn(same_conversion_only: bool):
    def keep(df: pd.DataFrame) -> pd.Series:
        from analysis.revision.vignette_targets import same_conversion_scenarios
        keys = _matched_keys(df.attrs.get("full_long", df), same_conversion_only)
        k = list(zip(df["from_opioid"], df["to_opioid"], df["from_opioid_unit"], df["to_opioid_unit"]))
        m = pd.Series([x in keys for x in k], index=df.index)
        if same_conversion_only and "scenario_key" in df.columns:
            m &= ~df["experiment"].eq("vignette") | df["scenario_key"].isin(same_conversion_scenarios())
        return m
    return keep


def _same_conversion(df: pd.DataFrame) -> pd.Series:
    """Vignette scenarios whose text and empty prompts ask for the same
    conversion (non-vignette rows kept)."""
    from analysis.revision.vignette_targets import same_conversion_scenarios
    if "scenario_key" not in df.columns:
        return pd.Series(True, index=df.index)
    return ~df["experiment"].eq("vignette") | df["scenario_key"].isin(same_conversion_scenarios())


DI2_AGENTS = ("methadone", "buprenorphine", "transdermal_fentanyl")


def _di2(df: pd.DataFrame) -> pd.Series:
    """Items involving methadone, buprenorphine or the fentanyl patch (DI 2)."""
    txt = df["from_opioid"].astype(str) + "|" + df["to_opioid"].astype(str)
    return txt.str.contains("|".join(DI2_AGENTS))


register_subset(Subset("all", "All conversions", lambda d: pd.Series(True, index=d.index)))
register_subset(Subset("ome", "Conversions involving oral morphine", _is_ome))
register_subset(Subset("nonome", "Opioid-to-opioid conversions", lambda d: ~_is_ome(d)))
register_subset(Subset("matched", "Conversions matched across experiments (figure 10)", _matched_fn(False)))
register_subset(Subset("matched_same_conversion",
                       "Matched conversions, scenarios where text and empty prompts ask the same conversion",
                       _matched_fn(True)))
register_subset(Subset("same_conversion", "Scenarios where text and empty prompts ask the same conversion",
                       _same_conversion))
register_subset(Subset("reduction", "Scenarios whose text asks for a clinical dose reduction",
                       lambda d: ~_same_conversion(d)))
register_subset(Subset("di2", "Methadone, buprenorphine and fentanyl-patch items", _di2))
register_subset(Subset("no_di2", "Excluding methadone, buprenorphine and fentanyl-patch items",
                       lambda d: ~_di2(d)))


def select(long: pd.DataFrame, condition: str, subset: str = "all") -> pd.DataFrame:
    """Rows of the (prepared) long table for one condition and subset."""
    c = CONDITIONS[condition]
    m = long["experiment"].eq(c.experiment)
    for col, val in c.filters.items():
        m &= long[col].eq(val)
    out = long[m]
    if subset != "all":
        out = out.copy()
        out.attrs["full_long"] = long
        out = out[SUBSETS[subset].keep(out).to_numpy()]
    return out
