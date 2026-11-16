"""The reported set (analysis/revision/reported_set.py): 3A, 3B and the tail tests compute only that set."""
import pandas as pd

from analysis.revision.reported_set import ANCHORED, ANY, REPORTED_SET
from analysis.revision.tolerances import BANDS

RR = "results_revision"


def test_set_is_consistent_with_registry():
    assert set(ANCHORED) <= set(BANDS)
    assert all(BANDS[b].anchor != "expected" for b in ANCHORED)
    assert all(BANDS[b].anchor == "expected" for b in ("tight", "broad"))


def test_distribution_summary_anchored_columns_only_on_any_rows():
    d = pd.read_csv(f"{RR}/distribution_summary.csv")
    cols = REPORTED_SET["distribution_summary"]["anchored_columns"]
    per_score, once = d[d.score != ANY], d[d.score == ANY]
    assert per_score[list(cols)].isna().all().all()
    pct = [c for c in cols if c.startswith("pct_")]
    assert once[pct].notna().all().all()
    assert once[once.model == "POOLED"]["n_ge_10x_highest"].notna().all()
    for s, conds in REPORTED_SET["distribution_summary"]["conditions_by_score"].items():
        assert set(d[d.score == s].condition) == set(conds), s
    assert set(once.condition) == set(REPORTED_SET["distribution_summary"]["anchored_conditions"])


def test_circularity_rows_are_the_reported_set():
    c = pd.read_csv(f"{RR}/circularity.csv")
    cfg = REPORTED_SET["circularity"]
    anc = c[c.outcome.isin(ANCHORED)]
    assert set(anc[anc.score == ANY].outcome) == set(ANCHORED) and set(anc[anc.score == ANY].subset) == {"all"}
    non = anc[anc.score != ANY]
    assert set(non.score) == set(cfg["anchored"]["coverage_restricted"])
    assert set(non.outcome) == set(cfg["anchored"]["coverage_restricted_outcomes"])
    acc = c[~c.outcome.isin(ANCHORED)]
    for s, subs in cfg["accuracy_grid"].items():
        assert set(acc[acc.score == s].subset) == set(subs), s
    assert set(acc.outcome) == set(cfg["accuracy_outcomes"])


def test_anchored_entries_are_score_independent():
    """The "any" rows equal what any full-coverage score gives (checked on the broad-ratio-free tail rate)."""
    from analysis.revision import stats
    L = stats.load_long()
    a = stats.model_summary(L, "nielsen", "rd_naive", "all", ["above_range"]).set_index("model")["pct_above_range"]
    b = stats.model_summary(L, "all", "rd_naive", "all", ["above_range"]).set_index("model")["pct_above_range"]
    assert (a - b).abs().max() < 1e-12


def test_tail_tests_cover_the_set():
    t = pd.read_csv(f"{RR}/safety_tests_by_model.csv")
    cfg = REPORTED_SET["safety_tests"]
    assert set(t.outcome) == set(cfg["outcomes"])
    assert {(a, b) for a, b in zip(t.cond_a, t.cond_b)} == {tuple(x) for x in cfg["tests"]}
