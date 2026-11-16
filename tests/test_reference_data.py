"""Task 1.1: reference-standard JSON files, index, loader and combined CSV."""
import json
import math

import pandas as pd

from resources.nielsen_data import NIELSEN_CONVERSIONS
from resources.reference_data import (
    COMBINED_CSV, CSV_COLUMNS, REFERENCES, combined_table, load_index,
    load_reference, load_reference_raw,
)

# The values table in REVISION_PLAN_REANALYSIS.md section 1.1 (blank = no factor).
PLAN_TABLE = """\
oral_codeine | 0.1 | 0.15 | 0.15 | 0.15 |
oral_hydrocodone | 1.2 | 1 | 1 | 1 | 1
oral_hydromorphone | 5 | 4 | 5 | 5 | 5
oral_methadone | 4.7 | | 4.7 | 4.7 |
oral_morphine | 1 | 1 | 1 | 1 | 1
oral_oxycodone | 1.5 | 1.5 | 1.5 | 1.5 | 1.25
oral_oxymorphone | 3 | 3 | 3 | 3 | 2.5
oral_tapentadol | 0.4 | 0.4 | 0.4 | 0.3 |
oral_tramadol | 0.2 | 0.1 | 0.2 | 0.2 |
sublingual_buprenorphine | 38.8 | | | 38.8 |
transdermal_buprenorphine | 2.2 | | | 2.2 |
transdermal_fentanyl | 2.7 | 2.4 | 2.4 | 2.4 |
iv_hydromorphone | 17.5 | | | | 12.5
iv_morphine | 3 | | | | 2.5
iv_fentanyl | 0.2 | | | | 0.1667"""


def _plan_rows():
    cols = list(CSV_COLUMNS.values())
    out = {}
    for line in PLAN_TABLE.splitlines():
        cells = [c.strip() for c in line.split("|")]
        cells += [""] * (1 + len(cols) - len(cells))
        out[cells[0]] = {c: (float(v) if v else None) for c, v in zip(cols, cells[1:])}
    return out


def _same(a, b):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    return b is not None and abs(float(a) - float(b)) < 1e-12


def test_index_lists_every_standard_with_required_fields():
    idx = load_index()
    assert set(idx) == set(CSV_COLUMNS)
    for name, e in idx.items():
        for field in ("file", "citation", "url", "year", "role", "verified", "notes"):
            assert field in e, (name, field)
        assert e["role"] in ("scoring", "recall_check_only")
    assert idx["cdc_2016"]["role"] == "recall_check_only"
    assert idx["nielsen"]["verified"] is True


def test_alternative_keys_exist_in_nielsen_with_same_unit_and_route_fields():
    nielsen_raw = json.load(open("resources/nielsen_ome.json"))
    for name in CSV_COLUMNS:
        if name == "nielsen":
            continue
        for key, e in load_reference_raw(name).items():
            assert key in nielsen_raw, (name, key)
            n = nielsen_raw[key]
            for field in ("administration_method", "substance", "unit", "priority"):
                assert e[field] == n[field], (name, key, field)
            assert e["LQ"] is None and e["UQ"] is None, (name, key)


def test_nielsen_loader_unchanged():
    assert load_reference("nielsen") == NIELSEN_CONVERSIONS
    assert REFERENCES["nielsen"] == NIELSEN_CONVERSIONS


def test_combined_table_matches_plan_table_exactly():
    plan = _plan_rows()
    table = combined_table().set_index("opioid_route")
    for route, row in plan.items():
        for col, v in row.items():
            assert _same(table.loc[route, col], v), (route, col, table.loc[route, col], v)
    # Every other agent: Nielsen only.
    others = table.drop(index=list(plan))
    assert others[[c for c in CSV_COLUMNS.values() if c != "Nielsen"]].isna().all().all()
    assert others["Nielsen"].notna().all()


def test_committed_csv_is_current():
    on_disk = pd.read_csv(COMBINED_CSV)
    fresh = combined_table()
    assert list(on_disk.columns) == list(fresh.columns)
    assert len(on_disk) == len(fresh)
    for col in fresh.columns:
        for a, b in zip(on_disk[col], fresh[col]):
            assert (a == b) or _same(a, b), (col, a, b)
