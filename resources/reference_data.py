"""
Loader for every published reference standard (oral-morphine-equivalent factors).

Each standard is one JSON file with the same schema and ``opioid_route`` keys as
``resources/nielsen_ome.json``; metadata lives in
``resources/reference_standards_index.json``. The JSON files are the single
source of truth: to correct a value, edit its JSON file and rerun from task 1.3.

    from resources.reference_data import REFERENCES, load_reference
    REFERENCES["cdc_2022"]["oral_codeine"]["conversion_factor"]   # 0.15

Running this module writes the combined, read-only view
``resources/reference_standards.csv`` (one row per opioid_route, one column per
standard):

    python -m resources.reference_data
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "resources" / "reference_standards_index.json"
COMBINED_CSV = ROOT / "resources" / "reference_standards.csv"

# Column names in the combined CSV (and the plan's values table).
CSV_COLUMNS = {
    "nielsen": "Nielsen",
    "cdc_2016": "CDC_2016",
    "cdc_2022": "CDC_2022",
    "nih_heal": "NIH_HEAL",
    "mcpherson_2018": "McPherson_2018",
}


def load_index() -> dict:
    with open(INDEX_PATH) as f:
        return json.load(f)


def _parse(data: dict) -> dict:
    # Same return shape as resources.nielsen_data.load_nielsen_conversions.
    return {
        key: {
            "conversion_factor": value["conversion_factor"],
            "unit": value.get("unit", "mg/day"),
            "priority": value.get("priority", None),
        }
        for key, value in data.items()
    }


def load_reference_raw(name: str) -> dict:
    """The standard's JSON entries as stored (including LQ / UQ and route fields)."""
    entry = load_index()[name]
    with open(ROOT / entry["file"]) as f:
        return json.load(f)


def load_reference(name: str) -> dict:
    """{opioid_route: {conversion_factor, unit, priority}} for one standard."""
    return _parse(load_reference_raw(name))


def scoring_standards() -> list[str]:
    """Names of the standards with role 'scoring' (CDC 2016 is recall-check only)."""
    return [k for k, v in load_index().items() if v["role"] == "scoring"]


REFERENCES: dict[str, dict] = {name: load_reference(name) for name in load_index()}


def combined_table() -> pd.DataFrame:
    """One row per opioid_route (Nielsen order), one factor column per standard."""
    routes = list(REFERENCES["nielsen"].keys())
    rows = []
    for route in routes:
        row = {"opioid_route": route}
        for name, col in CSV_COLUMNS.items():
            entry = REFERENCES.get(name, {}).get(route)
            row[col] = entry["conversion_factor"] if entry else None
        rows.append(row)
    return pd.DataFrame(rows, columns=["opioid_route", *CSV_COLUMNS.values()])


def write_combined_csv(path: Path = COMBINED_CSV) -> Path:
    tmp = Path(str(path) + ".tmp")
    combined_table().to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


if __name__ == "__main__":
    print(f"Wrote {write_combined_csv()}")
