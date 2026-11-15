"""One-time migration: add has_vignette_text column to results CSVs."""
from common.config import RESULTS_DIR
import csv

CSV_FILES = [
    f"{RESULTS_DIR}/reworded_vignettes_results.csv",
    f"{RESULTS_DIR}/scenario_experiment_results.csv",
]

for csv_file in CSV_FILES:
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames)
    except FileNotFoundError:
        print(f"{csv_file}: not found, skipping")
        continue

    if "has_vignette_text" in fieldnames:
        print(f"{csv_file}: already has has_vignette_text column")
        continue

    if "include_vignette" in fieldnames:
        idx = fieldnames.index("include_vignette")
        fieldnames[idx] = "has_vignette_text"
        for row in rows:
            row["has_vignette_text"] = row.pop("include_vignette", "True")
        print(f"{csv_file}: renamed include_vignette -> has_vignette_text ({len(rows)} rows)")
    else:
        idx = fieldnames.index("explain_units") + 1
        fieldnames.insert(idx, "has_vignette_text")
        for row in rows:
            row["has_vignette_text"] = "True"
        print(f"{csv_file}: added has_vignette_text column ({len(rows)} rows)")

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
