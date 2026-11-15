"""Compare every derived CSV in two results directories (reproduction check).

Usage (from the repository root):

    python -m analysis.revision.compare_results_dirs results results_nielsen \
        --out results_revision/repro_check_nielsen.csv

Each CSV present in either directory (except the raw inputs) is classed as:

  identical            same rows and values (numeric tolerance 1e-9, relative)
  extra_models_only    equal once rows for models outside CONSISTENT_MODELS are
                       dropped from the reference copy (the 13-model filter)
  expected_filter      differs, but is an output of the ranking or per-model
                       Wilcoxon step, whose statistics change with the filter
  missing_in_new / missing_in_ref
  (an empty file on one side only counts as different)
  different            any other difference: investigate

The output CSV has one row per file with the class and a short detail string.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from common.models import CONSISTENT_MODELS

RAW = {
    "experiment_results.csv",
    "explain_units_experiment_results.csv",
    "reworded_vignettes_results.csv",
}
MODEL_COLS = ("model", "model_a", "model_b")
EXPECTED_PREFIXES = ("ranking_", "wilcoxon_")
KEEP_LABELS = {"POOLED"}


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns):
        only_a = sorted(set(a.columns) - set(b.columns))
        only_b = sorted(set(b.columns) - set(a.columns))
        return False, f"columns differ (ref only {only_a}, new only {only_b})"
    if len(a) != len(b):
        return False, f"rows {len(a)} vs {len(b)}"
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    worst = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            close = np.isclose(xv, yv, rtol=1e-9, atol=1e-12) | both_nan
            if not close.all():
                gaps = np.abs(xv - yv)[~close]
                d = np.nanmax(gaps) if np.isfinite(gaps).any() else np.nan  # nan: one side missing
                worst.append(f"{c}: {int((~close).sum())} cells, max |diff| {d:.3g}")
        else:
            xs, ys = x.astype(str), y.astype(str)
            neq = (xs != ys) & ~(x.isna() & y.isna())
            if neq.any():
                worst.append(f"{c}: {int(neq.sum())} cells")
    return (not worst), "; ".join(worst[:6])


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.sort_values(list(df.columns), kind="mergesort").reset_index(drop=True)
    except TypeError:
        return df.astype(str).sort_values(list(df.columns), kind="mergesort").reset_index(drop=True)


def _drop_extra_models(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in MODEL_COLS if c in df.columns]
    if not cols:
        return df
    ok = np.ones(len(df), dtype=bool)
    allowed = set(CONSISTENT_MODELS) | KEEP_LABELS
    for c in cols:
        ok &= df[c].isin(allowed) | df[c].isna()
    return df[ok]


def _read(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return None


def compare_file(ref_path: str, new_path: str, name: str) -> dict:
    ref, new = _read(ref_path), _read(new_path)
    if ref is None or new is None:
        if ref is None and new is None:
            return {"file": name, "class": "identical", "detail": "both empty"}
        which = "reference" if ref is None else "new"
        return {"file": name, "class": "different", "detail": f"{which} copy is empty"}
    eq, detail = _frames_equal(ref, new)
    if eq:
        return {"file": name, "class": "identical", "detail": ""}
    # Order-insensitive comparison
    if list(ref.columns) == list(new.columns) and len(ref) == len(new):
        eq_sorted, _ = _frames_equal(_sort(ref), _sort(new))
        if eq_sorted:
            return {"file": name, "class": "identical", "detail": "same rows, different order"}
    ref_f = _drop_extra_models(ref)
    if len(ref_f) != len(ref):
        eq_f, detail_f = _frames_equal(ref_f, new)
        if not eq_f and list(ref_f.columns) == list(new.columns) and len(ref_f) == len(new):
            eq_f, detail_f = _frames_equal(_sort(ref_f), _sort(new))
        if eq_f:
            return {"file": name, "class": "extra_models_only",
                    "detail": f"{len(ref) - len(ref_f)} extra-model rows dropped from reference"}
        detail = f"after dropping {len(ref) - len(ref_f)} extra-model rows: {detail_f}"
    cls = "expected_filter" if name.startswith(EXPECTED_PREFIXES) else "different"
    return {"file": name, "class": cls, "detail": detail}


def compare_dirs(ref_dir: str, new_dir: str) -> pd.DataFrame:
    ref_files = {f for f in os.listdir(ref_dir) if f.endswith(".csv")} - RAW
    new_files = {f for f in os.listdir(new_dir) if f.endswith(".csv")} - RAW
    rows = []
    for name in sorted(ref_files | new_files):
        if name not in new_files:
            rows.append({"file": name, "class": "missing_in_new", "detail": ""})
        elif name not in ref_files:
            rows.append({"file": name, "class": "missing_in_ref", "detail": ""})
        else:
            rows.append(compare_file(os.path.join(ref_dir, name), os.path.join(new_dir, name), name))
    return pd.DataFrame(rows, columns=["file", "class", "detail"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--out", help="write the per-file table here (atomic)")
    args = ap.parse_args(argv)
    df = compare_dirs(args.ref_dir, args.new_dir)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, args.out)
    with pd.option_context("display.max_colwidth", 140, "display.width", 220):
        print(df.groupby("class").size().to_string())
        print(df[df["class"] != "identical"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
