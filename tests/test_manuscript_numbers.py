"""Check that every number in the private manuscript-number tables matches its source.

Reads revision_private/part2/manuscript_numbers.csv and new_sections_numbers.csv.
Each row has a locator naming a source file, a column or key, a row filter and an
optional aggregation. The stored value must equal the value recomputed from the
source exactly. Markdown tables in qualified_claims.md and new_sections_numbers.md
are checked against the CSVs by key. The test returns without doing anything when
revision_private/ is absent. It prints only pass/fail counts.

Locator formats (fields separated by '::'):
  FILE.json::KEY
  FILE.csv::COLUMN::FILTER::AGG::SCALE
  @changed_count / @n_keys
  @v16::KIND::SCORE::SUBSET   (kind: tight, broad, mfe, gain_tight, gain_broad)
FILTER is a ';'-separated list of COLUMN OP VALUE with OP in = != < > and the
values @notna and @models13 for '='. AGG is empty (one row), mean, median, sum,
count, max, min, nunique, num (float of a text cell) or jsonlen.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIV = os.path.join(ROOT, "revision_private", "part2")
FILTER_RE = re.compile(r"^(\w+)(!=|=|<|>)(.*)$")
_CACHE: dict = {}


def _csv(path):
    if path not in _CACHE:
        _CACHE[path] = pd.read_csv(os.path.join(ROOT, path))
    return _CACHE[path]


def _models13():
    sys.path.insert(0, ROOT)
    from common.models import CONSISTENT_MODELS
    return set(CONSISTENT_MODELS)


def _apply_filter(df, filt):
    for part in [p for p in filt.split(";") if p]:
        m = FILTER_RE.match(part)
        col, op, val = m.groups()
        s = df[col]
        if val == "@notna":
            df = df[s.notna()] if op == "=" else df[s.isna()]
            continue
        if val == "@models13":
            df = df[s.isin(_models13())]
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = float(val)
            mask = {"=": s == v, "!=": s != v, "<": s < v, ">": s > v}[op]
        else:
            mask = {"=": s.astype(str) == val, "!=": s.astype(str) != val}[op]
        df = df[mask]
    return df


def _py(x):
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return str(x)


def _v16(kind, score, sub):
    sys.path.insert(0, ROOT)
    from analysis.revision import stats as st
    if "long" not in _CACHE:
        _CACHE["long"] = st.load_long()
    long = _CACHE["long"]
    bands = ["tight", "broad"]
    n = st.model_summary(long, score, "vignette_naive", sub, bands)
    g = st.model_summary(long, score, "vignette_nielsen", sub, bands, average="cell")
    n0 = st.model_summary(long, score, "vignette_naive", sub, bands, average="cell")
    return float({"tight": n.pct_tight.mean(), "broad": n.pct_broad.mean(),
                  "mfe": n.median_fold_error.mean(),
                  "gain_tight": g.pct_tight.mean() - n0.pct_tight.mean(),
                  "gain_broad": g.pct_broad.mean() - n0.pct_broad.mean()}[kind])


def resolve(loc):
    """Recompute the value a locator points to."""
    if loc == "@changed_count":
        a = json.load(open(os.path.join(ROOT, "summary_w_figures", "figure_stats.json")))
        b = json.load(open(os.path.join(ROOT, "figures_nielsen", "figure_stats.json")))
        return sum(a[k] != b[k] for k in a)
    if loc == "@n_keys":
        return len(json.load(open(os.path.join(ROOT, "summary_w_figures", "figure_stats.json"))))
    if loc.startswith("@v16::"):
        _, kind, score, sub = loc.split("::")
        return _v16(kind, score, sub)
    parts = loc.split("::")
    parts += [""] * (5 - len(parts))
    path, col, filt, agg, scale = parts[:5]
    if path.endswith(".json"):
        return _py(json.load(open(os.path.join(ROOT, path)))[col])
    df = _apply_filter(_csv(path), filt)
    if agg == "":
        assert len(df) == 1, f"{loc}: {len(df)} rows"
        v = _py(df[col].iloc[0])
    elif agg == "count":
        v = int(len(df))
    elif agg == "num":
        assert len(df) == 1, loc
        f = float(df[col].iloc[0])
        v = int(f) if f == int(f) else f
    elif agg == "jsonlen":
        assert len(df) == 1, loc
        v = len(json.loads(df[col].iloc[0]))
    else:
        s = df[col]
        v = _py({"mean": s.mean, "median": s.median, "sum": s.sum, "max": s.max,
                 "min": s.min, "nunique": s.nunique}[agg]())
    if scale:
        v = float(v) * float(scale)
    return v


def fmt(v):
    """Text form stored in the CSVs (exact: repr round-trips floats)."""
    return repr(v) if isinstance(v, float) else str(v)


def same(stored, v):
    if isinstance(v, bool):
        return str(stored) == str(v)
    if isinstance(v, int):
        return str(stored) == str(v)
    if isinstance(v, float):
        return float(stored) == v
    return str(stored) == v


def _read(name):
    return pd.read_csv(os.path.join(PRIV, name), dtype=str, keep_default_na=False)


def _md_rows(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\|\s*`([^`]+)`\s*\|(.*)\|\s*$", line)
        if m:
            cells = [c.strip() for c in m.group(2).split("|")]
            rows[m.group(1)] = cells
    return rows


def test_manuscript_numbers():
    if not os.path.isfile(os.path.join(PRIV, "manuscript_numbers.csv")):
        return
    bad, n = [], 0
    mn = _read("manuscript_numbers.csv")
    assert len(mn) > 0
    for _, r in mn.iterrows():
        for side in ("published", "new"):
            loc, stored = r[f"{side}_locator"], r[f"{side}_value"]
            if not loc:
                assert stored == "", r["key"]
                continue
            n += 1
            if not same(stored, resolve(loc)):
                bad.append((r["key"], side))
        if r["published_value"] and r["new_value"]:
            diff = r["published_value"] != r["new_value"]
            if r["changed"] != ("yes" if diff else "no"):
                bad.append((r["key"], "changed flag"))
    fs = mn[mn["key"].isin(json.load(open(os.path.join(ROOT, "summary_w_figures", "figure_stats.json"))))]
    assert len(fs) == 83
    assert int((fs["changed"] == "yes").sum()) == resolve("@changed_count")
    assert len(set(mn["key"])) == len(mn)
    keyed = {}
    ns_path = os.path.join(PRIV, "new_sections_numbers.csv")
    if os.path.isfile(ns_path):
        ns = _read("new_sections_numbers.csv")
        assert len(set(ns["key"])) == len(ns)
        for _, r in ns.iterrows():
            n += 1
            if not same(r["value"], resolve(r["locator"])):
                bad.append((r["key"], "value"))
            keyed[r["key"]] = r["value"]
    for _, r in mn.iterrows():
        keyed.setdefault(r["key"], r["new_value"] or r["published_value"])
        keyed.setdefault(r["key"] + "@published", r["published_value"])
    m = 0
    for md in ("qualified_claims.md", "new_sections_numbers.md"):
        p = os.path.join(PRIV, md)
        if not os.path.isfile(p):
            continue
        for key, cells in _md_rows(p).items():
            assert key in keyed, f"{md}: unknown key {key}"
            m += 1
            if cells[0] != keyed[key]:
                bad.append((key, md))
    print(f"manuscript numbers: {n} source checks, {m} document rows, {len(bad)} mismatches")
    assert not bad, f"{len(bad)} mismatches (first keys: {[b[0] for b in bad[:5]]})"
