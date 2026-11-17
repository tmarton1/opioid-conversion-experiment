"""Check every number in a manuscript (.docx; .doc is converted locally with macOS textutil) against the
displayed values of figure_stats (+ the submitted values, to spot stale numbers).

This file contains no manuscript content. Run from the repository root:

    venv/bin/python orchestration/check_manuscript_numbers.py \
        [--manuscript "revision_private/manuscript/<file>.docx|.doc"] \
        [--stats-dir revision_private/figures_appendix/stats] \
        [--out revision_private/figures_appendix/manuscript_number_check.csv]

Inputs (stats dir): figure_stats_revision.json (flat key -> value), figure_stats_revision_meta.json (format,
display, structured fields), figure_stats_submitted.json (submitted values, flat).

Coverage: abstract, body paragraphs, tables (w:tbl cells, with table caption and row/column context), figure
legends, footnotes and endnotes. Each number gets: part (abstract / body / table N / figure legend N / footnote k /
endnote k), section (heading path, from paragraph styles; when the file has no styles, as converted by textutil, from
bold/italic/size heuristics), the sentence, and a status:

    matches current  one key of the revision figure_stats displays this number
    stale            equals a submitted value whose key changed, and no current key displays it
    ambiguous        several current keys display it (or it also equals a changed submitted value)
    no match         no key displays it
    skipped          (not written) whitelisted: definitions, years, citations, figure/table numbers, dates, model names

Numbers in words (e.g. "thirteen") are not checked. Everything before the Abstract heading (title page) and the
reference list is skipped.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from xml.etree import ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DEFAULT_MANUSCRIPT_DIR = "revision_private/manuscript"
DEFAULT_STATS_DIR = "revision_private/figures_appendix/stats"
DEFAULT_OUT = "revision_private/figures_appendix/manuscript_number_check.csv"
COLUMNS = ["status", "number", "value", "kind", "part", "section", "sentence", "matched_keys", "stale_keys",
           "context_best_key", "note", "manuscript", "drive_modified"]
STATUS_ORDER = {"stale": 0, "no match": 1, "ambiguous": 2, "matches current": 3}


# ───────────────────────────── reading the .docx ─────────────────────────────
def ensure_docx(path: str, tmpdir: str) -> str:
    if path.lower().endswith(".docx"):
        return path
    out = os.path.join(tmpdir, "manuscript.docx")
    subprocess.run(["textutil", "-convert", "docx", "-output", out, path], check=True)
    return out


def _style_names(z: zipfile.ZipFile) -> dict:
    if "word/styles.xml" not in z.namelist():
        return {}
    root = ET.fromstring(z.read("word/styles.xml"))
    names = {}
    for st in root.iter(W + "style"):
        n = st.find(W + "name")
        if n is not None:
            names[st.get(W + "styleId")] = n.get(W + "val")
    return names


def _run_props(r):
    rp = r.find(W + "rPr")
    if rp is None:
        return False, False, 0, False
    b = rp.find(W + "b") is not None and rp.find(W + "b").get(W + "val") not in ("0", "false")
    i = rp.find(W + "i") is not None and rp.find(W + "i").get(W + "val") not in ("0", "false")
    sz = rp.find(W + "sz")
    va = rp.find(W + "vertAlign")
    return b, i, int(sz.get(W + "val")) if sz is not None else 0, (va is not None and va.get(W + "val") == "superscript")


def read_paragraph(p, styles: dict) -> dict:
    """Text (superscript runs dropped; tabs and breaks kept), style name and formatting summary."""
    text, runs = [], []
    for r in p.iter(W + "r"):
        b, i, sz, sup = _run_props(r)
        t = ""
        for ch in r:
            if ch.tag == W + "t":
                t += ch.text or ""
            elif ch.tag == W + "tab":
                t += "\t"
            elif ch.tag in (W + "br", W + "cr"):
                t += "\n"
        if sup or re.match(r"\s*(?:ADDIN|HYPERLINK|PAGEREF|REF|SEQ|EMBED)\s", t):   # superscript citations; field codes
            continue
        text.append(t)
        if t.strip():
            runs.append((b, i, sz))
    ppr = p.find(W + "pPr")
    style = ""
    if ppr is not None and ppr.find(W + "pStyle") is not None:
        sid = ppr.find(W + "pStyle").get(W + "val")
        style = styles.get(sid, sid)
    return dict(text="".join(text), style=style, runs=runs)


def heading_level(par: dict, doc_has_styles: bool) -> int:
    """0 = not a heading. Style names first; without styles, short all-bold / all-italic / large paragraphs."""
    s = par["style"].lower()
    m = re.match(r"heading\s*(\d)", s)
    if m:
        return int(m.group(1))
    if s == "title":
        return 1
    if doc_has_styles and s:
        return 0
    t = par["text"].strip()
    if not t or len(t) > 90 or t.endswith((".", ";", ",")) or re.match(r"(Table|Figure|Multimedia Appendix)\s*\d", t):
        return 0
    runs = par["runs"]
    if not runs:
        return 0
    if all(sz >= 28 for _, _, sz in runs) or (all(b for b, _, _ in runs) and max(sz for _, _, sz in runs) >= 28):
        return 1
    if all(b for b, _, _ in runs) and len(t.split()) <= 8:
        return 1
    if all(i for _, i, _ in runs):
        return 2
    return 0


def read_blocks(path: str) -> list:
    """Ordered blocks: dict(kind='para'|'cell'|'footnote'|'endnote', text, heading_level, ...)."""
    with zipfile.ZipFile(path) as z:
        styles = _style_names(z)
        body = ET.fromstring(z.read("word/document.xml")).find(W + "body")
        foot = {n: ET.fromstring(z.read(f"word/{n}.xml")) for n in ("footnotes", "endnotes") if f"word/{n}.xml" in z.namelist()}
    has_styles = bool(styles) and any(s.lower().startswith("heading") for s in styles.values())
    blocks = []

    def para(p):
        d = read_paragraph(p, styles)
        d["kind"] = "para"
        d["heading"] = heading_level(d, has_styles)
        return d

    def walk(parent, in_table=None):
        for ch in parent:
            if ch.tag == W + "p":
                blocks.append(para(ch))
            elif ch.tag == W + "tbl":
                rows = []
                for tr in ch.iter(W + "tr"):
                    rows.append([" ".join(read_paragraph(p, styles)["text"] for p in tc.iter(W + "p")).strip()
                                 for tc in tr.findall(W + "tc")])
                blocks.append(dict(kind="table", rows=rows))
            elif ch.tag in (W + "sdt", W + "sdtContent", W + "ins"):
                walk(ch)

    walk(body)
    for kind, root in foot.items():
        for note in root:
            if note.get(W + "type") in ("separator", "continuationSeparator", "continuationNotice"):
                continue
            t = " ".join(read_paragraph(p, styles)["text"] for p in note.iter(W + "p")).strip()
            if t:
                blocks.append(dict(kind="footnote" if kind == "footnotes" else "endnote",
                                   text=t, id=note.get(W + "id")))
    return blocks


# ───────────────────────────── numbers ─────────────────────────────
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
REFHEAD = (r"(?:Figures?|Figs?\.?|Tables?|Multimedia Appendix(?:es)?|Appendix|Supplementary (?:Figure|Table)s?|Panels?|"
           r"Sections?|Steps?|Equations?|Eq\.)")
MASKS = [
    r"https?://\S+|doi:\S+|\S+@\S+",                                           # urls, doi, e-mail
    r"\[\d+(?:\s*[,–-]\s*\d+)*\]",                                             # citations [1], [2-4]
    REFHEAD + r"\s*\d+[A-Za-z]?(?:\s*(?:[-–,]|and|&)\s*\d+[A-Za-z]?)*",        # figure / table / appendix numbers
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",                      # dates
    MONTHS + r"\.?\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}\s+" + MONTHS + r"(?:\s+\d{4})?|" + MONTHS + r"\s+\d{4}",
    r"\b(?:19|20)\d{2}\b(?![\d%.,]\d)(?!\s*(?:responses|prompts|items|pairs|models|%))",   # years
    r"\b(?:GPT|Claude|Gemini|Llama|Mistral|Mixtral|Jamba|Command|DeepSeek|Qwen|Grok|PaLM|Bard|Phi|Sonnet|Opus|Haiku)"
    r"(?:[- ][A-Za-z+][\w]*)*[- ]?v?\d[\w.\-:]*(?:\s?[A-Za-z]\w*)?",                       # model names and versions
    r"\b[A-Za-z][\w]*-(?:[\w.]+-)*\d[\w.:]*(?:-[A-Za-z0-9]+)*",                               # model ids (gemini-2.5-pro, gpt-4.1)
    r"\b(?:Python|pandas|NumPy|SciPy|statsmodels|Bokeh|matplotlib|scikit-learn|OpenRouter)\s+v?\d[\w.]*|\bv\d+(?:\.\d+)+\b",   # software versions
    r"\d+(?:\.\d+)?(?:\s*(?:[-–/]|to)\s*\d+(?:\.\d+)?)*\s?-?\s?(?:mg|mcg|µg|ug|mL|kg|milligrams?|micrograms?|hours?|hrs?|h|"
    r"days?|minutes?|mins?|years?|weeks?|months?)\b",                                         # doses, durations, ages (design inputs)
    r"\b(?:day|days|POD)\s+\d+\b",
    r"\b(?:13|thirteen)\s+(?:LLMs?|models?|large language models)\b",           # the number of models (design)
    r"[±]\s*15\s*%|\bwithin\s+15\s*%|\b15\s*%\s+(?:band|tolerance|range|of the (?:expected|reference))",   # tight band
    r"\b0?\.6\s*(?:[–-]|to)\s*1\.7\s*×?|\b0?\.85\s*(?:[–-]|to)\s*1\.15\s*×?",    # band limits
    r"\b[24]\s*×|½\s*×|\b0?\.5\s*×",                                            # extreme-error cutoffs
    r"\b\d+(?:st|nd|rd|th)\b",                                                  # ordinals
    r"[Pp]\s*(?:value\s*)?[<≤]\s*0?\.05\b|\balpha\s*=?\s*0?\.05\b|\b95\s*%\s*(?:CI|confidence)",  # alpha, CI level
]
NUM_RE = re.compile(r"(?<![\w.])(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-−]?\d+)?(?:\s?%)?")
P_PREFIX = re.compile(r"[Pp]\s*(?:value|-value)?\s*(?:of\s*)?(<|=|≤|>|≥)\s*$")
BAND_VALUES = {"0.85", "1.15", "0.6", "1.7"}
LIST_MARKER = re.compile(r"^\s*\(?\d+[.)]\s+")
ABBREV = ["e.g.", "i.e.", "et al.", "vs.", "Fig.", "Figs.", "approx.", "Dr.", "St.", "No.", "etc.", "cf."]


def mask(sentence: str) -> str:
    s = sentence
    for pat in MASKS:
        s = re.sub(pat, lambda m: " " * len(m.group(0)), s)
    return s


def split_sentences(text: str) -> list:
    t = text
    for a in ABBREV:
        t = t.replace(a, a.replace(".", "․"))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[“\"])|\n+|\t+", t)
    return [p.replace("․", ".").strip() for p in parts if p.strip()]


def parse_token(tok: str, before: str, after: str) -> dict | None:
    raw = tok.strip()
    pct = raw.endswith("%")
    num = raw.rstrip("%").strip().replace(",", "").replace("−", "-")
    try:
        v = float(num)
    except ValueError:
        return None
    nd = len(num.split(".")[1].split("e")[0].split("E")[0]) if "." in num else 0
    kind, rel = "num", ""
    pm = P_PREFIX.search(before)
    if pm and not pct:
        kind, rel = "p", pm.group(1)
    elif pct:
        kind = "pct"
    elif re.match(r"\s*(?:pp\b|percentage points?)", after):
        kind = "pp"
    return dict(text=(f"P{rel}" + raw) if kind == "p" else raw, value=v, nd=nd, kind=kind, rel=rel)


def extract_numbers(sentence: str) -> list:
    """Numbers of a sentence outside the whitelist; each dict(text, value, nd, kind, rel)."""
    s = mask(sentence)
    s = LIST_MARKER.sub(lambda m: " " * len(m.group(0)), s)
    out = []
    for m in NUM_RE.finditer(s):
        t = parse_token(m.group(0), s[max(0, m.start() - 14):m.start()], s[m.end():m.end() + 20])
        if t is None:
            continue
        if t["kind"] == "num" and f"{t['value']:g}" in BAND_VALUES and t["nd"] >= 1:
            continue                                   # band limit written alone (0.85, 1.15, 0.6, 1.7)
        if t["kind"] == "p" and abs(t["value"] - 0.05) < 1e-12:
            continue
        out.append(t)
    return out


# ───────────────────────────── matching ─────────────────────────────
def _r(x: float, nd: int) -> float:
    return round(float(x) + 0.0, nd)


def matches(tok: dict, raw, fmt: str) -> bool:
    """Does the number as written display ``raw`` (formatted with ``fmt``)?"""
    if isinstance(raw, str):
        return False
    kind, _, arg = fmt.partition(":")
    n = int(arg) if arg else 0
    v, nd, tk = tok["value"], tok["nd"], tok["kind"]
    if kind == "percent":
        return tk in ("pct", "pp") and nd <= n and abs(_r(raw, nd) - v) < 1e-9
    if kind == "pp":
        return tk in ("pp", "pct") and nd <= n and abs(_r(raw, nd) - v) < 1e-9
    if kind == "count":
        return tk == "num" and nd == 0 and abs(round(float(raw)) - v) < 1e-9
    if kind == "ratio":
        return tk == "num" and nd <= n and abs(_r(raw, nd) - v) < 1e-9
    if kind in ("p", "p_caption"):
        if tk != "p":
            return False
        if tok["rel"] in ("<", "≤"):
            return float(raw) < v + 1e-12 and (v <= 0.001 or float(raw) < v)
        return nd >= 2 and abs(_r(raw, nd) - v) < 1e-9 or (v < 0.001 and float(raw) < 0.001)
    return False


CONTEXT_WORDS = {  # sentence word -> structured-field words
    "vignette": "vignettes", "vignettes": "vignettes", "random": "random doses", "naive": "naive", "tight": "tight",
    "broad": "broad", "nielsen": "Nielsen", "units": "unit-augmented", "bonferroni": "Bonferroni",
    "adjusted": "Bonferroni", "kendall": "kendall_w", "icc": "icc", "spearman": "rho", "mean": "mean", "median": "median",
    "best": "best model", "worst": "worst model",
}


def context_score(sentence: str, section: str, meta: dict) -> int:
    words = set(re.findall(r"[a-z]+", (sentence + " " + section).lower()))
    fields = " ".join(str(meta.get(f, "")) for f in ("measure", "experiment", "condition", "reference", "statistic", "adjusted")).lower()
    score = sum(1 for w, f in CONTEXT_WORDS.items() if w in words and f.lower() in fields)
    fig = re.findall(r"\d+", meta.get("figure", ""))
    for f in re.findall(r"(?:Figure|Fig\.?)\s*(\d+)", sentence + " " + section):
        if f in fig:
            score += 2
    return score


class Stats:
    def __init__(self, stats_dir: str):
        j = lambda n: json.load(open(os.path.join(stats_dir, n)))  # noqa: E731
        self.cur = j("figure_stats_revision.json")
        self.meta = j("figure_stats_revision_meta.json")
        self.sub = j("figure_stats_submitted.json")
        self.pools = (self.pools_for(self.cur, self.meta, False), self.pools_for(self.cur, self.meta, True))
        self.changed = {k: v for k, v in self.sub.items() if k in self.cur and self.cur[k] != v}

    @staticmethod
    def pools_for(cur, meta, caption_only):
        return [(k, cur[k], meta[k]["format"]) for k in cur if (meta[k].get("scope") == "caption-only") == caption_only]

    def current_matches(self, tok):
        hit = [k for k, raw, fmt in self.pools[0] if matches(tok, raw, fmt)]
        return hit, "figure_stats"

    def caption_matches(self, tok):
        return [k for k, raw, fmt in self.pools[1] if matches(tok, raw, fmt)]

    def stale_matches(self, tok):
        return [k for k, raw in self.changed.items() if matches(tok, raw, self.meta[k]["format"])]


def classify(tok: dict, sentence: str, section: str, S: Stats) -> dict:
    cur, _ = S.current_matches(tok)
    stale = S.stale_matches(tok)
    note = ""
    if cur and not stale:
        status = "matches current" if len(cur) == 1 else "ambiguous"
    elif cur and stale:
        status, note = "ambiguous", "also equals the submitted value of a key that changed"
    elif stale:
        status = "stale"
    else:
        status = "no match"
        cap = S.caption_matches(tok)
        if cap:
            note = "matches only caption-only keys: " + ", ".join(cap[:5])
    cands = cur or stale
    best = ""
    if cands:
        sc = sorted(((context_score(sentence, section, S.meta[k]), k) for k in cands), reverse=True)
        best = sc[0][1] if sc[0][0] > 0 and (len(sc) == 1 or sc[0][0] > sc[1][0]) else ""
        if status == "matches current" and sc[0][0] == 0:
            note = "single value match, no context overlap with the key: check it is the same quantity"
        if status == "matches current" and tok["kind"] in ("pct", "pp") and "." not in tok["text"] and S.meta[cands[0]]["format"].split(":")[-1] not in ("0",):
            note = (note + "; " if note else "") + "matches after rounding to a whole number"
    return dict(status=status, matched_keys="; ".join(f"{k}={S.meta[k]['display']}" for k in cur),
                stale_keys="; ".join(f"{k}: {fmt_old(S, k)} -> {S.meta[k]['display']}" for k in stale),
                context_best_key=best, note=note)


def fmt_old(S: Stats, k: str) -> str:
    from orchestration.extract_figure_stats import fmt_value
    return fmt_value(S.sub[k], S.meta[k]["format"])


# ───────────────────────────── the walk ─────────────────────────────
def check(path: str, S: Stats) -> list:
    blocks = read_blocks(path)
    rows = []
    heads = {}                    # level -> text
    seen_abstract = False
    skipping = False
    table_no = None
    table_after = None            # table number awaiting a flattened table paragraph
    last_table_caption = None

    def section():
        return " > ".join(heads[l] for l in sorted(heads))

    def emit(text, part, ctx=""):
        for sent in split_sentences(text):
            for tok in extract_numbers(sent):
                r = classify(tok, sent, section() + " " + ctx, S)
                rows.append(dict(number=tok["text"], value=tok["value"], kind=tok["kind"], part=part,
                                 section=section(), sentence=(ctx + " | " if ctx else "") + sent, **r))

    for b in blocks:
        if b["kind"] == "para":
            t = b["text"].strip()
            if b["heading"]:
                lvl = b["heading"]
                heads = {l: h for l, h in heads.items() if l < lvl}
                heads[lvl] = t
                low = t.lower().strip(" :")
                if low == "abstract":
                    seen_abstract = True
                skipping = low in ("references", "reference list", "bibliography")
                table_after = None
                continue
            if not seen_abstract or skipping or not t:
                continue
            m = re.match(r"(Table|Figure|Multimedia Appendix)\s*(\d+)", t)
            if m:
                if m.group(1) == "Table":
                    last_table_caption = m.group(2)
                    table_after = m.group(2)
                    emit(t, f"table {m.group(2)} caption")
                else:
                    table_after = None
                    emit(t, f"figure legend {m.group(2)}" if m.group(1) == "Figure" else f"{m.group(1).lower()} {m.group(2)} legend")
                continue
            if table_after and b["runs"] and max(sz for _, _, sz in b["runs"]) <= 20:
                emit(t, f"table {table_after}")        # table flattened to one paragraph (textutil)
                table_after = None
                continue
            table_after = None
            part = "abstract" if "abstract" in section().lower() else "body"
            emit(t, part)
        elif b["kind"] == "table" and seen_abstract and not skipping:
            rows_ = b["rows"]
            header = rows_[0] if rows_ else []
            tno = last_table_caption or "?"
            for ri, row in enumerate(rows_):
                for ci, cell in enumerate(row):
                    if not cell:
                        continue
                    ctx = f"table {tno}, row: {row[0] if row else ''}; column: {header[ci] if ci < len(header) else ''}"
                    emit(cell, f"table {tno}", ctx if ri else f"table {tno}, header")
        elif b["kind"] in ("footnote", "endnote") and seen_abstract:
            emit(b["text"], f"{b['kind']} {b.get('id', '')}")
    rows.sort(key=lambda r: (STATUS_ORDER[r["status"]], r["part"], r["section"]))
    return rows


def write_csv_atomic(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    os.replace(tmp, path)


def find_manuscript(d: str) -> tuple:
    """(path, name, drive modified time): the file named in manuscript_manifest.json (written by
    revision_private/fetch_manuscript.py); fallback: the newest .doc/.docx in the folder."""
    mf = os.path.join(d, "manuscript_manifest.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        p = m.get("local_path") or os.path.join(d, m["name"])
        if not os.path.isabs(p) and not os.path.exists(p):
            p = os.path.join(d, m["name"])
        if os.path.exists(p):
            return p, m.get("name", os.path.basename(p)), m.get("modifiedTime", "")
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".docx", ".doc")) and not f.startswith("~")]
    p = max(files, key=os.path.getmtime)
    return p, os.path.basename(p), "(no manifest)"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript")
    ap.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.manuscript:
        src, name, mod = a.manuscript, os.path.basename(a.manuscript), ""
    else:
        src, name, mod = find_manuscript(DEFAULT_MANUSCRIPT_DIR)
    tmp = tempfile.mkdtemp(prefix="tmp_docx_", dir=os.path.dirname(os.path.abspath(a.out)) if src.lower().endswith(".doc") else None)
    try:
        rows = check(ensure_docx(src, tmp), Stats(a.stats_dir))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for r in rows:
        r["manuscript"], r["drive_modified"] = name, mod
    write_csv_atomic(a.out, rows)
    print("manuscript:", name, "| Drive modified:", mod)
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"{len(rows)} numbers:", ", ".join(f"{k} {c.get(k, 0)}" for k in STATUS_ORDER), "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
