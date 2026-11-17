"""Decision 48: orchestration/check_manuscript_numbers.py on a small synthetic .docx (no real manuscript content)."""
import json
import os
import tempfile
import zipfile
from xml.sax.saxutils import escape

from orchestration.extract_figure_stats import fmt_value

from orchestration import check_manuscript_numbers as C

NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _p(text, style=None, sup=None, bold=False, sz=None):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    rpr = "<w:rPr>" + ("<w:b/>" if bold else "") + (f'<w:sz w:val="{sz}"/>' if sz else "") + "</w:rPr>"
    r = f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    if sup:
        r += f'<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>{sup}</w:t></w:r>'
    return f"<w:p>{ppr}{r}</w:p>"


def _docx(path, body, footnotes=None, styles=True):
    st = ('<w:styles %s><w:style w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
          '<w:style w:styleId="Heading2"><w:name w:val="heading 2"/></w:style></w:styles>' % NS)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f"<w:document {NS}><w:body>{body}</w:body></w:document>")
        if styles:
            z.writestr("word/styles.xml", st)
        if footnotes:
            z.writestr("word/footnotes.xml", f'<w:footnotes {NS}><w:footnote w:id="1">{_p(footnotes)}</w:footnote></w:footnotes>')


def _stats(d):
    cur = {"mean_x_pct": 28.8, "n_sig": 9, "n_sig_bonf": 5, "p_x": 0.0, "rho": 0.7, "new_pct": 50.0}
    meta = {k: dict(value=v, format=f, display=fmt_value(v, f), measure="m", experiment="e", condition="c",
                    reference="Nielsen", statistic="s", adjusted="n/a", figure="Figure 1", scope="figure_stats")
            for (k, v), f in zip(cur.items(), ["percent:1", "count", "count", "p", "ratio:2", "percent:1"])}
    sub = {"mean_x_pct": 27.0, "n_sig": 12, "p_x": 0.0, "rho": 0.7, "new_pct": 50.0}
    for n, o in (("figure_stats_revision.json", cur), ("figure_stats_revision_meta.json", meta), ("figure_stats_submitted.json", sub)):
        json.dump(o, open(os.path.join(d, n), "w"))


def _run(body, footnotes=None):
    with tempfile.TemporaryDirectory() as d:
        _stats(d)
        _docx(os.path.join(d, "m.docx"), body, footnotes)
        return C.check(os.path.join(d, "m.docx"), C.Stats(d))


def test_statuses_parts_and_whitelist():
    body = (_p("Title page 12345", None) + _p("Abstract", "Heading1")
            + _p("The mean was 28.8% in 2019 [3] (Figure 2) with 13 models and band 0.85-1.15.", sup="7")
            + _p("Results", "Heading1") + _p("Sub", "Heading2")
            + _p("The old mean 27.0% and 12 models differed; P<.001 and 0.70 and 999 appear. Table 3 shows 5%.")
            + _p("Figure 4: legend with 9 models.") + _p("References", "Heading1") + _p("Ref 77 and 88."))
    rows = _run(body, footnotes="A footnote with 12 models.")
    by = {(r["number"], r["part"]): r for r in rows}
    assert not any(r["number"] in ("2019", "3", "13", "0.85", "1.15", "7", "12345", "77", "88") and r["part"] != "footnote 1"
                   for r in rows)
    assert by[("28.8%", "abstract")]["status"] == "matches current"
    assert by[("27.0%", "body")]["status"] == "stale" and "mean_x_pct" in by[("27.0%", "body")]["stale_keys"]
    assert by[("12", "body")]["status"] == "stale"
    assert by[("P<.001", "body")]["status"] == "matches current"
    assert by[("0.70", "body")]["status"] == "matches current"
    assert by[("999", "body")]["status"] == "no match"
    assert by[("5%", "body")]["section"] == "Results > Sub"
    assert by[("9", "figure legend 4")]["status"] == "matches current"      # n_sig = 9
    assert by[("12", "footnote 1")]["status"] == "stale"
    assert [r["status"] for r in rows] == sorted((r["status"] for r in rows), key=lambda s: C.STATUS_ORDER[s])


def test_ambiguous_and_table_cells():
    tbl = ('<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Measure</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>'
           '<w:tr><w:tc><w:p><w:r><w:t>Mean</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>50.0%</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')
    body = _p("Abstract", "Heading1") + _p("Table 1: Something") + tbl + _p("Five of 5 in total; 5 sig.")
    rows = _run(body)
    cell = [r for r in rows if r["part"] == "table 1"][0]
    assert cell["number"] == "50.0%" and "row: Mean" in cell["sentence"] and "column: Value" in cell["sentence"]
    assert cell["status"] == "matches current"


def test_heading_heuristic_without_styles():
    body = (_p("Abstract", bold=True, sz=32) + _p("Intro", bold=True, sz=32) + _p("Plain text with 28.8% here."))
    with tempfile.TemporaryDirectory() as d:
        _stats(d)
        _docx(os.path.join(d, "m.docx"), body, styles=False)
        rows = C.check(os.path.join(d, "m.docx"), C.Stats(d))
    assert rows[0]["section"] == "Intro" and rows[0]["status"] == "matches current"


def test_csv_is_written_atomically_and_sorted():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "o.csv")
        C.write_csv_atomic(p, [dict(status="x", number="1")])
        assert open(p).read().splitlines()[0].startswith("status,number") and not os.path.exists(p + ".tmp")
