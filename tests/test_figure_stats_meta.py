"""Decision 48: display formats and structured fields for figure_stats.json (orchestration/extract_figure_stats.py)."""
import json
import os
import re

from orchestration import extract_figure_stats as X


def test_p_values_jmir_style():
    assert X.fmt_p(0.0) == "P<.001"
    assert X.fmt_p(0.0009) == "P<.001"
    assert X.fmt_p(0.001) == "P=.001"
    assert X.fmt_p(0.007) == "P=.007"
    assert X.fmt_p(0.03) == "P=.03"
    assert X.fmt_p(0.2439) == "P=.24"
    assert X.fmt_p(0.4043) == "P=.40"
    assert X.fmt_p(0.999) == "P>.99"
    assert "0." not in X.fmt_p(0.05)
    assert X.fmt_value(0.03, "p_caption") == "p = 0.030"
    assert X.fmt_value(0.0004, "p_caption") == "p < 0.001"
    assert X.fmt_value(0.24, "p_caption") == "p = 0.24"
    assert X.fmt_html(0.0, "p") == "<i>P</i>&lt;.001"
    assert X.fmt_html(0.03, "p") == "<i>P</i>=.03"


def test_percent_count_ratio_text():
    assert X.fmt_value(28.8, "percent:1") == "28.8%"
    assert X.fmt_value(100.0, "percent:0") == "100%"
    assert X.fmt_value(40.1, "pp:1") == "40.1 pp"
    assert X.fmt_value(13, "count") == "13"
    assert X.fmt_value(12.0, "count") == "12"
    assert X.fmt_value(0.7, "ratio:2") == "0.70"
    assert X.fmt_value(0.055, "ratio:3") == "0.055"
    assert X.fmt_value("gemini-2.5-pro", "text") == "gemini-2.5-pro"
    assert X.fmt_value("?", "percent:1") == "?"


def test_meta_covers_every_key_and_adds_bonferroni():
    stats = json.load(open("figures_nielsen/figure_stats.json"))   # extended map: base keys + *_bonf
    base = {k: v for k, v in stats.items() if k in X.KEY_META}      # the 83 submitted keys
    assert len(base) == 83
    bonf = X.bonferroni_stats()
    assert bonf["n_sig_rho_models_bonf"] == 5
    assert bonf["nielsen_n_sig_tight_vignettes_bonf"] == 5
    assert "icc_total_sig_pairs_bonf" not in bonf          # already adjusted within each cell
    assert {k: stats[k] for k in bonf} == bonf             # the flat map carries the Bonferroni keys
    meta = X.build_meta({**base, **bonf})
    assert len(meta) == len(base) + len(bonf)
    assert {k: stats[k] for k in bonf} == bonf
    assert meta["n_sig_rho_models_bonf"]["adjusted"] == "Bonferroni" and meta["n_sig_rho_models"]["adjusted"] == "unadjusted"
    assert meta["naive_rank_spearman_p"]["display"] == "P=.007"
    for k, v in meta.items():
        for f in ("format", "display", "measure", "experiment", "condition", "reference", "statistic", "adjusted", "figure"):
            assert v[f] != "", (k, f)
        assert X.make_description(v)


def test_unknown_key_rejected():
    try:
        X.build_meta({"no_such_key": 1})
    except KeyError:
        return
    raise AssertionError("unknown key accepted")


# ---- decision 47: every number programmatic; no stray digits (shared whitelist: orchestration/figure_stats_text.py) --------
import ast
import glob

from orchestration import figure_stats_text as F

STATS_DIR = "figures_nielsen"
STATS_PRIV = "revision_private/figures_appendix/stats"
DRAFTS = "revision_private/figures_appendix/drafts"


def test_stray_digit_checker():
    assert F.stray_digits("Mean {a}% of 14 models; within 0.6–1.7× (Figure 11, ±15%, p < 0.05)") != []
    assert F.stray_digits("Mean {a}% of {n_models} models; within 0.6–1.7× (Figure 11, ±15%, p < 0.05); CDC 2022; 13 models") == []
    assert F.stray_digits("half (½×), 2× and 4× highest; 0.6 ≈ 1/1.7") == []
    st, me = {"a": 28.8}, {"a": {"format": "percent:1"}}
    assert F.render("x {a}", st, me) == "x 28.8%"
    for bad in (float("nan"), None, "", "?"):
        try:
            F.render("x {a}", {"a": bad}, me)
        except (ValueError, TypeError, KeyError):
            continue
        raise AssertionError(f"bad value {bad!r} rendered")
    try:
        F.render("x {nokey}", st, me)
    except KeyError:
        pass
    else:
        raise AssertionError("missing key rendered")
    assert F.unexplained_numbers("value 28.8% and 99.9", {"a": 28.8}, me)[0][0] == "99.9"


def test_figure_stats_complete_and_clean():
    """Every key has a value (no missing, None, NaN or empty), a format, a source and structured fields."""
    stats, meta = F.load(STATS_DIR)
    assert F.check_stats(stats) == []
    assert set(meta) == set(stats)
    for k, m in meta.items():
        for f in ("format", "display", "figure", "measure", "statistic", "adjusted", "source"):
            assert m.get(f) not in (None, ""), (k, f)
        assert "nan" not in m["display"].lower() and m["display"] != "?", k
    assert len(stats) >= 200 and "ma1_n" in stats and "f9_sign_test_p" in stats and "std_match_rd_chance_pct" in stats


def test_caption_templates_have_no_stray_digits():
    p = os.path.join(STATS_PRIV, "caption_templates.json")
    if not os.path.exists(p):          # private file; absent in a public checkout
        return
    stats, meta = F.load(STATS_DIR)
    for it in json.load(open(p)):
        assert F.stray_digits(it["template"]) == [], (it["stem"], F.stray_digits(it["template"]))
        assert F.missing_keys(it["template"], stats) == [], it["stem"]
        assert F.render(it["template"], stats, meta) == it["filled"], it["stem"]


def _pdf_text(path):
    import PyPDF2
    return "\n".join(pg.extract_text() or "" for pg in PyPDF2.PdfReader(path).pages)


def test_repo_pdf_caption_text_numbers_come_from_figure_stats():
    """The caption / annotation text of the rebuilt all_comprehensive_figures.pdf: every number is a key's display
    or a whitelisted definition (the page header and the stats listing of the PDF carry none of their own)."""
    p = f"{DRAFTS}/rebuild_2026-11-16/all_comprehensive_figures.pdf"
    if not os.path.exists(p):
        return
    stats, meta = F.load(STATS_DIR)
    text = "\n".join(F.prose_lines(_pdf_text(p), min_words=5))
    u = F.unexplained_numbers(text, stats, meta)
    assert u == [], u[:10]


def test_summary_tables_builder_has_no_typed_numbers():
    """accuracy_summary_tables.pdf is computed from CSVs: its builder's literal text holds no typed number
    (HTML, CSS and format specifications aside)."""
    tree = ast.parse(open("visualize/summary_tables.py").read())
    doc = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(n)}
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in doc and n.value != "1":
            t = re.sub(r"<style.*?</style>", " ", n.value, flags=re.S)
            t = re.sub(r"<[^>]*>", " ", t)
            t = re.sub(r"\{[^{}]*\}", " ", t)
            t = re.sub(r"&[a-z]+;|#[0-9a-fA-F]{3,6}\b|^[+-]?\.\d[a-z]$|\.\d+[a-z]\b|\b\w+_\w+\.csv|px|rem|em\b", " ", t)
            bad += [(n.lineno, d) for d in F.stray_digits(t)]
    assert bad == [], bad[:8]


def test_latest_yian_pdf_built_after_the_templates():
    """A Yian PDF built after the current caption templates must contain only numbers from figure_stats (or the
    whitelist, or the commit / tag / decision labels it quotes)."""
    tp = os.path.join(STATS_PRIV, "caption_templates.json")
    pdfs = sorted(glob.glob(f"{DRAFTS}/figures_for_yian_*.pdf"), key=os.path.getmtime)
    if not (os.path.exists(tp) and pdfs) or os.path.getmtime(pdfs[-1]) < os.path.getmtime(tp):
        return                          # stale or absent: the builder reruns it after the templates change
    stats, meta = F.load(STATS_DIR)
    text = _pdf_text(pdfs[-1])
    labels = [r"\b[0-9a-f]{7}\b", r"submitted-\d{4}-\d{2}", r"\b(?:decisions?|DZ|DI)\s?\d+(?:\s?[–-]\s?\d+)?",
              r"\b\d{4}-\d{2}-\d{2}\b", r"\bv\d+\b", r"\b\d+ pages?\b",
              r"\bvignettes \d+(?:, \d+)*"]
    u = F.unexplained_numbers(text, stats, meta, extra_allowed=labels)
    assert u == [], u[:10]
