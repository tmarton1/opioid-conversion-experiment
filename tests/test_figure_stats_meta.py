"""Decision 48: display formats and structured fields for figure_stats.json (orchestration/extract_figure_stats.py)."""
import json
import os

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
    base = {k: v for k, v in stats.items() if not k.endswith("_bonf")}
    assert set(base) <= set(X.KEY_META) and len(base) == 83
    bonf = X.bonferroni_stats()
    assert bonf["n_sig_rho_models_bonf"] == 5
    assert bonf["nielsen_n_sig_tight_vignettes_bonf"] == 5
    assert "icc_total_sig_pairs_bonf" not in bonf          # already adjusted within each cell
    assert {k: stats[k] for k in bonf} == bonf             # the flat map carries the Bonferroni keys
    meta = X.build_meta({**base, **bonf})
    assert len(meta) == len(base) + len(bonf)
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


# ---- caption templates: no stray digits (decision 48c) ------------------------------------------------------
import re

# Fixed definitions allowed as literal digits in a caption template (everything else must be a {key} placeholder):
CAPTION_DIGIT_WHITELIST = [
    r"\b(?:Figure|Figures|Multimedia Appendix|Table|Panel)s? \d+",        # figure / appendix / table numbers
    r"0\.6–1\.7×", r"0\.2–5×", r"0\.1–10×", r"0\.6", r"1\.7×",             # band limits, plot limits
    r"±15%", r"p < 0\.05", r"\bwithin 1%",                                 # tight band; alpha; within-1% definition
    r"½×", r"\b[24]×",                                                     # extreme-error cutoffs (conventions)
    r"CDC 20(?:16|22)", r"McPherson 2018",                                 # standards' names
    r"mg/4 h", r"CV = 0",                                                  # dosing unit; precise-answer definition
    r"(?:GPT|Gemini|Claude|Llama|Jamba|Mistral|Mixtral)[ -][\w. -]*?\d[\w.]*", # model names
]


def stray_digits(template: str) -> list:
    """Digits left in a caption template after removing {key} placeholders and whitelisted definitions."""
    t = re.sub(r"\{[A-Za-z0-9_]+\}", "", template)
    for pat in CAPTION_DIGIT_WHITELIST:
        t = re.sub(pat, "", t)
    return re.findall(r".{0,15}\d.{0,10}", t)


def test_stray_digit_checker():
    assert stray_digits("Mean {a}% of 13 models; within 0.6–1.7× (Figure 11, ±15%, p < 0.05)") != []
    assert stray_digits("Mean {a}% of {n_models} models; within 0.6–1.7× (Figure 11, ±15%, p < 0.05); CDC 2022") == []


def test_caption_templates_have_no_stray_digits():
    p = "revision_private/figures_appendix/stats/caption_templates.json"
    if not os.path.exists(p):          # private file; absent in a public checkout
        return
    for it in json.load(open(p)):
        assert stray_digits(it["template"]) == [], (it["stem"], stray_digits(it["template"]))
