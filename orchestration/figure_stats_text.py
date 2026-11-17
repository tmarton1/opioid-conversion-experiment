"""Decision 47: render text from {key} templates filled with figure_stats.json, and the one shared check that
no digit appears in rendered text unless it came from a placeholder or is a whitelisted fixed definition.

    from orchestration.figure_stats_text import render, load, stray_digits, check_stats, missing_keys

    stats, meta = load("figures_nielsen")            # figure_stats.json + figure_stats_meta.json (flat map + displays)
    text = render("Mean {naive_vig_mean_tight_pct} of {n_models} models", stats, meta)
    assert not stray_digits("Mean {naive_vig_mean_tight_pct} of 13 models")   # whitelisted: "13 models"

``render`` formats each value with its per-key format (percent keys print the percent sign themselves, so a
template never types it). ``{{`` and ``}}`` are not special: braces in a template always delimit keys.
No manuscript content lives in this file.
"""
from __future__ import annotations

import json
import math
import os
import re

KEY_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")

# ── the one whitelist of literal digits allowed in rendered text (everything else must come from a key) ──────
# Each entry is a regular expression; matched text is removed before the digit check.
WHITELIST = [
    # figure, appendix, table, panel and section numbers
    r"\b(?:Figures?|Fig\.|Multimedia Appendix|Multimedia Appendices|Appendix|Tables?|Panels?|Sections?|DZ|DI)\s?\d+(?:\s?[–-]\s?\d+)?",
    # the two accuracy bands and the extreme-error cutoffs (definitions, not results)
    r"±\s?15\s?%", r"0\.6\s?[–-]\s?1\.7\s?×?", r"0\.6\s?≈\s?1/1\.7", r"1/1\.7", r"\b0\.6\b", r"\b1\.7×",
    r"½\s?×", r"\b[24]\s?×", r"\bwithin 1%", r"\b1%",
    r"p\s?<\s?0\.05", r"\bP\s?<\s?\.05", r"\b0\.05\b",
    # plot limits stated in captions
    r"0\.2\s?[–-]\s?5\s?×", r"0\.1\s?[–-]\s?10\s?×",
    # the number of models (design) and fixed technical terms
    r"\b13 models\b", r"\bthirteen\b",
    r"\bmg/4 h\b", r"\bCV\s?=\s?0\b", r"\b4 h\b",
    # published standards and their years
    r"CDC 20(?:16|22)", r"McPherson 2018", r"NIH HEAL", r"\bNielsen et al\b",
    # model names with versions
    r"(?:GPT|Gemini|Claude|Llama|Jamba|Mistral|Mixtral|Qwen|DeepSeek|Command)[ -][\w.+ -]*?\d[\w.]*",
    r"\bR\+", r"\bχ²",
    # bullets and enumerations in captions: (A), (B) ... are letters; "1." list markers
    r"^\s*\d+\.\s", r"\bTop row\b",
]
_WL = [re.compile(p, re.M) for p in WHITELIST]


def load(stats_dir: str = "figures_nielsen") -> tuple[dict, dict]:
    """(figure_stats.json, figure_stats_meta.json) of a directory."""
    stats = json.load(open(os.path.join(stats_dir, "figure_stats.json")))
    meta_p = os.path.join(stats_dir, "figure_stats_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    return stats, meta


def template_keys(template: str) -> list:
    return sorted(set(KEY_RE.findall(template)))


def missing_keys(template: str, stats: dict) -> list:
    return [k for k in template_keys(template) if k not in stats]


def bad_values(stats: dict, keys=None) -> list:
    """Keys whose value is missing, None, NaN, an empty string or the placeholder '?'."""
    out = []
    for k in (keys if keys is not None else stats):
        if k not in stats:
            out.append((k, "missing"))
            continue
        v = stats[k]
        if v is None:
            out.append((k, "None"))
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out.append((k, "NaN/inf"))
        elif isinstance(v, str) and v.strip() in ("", "?", "nan", "None"):
            out.append((k, f"empty or placeholder {v!r}"))
    return out


def check_stats(stats: dict) -> list:
    """Problems in the whole flat map (the build fails on any)."""
    return bad_values(stats)


def display(key: str, stats: dict, meta: dict | None = None) -> str:
    from orchestration.extract_figure_stats import fmt_value
    fmt = (meta or {}).get(key, {}).get("format")
    if fmt is None:
        raise KeyError(f"no format for key {key!r} (figure_stats_meta.json)")
    return fmt_value(stats[key], fmt)


def render(template: str, stats: dict, meta: dict | None = None) -> str:
    """Fill {key} placeholders with the displayed values. Raises on a missing key, NaN, None or empty value."""
    miss = missing_keys(template, stats)
    if miss:
        raise KeyError(f"keys missing from figure_stats: {miss}")
    bad = bad_values(stats, template_keys(template))
    if bad:
        raise ValueError(f"unusable values: {bad}")
    return KEY_RE.sub(lambda m: display(m.group(1), stats, meta), template)


def strip_allowed(text: str) -> str:
    for rx in _WL:
        text = rx.sub(" ", text)
    return text


def stray_digits(text: str) -> list:
    """Digits left in ``text`` after removing {key} placeholders and whitelisted definitions (context snippets).
    Use on a template (placeholders removed) to prove every number is a key."""
    t = strip_allowed(KEY_RE.sub(" ", text))
    return [m.group(0) for m in re.finditer(r".{0,14}\d.{0,10}", t)]


def digits_not_from_keys(rendered: str, template: str, stats: dict, meta: dict | None = None) -> list:
    """Digits in a rendered text that neither come from a placeholder value nor are whitelisted: the template with
    placeholders removed must be clean (same rule as stray_digits)."""
    return stray_digits(template)


# ── numbers in finished text (PDF caption / annotation text): each must be a displayed key value or whitelisted ──
NUM_RE = re.compile(r"(?<![\w.])(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-−+]?\d+)?")


def display_tokens(stats: dict, meta: dict) -> set:
    """Number tokens of every key's display (e.g. '28.8%' -> '28.8'; 'P<.001' -> '.001')."""
    from orchestration.extract_figure_stats import fmt_value
    out = set()
    for k, v in stats.items():
        fmt = meta.get(k, {}).get("format")
        if fmt is None or isinstance(v, bool):
            continue
        d = fmt_value(v, fmt) if not isinstance(v, str) else v
        out.update(t.replace(",", "") for t in NUM_RE.findall(d))
        if fmt.startswith(("percent", "ratio", "pp")) and not isinstance(v, str):
            out.add(f"{float(v):.0f}")      # a rounded form used in a range or a rounded sentence is a different key
    return out


def prose_lines(text: str, min_words: int = 6) -> list:
    """Lines that read as prose (at least ``min_words`` words of three or more letters), not table rows."""
    return [ln for ln in text.splitlines() if len(re.findall(r"\b[A-Za-z]{3,}\b", ln)) >= min_words]


def unexplained_numbers(text: str, stats: dict, meta: dict, extra_allowed: list | None = None) -> list:
    """Numbers in ``text`` that are neither whitelisted definitions nor the display of a figure_stats key."""
    allowed = display_tokens(stats, meta)
    t = strip_allowed(text)
    for pat in extra_allowed or []:
        t = re.sub(pat, " ", t)
    out = []
    for m in NUM_RE.finditer(t):
        tok = m.group(0).replace(",", "").rstrip(".")
        if tok not in allowed and tok.lstrip("0") not in allowed:
            out.append((tok, t[max(0, m.start() - 25):m.end() + 15].replace("\n", " ")))
    return out
