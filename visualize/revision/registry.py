"""
Figure registry for the revision framework.

Each entry: the figure's number and slug (as in FIGURES.md and
ComprehensiveFigureBuilder.FIGURE_METADATA), its title, whether its content
depends on the reference dose (``rescorable``: figures 1, 2, 3, 8-15), the views
it supports, and the panel specs the new (matplotlib) views draw. The bands
view of a single score is the original Bokeh builder (visualize/
comprehensive_figures.py) run on that score's results directory, so
``--scores nielsen --views bands`` reproduces the published figures; the
distribution view, multi-score comparisons (``--compare``) and non-default
bands use the panel specs. A new figure is one entry here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from visualize.revision.panels import Panel


@dataclass(frozen=True)
class FigureSpec:
    num: int
    slug: str
    title: str
    rescorable: bool
    panels: tuple = ()
    views: tuple = ("bands", "distribution")

    @property
    def stem(self) -> str:
        return f"fig{self.num:02d}_{self.slug}"


A, Q, P = "accuracy", "sequence", "precision"

FIGURES: dict[int, FigureSpec] = {f.num: f for f in [
    FigureSpec(1, "naive_vignette_bars", "Naive accuracy, vignettes", True,
               (Panel(A, ("vignette_naive",), title="Vignettes, naive"),)),
    FigureSpec(2, "cv_strip_vignettes", "Precision and accuracy, vignettes", True,
               (Panel(P, ("vignette_naive",), title="Vignettes, naive"),)),
    FigureSpec(3, "vignette_performance_profile", "Performance profile, vignettes", True,
               (Panel(P, ("vignette_naive",), title="Vignettes, naive"),)),
    FigureSpec(4, "variance_stacked", "Variance decomposition", False, views=("bands",)),
    FigureSpec(5, "spearman_variance_icc", "Wording variance vs ICC", False, views=("bands",)),
    FigureSpec(6, "icc_bubble_grid", "Scenario-level variance decomposition", False, views=("bands",)),
    FigureSpec(7, "kendall_w_summary", "Wording effects (Kendall's W)", False, views=("bands",)),
    FigureSpec(8, "naive_random_dose_bars", "Naive accuracy, random doses", True,
               (Panel(A, ("rd_naive",), title="Random doses, naive"),)),
    FigureSpec(9, "cv_strip_random_doses", "Precision and accuracy, random doses", True,
               (Panel(P, ("rd_naive",), title="Random doses, naive"),)),
    FigureSpec(10, "cross_experiment_bars", "Accuracy across prompt formats (matched conversions)", True,
               (Panel(A, ("rd_naive",), "matched", "Random doses, naive (matched)"),
                Panel(A, ("vignette_empty_naive",), "matched", "Empty vignettes, naive (matched)"),
                Panel(A, ("vignette_naive",), "matched", "Text vignettes, naive (matched)"))),
    FigureSpec(11, "nielsen_effect_combined", "Reference-grounding effect", True,
               (Panel(Q, ("vignette_naive", "vignette_nielsen"), title="Vignettes"),
                Panel(Q, ("rd_naive", "rd_nielsen"), title="Random doses"))),
    FigureSpec(12, "nielsen_ranked_random_doses", "Accuracy with Nielsen, random doses", True,
               (Panel(A, ("rd_nielsen",), title="Random doses, Nielsen"),)),
    FigureSpec(13, "nielsen_by_ome_combined", "Reference-grounding effect by conversion type", True,
               (Panel(Q, ("vignette_naive", "vignette_nielsen"), "ome", "Vignettes, involving oral morphine"),
                Panel(Q, ("vignette_naive", "vignette_nielsen"), "nonome", "Vignettes, opioid to opioid"),
                Panel(Q, ("rd_naive", "rd_nielsen"), "ome", "Random doses, to oral morphine"),
                Panel(Q, ("rd_naive", "rd_nielsen"), "nonome", "Random doses, opioid to opioid"))),
    FigureSpec(14, "explain_units_full_treatment", "Dosing-unit effect and full treatment", True,
               (Panel(Q, ("eu_4h_nielsen", "eu_4h_nielsen_units"), title="mg/4 h: units explained"),
                Panel(Q, ("rd_naive", "rd_nielsen", "eu_4h_nielsen", "eu_4h_nielsen_units"), "ome",
                      "Full treatment, to oral morphine"))),
    FigureSpec(15, "nielsen_units_ranked", "Best achievable accuracy", True,
               (Panel(A, ("eu_4h_nielsen_units",), title="Random doses (mg/4 h), Nielsen + units"),
                Panel(A, ("vignette_nielsen_units",), title="Vignettes, Nielsen + units"))),
]}


def parse_selection(text: str) -> list[int]:
    """'1,11' / '1-3,8' / 'all' -> figure numbers."""
    if text in ("", "all"):
        return sorted(FIGURES)
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [n for n in out if n not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figures {bad}; registered: {sorted(FIGURES)}")
    return out
