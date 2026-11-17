"""
Generate an HTML summary of accuracy-comparison and model-ranking results,
then convert to PDF.

Usage
-----
    from visualize.summary_tables import SummaryTablesBuilder
    builder = SummaryTablesBuilder()
    await builder.save(
        "summary_w_figures/accuracy_summary_tables.html",
        "summary_w_figures/accuracy_summary_tables.pdf",
    )
"""

from common.config import RESULTS_DIR
import html as _html
import os
import pandas as pd
import numpy as np

# Revision (decision 45): Bonferroni-adjusted p beside the unadjusted p (families from
# results_revision/multiple_comparisons_tests.csv), 0.6-1.7x-only groups beside the published
# ranks, condition names "naive" / "reference-grounded" / "unit-augmented", and sections for
# median fold error, scoring against the other references, range rates and extreme errors.
# OPIOID_FIGURES_SUBMITTED=1 reproduces the submitted tables.
SUBMITTED = os.environ.get("OPIOID_FIGURES_SUBMITTED", "") == "1"
REV_DIR = "results_revision"
_MC = {}


def _mc_family(fam: str) -> dict:
    """{unit: (p, p_bonf)} for one multiple-comparisons family ({} if unavailable)."""
    if fam not in _MC:
        path = os.path.join(REV_DIR, "multiple_comparisons_tests.csv")
        out = {}
        if os.path.exists(path):
            if "_df" not in _MC:
                _MC["_df"] = pd.read_csv(path)
            d = _MC["_df"][_MC["_df"]["family"] == fam]
            out = {u: (float(p), float(b)) for u, p, b in zip(d["unit"], d["p"], d["p_bonf"])}
        _MC[fam] = out
    return _MC[fam]


def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def _mark(p, pb) -> str:
    """★ significant after Bonferroni within the family; ☆ p < 0.05 unadjusted only."""
    if pb is not None and np.isfinite(pb) and pb < 0.05:
        return "★"
    if p is not None and np.isfinite(p) and p < 0.05:
        return "☆"
    return ""


RENAMES = [  # DZ 6 condition names (revision mode only)
    ("Effect of Nielsen Reference", "Effect of reference grounding (Nielsen's factors in the prompt)"),
    ("+ Nielsen + Explain Units", ", unit-augmented"),
    ("+ Nielsen", ", reference-grounded"),
    (">nielsen↑<", ">grounded↑<"), (">nielsen<", ">reference-grounded<"),
]


class SummaryTablesBuilder:
    """
    Reads the comparison CSVs and ranking CSVs and renders them as styled
    HTML tables that can be converted to PDF.
    """

    COMPARISON_SPECS = [
        {
            "csv": f"{RESULTS_DIR}/comparison_text_vs_empty.csv",
            "title": "Vignette Text vs Empty (no clinical context)",
            "a_label": "text",
            "b_label": "empty",
            "hold_constant": ["include_nielsen", "explain_units"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_nielsen_effect_vignettes.csv",
            "title": "Effect of Nielsen Reference — Vignettes (All)",
            "a_label": "nielsen",
            "b_label": "naive",
            "hold_constant": ["has_vignette_text", "explain_units"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_nielsen_effect_vignettes_ome.csv",
            "title": "Effect of Nielsen Reference — Vignettes (Oral Morphine Only)",
            "a_label": "nielsen",
            "b_label": "naive",
            "hold_constant": ["has_vignette_text", "explain_units"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_nielsen_effect_vignettes_nonome.csv",
            "title": "Effect of Nielsen Reference — Vignettes (Non–Oral Morphine)",
            "a_label": "nielsen",
            "b_label": "naive",
            "hold_constant": ["has_vignette_text", "explain_units"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_units_effect_vignettes.csv",
            "title": "Effect of Explaining Units — Vignettes",
            "a_label": "explain_units",
            "b_label": "no_units",
            "hold_constant": ["has_vignette_text", "include_nielsen"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_nielsen_effect_random_dose.csv",
            "title": "Effect of Nielsen Reference — Random Doses (All)",
            "a_label": "nielsen",
            "b_label": "naive",
            "hold_constant": ["include_examples"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_nielsen_effect_random_dose_ome.csv",
            "title": "Effect of Nielsen Reference — Random Doses (Oral Morphine Only)",
            "a_label": "nielsen",
            "b_label": "naive",
            "hold_constant": ["include_examples"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_nielsen_effect_random_dose_nonome.csv",
            "title": "Effect of Nielsen Reference — Random Doses (Non–Oral Morphine)",
            "a_label": "nielsen",
            "b_label": "naive",
            "hold_constant": ["include_examples"],
        },
        {
            "csv": f"{RESULTS_DIR}/comparison_units_effect_random_dose.csv",
            "title": "Effect of Explaining Units — Random Doses",
            "a_label": "explain_units",
            "b_label": "no_units",
            "hold_constant": [],
        },
    ]

    RANKING_CSVS = [
        (f"{RESULTS_DIR}/ranking_naive_vignettes.csv", "Model Ranking: Naive Vignettes"),
        (f"{RESULTS_DIR}/ranking_vignettes_nielsen_units.csv", "Model Ranking: Vignettes + Nielsen + Explain Units"),
        (f"{RESULTS_DIR}/ranking_naive_random_doses.csv", "Model Ranking: Naive Random Doses"),
        (f"{RESULTS_DIR}/ranking_random_doses_nielsen.csv", "Model Ranking: Random Doses + Nielsen"),
        (f"{RESULTS_DIR}/ranking_random_doses_nielsen_units.csv", "Model Ranking: Random Doses + Nielsen + Explain Units"),
    ]

    CSS = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            margin: 30px;
            color: #1a1a2e;
            background: #fff;
        }
        h1 { font-size: 22px; margin-top: 40px; border-bottom: 2px solid #333; padding-bottom: 6px; }
        h2 { font-size: 17px; margin-top: 30px; color: #444; }
        h3 { font-size: 14px; margin-top: 20px; color: #555; }
        table {
            border-collapse: collapse;
            margin: 10px 0 20px 0;
            font-size: 12px;
            min-width: 500px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 5px 10px;
            text-align: right;
        }
        th {
            background-color: #f0f0f0;
            font-weight: 600;
            text-align: center;
        }
        td:first-child, th:first-child {
            text-align: left;
        }
        tr:nth-child(even) { background-color: #fafafa; }
        .sig-better { background-color: #d4edda; font-weight: 600; }
        .sig-worse { background-color: #f8d7da; }
        .ns { color: #888; }
        .step-sig td { border-top: 3px solid #28a745 !important; }
        .summary-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px 18px;
            margin: 10px 0;
            font-size: 13px;
        }
        .verdict-box {
            border-radius: 6px;
            padding: 10px 16px;
            margin: 8px 0 14px 0;
            font-size: 13px;
            font-weight: 600;
        }
        .verdict-helps {
            background: #d4edda; border: 1px solid #b1dfbb; color: #155724;
        }
        .verdict-hurts {
            background: #f8d7da; border: 1px solid #f1aeb5; color: #58151c;
        }
        .verdict-mixed {
            background: #fff3cd; border: 1px solid #ffe69c; color: #664d03;
        }
        .verdict-none {
            background: #f8f9fa; border: 1px solid #dee2e6; color: #495057;
        }
        .rank-cell { font-weight: 700; font-size: 14px; }
        .page-break { page-break-before: always; }
    </style>
    """

    def _shorten_model(self, name: str) -> str:
        """Remove provider prefix for display."""
        return name.split("/")[-1] if "/" in name else name

    # ------------------------------------------------------------------
    # Comparison summary tables
    # ------------------------------------------------------------------

    def _build_comparison_overview(self, df: pd.DataFrame, a_label: str,
                                    b_label: str) -> str:
        """Top-level summary: counts of A-better, B-better, n.s. for tight & broad,
        with both Bonferroni-corrected and uncorrected results."""
        rows = []
        for rng in ("tight", "broad"):
            result_col = f"{rng}_result"
            result_col_unc = f"{rng}_result_uncorrected"
            if result_col not in df.columns:
                continue

            n_valid = int(df[f"{rng}_p_a_better"].notna().sum())

            for label_tag, rc in [
                ("Corrected", result_col),
                ("Uncorrected", result_col_unc),
            ]:
                if rc not in df.columns:
                    continue
                n_a = int((df[rc] == f"{a_label} sig. better").sum())
                n_b = int((df[rc] == f"{b_label} sig. better").sum())
                n_ns = n_valid - n_a - n_b

                rows.append(f"""
                    <tr>
                        <td>{rng}</td>
                        <td>{label_tag}</td>
                        <td>{n_valid}</td>
                        <td class='sig-better'>{n_a} ({n_a/n_valid*100:.1f}%)</td>
                        <td class='sig-worse'>{n_b} ({n_b/n_valid*100:.1f}%)</td>
                        <td class='ns'>{n_ns} ({n_ns/n_valid*100:.1f}%)</td>
                    </tr>""")

        return f"""
        <table>
            <tr>
                <th>Range</th><th>Correction</th><th># Tests</th>
                <th>{a_label} sig. better</th>
                <th>{b_label} sig. better</th>
                <th>n.s.</th>
            </tr>
            {''.join(rows)}
        </table>"""

    def _build_verdict(self, df: pd.DataFrame, a_label: str,
                       b_label: str, comparison_csv: str = "") -> str:
        """Overall verdict using pre-computed Wilcoxon signed-rank results.

        Reads pooled Wilcoxon p-values from a companion ``wilcoxon_*.csv``
        produced by ``analysis.per_model_wilcoxon``.  Positive Δ = A more
        accurate.
        """
        from analysis.per_model_wilcoxon import lookup_pooled_p

        wilcoxon_csv = comparison_csv.replace("comparison_", "wilcoxon_") if comparison_csv else ""

        parts = []   # HTML fragments for each range
        any_sig = False
        any_helps = False
        any_hurts = False

        for rng in ("tight", "broad"):
            diff_col = f"{rng}_diff"
            if diff_col not in df.columns:
                continue

            # Basic descriptive stats can still come from the raw df
            diffs = df[diff_col].dropna()
            if len(diffs) < 2:
                continue

            n = len(diffs)
            med = float(diffs.median())
            mean = float(diffs.mean())

            # Read pre-computed p-value
            p = None
            if wilcoxon_csv and os.path.exists(wilcoxon_csv):
                p, _, _ = lookup_pooled_p(wilcoxon_csv, rng)

            if p is None:
                parts.append(f"{rng}: n={n}, Wilcoxon CSV not found")
                continue

            sig = p < 0.05
            if sig:
                any_sig = True
                if med > 0:
                    any_helps = True
                elif med < 0:
                    any_hurts = True

            p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.4f}"
            star = " *" if sig else ""
            parts.append(
                f"{rng}: median Δ = {med:+.3f}, mean Δ = {mean:+.3f}, "
                f"{p_str}{star} (n = {n})"
            )

        if not parts:
            return ""

        detail_str = "; ".join(parts)

        if not any_sig:
            css = "verdict-none"
            msg = (f"No significant overall effect of {a_label} vs {b_label} "
                   f"(Wilcoxon signed-rank). {detail_str}")
        elif any_helps and not any_hurts:
            css = "verdict-helps"
            msg = (f"{a_label} significantly improves accuracy "
                   f"(Wilcoxon signed-rank). {detail_str}")
        elif any_hurts and not any_helps:
            css = "verdict-hurts"
            msg = (f"{a_label} significantly hurts accuracy "
                   f"(Wilcoxon signed-rank). {detail_str}")
        else:
            css = "verdict-mixed"
            msg = (f"Mixed effect of {a_label} "
                   f"(Wilcoxon signed-rank). {detail_str}")

        return (
            f"<div class='verdict-box {css}'>{msg}</div>"
            f"<div class='summary-box' style='font-size:11px; color:#555;'>"
            f"<b>Wilcoxon signed-rank test</b>: Tests whether the distribution "
            f"of accuracy-proportion differences (A − B) across all model × unit "
            f"pairs is symmetric around zero. Positive Δ = {a_label} more accurate. "
            + ("* = p &lt; 0.05." if SUBMITTED else
               "Per-model verdicts give the unadjusted p and p<sub>Bonf</sub> (Bonferroni within the family: "
               "the 13 per-model tests of this comparison and measure); ★ significant after Bonferroni, "
               "☆ p &lt; 0.05 unadjusted only.")
            + "</div>"
        )

    @staticmethod
    def _wilcoxon_verdict(model: str, range_type: str, wilcoxon_csv: str,
                          diffs: pd.Series) -> tuple[str, str]:
        """Look up pre-computed per-model Wilcoxon signed-rank result.

        Returns (verdict_html, css_class) where verdict_html is a short
        string like '↑ p=1.2e-03' and css_class is one of
        'sig-better', 'sig-worse', 'ns'.

        Falls back to descriptive-only output if the Wilcoxon CSV is
        unavailable.
        """
        from analysis.per_model_wilcoxon import lookup_per_model_p

        diffs = diffs.dropna()
        if len(diffs) < 2:
            return ("—", "ns")
        med = float(diffs.median())

        p = None
        if wilcoxon_csv and os.path.exists(wilcoxon_csv):
            p, _, _ = lookup_per_model_p(wilcoxon_csv, model, range_type)

        if p is None:
            return ("—", "ns")

        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        if not SUBMITTED and wilcoxon_csv:
            fam = _mc_family(f"{os.path.relpath(wilcoxon_csv)[:-4]}:{range_type}")
            if model in fam:
                pb = fam[model][1]
                p_str += f", p<sub>Bonf</sub>={_fmt_p(pb)}{(' ' + _mark(p, pb)) if _mark(p, pb) else ''}"

        if p >= 0.05:
            return (f"n.s. (p={p_str})", "ns")
        elif med > 0:
            return (f"↑ p={p_str}", "sig-better")
        elif med < 0:
            return (f"↓ p={p_str}", "sig-worse")
        else:
            # Significant but median exactly 0
            mean = float(diffs.mean())
            arrow = "↑" if mean > 0 else ("↓" if mean < 0 else "~")
            return (f"{arrow} p={p_str}", "sig-better" if mean > 0 else "sig-worse")

    def _build_comparison_by_model(self, df: pd.DataFrame, a_label: str,
                                     b_label: str,
                                     comparison_csv: str = "") -> str:
        """Per-model breakdown table with corrected/uncorrected counts
        and per-model Wilcoxon signed-rank verdicts.

        Reads pre-computed Wilcoxon p-values from a companion
        ``wilcoxon_*.csv``.
        """
        wilcoxon_csv = comparison_csv.replace("comparison_", "wilcoxon_") if comparison_csv else ""
        models = sorted(df["model"].unique())
        header = f"""
        <table>
            <tr>
                <th rowspan='2'>Model</th>
                <th colspan='3'>Tight (corrected)</th>
                <th colspan='3'>Tight (uncorr.)</th>
                <th colspan='3'>Broad (corrected)</th>
                <th colspan='3'>Broad (uncorr.)</th>
                <th colspan='2'>Wilcoxon</th>
            </tr>
            <tr>
                <th>{a_label}↑</th><th>{b_label}↑</th><th>n.s.</th>
                <th>{a_label}↑</th><th>{b_label}↑</th><th>n.s.</th>
                <th>{a_label}↑</th><th>{b_label}↑</th><th>n.s.</th>
                <th>{a_label}↑</th><th>{b_label}↑</th><th>n.s.</th>
                <th>Tight</th><th>Broad</th>
            </tr>"""
        rows = []
        for m in models:
            mdf = df[df["model"] == m]
            cells = []
            for rng in ("tight", "broad"):
                for suffix in ("", "_uncorrected"):
                    rc = f"{rng}_result{suffix}"
                    if rc not in df.columns:
                        cells.append("<td>—</td><td>—</td><td>—</td>")
                        continue
                    a_ct = int((mdf[rc] == f"{a_label} sig. better").sum())
                    b_ct = int((mdf[rc] == f"{b_label} sig. better").sum())
                    ns_ct = int((mdf[rc] == "n.s.").sum())
                    a_cls = " class='sig-better'" if a_ct else ""
                    b_cls = " class='sig-worse'" if b_ct else ""
                    ns_cls = " class='ns'" if not a_ct and not b_ct else ""
                    cells.append(f"<td{a_cls}>{a_ct}</td><td{b_cls}>{b_ct}</td>"
                                 f"<td{ns_cls}>{ns_ct}</td>")

            # Per-model Wilcoxon verdicts
            for rng in ("tight", "broad"):
                diff_col = f"{rng}_diff"
                if diff_col in mdf.columns:
                    verdict, css = self._wilcoxon_verdict(
                        m, rng, wilcoxon_csv, mdf[diff_col])
                    cells.append(f"<td class='{css}'>{verdict}</td>")
                else:
                    cells.append("<td>—</td>")

            rows.append(f"<tr><td>{self._shorten_model(m)}</td>{''.join(cells)}</tr>")

        return header + "\n".join(rows) + "</table>"

    def _build_comparison_by_condition(self, df: pd.DataFrame, a_label: str,
                                        b_label: str,
                                        hold_constant: list[str]) -> str:
        """Breakdown by hold_constant columns."""
        if not hold_constant:
            return ""

        header_cols = ", ".join(hold_constant)
        header = f"""
        <h3>By {header_cols}</h3>
        <table>
            <tr>
                <th>{header_cols}</th><th># Pairs</th>
                <th colspan='2'>Tight</th>
                <th colspan='2'>Broad</th>
            </tr>
            <tr>
                <th></th><th></th>
                <th>{a_label} ↑</th><th>{b_label} ↑</th>
                <th>{a_label} ↑</th><th>{b_label} ↑</th>
            </tr>"""

        rows = []
        for vals, grp in df.groupby(hold_constant):
            if not isinstance(vals, tuple):
                vals = (vals,)
            label = ", ".join(f"{c}={v}" for c, v in zip(hold_constant, vals))
            cells = []
            for rng in ("tight", "broad"):
                rc = f"{rng}_result"
                a_ct = int((grp[rc] == f"{a_label} sig. better").sum())
                b_ct = int((grp[rc] == f"{b_label} sig. better").sum())
                cells.append(f"<td>{a_ct}</td><td>{b_ct}</td>")
            rows.append(
                f"<tr><td>{label}</td><td>{len(grp)}</td>{''.join(cells)}</tr>"
            )

        return header + "\n".join(rows) + "</table>"

    def _build_comparison_section(self, spec: dict) -> str:
        """Full HTML section for one comparison."""
        if not os.path.exists(spec["csv"]):
            return f"<h2>{spec['title']}</h2><p><em>File not found: {spec['csv']}</em></p>"

        df = pd.read_csv(spec["csv"])
        a_label = spec["a_label"]
        b_label = spec["b_label"]

        html = f"<h2>{spec['title']}</h2>"
        html += self._build_verdict(df, a_label, b_label,
                                    comparison_csv=spec["csv"])
        html += self._build_comparison_overview(df, a_label, b_label)
        html += self._build_comparison_by_condition(
            df, a_label, b_label, spec["hold_constant"]
        )
        html += "<h3>By Model</h3>"
        html += self._build_comparison_by_model(df, a_label, b_label,
                                                comparison_csv=spec["csv"])
        return html

    # ------------------------------------------------------------------
    # Model ranking tables
    # ------------------------------------------------------------------

    def _build_ranking_section(self, csv_path: str, title: str) -> str:
        """HTML table for one model ranking.

        Reads the ranking CSV (one row per model with adjusted_rank) and,
        if available, the companion *_pairs.csv with all-pairs results.
        """
        if not os.path.exists(csv_path):
            return f"<h2>{title}</h2><p><em>File not found: {csv_path}</em></p>"

        df = pd.read_csv(csv_path)

        # Try to load the companion pairs CSV for test counts
        pairs_csv = csv_path.replace(".csv", "_pairs.csv")
        pairs_df = pd.read_csv(pairs_csv) if os.path.exists(pairs_csv) else pd.DataFrame()
        n_unit_tests = 0
        n_sig = 0
        if not pairs_df.empty:
            # CMH approach: one test per pair per range → total = n_pairs × 2
            n_unit_tests = len(pairs_df) * 2
            if "a_sig_better" in pairs_df.columns:
                n_sig = int(pairs_df["a_sig_better"].sum())
        eff_alpha = 0.05 / n_unit_tests if n_unit_tests else float("nan")

        tight_col = "mean_tight_prop" if "mean_tight_prop" in df.columns else "tight_prop"
        broad_col = "mean_broad_prop" if "mean_broad_prop" in df.columns else "broad_prop"

        groups = {}
        if not SUBMITTED and os.path.exists(os.path.join(REV_DIR, "rankings_fig12_fig15_broad.csv")):
            fig_of = {"ranking_naive_vignettes.csv": "Figure 1", "ranking_naive_random_doses.csv": "Figure 8",
                      "ranking_random_doses_nielsen.csv": "Figure 12",
                      "ranking_random_doses_nielsen_units.csv": "Figure 15 (top)",
                      "ranking_vignettes_nielsen_units.csv": "Figure 15 (bottom)"}.get(os.path.basename(csv_path))
            rb = pd.read_csv(os.path.join(REV_DIR, "rankings_fig12_fig15_broad.csv"))
            rb = rb[rb["figure"] == fig_of]
            groups = dict(zip(rb["model"], rb["group_broad"].astype(int)))
            if groups:
                df = df.assign(_g=df["model"].map(groups)).sort_values(["_g", broad_col], ascending=[True, False])
        html = f"""
        <h2>{title}</h2>
        <div class='summary-box'>
            {len(df)} models, {len(pairs_df)} model pairs,
            {n_unit_tests} Cochran–Mantel–Haenszel tests (tight + broad),
            Bonferroni α = {eff_alpha:.2e}<br>
            {n_sig} significant superiority relationships detected
        </div>
        <table>
            <tr>
                <th>Rank</th>{"<th>Group (0.6–1.7× only)</th>" if groups else ""}<th>Model</th>
                <th>Tight %</th><th>Broad %</th>
                <th>Units</th><th>n</th>
            </tr>"""
        if groups:
            html = html.replace("<div class='summary-box'>", "<div class='summary-box'>Rows sorted by the group "
                                "from the 0.6–1.7×-only ranking (one-sided CMH, Bonferroni over model pairs), then by "
                                "% within 0.6–1.7×; Rank = the published tight-primary rank.<br>", 1)

        rows = []
        prev_rank = None
        for _, row in df.iterrows():
            rank = int(row["adjusted_rank"])
            model = self._shorten_model(row["model"])
            tight_pct = f"{row[tight_col]*100:.1f}%"
            broad_pct = f"{row[broad_col]*100:.1f}%"
            n_units = int(row["n_units"]) if "n_units" in df.columns else ""
            n = int(row["n_total"])

            # Add visual separator between rank tiers (between groups in revision mode)
            tier = groups.get(row["model"], rank) if groups else rank
            border_class = " step-sig" if prev_rank is not None and tier != prev_rank else ""
            prev_rank = tier

            rows.append(f"""
                <tr class='{border_class}'>
                    <td class='rank-cell'>{rank}</td>{f"<td class='rank-cell'>{groups.get(row['model'], '')}</td>" if groups else ""}
                    <td>{model}</td>
                    <td>{tight_pct}</td><td>{broad_pct}</td>
                    <td>{n_units}</td><td>{n}</td>
                </tr>""")

        html += "\n".join(rows) + "</table>"
        return html

    # ------------------------------------------------------------------
    # revision sections (decision 45; not in the submitted tables)
    # ------------------------------------------------------------------

    _EXP = {"vignette": "text vignettes", "random_dose": "random doses"}
    _SCORE = {"nielsen": "scored against Nielsen", "all": "scored against all current standards",
              "non_nielsen": "scored against all except Nielsen", "any": "standard-anchored (same under every score)"}
    _OUT = {"tight": "% within ±15% of the reference", "broad": "% within 0.6–1.7× of the reference",
            "median_fold_error": "median fold error (lower is better)",
            "above_range": "% above the range of published standards (> 1.7× the highest; lower is better)",
            "below_range": "% below the range of published standards (< 0.6× the lowest; lower is better)",
            "overdose": "% above 2× the highest standard (extreme error; lower is better)",
            "severe_overdose": "% above 4× the highest standard (extreme error; lower is better)",
            "underdose": "% below ½× the lowest standard (extreme error; lower is better)"}

    def _revision_table(self, circ: pd.DataFrame, exp: str, score: str, outcome: str) -> str:
        d = circ[(circ["experiment"] == exp) & (circ["subset"] == "all") & (circ["score"] == score)
                 & (circ["outcome"] == outcome)].set_index("model")
        if d.empty:
            return ""
        kind = "tf" if outcome == "median_fold_error" else "band"
        fam = _mc_family(f"circularity:{exp}:all:{score}:{outcome}:{kind}")
        pcol = "tf_p" if kind == "tf" else "band_p"
        fmt = (lambda v: f"{v:.2f}×") if outcome == "median_fold_error" else (lambda v: f"{v:.1f}%")
        rows = []
        for m in [x for x in d.index if x != "ALL_MODELS"] + (["ALL_MODELS"] if "ALL_MODELS" in d.index else []):
            r = d.loc[m]
            pooled = m == "ALL_MODELS"
            p = float(r[pcol]) if pd.notna(r[pcol]) else np.nan
            pb = np.nan if pooled else fam.get(m, (np.nan, np.nan))[1]
            gain = float(r["gain"]) if pd.notna(r["gain"]) else np.nan
            mk = "" if pooled else _mark(p, pb)
            if mk and gain < 0:
                mk = {"★": "▲", "☆": "△"}[mk]
            name = "<b>all models (pooled)</b>" if pooled else self._shorten_model(m)
            rows.append(f"<tr><td>{name}</td><td>{fmt(r['before'])}</td><td>{fmt(r['after'])}</td>"
                        f"<td>{_fmt_p(p)}</td><td>{'(pooled, unadjusted)' if pooled else _fmt_p(pb)}</td>"
                        f"<td>{mk}</td></tr>")
        m_fam = len(fam) if fam else "—"
        return (f"<h3>{self._EXP[exp].capitalize()}, {self._SCORE[score]}: {self._OUT[outcome]}</h3>"
                f"<table><tr><th>Model</th><th>naive</th><th>reference-grounded</th><th>p</th>"
                f"<th>p, Bonferroni (family: this column, m = {m_fam})</th><th></th></tr>" + "".join(rows) + "</table>")

    def _build_revision_sections(self) -> str:
        path = os.path.join(REV_DIR, "circularity.csv")
        if not os.path.exists(path):
            return ""
        circ = pd.read_csv(path)
        intro = ("<div class='summary-box'>Naive → reference-grounded (Nielsen's factors in the prompt), per model, "
                 "cell means as in Figure 11 (results_revision/circularity.csv). p: per-model paired Wilcoxon over "
                 "items (threshold-free test for median fold error); Bonferroni within the family (one table = 13 "
                 "models). ★ significant after Bonferroni; ☆ p &lt; 0.05 unadjusted only; ▲ / △ the same for a "
                 "change in the worse direction. Pooled rows: test over model × item pairs, unadjusted. The cutoffs "
                 "are conventions.</div>")
        parts = ["<div class='page-break'></div>", "<h1>Median fold error (threshold-free)</h1>", intro]
        for exp in ("vignette", "random_dose"):
            for score in ("nielsen", "non_nielsen"):
                parts.append(self._revision_table(circ, exp, score, "median_fold_error"))
        parts += ["<div class='page-break'></div>", "<h1>Scoring against the other references</h1>", intro]
        for exp in ("vignette", "random_dose"):
            for score in ("all", "non_nielsen"):
                for outcome in ("broad", "tight"):
                    parts.append(self._revision_table(circ, exp, score, outcome))
        parts += ["<div class='page-break'></div>", "<h1>Answers above and below the range of published standards</h1>",
                  intro.replace("</div>", " Above / below the range: answers outside 0.6–1.7× of every current "
                                "standard, split by direction.</div>")]
        for exp in ("vignette", "random_dose"):
            for outcome in ("above_range", "below_range"):
                parts.append(self._revision_table(circ, exp, "any", outcome))
        parts += ["<div class='page-break'></div>", "<h1>Extreme errors</h1>",
                  intro.replace("</div>", " Extreme errors: below ½× the lowest, above 2× or above 4× the highest "
                                "current standard.</div>")]
        for exp in ("vignette", "random_dose"):
            for outcome in ("underdose", "overdose", "severe_overdose"):
                parts.append(self._revision_table(circ, exp, "any", outcome))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build_html(self) -> str:
        """Generate full HTML document."""
        sections = []
        sections.append("<h1>Accuracy Comparison Summaries</h1>")
        for spec in self.COMPARISON_SPECS:
            sections.append(self._build_comparison_section(spec))

        sections.append("<div class='page-break'></div>")
        sections.append("<h1>Model Rankings by Experimental Condition</h1>")
        for csv_path, title in self.RANKING_CSVS:
            sections.append(self._build_ranking_section(csv_path, title))
        if not SUBMITTED:
            sections.append(self._build_revision_sections())
            body = "".join(sections)
            for old, new in RENAMES:
                body = body.replace(old, new)
            sections = [body]

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Accuracy Summary Tables</title>
    {self.CSS}
</head>
<body>
    {''.join(sections)}
</body>
</html>"""

    async def save(self, html_path: str, pdf_path: str):
        """Save HTML and convert to PDF."""
        from visualize.visualization_config import convert_to_pdf_auto_size

        html = self.build_html()
        with open(html_path, "w") as f:
            f.write(html)
        print(f"✓ Saved HTML to {html_path}")

        await convert_to_pdf_auto_size(html_path, pdf_path, scale=2.0)
        print(f"✓ Saved PDF to {pdf_path}")
