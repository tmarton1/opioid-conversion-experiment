"""
Grand statistical methods summary — generates a hierarchical HTML document
listing every statistical test performed in the opioid-conversion-experiment,
with observed test statistics, sample sizes, p-values, figure references,
source-code links, and results-CSV links.

Usage:
    python -m analysis.statistical_methods          # writes raw_figures/statistical_methods.html
    python -m analysis.statistical_methods --pdf     # also writes a PDF version
"""
from __future__ import annotations

import os
import re
import html as html_mod
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Config ───────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "raw_figures"

GITHUB_REPO = "https://github.com/tmarton1/opioid-conversion-experiment"
GITHUB_BRANCH = "main"

def _gh(path: str, line: int | None = None) -> str:
    """GitHub permalink."""
    url = f"{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{path}"
    if line:
        url += f"#L{line}"
    return url

def _gh_csv(filename: str) -> str:
    return _gh(f"results/{filename}")

def _gh_code(path: str, line: int | None = None) -> str:
    return _gh(path, line)

def _p(val: float, threshold: float = 0.001) -> str:
    """Format a p-value."""
    if pd.isna(val):
        return "N/A"
    if val < threshold:
        return f"< {threshold}"
    return f"{val:.4f}"

def _sig(val: float) -> str:
    """Significance stars."""
    if pd.isna(val):
        return ""
    if val < 0.001:
        return "***"
    if val < 0.01:
        return "**"
    if val < 0.05:
        return "*"
    return "ns"

def _clean_model(m: str) -> str:
    """Strip provider prefix and date suffixes."""
    m = re.sub(r"^[^/]+/", "", m)
    m = re.sub(r"[:]?\d{8,}$", "", m)
    m = re.sub(r"-\d{2}-\d{4}$", "", m)
    m = re.sub(r"-\d{4}$", "", m)
    return m


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class TestRow:
    """One row in a sub-table: a single test result."""
    context: str          # e.g. model name, scenario, pair
    test_name: str        # e.g. "Wilcoxon signed-rank"
    statistic: str        # e.g. "W = 45"
    n: str                # sample size description
    p_value: str          # formatted p-value
    significance: str     # */**/***/ ns
    detail: str = ""      # extra info

@dataclass
class SubTable:
    """A sub-table grouping related test results."""
    title: str
    description: str
    csv_file: str | None = None
    code_file: str | None = None
    code_line: int | None = None
    rows: list[TestRow] = field(default_factory=list)
    summary: str = ""

@dataclass
class Section:
    """A top-level section (maps to a figure or analysis group)."""
    title: str
    figure_name: str | None = None      # PDF page reference
    figure_page: int | None = None
    description: str = ""
    subtables: list[SubTable] = field(default_factory=list)


# ── Helper: safe CSV read ───────────────────────────────────────────

def _read(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Section builders ────────────────────────────────────────────────

def _build_ranking_section(
    title: str,
    ranking_csv: str,
    pairs_csv: str,
    figure_name: str,
    figure_page: int,
    description: str,
) -> Section:
    """CMH pairwise ranking tests."""
    sec = Section(title=title, figure_name=figure_name,
                  figure_page=figure_page, description=description)

    pairs_df = _read(pairs_csv)
    if pairs_df is None:
        return sec

    rows = []
    for _, r in pairs_df.iterrows():
        ma = _clean_model(str(r["model_a"]))
        mb = _clean_model(str(r["model_b"]))
        n_strata = int(r.get("n_strata", 0))
        tp = r.get("tight_p_bonf", r.get("tight_p", np.nan))
        bp = r.get("broad_p_bonf", r.get("broad_p", np.nan))
        sig_t = _sig(tp)
        sig_b = _sig(bp)
        # Only show significant pairs to keep table manageable
        if sig_t in ("*", "**", "***") or sig_b in ("*", "**", "***"):
            rows.append(TestRow(
                context=f"{ma} vs {mb}",
                test_name="CMH (one-sided, Bonferroni)",
                statistic=f"strata = {n_strata}",
                n=f"{n_strata} strata",
                p_value=f"tight {_p(tp)}, broad {_p(bp)}",
                significance=f"tight {sig_t}, broad {sig_b}",
            ))

    n_total = len(pairs_df)
    n_sig = len(rows)
    sub = SubTable(
        title="Cochran–Mantel–Haenszel Pairwise Model Comparisons",
        description=(
            f"One-sided CMH tests comparing all model pairs, stratified by "
            f"conversion unit (scenario or opioid pair). Bonferroni-corrected "
            f"across {n_total} pairs. Showing {n_sig}/{n_total} significant pairs."
        ),
        csv_file=pairs_csv,
        code_file="analysis/model_ranking.py",
        rows=rows,
        summary=f"{n_sig} of {n_total} pairs significant after Bonferroni correction.",
    )
    sec.subtables.append(sub)
    return sec


def _build_wilcoxon_section(
    title: str,
    wilcoxon_csv: str,
    comparison_csv: str | None,
    figure_name: str,
    figure_page: int,
    description: str,
    comparison_description: str = "",
) -> Section:
    """Wilcoxon signed-rank + Fisher's exact tests."""
    sec = Section(title=title, figure_name=figure_name,
                  figure_page=figure_page, description=description)

    # Per-model Wilcoxon
    wdf = _read(wilcoxon_csv)
    if wdf is not None:
        rows = []
        for _, r in wdf.iterrows():
            model = _clean_model(str(r["model"]))
            tp = r.get("tight_p", np.nan)
            bp = r.get("broad_p", np.nan)
            tn = int(r.get("tight_n", 0))
            bn = int(r.get("broad_n", 0))
            rows.append(TestRow(
                context=model,
                test_name="Wilcoxon signed-rank",
                statistic="",
                n=f"tight n={tn}, broad n={bn}",
                p_value=f"tight {_p(tp)}, broad {_p(bp)}",
                significance=f"tight {_sig(tp)}, broad {_sig(bp)}",
            ))

        n_sig_tight = sum(1 for _, r in wdf.iterrows()
                          if r.get("tight_p", 1) < 0.05
                          and str(r["model"]) != "POOLED")
        n_sig_broad = sum(1 for _, r in wdf.iterrows()
                          if r.get("broad_p", 1) < 0.05
                          and str(r["model"]) != "POOLED")
        n_models = sum(1 for _, r in wdf.iterrows()
                       if str(r["model"]) != "POOLED")

        sub = SubTable(
            title="Per-Model Wilcoxon Signed-Rank Tests",
            description=(
                "Wilcoxon signed-rank test on accuracy differences between "
                "conditions, per model and pooled across all models."
            ),
            csv_file=wilcoxon_csv,
            code_file="analysis/per_model_wilcoxon.py",
            rows=rows,
            summary=(
                f"{n_sig_tight}/{n_models} models significant (tight), "
                f"{n_sig_broad}/{n_models} (broad)."
            ),
        )
        sec.subtables.append(sub)

    # Fisher's exact (summary only — too many rows for full table)
    if comparison_csv:
        cdf = _read(comparison_csv)
        if cdf is not None:
            n_total = len(cdf)
            n_sig_tight = int((cdf.get("tight_result", pd.Series(dtype=str))
                               .isin(["a_better", "b_better"])).sum())
            n_sig_broad = int((cdf.get("broad_result", pd.Series(dtype=str))
                               .isin(["a_better", "b_better"])).sum())

            sub = SubTable(
                title="Fisher's Exact Tests (per model × unit)",
                description=(
                    f"{comparison_description or 'Per-unit Fisher exact tests.'} "
                    f"Bonferroni-corrected across families. "
                    f"{n_total} total comparisons."
                ),
                csv_file=comparison_csv,
                code_file="analysis/accuracy_comparison.py",
                rows=[],  # summary only
                summary=(
                    f"{n_sig_tight}/{n_total} significant (tight), "
                    f"{n_sig_broad}/{n_total} (broad) after Bonferroni."
                ),
            )
            sec.subtables.append(sub)

    return sec


def _build_icc_section() -> Section:
    """ICC analysis section."""
    sec = Section(
        title="Intraclass Correlation — Vignette Wording Effect",
        figure_name="variance_stacked",
        figure_page=4,
        description=(
            "ICC(1,1) quantifies the proportion of response variance "
            "attributable to vignette wording (between-vignette) vs "
            "stochastic noise (within-vignette). An F-test evaluates "
            "H₀: σ²_between = 0."
        ),
    )

    icc_df = _read("icc_analysis.csv")
    if icc_df is None:
        return sec

    # Summary by model (mean ICC, % significant F-tests)
    rows = []
    for m in sorted(icc_df["model"].unique()):
        mdf = icc_df[icc_df["model"] == m]
        mean_icc = mdf["icc"].mean()
        n_scen = len(mdf)
        n_sig_f = int((mdf["f_pvalue"] < 0.05).sum())
        mean_f = mdf["f_stat"].mean()
        k_val = int(mdf["k"].mode().iloc[0]) if "k" in mdf.columns else "?"
        rows.append(TestRow(
            context=_clean_model(m),
            test_name="ICC(1,1) + F-test",
            statistic=f"mean ICC={mean_icc:.3f}, mean F={mean_f:.1f}",
            n=f"{n_scen} scenarios × {k_val} vignettes",
            p_value=f"{n_sig_f}/{n_scen} F-tests p<0.05",
            significance=f"{n_sig_f}/{n_scen} sig",
        ))

    sub = SubTable(
        title="ICC(1,1) and F-test per Model × Scenario",
        description=(
            "One-way random-effects ANOVA decomposing log-ratio variance "
            "into within- and between-vignette components. "
            f"{len(icc_df)} total model × scenario cells."
        ),
        csv_file="icc_analysis.csv",
        code_file="analysis/icc_analysis.py",
        rows=rows,
        summary=f"{len(icc_df)} model × scenario cells analysed.",
    )
    sec.subtables.append(sub)

    # Pairwise Mann–Whitney U (sig_pairs column)
    total_sig = icc_df["sig_pairs"].sum()
    total_pairs = icc_df["total_pairs"].sum()
    sub2 = SubTable(
        title="Pairwise Mann–Whitney U (Bonferroni-corrected)",
        description=(
            "Within each model × scenario cell, all (k choose 2) vignette "
            "pairs compared via Mann–Whitney U on log-ratio distributions, "
            "Bonferroni-corrected."
        ),
        csv_file="icc_analysis.csv",
        code_file="analysis/icc_analysis.py",
        rows=[],
        summary=f"{int(total_sig)}/{int(total_pairs)} pairwise comparisons significant.",
    )
    sec.subtables.append(sub2)

    return sec


def _build_spearman_section() -> Section:
    """Spearman ρ (variance↔ICC) section."""
    sec = Section(
        title="Spearman ρ — Variance vs ICC Correlation",
        figure_name="spearman_variance_icc",
        figure_page=5,
        description=(
            "Per-model and overall Spearman rank correlation between "
            "scenario total variance and ICC, plus a binomial sign test "
            "on the direction of per-model ρ."
        ),
    )

    sp_df = _read("variance_icc_spearman.csv")
    if sp_df is None:
        return sec

    rows = []
    per_model = sp_df[
        (sp_df["stat_type"] == "variance_vs_icc") &
        (sp_df["model"] != "OVERALL")
    ]
    for _, r in per_model.iterrows():
        rows.append(TestRow(
            context=_clean_model(str(r["model"])),
            test_name="Spearman ρ",
            statistic=f"ρ = {r['rho']:.3f}",
            n=f"n = {int(r['n'])}",
            p_value=_p(r["p_value"]),
            significance=_sig(r["p_value"]),
        ))

    # Overall
    overall = sp_df[
        (sp_df["stat_type"] == "variance_vs_icc") &
        (sp_df["model"] == "OVERALL")
    ]
    if not overall.empty:
        r = overall.iloc[0]
        rows.append(TestRow(
            context="OVERALL (pooled)",
            test_name="Spearman ρ",
            statistic=f"ρ = {r['rho']:.3f}",
            n=f"n = {int(r['n'])}",
            p_value=_p(r["p_value"]),
            significance=_sig(r["p_value"]),
        ))

    sub = SubTable(
        title="Per-Model and Overall Spearman ρ (Variance ↔ ICC)",
        description="Spearman rank correlation between total log-ratio variance and ICC.",
        csv_file="variance_icc_spearman.csv",
        code_file="analysis/variance_icc_stats.py",
        rows=rows,
    )
    sec.subtables.append(sub)

    # Sign test
    sign_row = sp_df[sp_df["stat_type"] == "sign_test"]
    if not sign_row.empty:
        sr = sign_row.iloc[0]
        n_pos = int(sr.get("n_positive", 0))
        n_neg = int(sr.get("n_negative", 0))
        n_sig = int(sr.get("n_individually_sig", 0))
        sp = float(sr["p_value"])
        sub2 = SubTable(
            title="Binomial Sign Test on ρ Direction",
            description="Tests whether per-model ρ values are consistently positive or negative.",
            csv_file="variance_icc_spearman.csv",
            code_file="analysis/variance_icc_stats.py",
            rows=[TestRow(
                context="All models",
                test_name="Binomial sign test",
                statistic=f"{n_pos} positive, {n_neg} negative",
                n=f"{n_pos + n_neg} models",
                p_value=_p(sp),
                significance=_sig(sp),
                detail=f"{n_sig} individually significant (p<0.05)",
            )],
        )
        sec.subtables.append(sub2)

    # Accuracy vs variance Spearman
    acc_row = sp_df[sp_df["stat_type"] == "accuracy_vs_variance"]
    if not acc_row.empty:
        ar = acc_row.iloc[0]
        sub3 = SubTable(
            title="Spearman ρ — Accuracy vs Total Variance",
            description="Correlation between model accuracy and mean total variance.",
            csv_file="variance_icc_spearman.csv",
            code_file="analysis/variance_icc_stats.py",
            rows=[TestRow(
                context="Across models",
                test_name="Spearman ρ",
                statistic=f"ρ = {ar['rho']:.3f}",
                n=f"n = {int(ar['n'])}",
                p_value=_p(ar["p_value"]),
                significance=_sig(ar["p_value"]),
            )],
        )
        sec.subtables.append(sub3)

    return sec


def _build_kendall_section() -> Section:
    """Kendall's W section."""
    sec = Section(
        title="Kendall's W — Cross-Model Concordance",
        figure_name="kendall_w_summary",
        figure_page=7,
        description=(
            "Kendall's coefficient of concordance (W) measures cross-model "
            "agreement on which vignette wordings produce higher or lower "
            "responses within each scenario."
        ),
    )

    kdf = _read("kendall_w_vignettes.csv")
    if kdf is None:
        return sec

    rows = []
    for _, r in kdf.iterrows():
        rows.append(TestRow(
            context=str(r["scenario_key"]),
            test_name="Kendall's W (χ² approx)",
            statistic=f"W = {r['kendall_w']:.3f}",
            n="13 models",
            p_value=_p(r["kendall_p"]),
            significance=_sig(r["kendall_p"]),
        ))

    n_sig = sum(1 for _, r in kdf.iterrows() if r["kendall_p"] < 0.05)
    sub = SubTable(
        title="Kendall's W per Scenario",
        description="Cross-model concordance on vignette-rank orderings.",
        csv_file="kendall_w_vignettes.csv",
        code_file="analysis/kendall_w.py",
        rows=rows,
        summary=f"{n_sig}/{len(kdf)} scenarios with significant concordance (p<0.05).",
    )
    sec.subtables.append(sub)
    return sec


def _build_chi2_section() -> Section:
    """Chi-squared independence tests."""
    sec = Section(
        title="χ² Independence Tests — CV × Accuracy Grids",
        figure_name="cv_strip_vignettes",
        figure_page=2,
        description=(
            "Chi-squared tests of independence between accuracy category "
            "and CV bin per model, evaluating whether precision and accuracy "
            "are associated."
        ),
    )

    chi2_df = _read("cv_accuracy_chi2.csv")
    if chi2_df is not None:
        rows = []
        for _, r in chi2_df.iterrows():
            if pd.isna(r.get("chi2", np.nan)):
                stat_str = "N/A"
            else:
                stat_str = f"χ² = {r['chi2']:.2f}, dof = {int(r['dof'])}"
            detail_str = ""
            if not pd.isna(r.get("min_expected", np.nan)):
                detail_str = (f"min expected = {r['min_expected']:.1f}"
                              + (" ⚠" if r.get("low_expected_warning", False) else ""))
            rows.append(TestRow(
                context=_clean_model(str(r["model"])),
                test_name="χ² independence",
                statistic=stat_str,
                n=f"n = {int(r['n_obs'])}",
                p_value=_p(r["p_value"]),
                significance=_sig(r["p_value"]),
                detail=detail_str,
            ))
        n_sig = sum(1 for _, r in chi2_df.iterrows()
                    if not pd.isna(r.get("p_value", np.nan)) and r["p_value"] < 0.05)
        sub = SubTable(
            title="CV × Accuracy Independence (Vignettes)",
            description="Per-model χ² test: are accuracy categories independent of CV bins?",
            csv_file="cv_accuracy_chi2.csv",
            code_file="analysis/variance_icc_stats.py",
            rows=rows,
            summary=f"{n_sig}/{len(chi2_df)} models significant (p<0.05).",
        )
        sec.subtables.append(sub)

    grid_df = _read("grid_independence_tests.csv")
    if grid_df is not None:
        for grid_type in grid_df["grid_type"].unique():
            gdf = grid_df[grid_df["grid_type"] == grid_type]
            if "chi2" not in gdf.columns:
                continue
            rows = []
            for _, r in gdf.iterrows():
                if pd.isna(r.get("chi2", np.nan)) or pd.isna(r.get("dof", np.nan)):
                    continue
                dof_val = int(r['dof']) if not pd.isna(r.get('dof', np.nan)) else '?'
                n_obs_val = int(r['n_obs']) if not pd.isna(r.get('n_obs', np.nan)) else '?'
                rows.append(TestRow(
                    context=_clean_model(str(r["model"])),
                    test_name="χ² independence",
                    statistic=f"χ² = {r['chi2']:.2f}, dof = {dof_val}",
                    n=f"n = {n_obs_val}",
                    p_value=_p(r["p_value"]),
                    significance=_sig(r["p_value"]),
                ))
            if rows:
                n_sig = sum(1 for _, r in gdf.iterrows()
                            if not pd.isna(r.get("p_value", np.nan))
                            and r["p_value"] < 0.05)
                sub = SubTable(
                    title=f"Grid Independence: {grid_type}",
                    description=f"χ² test for {grid_type} 3×3 grid per model.",
                    csv_file="grid_independence_tests.csv",
                    code_file="analysis/grid_independence_tests.py",
                    rows=rows,
                    summary=f"{n_sig}/{len(rows)} models significant.",
                )
                sec.subtables.append(sub)

    return sec


def _build_cross_experiment_section() -> Section:
    """Cross-experiment matched-pair tests."""
    sec = Section(
        title="Cross-Experiment Comparison (Matched Pairs)",
        figure_name="cross_experiment_bars",
        figure_page=10,
        description=(
            "Wilcoxon signed-rank tests and Spearman correlations comparing "
            "accuracy across random-dose, empty-vignette, and text-vignette "
            "paradigms on the 11 opioid pairs common to both experiments."
        ),
    )

    # Wilcoxon
    wdf = _read("cross_experiment_matched_wilcoxon.csv")
    if wdf is not None:
        rows = []
        for _, r in wdf.iterrows():
            model = _clean_model(str(r["model"]))
            comp = str(r.get("comparison", ""))
            tp = r.get("tight_wilcoxon_p", np.nan)
            bp = r.get("broad_wilcoxon_p", np.nan)
            tn = int(r.get("tight_n_pairs", 0))
            rows.append(TestRow(
                context=f"{model} ({comp})",
                test_name="Wilcoxon signed-rank (paired)",
                statistic="",
                n=f"n = {tn} pairs",
                p_value=f"tight {_p(tp)}, broad {_p(bp)}",
                significance=f"tight {_sig(tp)}, broad {_sig(bp)}",
            ))

        sub = SubTable(
            title="Matched-Pair Wilcoxon Tests",
            description="Per-model paired comparison of matched opioid pairs across experiments.",
            csv_file="cross_experiment_matched_wilcoxon.csv",
            code_file="analysis/cross_experiment_matched.py",
            rows=rows,
        )
        sec.subtables.append(sub)

    # Spearman
    sdf = _read("cross_experiment_matched_spearman.csv")
    if sdf is not None:
        rows = []
        for _, r in sdf.iterrows():
            model = _clean_model(str(r["model"]))
            comp = str(r.get("comparison", ""))
            rho = r.get("spearman_rho", np.nan)
            sp = r.get("spearman_p", np.nan)
            n = int(r.get("n_pairs", 0))
            rows.append(TestRow(
                context=f"{model} ({comp})",
                test_name="Spearman ρ",
                statistic=f"ρ = {rho:.3f}" if not pd.isna(rho) else "N/A",
                n=f"n = {n}",
                p_value=_p(sp),
                significance=_sig(sp),
            ))

        sub = SubTable(
            title="Cross-Experiment Spearman Correlations",
            description="Per-model Spearman ρ on matched accuracy across experimental formats.",
            csv_file="cross_experiment_matched_spearman.csv",
            code_file="analysis/cross_experiment_matched.py",
            rows=rows,
        )
        sec.subtables.append(sub)

    return sec


def _build_regression_section() -> Section:
    """Linear regression (dose-dependence) section."""
    sec = Section(
        title="Dose-Dependence Regression",
        figure_name="cv_strip_random_doses",
        figure_page=9,
        description=(
            "Per (model, opioid pair, condition) linear regression of "
            "log(observed coefficient) on expected dose, testing for "
            "dose-dependent conversion bias."
        ),
    )

    df = _read("pair_model_analysis.csv")
    if df is None:
        return sec

    # Filter to naive condition and get significant regressions
    cols_needed = ["model", "opioid_pair", "regression_slope",
                   "regression_slope_pvalue", "r_squared"]
    if not all(c in df.columns for c in cols_needed):
        return sec

    # Only show naive condition significant slopes
    naive = df
    if "include_nielsen" in df.columns:
        naive = df[(df["include_nielsen"] == False)]
    if "include_examples" in df.columns:
        naive = naive[(naive["include_examples"] == False)]

    sig = naive[naive["regression_slope_pvalue"] < 0.05].copy()

    rows = []
    for _, r in sig.head(30).iterrows():  # cap at 30 for readability
        rows.append(TestRow(
            context=f"{_clean_model(str(r['model']))} / {r['opioid_pair']}",
            test_name="Linear regression",
            statistic=f"slope = {r['regression_slope']:.4f}, R² = {r['r_squared']:.3f}",
            n="",
            p_value=_p(r["regression_slope_pvalue"]),
            significance=_sig(r["regression_slope_pvalue"]),
        ))

    n_total = len(naive.dropna(subset=["regression_slope_pvalue"]))
    n_sig = len(sig)
    sub = SubTable(
        title="Significant Dose-Dependent Slopes (Naive Condition)",
        description=(
            f"Linear regression of log(coefficient) on expected dose. "
            f"Showing significant results (p<0.05) out of {n_total} total."
        ),
        csv_file="pair_model_analysis.csv",
        code_file="analysis/opioid_pair_model_result.py",
        rows=rows,
        summary=f"{n_sig}/{n_total} regressions significant (p<0.05).",
    )
    sec.subtables.append(sub)
    return sec


def _build_consensus_section() -> Section:
    """Consensus t-tests."""
    sec = Section(
        title="Consensus Coefficient T-Tests",
        figure_name=None,
        figure_page=None,
        description=(
            "Weighted t-tests (in log-space) comparing model consensus "
            "conversion coefficients to expected literature values."
        ),
    )

    for csv_name, sub_title, desc in [
        ("opioid_pair_consensus.csv",
         "Opioid Pair Consensus vs Expected",
         "Per (model, opioid pair): weighted t-test of consensus coefficient vs expected."),
        ("vignette_model_consensus.csv",
         "Vignette Consensus vs Expected Dose",
         "Per (model, scenario): weighted t-test of consensus dose vs expected."),
    ]:
        df = _read(csv_name)
        if df is None:
            continue

        pcol = "p_value_log"
        tcol = "t_score_log"
        if pcol not in df.columns:
            continue

        sig = df[df[pcol] < 0.05]
        n_total = len(df.dropna(subset=[pcol]))
        n_sig = len(sig)

        # Show summary row per model
        model_col = "model" if "model" in df.columns else df.columns[0]
        rows = []
        for m in sorted(df[model_col].unique()):
            mdf = df[df[model_col] == m].dropna(subset=[pcol])
            m_sig = (mdf[pcol] < 0.05).sum()
            m_total = len(mdf)
            if m_total > 0:
                rows.append(TestRow(
                    context=_clean_model(str(m)),
                    test_name="Weighted t-test (log)",
                    statistic=f"median |t| = {mdf[tcol].abs().median():.2f}",
                    n=f"n = {m_total} units",
                    p_value=f"{m_sig}/{m_total} sig",
                    significance=f"{m_sig}/{m_total}",
                ))

        sub = SubTable(
            title=sub_title,
            description=desc,
            csv_file=csv_name,
            code_file="analysis/opioid_pair_consensus.py",
            rows=rows,
            summary=f"{n_sig}/{n_total} total tests significant (p<0.05).",
        )
        sec.subtables.append(sub)

    return sec


# ── Build all sections ──────────────────────────────────────────────

FIGURE_PAGES = {
    "naive_vignette_bars": 1,
    "cv_strip_vignettes": 2,
    "vignette_performance_profile": 3,
    "variance_stacked": 4,
    "spearman_variance_icc": 5,
    "icc_bubble_grid": 6,
    "kendall_w_summary": 7,
    "naive_random_dose_bars": 8,
    "cv_strip_random_doses": 9,
    "cross_experiment_bars": 10,
    "nielsen_effect_combined": 11,
    "nielsen_ranked_random_doses": 12,
    "nielsen_by_ome_combined": 13,
    "explain_units_full_treatment": 14,
    "nielsen_units_ranked": 15,
}


def build_all_sections() -> list[Section]:
    """Build every section for the grand table."""
    sections = []

    # ── Clinical Vignettes ────────────────────────────────────────

    sections.append(_build_ranking_section(
        "Model Ranking — Naive Vignettes",
        "ranking_naive_vignettes.csv",
        "ranking_naive_vignettes_pairs.csv",
        "naive_vignette_bars", 1,
        "Pairwise CMH ranking of all 13 models on 20 clinical vignette "
        "scenarios in the naive condition (no reference data provided). "
        "Each scenario serves as a stratum, controlling for the "
        "hierarchical nesting of trials within items.",
    ))

    sections.append(_build_chi2_section())

    sections.append(_build_icc_section())

    sections.append(_build_spearman_section())

    sections.append(_build_kendall_section())

    # ── Random Doses ──────────────────────────────────────────────

    sections.append(_build_ranking_section(
        "Model Ranking — Naive Random Doses",
        "ranking_naive_random_doses.csv",
        "ranking_naive_random_doses_pairs.csv",
        "naive_random_dose_bars", 8,
        "Pairwise CMH ranking on 49 random-dose opioid pairs in the naive "
        "condition. This controlled experiment removes vignette-wording "
        "variability, testing models on purely numeric prompts.",
    ))

    sections.append(_build_regression_section())

    sections.append(_build_cross_experiment_section())

    # ── Reference & Unit Augmentation ─────────────────────────────

    sections.append(_build_wilcoxon_section(
        "Nielsen Effect — Vignettes",
        "wilcoxon_nielsen_effect_vignettes.csv",
        "comparison_nielsen_effect_vignettes.csv",
        "nielsen_effect_combined", 11,
        "Paired comparison of accuracy before and after providing the "
        "Nielsen equianalgesic reference table in the prompt, on vignette "
        "scenarios. Wilcoxon signed-rank tests quantify the per-model "
        "improvement; Fisher's exact tests identify specific scenarios "
        "where the reference data changed the outcome.",
        "Fisher's exact test per (model, scenario): nielsen vs naive.",
    ))

    sections.append(_build_wilcoxon_section(
        "Nielsen Effect — Random Doses",
        "wilcoxon_nielsen_effect_random_dose.csv",
        "comparison_nielsen_effect_random_dose.csv",
        "nielsen_effect_combined", 11,
        "Same paired comparison on random-dose conversions. Because random "
        "doses remove vignette-wording confounds, this tests the pure "
        "effect of reference-data augmentation.",
        "Fisher's exact test per (model, opioid pair): nielsen vs naive.",
    ))

    sections.append(_build_ranking_section(
        "Model Ranking — Nielsen Random Doses",
        "ranking_random_doses_nielsen.csv",
        "ranking_random_doses_nielsen_pairs.csv",
        "nielsen_ranked_random_doses", 12,
        "Re-ranking models after providing the Nielsen reference table, "
        "testing whether the relative model ordering changes when "
        "reference data is available.",
    ))

    sections.append(_build_wilcoxon_section(
        "Nielsen by OME — Vignettes (OME)",
        "wilcoxon_nielsen_effect_vignettes_ome.csv",
        "comparison_nielsen_effect_vignettes_ome.csv",
        "nielsen_by_ome_combined", 13,
        "Nielsen effect restricted to vignette scenarios involving oral "
        "morphine equivalents as the target — the unit most directly "
        "addressed by the reference table.",
    ))

    sections.append(_build_wilcoxon_section(
        "Nielsen by OME — Vignettes (Non-OME)",
        "wilcoxon_nielsen_effect_vignettes_nonome.csv",
        "comparison_nielsen_effect_vignettes_nonome.csv",
        "nielsen_by_ome_combined", 13,
        "Nielsen effect on vignette scenarios with non-oral-morphine "
        "targets, testing whether the reference table helps with "
        "conversions that require multi-step reasoning.",
    ))

    sections.append(_build_wilcoxon_section(
        "Nielsen by OME — Random Doses (OME)",
        "wilcoxon_nielsen_effect_random_dose_ome.csv",
        "comparison_nielsen_effect_random_dose_ome.csv",
        "nielsen_by_ome_combined", 13,
        "Nielsen effect restricted to random-dose pairs involving oral "
        "morphine equivalents (direct lookup conversions).",
    ))

    sections.append(_build_wilcoxon_section(
        "Nielsen by OME — Random Doses (Non-OME)",
        "wilcoxon_nielsen_effect_random_dose_nonome.csv",
        "comparison_nielsen_effect_random_dose_nonome.csv",
        "nielsen_by_ome_combined", 13,
        "Nielsen effect on non-oral-morphine random-dose pairs, requiring "
        "the model to chain two conversion factors through morphine "
        "equivalents.",
    ))

    sections.append(_build_wilcoxon_section(
        "Dosing Unit Effect",
        "wilcoxon_dosing_unit_effect.csv",
        "comparison_dosing_unit_effect.csv",
        "explain_units_full_treatment", 14,
        "Paired comparison testing whether models perform worse when the "
        "target uses non-daily dosing units (e.g., mg/4hr vs mg/day). "
        "This isolates a systematic error where models fail to account "
        "for time-unit conversions.",
    ))

    sections.append(_build_wilcoxon_section(
        "Explain Units Effect — Random Doses",
        "wilcoxon_units_effect_random_dose.csv",
        "comparison_units_effect_random_dose.csv",
        "explain_units_full_treatment", 14,
        "Paired comparison testing whether adding an explicit dosing-unit "
        "explanation to the prompt (e.g., 'the dose over 4 hours should "
        "be 1/6 the daily dose') rescues the unit-conversion error.",
    ))

    sections.append(_build_ranking_section(
        "Model Ranking — Nielsen + Units (Random Doses)",
        "ranking_random_doses_nielsen_units.csv",
        "ranking_random_doses_nielsen_units_pairs.csv",
        "nielsen_units_ranked", 15,
        "Final model ranking on random doses after applying both the "
        "Nielsen reference table and dosing-unit explanation — the best "
        "achievable prompt augmentation.",
    ))

    sections.append(_build_ranking_section(
        "Model Ranking — Nielsen + Units (Vignettes)",
        "ranking_vignettes_nielsen_units.csv",
        "ranking_vignettes_nielsen_units_pairs.csv",
        "nielsen_units_ranked", 15,
        "Final model ranking on vignettes after applying both the "
        "Nielsen reference table and dosing-unit explanation.",
    ))

    # ── Supplementary / not tied to a single figure ───────────────

    sections.append(_build_consensus_section())

    return sections


# ── HTML Rendering ──────────────────────────────────────────────────

_CSS = """
<style>
  :root { --bg: #fafafa; --border: #ddd; --accent: #2563eb; --sig: #16a34a; }
  body { font-family: "Inter", system-ui, sans-serif; max-width: 1200px;
         margin: 0 auto; padding: 20px; background: var(--bg); color: #1a1a1a;
         line-height: 1.55; }
  h1 { font-size: 22pt; border-bottom: 3px solid var(--accent); padding-bottom: 8px; }
  h2 { font-size: 15pt; margin-top: 28px; cursor: pointer; }
  h2:hover { color: var(--accent); }
  h3 { font-size: 12pt; margin-top: 16px; color: #444; }
  h3:hover { color: var(--accent); }
  .section { border: 1px solid var(--border); border-radius: 8px;
             margin: 16px 0; padding: 16px; background: white; }
  .section-header { display: flex; justify-content: space-between; align-items: center; }
  .page-badge { background: var(--accent); color: white; border-radius: 12px;
                padding: 2px 10px; font-size: 9pt; white-space: nowrap; }
  .desc { font-size: 10pt; color: #555; margin: 4px 0 12px 0; }
  .methods-prose { font-size: 10.5pt; color: #222; line-height: 1.6;
                   margin: 8px 0 16px 0; }
  .methods-prose p { margin: 6px 0; }
  .methods-block { background: white; border: 1px solid var(--border);
                   border-radius: 8px; padding: 20px; margin: 16px 0; }
  .methods-block h3 { margin-top: 18px; color: #1a1a1a; font-size: 12pt; }
  .methods-block h3:first-child { margin-top: 0; }
  .glossary-entry { margin: 12px 0 18px 0; padding-left: 12px;
                    border-left: 3px solid var(--accent); }
  .glossary-entry h4 { margin: 0 0 4px 0; font-size: 11pt; color: #1a1a1a; }
  .glossary-entry p { font-size: 10pt; margin: 4px 0; color: #333; }
  .summary { font-size: 10pt; font-weight: 600; color: var(--sig); margin: 8px 0; }
  .links { font-size: 9pt; margin: 4px 0; }
  .links a { color: var(--accent); margin-right: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; }
  th { background: #f0f4ff; text-align: left; padding: 6px 8px;
       border-bottom: 2px solid var(--border); font-weight: 600; }
  td { padding: 5px 8px; border-bottom: 1px solid #eee; }
  tr:hover td { background: #f8f9ff; }
  .sig-cell { font-weight: 600; }
  .sig-cell.sig { color: var(--sig); }
  .sig-cell.ns { color: #999; }
  details { margin: 4px 0; }
  details summary { cursor: pointer; font-weight: 600; font-size: 10pt; }
  details summary:hover { color: var(--accent); }
  .overview-table { margin: 20px 0; }
  .overview-table th { font-size: 10pt; }
  .overview-table td { font-size: 10pt; padding: 8px; }
  .toc { background: white; border: 1px solid var(--border); border-radius: 8px;
         padding: 16px; margin: 16px 0; }
  .toc ol { padding-left: 20px; }
  .toc li { margin: 4px 0; font-size: 10pt; }
  .toc a { color: var(--accent); text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  @media print {
    .section { break-inside: avoid; }
    details { display: block; }
    details[open] summary ~ * { display: block; }
  }
</style>
"""

_JS = """
<script>
function toggleAll(open) {
  document.querySelectorAll('details').forEach(d => d.open = open);
}
</script>
"""


# ── Methods Overview Prose ──────────────────────────────────────────

def _render_methods_overview() -> str:
    """Render the general Methods overview (non-statistical)."""
    gh = GITHUB_REPO
    return f'''
<div class="methods-block">
<h2 id="methods-overview">Methods Overview</h2>

<h3>Study Design</h3>
<div class="methods-prose">
<p>We evaluated the ability of 13 large language models (LLMs) to perform
equianalgesic opioid dose conversions — a common and safety-critical clinical
task. Each model was presented with a source opioid, a dose, and a target opioid,
and asked to calculate the equivalent dose in the target formulation. Two
complementary experimental paradigms were used:</p>

<p>(1)&nbsp;<b>Clinical vignette conversions</b> across 20 clinical scenarios
(e.g., post-surgical cancer patients, pediatric burn patients, elderly
hip-fracture patients), each re-stated in 4 distinct phrasings plus 1
empty-text control, yielding 100 vignettes in total.</p>

<p>(2)&nbsp;<b>Random-dose conversions</b> across 49 directed opioid-pair
conversions drawn from 20 opioid substances (administration route included),
comprising 15 bidirectional pairs tested in both directions and 19
one-directional conversions to oral morphine. Each opioid pair was
assigned a clinician-defined priority score (priority 1 or 2) reflecting
clinical relevance. Doses were sampled log-uniformly from an
oral-morphine-equivalent range of 10–120 mg. All random-dose prompts used
the standard unit <code>mg/day</code>. A subset of 15 pairs was additionally
tested with a non-daily target unit (<code>mg/4 hours</code>) to isolate
time-unit conversion errors; in those trials, the prompt optionally included
an explicit dosing-unit explanation (e.g., "the dose over 4 hours should be
1/6 the daily dose").</p>

<p>Prompt conditions were factorially varied to test the effect of providing
a published reference table — the Nielsen equianalgesic conversion table
(Nielsen et al., 2015). We deliberately included opioid pairs that require
extra reasoning steps beyond a simple lookup in the Nielsen table: pairs
whose target is not oral morphine require chaining two conversion factors
through a morphine-equivalent intermediate, and pairs with non-daily units
require an additional time-unit conversion.</p>

<p>Accuracy was assessed as the ratio of observed to expected dose
(ratio<sub>obs/exp</sub>). Two clinically-motivated accuracy ranges were defined:
<b>tight</b> (0.85–1.15× expected, i.e., ±15%) and <b>broad</b>
(0.60–1.70× expected). Each model × item × condition combination was repeated
multiple times (median 7 for vignettes; median 10 for random doses) to quantify
response variability. All API calls used temperature&nbsp;=&nbsp;0.0.</p>
</div>

<h3>Implementation &amp; Code Organization</h3>
<div class="methods-prose">
<p>The entire experimental pipeline — from prompt construction through data
collection, statistical analysis, and figure generation — was implemented in
Python and is publicly available at
<a href="{gh}" target="_blank">{gh}</a>.
The experiment code was developed with the assistance of GPT-4.1 and
Claude Sonnet 4.5; the analysis and visualization code was developed with
Claude Opus 4.6 via GitHub Copilot. The implementation uses
Python&nbsp;3.14.0, pandas&nbsp;2.3.3, SciPy&nbsp;1.16.3,
NumPy&nbsp;2.3.4, statsmodels&nbsp;0.14.6, and Bokeh&nbsp;3.8.1.
The repository is organized into five packages:</p>
<ul style="font-size:10.5pt; margin:4px 0 8px 20px;">
<li><b><a href="{gh}/tree/{GITHUB_BRANCH}/common" target="_blank">common/</a></b>
 — Opioid data models, the Nielsen conversion table, unit-aware dose
 conversion logic, and the opioid-pair generator.</li>
<li><b><a href="{gh}/tree/{GITHUB_BRANCH}/experiment" target="_blank">experiment/</a></b>
 — Prompt construction, API routing, experiment runners for both random-dose
 and clinical-vignette paradigms.</li>
<li><b><a href="{gh}/tree/{GITHUB_BRANCH}/resources" target="_blank">resources/</a></b>
 — The Nielsen equianalgesic table (JSON), scenario definitions, and
 vignette text.</li>
<li><b><a href="{gh}/tree/{GITHUB_BRANCH}/analysis" target="_blank">analysis/</a></b>
 — Statistical analysis modules that read raw result CSVs and produce
 summary statistics, test results, and derived CSVs.</li>
<li><b><a href="{gh}/tree/{GITHUB_BRANCH}/visualize" target="_blank">visualize/</a></b>
 — Visualization code generating publication-ready figures.</li>
</ul>
</div>

<h3>API Infrastructure &amp; Data Collection</h3>
<div class="methods-prose">
<p>All 13 models were queried through the
<a href="https://openrouter.ai" target="_blank">OpenRouter</a> API, which
provides a unified interface to models from OpenAI, Anthropic, Google, Mistral,
Meta, Cohere, AI21, Alibaba (Qwen), and DeepSeek. Each API call used
temperature = 0.0 and a maximum token budget of 500 tokens. Models were
instructed to return their answer as a Python dictionary literal
(<code>{{"calculated dose": <i>value</i>}}</code>), and the numeric dose was
extracted from the response text via regular expression.</p>

<p>The 13 models tested were: GPT-4.1, GPT-4.1-mini, GPT-3.5-turbo
(OpenAI); Claude 3.5 Sonnet (Anthropic); Mistral Large, Mixtral 8×22B
(Mistral); Gemini 2.5 Pro, Gemini 2.5 Flash (Google); Command R+ (Cohere);
Jamba Large (AI21); Qwen 2.5 72B (Alibaba); Llama 3.3 70B (Meta); and
DeepSeek Chat (DeepSeek).</p>

<p>Results were saved row-by-row to CSV files during each experiment run. The
random-dose experiment produced 9,667 rows for 13 models
(<a href="{_gh("results/experiment_results.csv")}" target="_blank">results/experiment_results.csv</a>;
26,871 total including preliminary models),
the vignette experiment produced 27,166 rows for 13 models
(<a href="{_gh("results/reworded_vignettes_results.csv")}" target="_blank">results/reworded_vignettes_results.csv</a>;
28,814 total),
and the dosing-unit experiment produced 9,743 rows for 13 models
(<a href="{_gh("results/explain_units_experiment_results.csv")}" target="_blank">results/explain_units_experiment_results.csv</a>;
10,493 total).</p>
</div>

<h3>Analysis Pipeline</h3>
<div class="methods-prose">
<p>Raw result CSVs were processed by analysis modules in the
<a href="{gh}/tree/{GITHUB_BRANCH}/analysis" target="_blank">analysis/</a>
package. Each module reads the appropriate experiment CSV(s), groups trials by
model, opioid pair (or vignette), and condition, computes accuracy rates and
summary statistics, performs the relevant statistical tests, and writes the
results to derived CSV files in the
<a href="{gh}/tree/{GITHUB_BRANCH}/results" target="_blank">results/</a>
directory. Vignette analyses group by <code>vignette_id</code> and
<code>scenario_key</code> to distinguish wording variants within the same
clinical scenario. This two-stage design (experiment → CSV → analysis → CSV)
ensures full reproducibility: all statistical results reported below can be
regenerated from the raw data using the analysis modules.</p>
</div>

<h3>Visualization</h3>
<div class="methods-prose">
<p>All figures were generated using
<a href="https://bokeh.org" target="_blank">Bokeh&nbsp;3.8.1</a>, a Python
interactive visualization library. The
<a href="{_gh("visualize/comprehensive_figures.py")}" target="_blank">ComprehensiveFigureBuilder</a>
class produces 15 publication-ready figures organized in three groups:
Figures 1–7 examine model accuracy and variance on clinical vignettes,
Figures 8–10 replicate the analysis on controlled random-dose prompts,
and Figures 11–15 test whether providing reference data or dosing-unit
explanations improves performance. Each figure is saved as an interactive
HTML file and a static PNG.</p>
</div>
</div>
'''


# ── Statistical Methods Glossary ────────────────────────────────────

def _render_glossary(sections: list[Section]) -> str:
    """Render a detailed glossary of each statistical test family."""

    # Dynamically gather numbers from data
    icc_df = _read("icc_analysis.csv")
    n_models = int(icc_df["model"].nunique()) if icc_df is not None else 13
    n_scenarios = int(icc_df["scenario_key"].nunique()) if icc_df is not None and "scenario_key" in icc_df.columns else 20
    n_icc_cells = len(icc_df) if icc_df is not None else 260
    k_val = int(icc_df["k"].mode().iloc[0]) if icc_df is not None and "k" in icc_df.columns else 4
    n_sig_f = int((icc_df["f_pvalue"] < 0.05).sum()) if icc_df is not None else "?"

    chi2_df = _read("cv_accuracy_chi2.csv")
    n_chi2_valid = int(chi2_df["p_value"].dropna().shape[0]) if chi2_df is not None else "?"
    n_chi2_sig = int((chi2_df["p_value"].dropna() < 0.05).sum()) if chi2_df is not None else "?"

    kw_df = _read("kendall_w_vignettes.csv")
    n_kw = len(kw_df) if kw_df is not None else 20
    n_kw_sig = int((kw_df["kendall_p"] < 0.05).sum()) if kw_df is not None else 0

    sp_df = _read("variance_icc_spearman.csv")
    overall_sp = sp_df[(sp_df["stat_type"] == "variance_vs_icc") & (sp_df["model"] == "OVERALL")] if sp_df is not None else pd.DataFrame()
    sp_rho = f'{overall_sp.iloc[0]["rho"]:.3f}' if not overall_sp.empty else "?"
    sp_n = int(overall_sp.iloc[0]["n"]) if not overall_sp.empty else "?"

    pma_df = _read("pair_model_analysis.csv")
    n_reg_valid = int(pma_df["regression_slope_pvalue"].dropna().shape[0]) if pma_df is not None and "regression_slope_pvalue" in pma_df.columns else "?"
    n_reg_sig = int((pma_df["regression_slope_pvalue"].dropna() < 0.05).sum()) if pma_df is not None and "regression_slope_pvalue" in pma_df.columns else "?"

    # Count ranking pairs across all ranking CSVs
    ranking_csvs = list(RESULTS_DIR.glob("ranking_*_pairs.csv"))
    n_ranking_files = len(ranking_csvs)

    # Count wilcoxon CSVs
    wilcoxon_csvs = list(RESULTS_DIR.glob("wilcoxon_*.csv"))
    n_wilcoxon_files = len(wilcoxon_csvs)

    entries = []

    # 1. CMH
    entries.append(f'''
<div class="glossary-entry">
<h4>Cochran–Mantel–Haenszel (CMH) Test</h4>
<p><b>Purpose.</b> To rank-order models by pairwise accuracy comparisons while
controlling for the hierarchical structure of the data.</p>

<p><b>Why CMH.</b> The conversion data has a hierarchical structure: individual
trials are nested within conversion items (opioid pairs or vignette scenarios).
A simple pooled comparison would treat every trial as independent, when in fact
trials within the same item are correlated. The CMH test respects this nesting
by forming a separate 2×2 table within each conversion item, then testing
whether one model is consistently more accurate across all strata — pooling
evidence across independent item-level comparisons rather than across
non-independent trial-level observations. The one-sided formulation tests
whether model A is systematically more accurate than model B. Because the
number of pairwise comparisons is large, all p-values are
Bonferroni-corrected.</p>

<p><b>Application.</b> We computed CMH tests for every pair of models in
{n_ranking_files} ranking analyses, with stratification counts ranging from
15 to 49 strata depending on the item set. Results are stored in the
<code>ranking_*_pairs.csv</code> files.</p>
</div>''')

    # 2. Wilcoxon
    entries.append(f'''
<div class="glossary-entry">
<h4>Wilcoxon Signed-Rank Test</h4>
<p><b>Purpose.</b> To test whether a prompt augmentation (e.g., providing the
Nielsen reference table, adding a dosing-unit explanation) significantly changes
a model's accuracy on matched conversion items.</p>

<p><b>Why Wilcoxon.</b> Each model is tested on the same set of opioid pairs
(or scenarios) under two conditions, yielding naturally paired observations. The
Wilcoxon signed-rank test is the appropriate non-parametric paired test when the
distribution of differences may be non-normal. It tests H₀: the median
difference in accuracy between conditions is zero. We perform this test
separately for each model and pooled across all models.</p>

<p><b>Application.</b> Wilcoxon tests are computed in {n_wilcoxon_files}
separate comparison analyses (e.g., Nielsen effect on vignettes, Nielsen effect
on random doses, dosing-unit explanation effect, OME-stratified subsets). Each
CSV in the <code>wilcoxon_*.csv</code> family contains per-model test results
plus a POOLED row aggregating across all models.</p>
</div>''')

    # 3. Fisher's exact
    entries.append('''
<div class="glossary-entry">
<h4>Fisher's Exact Test (with Bonferroni Correction)</h4>
<p><b>Purpose.</b> To test whether the accuracy rate on a specific conversion
item differs between two prompt conditions for a given model.</p>

<p><b>Why Fisher's exact.</b> At the individual model × item level, each
condition yields only ~7–10 binomial trials (repetitions). With such small cell
counts, the χ² approximation is unreliable; Fisher's exact test computes the
exact hypergeometric probability. Because hundreds of such tests are performed in
parallel (one per model × item combination), Bonferroni correction is applied
within each model's family of items to control the family-wise error rate.</p>

<p><b>Application.</b> Fisher's exact tests are computed in the
<code>comparison_*.csv</code> files, one per condition-pair analysis. The
<code>tight_result</code> and <code>broad_result</code> columns indicate whether
condition A or B was significantly better, or whether no significant difference
was found.</p>
</div>''')

    # 4. ICC
    entries.append(f'''
<div class="glossary-entry">
<h4>Intraclass Correlation Coefficient — ICC(1,1) with F-test</h4>
<p><b>Purpose.</b> To quantify how much of the variance in model responses is
attributable to vignette wording (between-vignette variance) versus stochastic
noise (within-vignette variance). A high ICC indicates that the model's answer
depends strongly on how the question is phrased, even when the underlying
clinical scenario is identical.</p>

<p><b>Why ICC(1,1).</b> The ICC(1,1) from a one-way random-effects ANOVA is the
standard measure of rater agreement/consistency when the grouping factor
(vignette wording) is treated as a random sample from a population of possible
phrasings. The associated F-test evaluates H₀: σ²<sub>between</sub> = 0, i.e.,
all vignette wordings produce statistically indistinguishable response
distributions.</p>

<p><b>Application.</b> ICC is computed for each of the {n_icc_cells} model ×
scenario cells ({n_models} models × {n_scenarios} scenarios), with k = {k_val}
vignettes per scenario. The F-test rejects H₀ in {n_sig_f}/{n_icc_cells} cells,
indicating pervasive wording sensitivity. Results are in
<code>results/icc_analysis.csv</code>.</p>
</div>''')

    # 5. Mann-Whitney U
    entries.append('''
<div class="glossary-entry">
<h4>Mann–Whitney U Test (Bonferroni-corrected)</h4>
<p><b>Purpose.</b> To identify specific pairs of vignette wordings whose
response distributions differ significantly within a given model × scenario
cell.</p>

<p><b>Why Mann–Whitney U.</b> Within each ICC cell, the k vignettes' response
distributions are compared pairwise. Because the responses are log-ratios with
potentially skewed distributions and small per-vignette sample sizes (~7
repetitions), the non-parametric Mann–Whitney U test is preferred over a
parametric alternative. Bonferroni correction is applied across the (k choose 2)
pairs within each cell to control family-wise error.</p>

<p><b>Application.</b> Pairwise Mann–Whitney results are summarized in the
<code>sig_pairs</code> and <code>total_pairs</code> columns of
<code>results/icc_analysis.csv</code>.</p>
</div>''')

    # 6. Spearman
    entries.append(f'''
<div class="glossary-entry">
<h4>Spearman Rank Correlation (ρ)</h4>
<p><b>Purpose.</b> To test monotonic associations between continuous measures —
in particular, whether scenarios with higher total response variance also have
higher ICC (i.e., more wording-driven variance), and whether more accurate
models tend to have lower response variance.</p>

<p><b>Why Spearman.</b> Spearman's ρ is a non-parametric rank-based correlation
that does not assume linearity or normality. It is robust to outliers and
appropriate when the relationship may be monotonic but not necessarily linear.
The associated p-value tests H₀: ρ = 0 (no monotonic association).</p>

<p><b>Application.</b> (1) Per-model and overall correlation between scenario
total variance and ICC (overall ρ = {sp_rho}, n = {sp_n} model × scenario
observations). (2) Correlation between model-level accuracy and mean total
variance across models (n = {n_models}). (3) Cross-experiment Spearman
correlations between matched-pair accuracies across experimental paradigms.
Results are in <code>results/variance_icc_spearman.csv</code> and
<code>results/cross_experiment_matched_spearman.csv</code>.</p>
</div>''')

    # 7. Binomial sign test
    entries.append('''
<div class="glossary-entry">
<h4>Binomial Sign Test</h4>
<p><b>Purpose.</b> To test whether the per-model Spearman ρ values (variance vs.
ICC) are consistently positive across models — i.e., whether the association
between total variance and wording sensitivity is a general phenomenon rather
than an artefact of a few outlier models.</p>

<p><b>Why the sign test.</b> The sign test is the simplest non-parametric test
for a consistent direction: it counts the number of positive vs. negative
values among independent observations and compares to a binomial distribution
with p = 0.5. It makes no assumptions about the magnitude or distribution of the
ρ values, only their sign.</p>

<p><b>Application.</b> Of the 13 per-model ρ values, 12 are positive and 1 is
negative, yielding a sign-test p = 0.003, confirming that the variance–ICC
association is consistent across models.</p>
</div>''')

    # 8. Kendall's W
    entries.append(f'''
<div class="glossary-entry">
<h4>Kendall's Coefficient of Concordance (W)</h4>
<p><b>Purpose.</b> To test whether different models agree on <i>which</i>
vignette wordings produce higher or lower responses within each scenario. A
significant W would indicate that wording effects are not merely model-specific
noise, but reflect some shared sensitivity to phrasing.</p>

<p><b>Why Kendall's W.</b> Kendall's W is the standard non-parametric measure
of inter-rater agreement when multiple raters (here, models) rank the same set
of items (vignette wordings) within each scenario. It ranges from 0 (no
agreement) to 1 (perfect agreement). The associated χ² approximation tests
H₀: W = 0.</p>

<p><b>Application.</b> W is computed for each of the {n_kw} scenarios, with
{n_models} models as raters ranking {k_val} vignettes by mean log-ratio.
{n_kw_sig}/{n_kw} scenarios reached significance (p < 0.05), indicating that
wording sensitivity is largely model-specific rather than reflecting a shared
vulnerability to particular phrasings. Results are in
<code>results/kendall_w_vignettes.csv</code>.</p>
</div>''')

    # 9. Chi-squared independence
    entries.append(f'''
<div class="glossary-entry">
<h4>Pearson's χ² Test of Independence</h4>
<p><b>Purpose.</b> To test whether accuracy category and response precision
(coefficient of variation) are statistically associated for each model — i.e.,
whether models that are imprecise on a given item also tend to be inaccurate.</p>

<p><b>Why χ².</b> The test cross-tabulates two categorical variables (accuracy
bin × CV bin) in a contingency table and tests H₀: the row and column variables
are independent. The Pearson χ² statistic follows an approximate χ² distribution
with (r−1)(c−1) degrees of freedom when expected cell counts are sufficiently
large (≥ 5). When minimum expected counts fall below this threshold, a warning is
flagged. For one model (Gemini 2.5 Flash), data sparsity precluded a valid test
and χ² is reported as N/A.</p>

<p><b>Application.</b> Per-model χ² tests on the vignette CV × accuracy grid
found {n_chi2_sig}/{n_chi2_valid} models with a significant association. Additionally,
3×3 grid independence tests were computed for accuracy × variance, variance × CV,
and variance × ICC grids across all models, stored in
<code>results/grid_independence_tests.csv</code>.</p>
</div>''')

    # 10. Linear regression
    entries.append(f'''
<div class="glossary-entry">
<h4>Linear Regression (Dose-Dependence)</h4>
<p><b>Purpose.</b> To test whether model conversion errors are dose-dependent —
i.e., whether the log(observed coefficient) changes systematically with the
expected dose level. A significant slope would indicate that the model's
conversion factor is not constant across doses, suggesting a failure to apply
linear proportional conversion.</p>

<p><b>Why linear regression.</b> The dose-dependence hypothesis is naturally
expressed as a linear relationship: log(ratio<sub>obs/exp</sub>) = β₀ + β₁ ×
dose + ε. Ordinary least-squares regression directly tests H₀: β₁ = 0 via a
t-test on the slope. The R² statistic quantifies what fraction of the
log-ratio variance is explained by dose.</p>

<p><b>Application.</b> Regression was fit for each model × opioid pair ×
condition combination under the naive condition. Of {n_reg_valid} valid
regressions, {n_reg_sig} showed a significant dose-dependent slope (p < 0.05).
Results are in <code>results/pair_model_analysis.csv</code>.</p>
</div>''')

    # 11. Weighted t-test
    entries.append('''
<div class="glossary-entry">
<h4>Weighted t-test (Log-Space)</h4>
<p><b>Purpose.</b> To test whether a model's consensus conversion coefficient
(the precision-weighted mean across repetitions) differs significantly from the
expected literature value for each opioid pair or vignette scenario.</p>

<p><b>Why weighted t-test in log-space.</b> Conversion coefficients are
inherently ratio-scale and right-skewed; log-transformation produces an
approximately normal distribution suitable for a t-test. Weighting by the inverse
variance of each repetition's log-ratio ensures that more precise estimates
contribute more to the consensus. The test statistic is
t = (x̄<sub>w</sub> − μ₀) / SE<sub>w</sub>, where μ₀ = 0 in log-space
(i.e., perfect agreement with the expected coefficient).</p>

<p><b>Application.</b> Consensus t-tests are computed (1) per model × opioid
pair in <code>results/opioid_pair_consensus.csv</code>, and (2) per model ×
vignette scenario in <code>results/vignette_model_consensus.csv</code>.</p>
</div>''')

    return (
        '<div class="methods-block">\n'
        '<h2 id="stat-glossary">Statistical Test Glossary</h2>\n'
        '<div class="methods-prose">\n'
        '<p>The following tests were used throughout the analyses. Each entry '
        'describes the test, its statistical justification, and how it was applied '
        'in this study. Specific test results are detailed in the per-figure '
        'sections below.</p>\n'
        '</div>\n'
        + "\n".join(entries)
        + '\n</div>\n'
    )


def _esc(s: str) -> str:
    return html_mod.escape(str(s))


def render_overview_table(sections: list[Section]) -> str:
    """Master overview table of all test families."""
    # Collect unique test types across all sections/subtables
    test_families: dict[str, dict] = {}
    for sec in sections:
        for sub in sec.subtables:
            key = sub.title
            if key not in test_families:
                test_families[key] = {
                    "test": sub.title,
                    "n_results": len(sub.rows),
                    "sections": [],
                    "csv_files": set(),
                    "code_files": set(),
                }
            else:
                test_families[key]["n_results"] += len(sub.rows)
            test_families[key]["sections"].append(sec.title)
            if sub.csv_file:
                test_families[key]["csv_files"].add(sub.csv_file)
            if sub.code_file:
                test_families[key]["code_files"].add(sub.code_file)

    rows_html = []
    for tf in test_families.values():
        csvs = ", ".join(
            f'<a href="{_gh_csv(c)}" target="_blank">results/{c}</a>'
            for c in sorted(tf["csv_files"])
        )
        codes = ", ".join(
            f'<a href="{_gh_code(c)}" target="_blank">{c}</a>'
            for c in sorted(tf["code_files"])
        )
        secs = ", ".join(sorted(set(tf["sections"])))
        rows_html.append(
            f"<tr><td>{_esc(tf['test'])}</td>"
            f"<td>{tf['n_results']}</td>"
            f"<td style='font-size:8pt'>{secs}</td>"
            f"<td style='font-size:8pt'>{csvs}</td>"
            f"<td style='font-size:8pt'>{codes}</td></tr>"
        )

    return (
        '<table class="overview-table">'
        "<thead><tr>"
        "<th>Test Family</th><th>N Results</th>"
        "<th>Used In</th><th>CSV Files</th><th>Code</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows_html)
        + "</tbody></table>"
    )


def render_section(sec: Section, idx: int) -> str:
    """Render one collapsible section."""
    parts = []
    sec_id = f"sec-{idx}"

    # Header
    page_badge = ""
    if sec.figure_page:
        page_badge = (
            f'<span class="page-badge">Figure {sec.figure_page}: '
            f'{_esc(sec.figure_name or "")}</span>'
        )

    parts.append(f'<div class="section" id="{sec_id}">')
    parts.append(f'<div class="section-header">')
    parts.append(f'<h2>{idx}. {_esc(sec.title)}</h2>')
    parts.append(page_badge)
    parts.append('</div>')
    parts.append(f'<p class="desc">{_esc(sec.description)}</p>')

    # Subtables
    for si, sub in enumerate(sec.subtables):
        sub_id = f"{sec_id}-sub-{si}"
        parts.append(f'<details id="{sub_id}">')
        parts.append(f'<summary>{_esc(sub.title)}</summary>')
        parts.append(f'<p class="desc">{_esc(sub.description)}</p>')

        # Links
        links = []
        if sub.csv_file:
            links.append(
                f'<a href="{_gh_csv(sub.csv_file)}" target="_blank">'
                f'📊 results/{sub.csv_file}</a>'
            )
        if sub.code_file:
            url = _gh_code(sub.code_file, sub.code_line)
            links.append(
                f'<a href="{url}" target="_blank">'
                f'💻 {sub.code_file}'
                + (f"#L{sub.code_line}" if sub.code_line else "")
                + '</a>'
            )
        if links:
            parts.append(f'<p class="links">{" ".join(links)}</p>')

        # Summary
        if sub.summary:
            parts.append(f'<p class="summary">{_esc(sub.summary)}</p>')

        # Table of results
        if sub.rows:
            parts.append('<table><thead><tr>')
            parts.append('<th>Context</th><th>Test</th><th>Statistic</th>')
            parts.append('<th>N</th><th>p-value</th><th>Sig</th>')
            if any(r.detail for r in sub.rows):
                parts.append('<th>Detail</th>')
            parts.append('</tr></thead><tbody>')

            for r in sub.rows:
                sig_class = "sig" if any(
                    c in r.significance for c in ("*",)
                ) else "ns"
                parts.append(
                    f'<tr>'
                    f'<td>{_esc(r.context)}</td>'
                    f'<td>{_esc(r.test_name)}</td>'
                    f'<td>{_esc(r.statistic)}</td>'
                    f'<td>{_esc(r.n)}</td>'
                    f'<td>{_esc(r.p_value)}</td>'
                    f'<td class="sig-cell {sig_class}">{_esc(r.significance)}</td>'
                )
                if any(rr.detail for rr in sub.rows):
                    parts.append(f'<td>{_esc(r.detail)}</td>')
                parts.append('</tr>')

            parts.append('</tbody></table>')

        parts.append('</details>')

    parts.append('</div>')
    return "\n".join(parts)


def render_html(sections: list[Section]) -> str:
    """Full HTML document."""
    parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="UTF-8">',
        "<title>Methods — Opioid Conversion Experiment</title>",
        _CSS,
        _JS,
        "</head><body>",
        "<h1>Methods</h1>",
        '<p class="desc">This document describes the experimental design, '
        "implementation, analysis pipeline, and statistical methods used in "
        "the opioid-conversion-experiment study. All source code and data are "
        f'available at <a href="{GITHUB_REPO}" target="_blank">{GITHUB_REPO}</a>.</p>',
    ]

    # Methods overview
    parts.append(_render_methods_overview())

    # Table of contents for statistical sections
    parts.append('<div class="toc"><strong>Statistical Analyses by Figure</strong><ol>')
    for i, sec in enumerate(sections, 1):
        page = f" (Fig. {sec.figure_page})" if sec.figure_page else ""
        parts.append(
            f'<li><a href="#sec-{i}">{_esc(sec.title)}{page}</a></li>'
        )
    parts.append('</ol></div>')

    # Master test inventory table
    parts.append("<h2>Master Test Inventory</h2>")
    parts.append(
        '<div class="methods-prose"><p>The table below summarizes every '
        "statistical test family, the number of individual test results "
        "produced, the figures they support, and the CSV and code files "
        "where they are computed.</p></div>"
    )
    parts.append(render_overview_table(sections))

    # Statistical glossary
    parts.append(_render_glossary(sections))

    # Per-figure sections
    parts.append('<h2 id="per-figure">Detailed Results by Figure</h2>')
    parts.append(
        '<div class="methods-prose"><p>Each section below corresponds to one '
        "or more figures in the comprehensive figures PDF. Expand a section to "
        "see individual test results with observed statistics, sample sizes, "
        "and p-values.</p></div>"
    )
    parts.append(
        '<p><button onclick="toggleAll(true)">Expand All</button> '
        '<button onclick="toggleAll(false)">Collapse All</button></p>'
    )
    for i, sec in enumerate(sections, 1):
        parts.append(render_section(sec, i))

    parts.append("</body></html>")
    return "\n".join(parts)


# ── Main ────────────────────────────────────────────────────────────

def generate(output_html: str | None = None, output_pdf: bool = False):
    """Generate the statistical methods HTML (and optionally PDF)."""
    sections = build_all_sections()
    html_content = render_html(sections)

    if output_html is None:
        output_html = str(FIGURES_DIR / "statistical_methods.html")

    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"✓ Saved {output_html}")

    if output_pdf:
        _generate_pdf(output_html)


def _generate_pdf(html_path: str):
    """Convert HTML to PDF via playwright."""
    import asyncio

    async def _convert():
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.goto(f"file://{os.path.abspath(html_path)}")
            # Expand all details
            await page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
            await page.wait_for_timeout(500)
            pdf_path = html_path.replace(".html", ".pdf")
            await page.pdf(
                path=pdf_path,
                format="Letter",
                margin={"top": "0.5in", "bottom": "0.5in",
                        "left": "0.5in", "right": "0.5in"},
                print_background=True,
            )
            await browser.close()
            print(f"✓ Saved {pdf_path}")

    asyncio.run(_convert())


if __name__ == "__main__":
    import sys
    generate(output_pdf="--pdf" in sys.argv)
