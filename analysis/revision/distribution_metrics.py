"""
Task 3A: distribution metrics, extreme errors and threshold-free paired tests.

    python -m analysis.revision.distribution_metrics

Writes (atomically, deterministic, no randomness):
  results_revision/distribution_summary.csv     model x experiment x condition x score (+ POOLED)
  results_revision/extreme_errors.csv           DZ 7 table, Nielsen score
  results_revision/wilcoxon_threshold_free.csv  paired Wilcoxon, median |log ratio| next to tight proportion
  (the safety tails overdose, severe_overdose and underdose are registry entries
   in analysis/revision/tolerances.py and appear as pct_<entry> columns of
   distribution_summary.csv like tight and broad; POOLED rows also give
   n_ge_10x_highest, the count of responses >= 10x the highest standard)
  results_revision/safety_exceedance.csv        % of responses above each multiple (1x-10x) of the
                                                highest current standard, per condition x model (+ POOLED)
  extreme_errors.csv is the supplementary DZ 7 table: > 2x / < 0.5x of the score's
  expected dose (the reviewer's literal question).

Conventions
  * Text vignettes are the vignette experiment; empty prompts are separate
    supplementary conditions (role = supplementary_empty_prompts).
  * subset "ome" = oral morphine is source or target (the pipeline's flag).
  * Model rows: pct_<band> = mean over items of the item's mean cell proportion
    (the ranking average); distribution metrics are over responses.
    POOLED rows: median_fold_error is pooled over all responses
    (median_fold_error_basis = "pooled_responses"); mean_model_median_fold_error
    is the mean of the 13 per-model medians (the basis used in the report's
    findings and the circularity tables); pct_<band> = mean of the 13 model values.
  * Paired tests: pairing unit = conversion item; positive diff = condition b better.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.revision import stats
from analysis.revision.conditions import CONDITIONS
from analysis.revision.scores import SCORES
from analysis.revision.tolerances import BANDS

OUT = "results_revision"
MAIN = ["vignette_naive", "vignette_nielsen", "vignette_nielsen_units",
        "rd_naive", "rd_nielsen", "rd_nielsen_examples",
        "eu_4h_nielsen", "eu_4h_nielsen_units"]
SUPP = ["vignette_empty_naive", "vignette_empty_nielsen", "vignette_empty_nielsen_units"]


def write_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUT, name)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, path)


def role(cond: str) -> str:
    return "supplementary_empty_prompts" if cond in SUPP else "main"


def _dist(d: pd.DataFrame) -> dict:
    lr2 = d["log_ratio"].dropna() / np.log(2)
    return {"n_responses": len(d),
            "median_fold_error": float(np.nanmedian(d["fold_error"])) if d["fold_error"].notna().any() else np.nan,
            "median_log2_ratio": float(lr2.median()) if len(lr2) else np.nan,
            "iqr_log2_ratio": float(lr2.quantile(.75) - lr2.quantile(.25)) if len(lr2) else np.nan,
            "pct_over_2x": 100 * float((d["ratio"] > stats.OVERDOSE).mean()),
            "pct_under_half": 100 * float((d["ratio"] < stats.UNDERDOSE).mean())}


def summary_rows(long, score, cond) -> list[dict]:
    bands = list(BANDS)
    ms = stats.model_summary(long, score, cond, "all", bands, average="item")
    c = CONDITIONS[cond]
    base = {"experiment": c.experiment, "condition": cond, "condition_label": c.label,
            "role": role(cond), "score": score}
    rows = [{**base, "model": r["model"], **{k: r[k] for k in ms.columns
            if k not in ("score", "condition", "subset", "model")}} for _, r in ms.iterrows()]
    d = stats.responses(long, score, cond, "all", bands)
    pooled = {**base, "model": "POOLED", **_dist(d), "n_items": int(d[c.item].nunique()),
              "median_fold_error_basis": "pooled_responses",
              "mean_model_median_fold_error": float(ms["median_fold_error"].mean()),
              "n_ge_10x_highest": int(stats.gross_error_count(long, cond).set_index("model").loc["POOLED", "n_ge_10x_highest"])}
    for r in rows:
        r["median_fold_error_basis"] = "model_responses"
    for b in bands:
        pooled[f"pct_{b}"] = float(ms[f"pct_{b}"].mean())
    rows.append(pooled)
    return rows


def build_summary(long) -> pd.DataFrame:
    rows = []
    for score in SCORES:
        for cond in MAIN + SUPP:
            rows += summary_rows(long, score, cond)
    cols = ["experiment", "condition", "condition_label", "role", "score", "model", "n_responses", "n_items",
            "median_fold_error", "median_fold_error_basis", "mean_model_median_fold_error",
            "median_log2_ratio", "iqr_log2_ratio"] + [f"pct_{b}" for b in BANDS] + \
           ["pct_over_2x", "pct_under_half", "n_ge_10x_highest"]
    return pd.DataFrame(rows).reindex(columns=cols)


def build_extreme(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[summary.score == "nielsen"]
    cols = ["experiment", "condition", "condition_label", "role", "model", "n_responses", "pct_over_2x", "pct_under_half"]
    out = s[cols].copy()
    out["pct_extreme_total"] = out["pct_over_2x"] + out["pct_under_half"]
    out.insert(0, "score", "nielsen")
    return out


# ── threshold-free paired tests ──────────────────────────────────────────

# (name, label, cond_a, subset_a, cond_b, subset_b, subsets for the split, role)
COMPARISONS = [
    ("grounding_vignettes", "Reference grounding, text vignettes", "vignette_naive", "vignette_nielsen", ("all", "ome", "nonome"), "main"),
    ("grounding_random_doses", "Reference grounding, random doses", "rd_naive", "rd_nielsen", ("all", "ome", "nonome"), "main"),
    ("units_effect_vignettes", "Unit explanation, text vignettes", "vignette_nielsen", "vignette_nielsen_units", ("all", "ome", "nonome"), "main"),
    ("unit_explanation_random_doses", "Unit explanation, random doses (mg/4 h)", "eu_4h_nielsen", "eu_4h_nielsen_units", ("all",), "main"),
    ("dosing_unit", "Dosing unit: mg/day (random doses, ome) vs mg/4 h", "rd_nielsen", "eu_4h_nielsen", ("ome",), "main"),
    ("grounding_empty_prompts", "Reference grounding, empty prompts", "vignette_empty_naive", "vignette_empty_nielsen", ("all", "ome", "nonome"), "supplementary_empty_prompts"),
]


def _paired(long, score, ca, sa, cb, sb, outcome) -> pd.DataFrame:
    """Per-model and POOLED Wilcoxon; a and b may use different subsets
    (the dosing-unit design). Positive diff = b better."""
    bands = [] if outcome == stats.THRESHOLD_FREE else [outcome]
    col = outcome if outcome == stats.THRESHOLD_FREE else f"prop_{outcome}"
    a = stats.item_table(long, score, ca, sa, bands)[["model", "item", col]]
    b = stats.item_table(long, score, cb, sb, bands)[["model", "item", col]]
    m = a.merge(b, on=["model", "item"], suffixes=("_a", "_b"))
    m["diff"] = (m[f"{col}_a"] - m[f"{col}_b"]) if outcome == stats.THRESHOLD_FREE else (m[f"{col}_b"] - m[f"{col}_a"])
    rows = []
    for model, g in list(m.groupby("model")) + [("POOLED", m)]:
        p, med, mean, n = stats._wilcoxon(g["diff"])
        rows.append({"model": model, "n_pairs": n, "p": p, "median_diff": med, "mean_diff": mean,
                     "mean_a": float(g[f"{col}_a"].mean()), "mean_b": float(g[f"{col}_b"].mean())})
    return pd.DataFrame(rows)


def build_wilcoxon(long) -> pd.DataFrame:
    out = []
    for score in SCORES:
        for name, label, ca, cb, subsets, rl in COMPARISONS:
            for sub in subsets:
                sa = sub                                   # dosing-unit: rd subset ome vs all eu items
                sb = "all" if name == "dosing_unit" else sub
                tf = _paired(long, score, ca, sa, cb, sb, stats.THRESHOLD_FREE).set_index("model")
                ti = _paired(long, score, ca, sa, cb, sb, "tight").set_index("model")
                for model in tf.index:
                    t, b = tf.loc[model], ti.loc[model]
                    out.append({"score": score, "comparison": name, "comparison_label": label, "role": rl,
                                "cond_a": ca, "cond_b": cb, "subset": sub, "model": model,
                                "n_pairs": int(t.n_pairs),
                                "tf_mean_a": t.mean_a, "tf_mean_b": t.mean_b, "tf_median_diff": t.median_diff,
                                "tf_mean_diff": t.mean_diff, "tf_p": t.p,
                                "tight_n_pairs": int(b.n_pairs),
                                "tight_mean_a": b.mean_a, "tight_mean_b": b.mean_b, "tight_median_diff": b.median_diff,
                                "tight_mean_diff": b.mean_diff, "tight_p": b.p})
    df = pd.DataFrame(out)
    df["tf_sig_005"] = df["tf_p"] < 0.05
    df["tight_sig_005"] = df["tight_p"] < 0.05
    return df


def main() -> None:
    long = stats.load_long()
    summary = build_summary(long)
    write_csv(summary, "distribution_summary.csv")
    write_csv(build_extreme(summary), "extreme_errors.csv")
    write_csv(build_wilcoxon(long), "wilcoxon_threshold_free.csv")
    exc = []
    for cond in MAIN + SUPP:
        c = CONDITIONS[cond]
        e = stats.overdose_exceedance(long, cond)
        e.insert(0, "role", role(cond)); e.insert(0, "condition_label", c.label); e.insert(0, "experiment", c.experiment)
        exc.append(e)
    write_csv(pd.concat(exc, ignore_index=True), "safety_exceedance.csv")
    print("wrote distribution_summary.csv, extreme_errors.csv, wilcoxon_threshold_free.csv, safety_exceedance.csv")


if __name__ == "__main__":
    main()
