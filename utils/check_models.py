"""
Check which models have incomplete data across random-dose and vignette experiments.
Run from the repo root: python utils/check_models.py
"""

import pandas as pd
from common.models import CONSISTENT_MODELS


def check_completeness(
    pair_csv: str = "results/pair_model_analysis.csv",
    vignette_csv: str = "results/reworded_vignettes_model_analysis.csv",
    reference_model: str = "anthropic/claude-3.5-sonnet:20241022",
):
    pair_df = pd.read_csv(pair_csv)
    vig_df = pd.read_csv(vignette_csv)

    pair_models = set(pair_df["model"].unique())
    vig_models = set(vig_df["model"].unique())

    print("=== Random-dose models ===")
    naive = pair_df[
        (pair_df["include_nielsen"] == False) & (pair_df["include_examples"] == False)
    ]
    for m in sorted(pair_models):
        n_pairs = naive[naive["model"] == m]["opioid_pair"].nunique()
        total = len(pair_df[pair_df["model"] == m])
        flag = " ⚠" if n_pairs < 49 else ""
        print(f"  {m}: {n_pairs} pairs, {total} rows{flag}")

    print("\n=== Vignette models ===")
    ref = vig_df[vig_df["model"] == reference_model]
    ref_keys = set(
        ref.groupby(
            ["scenario_key", "include_nielsen", "explain_units", "has_vignette_text"]
        ).groups.keys()
    )
    for m in sorted(vig_models):
        mdf = vig_df[vig_df["model"] == m]
        n = len(mdf)
        m_keys = set(
            mdf.groupby(
                ["scenario_key", "include_nielsen", "explain_units", "has_vignette_text"]
            ).groups.keys()
        )
        missing = len(ref_keys - m_keys)
        flag = f" ⚠ missing {missing} combos" if missing else ""
        print(f"  {m}: {n} rows{flag}")

    print("\n=== Only in random-dose (not in vignettes) ===")
    print(f"  {pair_models - vig_models or 'none'}")

    print("\n=== Only in vignettes (not in random-dose) ===")
    print(f"  {vig_models - pair_models or 'none'}")

    print(f"\n=== CONSISTENT_MODELS ({len(CONSISTENT_MODELS)}) ===")
    for m in CONSISTENT_MODELS:
        in_pair = "✓" if m in pair_models else "✗"
        in_vig = "✓" if m in vig_models else "✗"
        print(f"  {in_pair} pair  {in_vig} vig  {m}")


if __name__ == "__main__":
    check_completeness()
