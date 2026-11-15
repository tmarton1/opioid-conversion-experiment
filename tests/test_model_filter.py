"""Task 0.2: ranking and Wilcoxon outputs use only the 13 CONSISTENT_MODELS."""
import pandas as pd

from analysis.model_ranking import ModelRanking
from analysis.per_model_wilcoxon import compute_per_model_wilcoxon
from common.models import CONSISTENT_MODELS


def test_ranking_restricted_to_consistent_models():
    r = ModelRanking(
        csv_path="results/pair_model_analysis.csv",
        filters={"include_nielsen": False, "include_examples": False},
        unit_col="opioid_pair",
        label="Naive Random Doses",
    )
    df = r.build()
    assert set(df["model"]) == set(CONSISTENT_MODELS)


def test_wilcoxon_restricted_to_consistent_models():
    src = "results/comparison_nielsen_effect_random_dose.csv"
    raw = pd.read_csv(src)
    assert set(raw["model"]) - set(CONSISTENT_MODELS), "fixture should contain extra models"
    w = compute_per_model_wilcoxon(src)
    assert set(w["model"]) - {"POOLED"} == set(CONSISTENT_MODELS)
    pooled_n = int(w.loc[w["model"] == "POOLED", "tight_n"].iloc[0])
    assert pooled_n == raw[raw["model"].isin(CONSISTENT_MODELS)]["tight_diff"].notna().sum()
