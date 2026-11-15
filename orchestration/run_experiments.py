"""
Run all LLM opioid conversion experiments and append results to CSV files.

This script replaces the experiment-running cells in opioid-conversion.ipynb.
It is designed to be re-run safely: each Experiment/ScenarioExperiment call
appends to its output CSV and assigns a new experiment_id, so partial runs are
never overwritten.

Sections
--------
1. Random-dose opioid pair experiment (main, covering all prompt conditions)
2. Explain-units experiment (mg/4 hours target unit)
3. Vignette scenario experiment (reworded + empty-text vignettes)

Usage
-----
    python run_experiments.py [--section SECTION]

    SECTION can be: pair | explain_units | vignettes | empty_vignettes | all
    Default: all

Run from the repository root:
    python orchestration/run_experiments.py [--section SECTION]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import RESULTS_DIR

import argparse
import importlib

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Run opioid conversion experiments")
    p.add_argument(
        "--section",
        choices=["pair", "explain_units", "vignettes", "empty_vignettes", "all"],
        default="all",
        help="Which experiment section to run (default: all)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Section 1 – Random-dose opioid pair experiment
# ---------------------------------------------------------------------------

def run_pair_experiment():
    """
    Run the main opioid-pair experiment across all prompt conditions and models.

    Experiments:
    - OME opioids (priority ≤ 4) vs oral morphine (mg/day) — naive, Nielsen
    - Bidirectional pairs (priority 1) — naive, Nielsen

    Output: results/experiment_results.csv
    """
    from common.opioid import OpioidPairGenerator
    from experiment.experiment import Experiment
    from experiment.prompt_service import PromptedOpioidConversion
    from common.models import CONSISTENT_MODELS

    PromptedOpioidConversion.initialize_client()

    opioid_list_ome = (
        OpioidPairGenerator(lambda priority: priority <= 4 if priority else False)
        .generate_opioid_list_or_pairs(pairs=False, use_permutations=False)
    )

    opioid_pairs = (
        OpioidPairGenerator(lambda priority: priority == 1 if priority else False)
        .generate_opioid_list_or_pairs(pairs=True, use_permutations=True)
    )

    prompt_conditions = [
        {"include_nielsen": False, "include_examples": False},
        {"include_nielsen": True, "include_examples": False},
        {"include_nielsen": True, "include_examples": True},
    ]

    output_csv = f"{RESULTS_DIR}/experiment_results.csv"

    print("── Running OME opioid pair experiment ──")
    exp_ome = Experiment(
        opioid_list_ome, prompt_conditions, CONSISTENT_MODELS,
        repetitions=25, filename=output_csv,
    )
    exp_ome.run()
    print(f"✓ OME experiment ID: {exp_ome.experiment_id}")

    print("── Running bidirectional pairs experiment ──")
    exp_pairs = Experiment(
        opioid_pairs, prompt_conditions, CONSISTENT_MODELS,
        repetitions=25, filename=output_csv,
    )
    exp_pairs.run()
    print(f"✓ Pairs experiment ID: {exp_pairs.experiment_id}")


# ---------------------------------------------------------------------------
# Section 2 – Explain-units experiment
# ---------------------------------------------------------------------------

def run_explain_units_experiment():
    """
    Run the explain-units experiment using a mg/4 hours target unit.

    Prompt conditions: Nielsen + ask_for_units, with and without explain_units.
    Covers all priority ≤ 4 OME opioids.

    Output: results/explain_units_experiment_results.csv
    """
    from common.opioid import Opioid, OpioidPairGenerator
    from experiment.experiment import Experiment
    from experiment.prompt_service import PromptedOpioidConversion
    from common.models import CONSISTENT_MODELS

    PromptedOpioidConversion.initialize_client()

    from_opioids = (
        OpioidPairGenerator(lambda priority: priority <= 4 if priority else False)
        .generate_opioid_list_or_pairs(pairs=False, use_permutations=False)
    )

    to_opioid = Opioid(
        administration_method="oral", substance="morphine", unit="mg/4 hours"
    )
    opioid_pairs = [(from_opioid, to_opioid) for from_opioid in from_opioids]

    prompt_conditions = [
        {"include_nielsen": True, "include_examples": False,
         "ask_for_units": True, "explain_units": True},
        {"include_nielsen": True, "include_examples": False,
         "ask_for_units": True, "explain_units": False},
    ]

    output_csv = f"{RESULTS_DIR}/explain_units_experiment_results.csv"

    print("── Running explain-units experiment ──")
    exp = Experiment(
        opioid_pairs, prompt_conditions, CONSISTENT_MODELS,
        repetitions=25, filename=output_csv,
    )
    exp.run()
    print(f"✓ Explain-units experiment ID: {exp.experiment_id}")


# ---------------------------------------------------------------------------
# Section 3 – Vignette scenario experiment (text vignettes)
# ---------------------------------------------------------------------------

def run_vignette_experiment():
    """
    Run the scenario vignette experiment across all models and prompt conditions.

    Covers vignettes v2–v81 (text vignettes with 4 wordings per scenario).
    Prompt conditions: naive (no reference).

    Output: results/reworded_vignettes_results.csv
    """
    from resources.scenario_vignettes import SCENARIO_VIGNETTES
    from experiment.scenario import Scenario, ScenarioPromptedConversion
    from experiment.scenario_experiment import ScenarioExperiment
    from experiment.prompt_service import PromptedOpioidConversion
    from common.models import CONSISTENT_MODELS

    PromptedOpioidConversion.initialize_client()

    vignette_ids = [f"v{i}" for i in range(2, 82)]

    scenarios = [
        ScenarioPromptedConversion(
            Scenario(SCENARIO_VIGNETTES[v_id]),
            model=None,
            include_nielsen=False,
        )
        for v_id in vignette_ids
    ]

    prompt_conditions = [
        {"include_nielsen": False, "include_examples": False},
        {"include_nielsen": True, "include_examples": False},
        {"include_nielsen": True, "include_examples": False,
         "ask_for_units": True, "explain_units": True},
    ]

    output_csv = f"{RESULTS_DIR}/reworded_vignettes_results.csv"

    print(f"── Running vignette experiment: {len(vignette_ids)} vignettes ──")
    exp = ScenarioExperiment(
        scenarios=scenarios,
        prompt_conditions=prompt_conditions,
        models=CONSISTENT_MODELS,
        repetitions=7,
        filename=output_csv,
    )
    exp.run()
    print(f"✓ Vignette experiment ID: {exp.experiment_id}")


# ---------------------------------------------------------------------------
# Section 4 – Empty-text vignette experiment
# ---------------------------------------------------------------------------

def run_empty_vignette_experiment():
    """
    Build empty-text vignette entries and run the experiment for them.

    build_scenarios() creates one empty-text vignette per scenario (IDs v82+)
    in resources/reworded_vignettes.json and returns their IDs.

    Output: appends to results/reworded_vignettes_results.csv
    """
    from resources.build_scenarios import build_scenarios

    print("── Building empty-text vignettes ──")
    empty_vignette_ids = build_scenarios("resources/reworded_vignettes.json")
    print(f"Empty-text vignettes: {empty_vignette_ids}")

    # Reload to pick up new entries
    import resources.scenario_vignettes
    importlib.reload(resources.scenario_vignettes)
    from resources.scenario_vignettes import SCENARIO_VIGNETTES

    from experiment.scenario import Scenario, ScenarioPromptedConversion
    from experiment.scenario_experiment import ScenarioExperiment
    from experiment.prompt_service import PromptedOpioidConversion
    from common.models import CONSISTENT_MODELS

    PromptedOpioidConversion.initialize_client()

    scenarios = [
        ScenarioPromptedConversion(
            Scenario(SCENARIO_VIGNETTES[v_id]),
            model=None,
            include_nielsen=False,
        )
        for v_id in empty_vignette_ids
    ]

    prompt_conditions = [
        {"include_nielsen": False, "include_examples": False},
        {"include_nielsen": True, "include_examples": False},
        {"include_nielsen": True, "include_examples": False,
         "ask_for_units": True, "explain_units": True},
    ]

    output_csv = f"{RESULTS_DIR}/reworded_vignettes_results.csv"

    print(f"── Running empty-vignette experiment: {len(empty_vignette_ids)} vignettes ──")
    exp = ScenarioExperiment(
        scenarios=scenarios,
        prompt_conditions=prompt_conditions,
        models=CONSISTENT_MODELS,
        repetitions=7,
        filename=output_csv,
    )
    exp.run()
    print(f"✓ Empty-vignette experiment ID: {exp.experiment_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    section = args.section

    if section in ("pair", "all"):
        run_pair_experiment()
    if section in ("explain_units", "all"):
        run_explain_units_experiment()
    if section in ("vignettes", "all"):
        run_vignette_experiment()
    if section in ("empty_vignettes", "all"):
        run_empty_vignette_experiment()
