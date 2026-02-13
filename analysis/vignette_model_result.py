import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import csv
from visualize.visualization_config import EPSILON_FOR_WEIGHT, TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER, BROAD_RANGE_LOWER, BROAD_RANGE_UPPER
from analysis.opioid_pair_consensus import OpioidPairConsensus

class VignetteModelResult:
    """
    Analyzes results for a specific vignette, model, and prompt conditions.
    Calculates observed/expected coefficients and summary statistics.
    """

    def __init__(self, 
                 vignette_id: str,
                 scenario_key: str,
                 from_opioid: str,
                 from_opioid_unit: str,
                 to_opioid: str,
                 to_opioid_unit: str,
                 multiple_from_opioids: bool,
                 ome: bool,
                 model: str,
                 prompt_conditions: Dict[str, bool],
                 results_df: pd.DataFrame,
                 has_vignette_text: bool = True):
        self.vignette_id = vignette_id
        self.scenario_key = scenario_key
        self.from_opioid = from_opioid
        self.from_opioid_unit = from_opioid_unit
        self.to_opioid = to_opioid
        self.to_opioid_unit = to_opioid_unit
        self.multiple_from_opioids = multiple_from_opioids
        self.ome = ome
        self.has_vignette_text = has_vignette_text
        self.model = model
        self.prompt_conditions = prompt_conditions
        self.results_df = results_df
        self.expected_to_dose = results_df['expected_to_dose'].iloc[0]

        self.observed_dose_mean = None
        self.observed_dose_std = None
        self.observed_dose_cv = None
        self.observed_log_dose_mean = None
        self.observed_log_dose_std = None
        self.observed_log_dose_cv = None
        self.weight = None

    def calculate_dose_stats(self):
        doses = self.results_df['observed_to_dose']
        doses = doses[doses > 0]
        if len(doses) == 0:
            raise ValueError(f"No positive doses for vignette {self.vignette_id}, model {self.model}")
        self.observed_dose_mean = np.mean(doses)
        self.observed_dose_std = np.std(doses, ddof=1)
        if self.observed_dose_mean != 0:
            self.observed_dose_cv = self.observed_dose_std / abs(self.observed_dose_mean)
        else:
            self.observed_dose_cv = np.nan
        log_doses = np.log(doses)
        self.observed_log_dose_mean = float(np.mean(log_doses))
        self.observed_log_dose_std = float(np.std(log_doses, ddof=1))
        if self.observed_log_dose_mean != 0:
            self.observed_log_dose_cv = self.observed_log_dose_std / abs(self.observed_log_dose_mean)
        else:
            self.observed_log_dose_cv = np.nan

    def calculate_weights(self):
        epsilon = EPSILON_FOR_WEIGHT
        if self.observed_log_dose_cv is not None:
            self.weight = 1.0 / (self.observed_log_dose_cv ** 2 + epsilon)
        else:
            self.weight = np.nan

    def calculate_proportion_within_range(self, lower: float, upper: float) -> float:
        observed = self.results_df['observed_to_dose']
        expected = self.results_df['expected_to_dose']
        ratios = observed / expected
        mask = (ratios >= lower) & (ratios <= upper)
        return mask.mean() if len(mask) > 0 else np.nan

    def calculate_all(self):
        self.calculate_dose_stats()
        self.calculate_weights()
        self.tight_proportion = self.calculate_proportion_within_range(TIGHT_RANGE_LOWER, TIGHT_RANGE_UPPER)
        self.broad_proportion = self.calculate_proportion_within_range(BROAD_RANGE_LOWER, BROAD_RANGE_UPPER)

    def to_dict(self):
        tight_col = f"tight({TIGHT_RANGE_LOWER},{TIGHT_RANGE_UPPER})"
        broad_col = f"broad({BROAD_RANGE_LOWER},{BROAD_RANGE_UPPER})"
        return {
            'vignette_id': self.vignette_id,
            'scenario_key': self.scenario_key,
            'from_opioid': self.from_opioid,
            'from_opioid_unit': self.from_opioid_unit,
            'to_opioid': self.to_opioid,
            'to_opioid_unit': self.to_opioid_unit,
            'multiple_from_opioids': self.multiple_from_opioids,
            'ome': self.ome,
            'has_vignette_text': self.has_vignette_text,
            'model': self.model,
            'include_nielsen': self.prompt_conditions.get('include_nielsen', False),
            'include_examples': self.prompt_conditions.get('include_examples', False),
            'ask_for_units': self.prompt_conditions.get('ask_for_units', False),
            'explain_units': self.prompt_conditions.get('explain_units', False),
            'expected_to_dose': self.expected_to_dose,
            'observed_dose_mean': self.observed_dose_mean,
            'observed_dose_std': self.observed_dose_std,
            'observed_dose_cv': self.observed_dose_cv,
            'observed_log_dose_mean': self.observed_log_dose_mean,
            'observed_log_dose_std': self.observed_log_dose_std,
            'observed_log_dose_cv': self.observed_log_dose_cv,  # <-- Add this line
            'weight': self.weight,
            'n_observations': len(self.results_df),
            tight_col: self.tight_proportion,
            broad_col: self.broad_proportion
        }

    def append_to_csv(self, csv_file: str):
        csv_path = Path(csv_file)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        result_dict = self.to_dict()
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_dict)

    @classmethod
    def from_csv_filter(cls,
                    csv_file: str,
                    vignette_id: str,
                    scenario_key: str,
                    from_opioid: str,
                    from_opioid_unit: str,
                    multiple_from_opioids: bool,
                    ome: bool,
                    model: str,
                    prompt_conditions: Dict[str, bool]):
        """
        Create a VignetteModelResult by filtering a CSV file.

        Args:
            csv_file: Path to scenario experiment results CSV
            vignette_id: Vignette identifier
            scenario_key: Scenario key
            from_opioid: Source opioid
            from_opioid_unit: Source opioid unit
            multiple_from_opioids: Multiple opioids flag
            ome: OME flag
            model: Model name
            prompt_conditions: Dict with 'include_nielsen' and 'include_examples' keys

        Returns:
            VignetteModelResult instance with filtered data
        """
        df = pd.read_csv(csv_file)
        filtered_df = df[
            (df['vignette_id'] == vignette_id) &
            (df['scenario_key'] == scenario_key) &
            (df['from_opioid'] == from_opioid) &
            (df['from_opioid_unit'] == from_opioid_unit) &
            (df['multiple_from_opioids'] == multiple_from_opioids) &
            (df['ome'] == ome) &
            (df['model'] == model) &
            (df['include_nielsen'] == prompt_conditions.get('include_nielsen', False)) &
            (df['include_examples'] == prompt_conditions.get('include_examples', False)) &
            (df.get('ask_for_units', False) == prompt_conditions.get('ask_for_units', False)) &
            (df.get('explain_units', False) == prompt_conditions.get('explain_units', False))
        ].copy()

        if len(filtered_df) == 0:
            raise ValueError(
                f"No data found for combination:\n"
                f"  vignette_id: {vignette_id}\n"
                f"  scenario_key: {scenario_key}\n"
                f"  from_opioid: {from_opioid}\n"
                f"  from_opioid_unit: {from_opioid_unit}\n"
                f"  multiple_from_opioids: {multiple_from_opioids}\n"
                f"  ome: {ome}\n"
                f"  model: {model}\n"
                f"  prompt_conditions: {prompt_conditions}"
            )

        return cls(
            vignette_id=vignette_id,
            scenario_key=scenario_key,
            from_opioid=from_opioid,
            from_opioid_unit=from_opioid_unit,
            multiple_from_opioids=multiple_from_opioids,
            ome=ome,
            model=model,
            prompt_conditions=prompt_conditions,
            results_df=filtered_df
        )
    
    @classmethod
    def process_all_combinations(cls, csv_file: str, output_file: str, verbose: bool = True):
        df = pd.read_csv(csv_file)
        groups = df.groupby([
            'vignette_id', 'scenario_key', 'from_opioid', 'from_opioid_unit', 'to_opioid', 'to_opioid_unit','multiple_from_opioids', 'ome', 'has_vignette_text',
            'model', 'include_nielsen', 'include_examples', 'ask_for_units', 'explain_units'
        ])
        total = len(groups)
        processed = 0
        results_list = []
        output_path = Path(output_file)
        if output_path.exists():
            output_path.unlink()
        for (vignette_id, scenario_key, from_opioid, from_opioid_unit, to_opioid, to_opioid_unit, multiple_from_opioids, ome, has_vignette_text,
             model, include_nielsen, include_examples, ask_for_units, explain_units), group_df in groups:
            prompt_conditions = {
                'include_nielsen': include_nielsen,
                'include_examples': include_examples,
                'ask_for_units': ask_for_units,
                'explain_units': explain_units
            }
            analyzer = cls(
                vignette_id=vignette_id,
                scenario_key=scenario_key,
                from_opioid=from_opioid,
                from_opioid_unit=from_opioid_unit,
                to_opioid=to_opioid,
                to_opioid_unit=to_opioid_unit,
                multiple_from_opioids=multiple_from_opioids,
                ome=ome,
                has_vignette_text=has_vignette_text,
                model=model,
                prompt_conditions=prompt_conditions,
                results_df=group_df
            )
            try:
                analyzer.calculate_all()
                analyzer.append_to_csv(output_file)
                results_list.append(analyzer)
                processed += 1
                if verbose and processed % 50 == 0:
                    print(f"Processed {processed}/{total} combinations...")
            except Exception as e:
                print(f"Error processing vignette {vignette_id}, model {model}: {e}")
        if verbose:
            print(f"Completed! Processed {processed}/{total} combinations.")
            print(f"Results saved to: {output_file}")
        return results_list
    

class VignetteModelConsensus(OpioidPairConsensus):
    @staticmethod
    def calculate_consensus_stats(group_df):
        import numpy as np
        from scipy import stats

        weights = group_df['weight'].astype(float)
        log_doses = group_df['observed_log_dose_mean'].astype(float)
        expected_to_dose = group_df["expected_to_dose"].iloc[0] if "expected_to_dose" in group_df else np.nan

        if weights.sum() > 0:
            consensus_log_dose = np.average(log_doses, weights=weights)
            consensus_dose = np.exp(consensus_log_dose)
            weighted_variance_log = np.average((log_doses - consensus_log_dose) ** 2, weights=weights)
            effective_n_models = (weights.sum())**2 / (weights**2).sum()
        else:
            consensus_log_dose = np.nan
            consensus_dose = np.nan
            weighted_variance_log = np.nan
            effective_n_models = np.nan

        # Standard error and degrees of freedom
        if weights.sum() > 0 and effective_n_models > 1 and np.isfinite(weighted_variance_log):
            se_consensus_log_dose = np.sqrt(weighted_variance_log / effective_n_models)
            df_log_dose = effective_n_models - 1
        else:
            se_consensus_log_dose = np.nan
            df_log_dose = np.nan

        # t-test vs expected literature dose
        if (
            expected_to_dose is not None
            and expected_to_dose > 0
            and se_consensus_log_dose is not None
            and np.isfinite(se_consensus_log_dose)
            and se_consensus_log_dose > 0
        ):
            log_expected = np.log(expected_to_dose)
            t_score_log_dose = (consensus_log_dose - log_expected) / se_consensus_log_dose
            p_value_log_dose = 2 * (1 - stats.t.cdf(abs(t_score_log_dose), df=df_log_dose))
        else:
            t_score_log_dose = np.nan
            p_value_log_dose = np.nan

        return {
            'consensus_log_dose': consensus_log_dose,
            'consensus_dose': consensus_dose,
            'weighted_variance_log': weighted_variance_log,
            'se_consensus_log_dose': se_consensus_log_dose,
            'df_log': df_log_dose,
            't_score_log': t_score_log_dose,
            'p_value_log': p_value_log_dose,
            'effective_n_models': effective_n_models,
            'expected_to_dose': expected_to_dose,
        }

    @classmethod
    def process_all_combinations(cls, csv_file, output_file, verbose=True):
        import pandas as pd
        from pathlib import Path
        import csv

        df = pd.read_csv(csv_file)
        groups = df.groupby([
            'vignette_id', 'scenario_key', 'from_opioid', 'from_opioid_unit', 'to_opioid', 'to_opioid_unit',
            'multiple_from_opioids', 'ome', 'has_vignette_text', 'include_nielsen', 'include_examples', 'ask_for_units', 'explain_units'
        ])
        total = len(groups)
        processed = 0
        results_list = []
        output_path = Path(output_file)
        if output_path.exists():
            output_path.unlink()
        for keys, group_df in groups:
            (vignette_id, scenario_key, from_opioid, from_opioid_unit, to_opioid, to_opioid_unit,
             multiple_from_opioids, ome, has_vignette_text, include_nielsen, include_examples, ask_for_units, explain_units) = keys

            stats_dict = cls.calculate_consensus_stats(group_df)

            consensus_dict = {
                'vignette_id': vignette_id,
                'scenario_key': scenario_key,
                'from_opioid': from_opioid,
                'from_opioid_unit': from_opioid_unit,
                'to_opioid': to_opioid,
                'to_opioid_unit': to_opioid_unit,
                'multiple_from_opioids': multiple_from_opioids,
                'ome': ome,
                'has_vignette_text': has_vignette_text,
                'include_nielsen': include_nielsen,
                'include_examples': include_examples,
                'ask_for_units': ask_for_units,
                'explain_units': explain_units,
                  **stats_dict,
            }
            file_exists = output_path.exists() and output_path.stat().st_size > 0
            with open(output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=consensus_dict.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(consensus_dict)
            results_list.append(consensus_dict)
            processed += 1
            if verbose and processed % 50 == 0:
                print(f"Processed {processed}/{total} consensus groups...")
        if verbose:
            print(f"Completed! Processed {processed}/{total} consensus groups.")
            print(f"Consensus results saved to: {output_file}")
        return results_list