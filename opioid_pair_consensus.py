import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import csv
from scipy import stats

class OpioidPairConsensus:
    """
    Calculates consensus conversion coefficients across models for a specific opioid pair
    and prompt condition, using weighted averaging based on model weights.
    """
    
    def __init__(self,
                 from_opioid: str,
                 to_opioid: str,
                 prompt_conditions: Dict[str, bool],
                 analysis_df: pd.DataFrame,
                 expected_coefficient: Optional[float] = None):
        """
        Initialize the consensus analyzer.
        
        Args:
            from_opioid: Source opioid (e.g., "iv_fentanyl")
            to_opioid: Target opioid (e.g., "oral_morphine")
            prompt_conditions: Dict with 'include_nielson' and 'include_examples' keys
            analysis_df: DataFrame from pair_model_analysis.csv with model-level results
            expected_coefficient: Expected coefficient from Nielson data (literature value)
        """
        self.from_opioid = from_opioid
        self.to_opioid = to_opioid
        self.prompt_conditions = prompt_conditions
        self.analysis_df = analysis_df
        self.expected_coefficient = expected_coefficient
        
        # Results to be calculated
        self.consensus_coefficient = None
        self.normalized_weights = None  # v values
        self.effective_n_models = None
        self.weighted_variance = None
        self.se_consensus = None
        self.t_score = None
        self.p_value = None
        self.df = None
        
    def calculate_consensus(self):
        """
        Calculate weighted consensus coefficient and related statistics.
        """
        # Get observed coefficients and weights from each model
        observed_coeffs = self.analysis_df['observed_coefficient_mean'].values
        weights = self.analysis_df['weight'].values

        # per-model SDs (intrinsic per-response noise for each model)
        if 'observed_coefficient_std' in self.analysis_df.columns:
            per_model_sd = self.analysis_df['observed_coefficient_std'].values
        else:
            per_model_sd = None
        
        if len(observed_coeffs) == 0:
            raise ValueError(f"No model data found for {self.from_opioid} -> {self.to_opioid}")
        
        # Handle NaN or zero weights
        valid_mask = ~np.isnan(weights) & (weights > 0)
        if not np.any(valid_mask):
            raise ValueError(f"No valid weights for {self.from_opioid} -> {self.to_opioid}")
        
        observed_coeffs = observed_coeffs[valid_mask]
        weights = weights[valid_mask]
        if per_model_sd is not None:
            per_model_sd = per_model_sd[valid_mask]
        
        # Normalize weights so they sum to 1 (these are the v values)
        total_weight = np.sum(weights)
        self.normalized_weights = weights / total_weight
        
        # Calculate consensus coefficient as weighted average
        self.consensus_coefficient = np.sum(self.normalized_weights * observed_coeffs)
        
        # Effective number of models: 1 / sum(v^2)
        sum_v_squared = np.sum(self.normalized_weights ** 2)
        self.effective_n_models = 1.0 / sum_v_squared
        
        # Weighted variance: Sum(v * (a - a*)^2) / (1 - Sum(v^2))
        deviations_squared = (observed_coeffs - self.consensus_coefficient) ** 2
        numerator = np.sum(self.normalized_weights * deviations_squared)
        denominator = 1.0 - sum_v_squared
        
        if denominator > 0:
            self.weighted_variance = numerator / denominator
        else:
            self.weighted_variance = np.nan
        
        # Standard error of consensus: SE = sqrt(weighted_variance / effective_n_models)
        if self.weighted_variance is not None and self.effective_n_models > 0:
            self.se_consensus = np.sqrt(self.weighted_variance / self.effective_n_models)
        else:
            self.se_consensus = np.nan
        
        # Degrees of freedom
        self.df = self.effective_n_models - 1
        
        # t-score and p-value (if expected coefficient is provided)
        if self.expected_coefficient is not None and self.se_consensus is not None and not np.isnan(self.se_consensus):
            if self.se_consensus > 0:
                self.t_score = (self.consensus_coefficient - self.expected_coefficient) / self.se_consensus
                # Two-tailed p-value
                self.p_value = 2 * (1 - stats.t.cdf(abs(self.t_score), df=self.df))
            else:
                self.t_score = np.nan
                self.p_value = np.nan
        else:
            self.t_score = np.nan
            self.p_value = np.nan

        if per_model_sd is not None:
            # Within-model variance term: sum_j v_j * sigma_j^2
            within_var = np.sum(self.normalized_weights * (per_model_sd ** 2))

            # Between-model variance term (no Bessel correction here)
            between_var = np.sum(self.normalized_weights * deviations_squared)

            self.sigma_cons_sq = within_var + between_var
            self.sigma_cons = np.sqrt(self.sigma_cons_sq)
        else:
            self.sigma_cons_sq = np.nan
            self.sigma_cons = np.nan
    
    def to_dict(self) -> Dict:
        """
        Convert results to dictionary for CSV writing.
        
        Returns:
            Dictionary with all result fields
        """
        return {
            'from_opioid': self.from_opioid,
            'to_opioid': self.to_opioid,
            'opioid_pair': f"{self.from_opioid} -> {self.to_opioid}",
            'include_nielson': self.prompt_conditions.get('include_nielson', False),
            'include_examples': self.prompt_conditions.get('include_examples', False),
            'expected_coefficient': self.expected_coefficient,
            'consensus_coefficient': self.consensus_coefficient,
            'effective_n_models': self.effective_n_models,
            'weighted_variance': self.weighted_variance,
            'se_consensus': self.se_consensus,
            'sigma_cons': self.sigma_cons,
            't_score': self.t_score,
            'df': self.df,
            'p_value': self.p_value,
            'n_models': len(self.analysis_df)
        }
    
    def append_to_csv(self, csv_file: str):
        """
        Append results as a new row to CSV file.
        Creates file with headers if it doesn't exist.
        
        Args:
            csv_file: Path to CSV file
        """
        csv_path = Path(csv_file)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        
        # Get results as dictionary
        result_dict = self.to_dict()
        
        # Write to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_dict.keys())
            
            # Write header if file is new or empty
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(result_dict)
    
    @classmethod
    def from_csv_filter(cls,
                       csv_file: str,
                       from_opioid: str,
                       to_opioid: str,
                       prompt_conditions: Dict[str, bool],
                       conversion_service=None):
        """
        Create an OpioidPairConsensus by filtering a pair_model_analysis CSV file.
        
        Args:
            csv_file: Path to pair_model_analysis CSV
            from_opioid: Source opioid (e.g., "iv_fentanyl")
            to_opioid: Target opioid (e.g., "oral_morphine")
            prompt_conditions: Dict with 'include_nielson' and 'include_examples' keys
            conversion_service: Optional ConversionService to calculate expected coefficient
            
        Returns:
            OpioidPairConsensus instance with filtered data
        """
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Filter for the specific combination (all models for this pair and prompt condition)
        filtered_df = df[
            (df['from_opioid'] == from_opioid) &
            (df['to_opioid'] == to_opioid) &
            (df['include_nielson'] == prompt_conditions.get('include_nielson', False)) &
            (df['include_examples'] == prompt_conditions.get('include_examples', False))
        ].copy()
        
        if len(filtered_df) == 0:
            raise ValueError(
                f"No data found for combination:\n"
                f"  from_opioid: {from_opioid}\n"
                f"  to_opioid: {to_opioid}\n"
                f"  prompt_conditions: {prompt_conditions}"
            )
        
        # Get expected coefficient (should be same for all models, just take first)
        expected_coeff = filtered_df['expected_coefficient'].iloc[0]
        
        # If conversion_service provided, recalculate to be sure
        if conversion_service:
            try:
                from opioid import Opioid
                
                from_parts = from_opioid.split('_', 1)
                to_parts = to_opioid.split('_', 1)
                
                from_opioid_obj = Opioid(
                    administration_method=from_parts[0],
                    substance=from_parts[1]
                )
                to_opioid_obj = Opioid(
                    administration_method=to_parts[0],
                    substance=to_parts[1]
                )
                
                expected_to_dose = conversion_service.calculate_expected_dose(
                    1.0, from_opioid_obj, to_opioid_obj
                )
                expected_coeff = expected_to_dose
                
            except Exception as e:
                print(f"Could not recalculate expected coefficient for {from_opioid} -> {to_opioid}: {e}")
        
        # Create instance
        return cls(
            from_opioid=from_opioid,
            to_opioid=to_opioid,
            prompt_conditions=prompt_conditions,
            analysis_df=filtered_df,
            expected_coefficient=expected_coeff
        )
    
    @classmethod
    def process_all_combinations(cls,
                                 csv_file: str,
                                 output_file: str,
                                 conversion_service=None,
                                 verbose: bool = True):
        """
        Process all unique opioid pair/prompt condition combinations from pair_model_analysis CSV.
        
        Args:
            csv_file: Path to pair_model_analysis CSV
            output_file: Path to output consensus CSV
            conversion_service: ConversionService to calculate expected coefficients
            verbose: Print progress messages
            
        Returns:
            List of OpioidPairConsensus instances
        """
        df = pd.read_csv(csv_file)
        
        # Group by opioid pair and prompt conditions (across all models)
        groups = df.groupby(['from_opioid', 'to_opioid', 'include_nielson', 'include_examples'])
        
        total = len(groups)
        processed = 0
        results_list = []
        
        # Remove output file if it exists to start fresh
        output_path = Path(output_file)
        if output_path.exists():
            output_path.unlink()
        
        for (from_opioid, to_opioid, include_nielson, include_examples), group_df in groups:
            prompt_conditions = {
                'include_nielson': include_nielson,
                'include_examples': include_examples
            }
            
            # Get expected coefficient
            expected_coeff = group_df['expected_coefficient'].iloc[0]
            
            # If conversion_service provided, recalculate
            if conversion_service:
                try:
                    from opioid import Opioid
                    
                    from_parts = from_opioid.split('_', 1)
                    to_parts = to_opioid.split('_', 1)
                    
                    from_opioid_obj = Opioid(
                        administration_method=from_parts[0],
                        substance=from_parts[1]
                    )
                    to_opioid_obj = Opioid(
                        administration_method=to_parts[0],
                        substance=to_parts[1]
                    )
                    
                    expected_to_dose = conversion_service.calculate_expected_dose(
                        1.0, from_opioid_obj, to_opioid_obj
                    )
                    expected_coeff = expected_to_dose
                    
                except Exception as e:
                    if verbose:
                        print(f"Could not recalculate expected coefficient for {from_opioid} -> {to_opioid}: {e}")
            
            # Create consensus analyzer
            consensus = cls(
                from_opioid=from_opioid,
                to_opioid=to_opioid,
                prompt_conditions=prompt_conditions,
                analysis_df=group_df,
                expected_coefficient=expected_coeff
            )
            
            # Calculate consensus
            try:
                consensus.calculate_consensus()
                
                # Append to CSV
                consensus.append_to_csv(output_file)
                
                results_list.append(consensus)
                processed += 1
                
                if verbose and processed % 10 == 0:
                    print(f"Processed {processed}/{total} combinations...")
                    
            except Exception as e:
                print(f"Error processing consensus for {from_opioid} -> {to_opioid}: {e}")
        
        if verbose:
            print(f"Completed! Processed {processed}/{total} combinations.")
            print(f"Results saved to: {output_file}")
        
        return results_list