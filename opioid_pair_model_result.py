import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import csv
from scipy import stats
from opioid import Opioid

class OpioidPairModelResult:
    """
    Analyzes results for a specific combination of opioid pair, model, and prompt conditions.
    Calculates expected vs observed coefficients, performs regressions, and computes weights.
    """
    
    def __init__(self, 
                 from_opioid: str,
                 to_opioid: str,
                 model: str,
                 prompt_conditions: Dict[str, bool],
                 results_df: pd.DataFrame,
                 expected_coefficient: Optional[float] = None):
        """
        Initialize the result analyzer.
        
        Args:
            from_opioid: Source opioid (e.g., "iv_fentanyl")
            to_opioid: Target opioid (e.g., "oral_morphine")
            model: Model name (e.g., "gpt-4.1-mini")
            prompt_conditions: Dict with 'include_nielson' and 'include_examples' keys
            results_df: DataFrame containing individual conversion results
            expected_coefficient: Expected coefficient from Nielson data (if available)
        """
        self.from_opioid = from_opioid
        self.to_opioid = to_opioid
        self.model = model
        self.prompt_conditions = prompt_conditions
        self.results_df = results_df
        self.expected_coefficient = expected_coefficient

        # Derived flag: ome means converting TO oral morphine
        self.ome: Optional[bool] = (self.to_opioid == "oral_morphine")

        # If the raw data has an 'ome' column, sanity-check consistency
        if "ome" in self.results_df.columns:
            raw_ome_values = self.results_df["ome"].dropna().unique()
            if len(raw_ome_values) > 0:
                # If data says something different than the naming convention, warn
                raw_ome_bool = bool(raw_ome_values[0])
                if raw_ome_bool != self.ome:
                    print(
                        f"Warning: 'ome' column ({raw_ome_bool}) disagrees with "
                        f"to_opioid == 'oral_morphine' ({self.ome}) for "
                        f"{self.from_opioid} -> {self.to_opioid}, model={self.model}."
                    )
        
        # Results to be calculated
        self.observed_coefficient_mean = None
        self.observed_coefficient_std = None
        self.observed_coefficient_cv = None
        self.observed_log_coefficient_mean = None
        self.observed_log_coefficient_std = None
        self.r_squared = None
        self.f = None
        self.weight = None
        
    def calculate_coefficients(self) -> Tuple[float, float, float]:
        """
        Calculate observed coefficient statistics from observed_to_dose / from_dose.
        
        Returns:
            (mean, std, cv) tuple
        """
        # Get the coefficients (ratios) for this specific combination
        coefficients = self.results_df['observed_to_dose'] / self.results_df['from_dose']
        positive_mask = coefficients > 0
        if not all(positive_mask):
            coefficients = coefficients[positive_mask]

        if len(coefficients) == 0:
            raise ValueError(f"No positive coefficients found for {self.from_opioid} -> {self.to_opioid}, {self.model}")
        
        if len(coefficients) == 0:
            raise ValueError(f"No data found for {self.from_opioid} -> {self.to_opioid}, {self.model}")
        
        self.observed_coefficient_mean = np.mean(coefficients)
        self.observed_coefficient_std = np.std(coefficients, ddof=1)  # Sample std

        log_coefficients = np.log(coefficients)
        self.observed_log_coefficient_mean = float(np.mean(log_coefficients))
        self.observed_log_coefficient_std = float(np.std(log_coefficients, ddof=1))
        
        # Coefficient of variation (CV)
        if self.observed_coefficient_mean != 0:
            self.observed_coefficient_cv = self.observed_coefficient_std / abs(self.observed_coefficient_mean)
        else:
            self.observed_coefficient_cv = np.nan
            
        return self.observed_coefficient_mean, self.observed_coefficient_std, self.observed_coefficient_cv
    
    def perform_regression(self) -> float:
        """
        Perform regression analysis:
        M0: log_c = log_a + error (null model)
        M1: log_c = beta_0 + beta_1 * dose_to + error
        
        Calculate R^2 = 1 - s1^2/s0^2
        
        Returns:
            R^2 value
        """
        coefficients = self.results_df['observed_to_dose'] / self.results_df['from_dose']
        positive_mask = coefficients > 0
        coefficients = coefficients[positive_mask]
        dose_to = self.results_df.loc[positive_mask, "expected_to_dose"].values
        log_coefficients = np.log(coefficients)
        
        # M0: Null model - variance around mean
        s0_squared = np.var(log_coefficients, ddof=1)
        
        # M1: Linear regression
        # Use scipy.stats.linregress for simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(dose_to, log_coefficients)
        
        # Calculate residuals and residual variance
        predictions = slope * dose_to + intercept
        residuals = log_coefficients - predictions
        s1_squared = np.var(residuals, ddof=2) if len(log_coefficients) > 2 else  np.var(residuals, ddof=1)# ddof=2 for two parameters (slope, intercept)
        
        # R^2
        if s0_squared != 0:
            self.r_squared = 1 - (s1_squared / s0_squared)
        else:
            self.r_squared = np.nan
            
        return self.r_squared
    
    def calculate_weights(self) -> Tuple[float, float]:
        """
        Calculate f and w:
        f = exp(-4 * R^2)
        w = f / (log_std^2 + epsilon)
        
        Returns:
            (f, w) tuple
        """
        if self.r_squared is not None:
            self.f = np.exp(-4 * self.r_squared)
        else:
            self.f = np.nan
        
        epsilon = 5e-2
        if self.observed_log_coefficient_std is not None and self.observed_log_coefficient_std != 0:
            denom = self.observed_log_coefficient_std ** 2 + epsilon
            self.weight = float(self.f / denom) if denom > 0 else np.nan
        else:
            self.weight = np.nan
            
        return self.f, self.weight
    
    def calculate_all(self):
        """
        Perform all calculations in sequence.
        """
        self.calculate_coefficients()
        self.perform_regression()
        self.calculate_weights()
    
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
            'model': self.model,
            'include_nielson': self.prompt_conditions.get('include_nielson', False),
            'include_examples': self.prompt_conditions.get('include_examples', False),
            'ome': self.ome,
            'expected_coefficient': self.expected_coefficient,
            'observed_coefficient_mean': self.observed_coefficient_mean,
            'observed_coefficient_std': self.observed_coefficient_std,
            'observed_coefficient_cv': self.observed_coefficient_cv,
            'observed_log_coefficient_mean': self.observed_log_coefficient_mean,
            'observed_log_coefficient_std': self.observed_log_coefficient_std,
            'r_squared': self.r_squared,
            'f': self.f,
            'weight': self.weight,
            'n_observations': len(self.results_df)
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
                    model: str,
                    prompt_conditions: Dict[str, bool],
                    conversion_service=None):
        """
        Create an OpioidPairModelResult by filtering a CSV file.
        
        Args:
            csv_file: Path to experiment results CSV
            from_opioid: Source opioid (e.g., "iv_fentanyl")
            to_opioid: Target opioid (e.g., "oral_morphine")
            model: Model name (e.g., "gpt-4.1-mini")
            prompt_conditions: Dict with 'include_nielson' and 'include_examples' keys
            conversion_service: Optional ConversionService to calculate expected coefficient
            
        Returns:
            OpioidPairModelResult instance with filtered data
        """
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Filter for the specific combination
        filtered_df = df[
            (df['from_opioid'] == from_opioid) &
            (df['to_opioid'] == to_opioid) &
            (df['model'] == model) &
            (df['include_nielson'] == prompt_conditions.get('include_nielson', False)) &
            (df['include_examples'] == prompt_conditions.get('include_examples', False))
        ].copy()
        
        if len(filtered_df) == 0:
            raise ValueError(
                f"No data found for combination:\n"
                f"  from_opioid: {from_opioid}\n"
                f"  to_opioid: {to_opioid}\n"
                f"  model: {model}\n"
                f"  prompt_conditions: {prompt_conditions}"
            )
        
        # Calculate expected coefficient from Nielson data
        expected_coeff = None
        if conversion_service:
            try:                
                # Parse opioid strings (format: "administration_substance")
                from_parts = from_opioid.split('_', 1)  # Split only on first underscore
                to_parts = to_opioid.split('_', 1)
                
                from_opioid_obj = Opioid(
                    administration_method=from_parts[0],
                    substance=from_parts[1]
                )
                to_opioid_obj = Opioid(
                    administration_method=to_parts[0],
                    substance=to_parts[1]
                )
                
                # Calculate expected coefficient using conversion service
                # Use a standard dose of 1.0 to get the conversion factor
                expected_to_dose = conversion_service.calculate_expected_dose(
                    1.0, from_opioid_obj, to_opioid_obj
                )
                expected_coeff = expected_to_dose  # This is the conversion factor
                
            except Exception as e:
                print(f"Could not calculate expected coefficient for {from_opioid} -> {to_opioid}: {e}")
                expected_coeff = None
        
        # Create instance
        return cls(
            from_opioid=from_opioid,
            to_opioid=to_opioid,
            model=model,
            prompt_conditions=prompt_conditions,
            results_df=filtered_df,
            expected_coefficient=expected_coeff
        )

    @classmethod
    def process_all_combinations(cls, 
                                csv_file: str,
                                output_file: str,
                                conversion_service=None,
                                verbose: bool = True):
        """
        Process all unique combinations from an experiment results CSV.
        
        Args:
            csv_file: Path to experiment results CSV
            output_file: Path to output analysis CSV
            conversion_service: ConversionService to calculate expected coefficients
            verbose: Print progress messages
            
        Returns:
            List of OpioidPairModelResult instances
        """
        df = pd.read_csv(csv_file)
        
        # Group by the key dimensions
        groups = df.groupby(['from_opioid', 'to_opioid', 'model', 'include_nielson', 'include_examples'])
        
        total = len(groups)
        processed = 0
        results_list = []
        
        # Remove output file if it exists to start fresh
        output_path = Path(output_file)
        if output_path.exists():
            output_path.unlink()
        
        for (from_opioid, to_opioid, model, include_nielson, include_examples), group_df in groups:
            prompt_conditions = {
                'include_nielson': include_nielson,
                'include_examples': include_examples
            }
            
            # Calculate expected coefficient from Nielson data
            expected_coeff = None
            if conversion_service:
                try:                    
                    # Parse opioid strings (format: "administration_substance")
                    from_parts = from_opioid.split('_', 1)  # Split only on first underscore
                    to_parts = to_opioid.split('_', 1)
                    
                    from_opioid_obj = Opioid(
                        administration_method=from_parts[0],
                        substance=from_parts[1]
                    )
                    to_opioid_obj = Opioid(
                        administration_method=to_parts[0],
                        substance=to_parts[1]
                    )
                    
                    # Calculate expected coefficient using conversion service
                    # Use a standard dose of 1.0 to get the conversion factor
                    expected_to_dose = conversion_service.calculate_expected_dose(
                        1.0, from_opioid_obj, to_opioid_obj
                    )
                    expected_coeff = expected_to_dose  # This is the conversion factor
                    
                except Exception as e:
                    if verbose:
                        print(f"Could not calculate expected coefficient for {from_opioid} -> {to_opioid}: {e}")
            
            # Create analyzer
            analyzer = cls(
                from_opioid=from_opioid,
                to_opioid=to_opioid,
                model=model,
                prompt_conditions=prompt_conditions,
                results_df=group_df,
                expected_coefficient=expected_coeff
            )
            
            # Calculate all metrics
            try:
                analyzer.calculate_all()
                
                # Append to CSV
                analyzer.append_to_csv(output_file)
                
                results_list.append(analyzer)
                processed += 1
                
                if verbose and processed % 10 == 0:
                    print(f"Processed {processed}/{total} combinations...")
                    
            except Exception as e:
                print(f"Error processing {from_opioid} -> {to_opioid}, {model}: {e}")
        
        if verbose:
            print(f"Completed! Processed {processed}/{total} combinations.")
            print(f"Results saved to: {output_file}")
        
        return results_list