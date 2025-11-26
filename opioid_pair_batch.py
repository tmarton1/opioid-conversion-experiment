from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence
from visualization_config import get_model_order, normalize_model_name
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairPromptKey:
    """
    Identifies a single 'row' in your visualization:
    a specific opioid pair + prompt condition (+ optional ome flag).
    """
    from_opioid: str
    to_opioid: str
    include_nielson: bool
    include_examples: bool
    ome: Optional[bool] = None  # derived in your pipeline but stored in CSV
    
    @property
    def pair_name(self) -> str:
        """Return formatted pair name."""
        return f"{self.from_opioid} -> {self.to_opioid}"

class OpioidPairBatch:
    """
    Manages a batch of opioid pair conversions with their associated data.
    
    Provides filtered access to:
    - pair_model_analysis.csv (weights, statistics per model)
    - experiment_results_by_ome.csv (raw ratio data)
    - opioid_pair_consensus.csv (consensus across models)
    """

    def __init__(
        self,
        experiment_results_csv: str,
        pair_model_csv: str,
        consensus_csv: str,
        keys: List[PairPromptKey],
    ):
        if not keys:
            raise ValueError("OpioidPairBatch requires at least one PairPromptKey.")

        self.experiment_results_csv = experiment_results_csv
        self.pair_model_csv = pair_model_csv
        self.consensus_csv = consensus_csv
        self._keys = keys
        
        # Lazy-load data only when needed
        self._pair_model_df: Optional[pd.DataFrame] = None
        self._experiment_df: Optional[pd.DataFrame] = None
        self._consensus_df: Optional[pd.DataFrame] = None

    # -------------------------- public API ---------------------------

    @property
    def keys(self) -> List[PairPromptKey]:
        """Return the list of keys (rows) in the batch, in order."""
        return self._keys
    
    @property
    def pair_model_data(self) -> pd.DataFrame:
        """
        Load and filter pair_model_analysis.csv to only include rows matching batch keys.
        Cached after first access.
        """
        if self._pair_model_df is None:
            df = pd.read_csv(self.pair_model_csv)
            self._pair_model_df = self._filter_dataframe(df)
        return self._pair_model_df
    
    @property
    def experiment_data(self) -> pd.DataFrame:
        """
        Load and filter experiment_results_by_ome.csv to only include rows matching batch keys.
        Cached after first access.
        """
        if self._experiment_df is None:
            df = pd.read_csv(self.experiment_results_csv)
            self._experiment_df = self._filter_dataframe(df)
        return self._experiment_df
    
    @property
    def consensus_data(self) -> pd.DataFrame:
        """
        Load and filter opioid_pair_consensus.csv to only include rows matching batch keys.
        Cached after first access.
        """
        if self._consensus_df is None:
            df = pd.read_csv(self.consensus_csv)
            self._consensus_df = self._filter_dataframe(df)
        return self._consensus_df
    
    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a DataFrame to only include rows matching batch keys."""
        masks = []
        
        for key in self._keys:
            mask = (
            (df['from_opioid'] == key.from_opioid) &
            (df.get('from_opioid_unit', None) == key.from_opioid_unit if 'from_opioid_unit' in df.columns else True) &
            (df['to_opioid'] == key.to_opioid) &
            (df.get('to_opioid_unit', None) == key.to_opioid_unit if 'to_opioid_unit' in df.columns else True) &
            (df['include_nielson'] == key.include_nielson) &
            (df['include_examples'] == key.include_examples)
            )
            if key.ome is not None and 'ome' in df.columns:
                mask &= (df['ome'] == key.ome)
            masks.append(mask)
        
        # Combine masks with OR
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask |= mask
            return df[combined_mask].copy()
        
        return pd.DataFrame()
    
    @staticmethod
    def add_mean_row_and_col(matrix: np.ndarray) -> np.ndarray:
        """
        Add mean column (across pairs) and mean row (across models) to a matrix.
        Returns new matrix with summary row/col.
        """
        # Mean column (mean across pairs for each model)
        col_means = np.nanmean(matrix, axis=1, keepdims=True)
        matrix = np.hstack([matrix, col_means])
        # Mean row (mean across models for each pair + overall mean)
        row_means = np.nanmean(matrix, axis=0, keepdims=True)
        matrix = np.vstack([matrix, row_means])
        return matrix

    def get_weight_matrix_with_summary(self) -> np.ndarray:
        """Return weight matrix with mean row and column."""
        matrix = self.compute_weight_matrix()
        return self.add_mean_row_and_col(matrix)

    def get_range_matrix_with_summary(self, range_type: str = "tight") -> np.ndarray:
        """Return range matrix with mean row and column."""
        matrix = self.compute_range_matrix(range_type)
        return self.add_mean_row_and_col(matrix)
    
    def get_mean_accuracy_by_model(self, model_order: List[str]) -> (List[float], List[float]):
        """
        Returns tight_means and caution_means for each model in model_order.
        """
        tight_matrix = self.get_range_matrix_with_summary("tight")
        broad_matrix = self.get_range_matrix_with_summary("broad")
        tight_means = []
        caution_means = []
        for i, model in enumerate(model_order):
            tight = np.nanmean(tight_matrix[i, :-1])
            broad = np.nanmean(broad_matrix[i, :-1])
            caution = broad - tight
            tight_means.append(tight)
            caution_means.append(caution)
        return tight_means, caution_means
    
    def get_models(self) -> List[str]:
        """
        Return the canonical ordered list of model names for visualization,
        using normalization and ordering logic consistent with other plots.
        """
        raw_models = list(self.pair_model_data['model'].unique())
        # Normalize model names if needed
        normalized_models = [normalize_model_name(m) for m in raw_models]
        # Get canonical order
        ordered_models = get_model_order(normalized_models)
        return ordered_models
    
    def get_pairs(self) -> List[str]:
        """Get list of unique opioid pairs in key order."""
        pairs = []
        for key in self._keys:
            pair_name = key.pair_name
            if pair_name not in pairs:
                pairs.append(pair_name)
        return pairs
    
    def get_data_for_pair_model(self, pair_name: str, model: str) -> Dict[str, Any]:
        """
        Get all data for a specific pair-model combination.
        
        Returns:
            dict with keys: weight, experiment_data, consensus_data
        """
        from_opioid, to_opioid = pair_name.split(' -> ')
        
        # Find matching key
        matching_key = None
        for key in self._keys:
            if key.from_opioid == from_opioid and key.to_opioid == to_opioid:
                matching_key = key
                break
        
        if matching_key is None:
            return {}
        
        # Get pair-model data (weight, etc.)
        pair_model_row = self.pair_model_data[
            (self.pair_model_data['from_opioid'] == from_opioid) &
            (self.pair_model_data['to_opioid'] == to_opioid) &
            (self.pair_model_data['model'] == model) &
            (self.pair_model_data['include_nielson'] == matching_key.include_nielson) &
            (self.pair_model_data['include_examples'] == matching_key.include_examples)
        ]
        
        # Get experiment data (individual trials)
        exp_rows = self.experiment_data[
            (self.experiment_data['from_opioid'] == from_opioid) &
            (self.experiment_data['to_opioid'] == to_opioid) &
            (self.experiment_data['model'] == model) &
            (self.experiment_data['include_nielson'] == matching_key.include_nielson) &
            (self.experiment_data['include_examples'] == matching_key.include_examples)
        ]
        
        # Get consensus data
        consensus_row = self.consensus_data[
            (self.consensus_data['from_opioid'] == from_opioid) &
            (self.consensus_data['to_opioid'] == to_opioid) &
            (self.consensus_data['include_nielson'] == matching_key.include_nielson) &
            (self.consensus_data['include_examples'] == matching_key.include_examples)
        ]
        
        result = {}
        
        if len(pair_model_row) > 0:
            result['weight'] = pair_model_row['weight'].iloc[0]
            result['pair_model_data'] = pair_model_row.iloc[0].to_dict()
        
        if len(exp_rows) > 0:
            result['experiment_data'] = exp_rows
            result['ratios'] = exp_rows['ratio_observed_expected'].dropna().values
        
        if len(consensus_row) > 0:
            result['consensus_data'] = consensus_row.iloc[0].to_dict()
        
        return result
    
    def compute_weight_matrix(self) -> np.ndarray:
        """
        Compute matrix of weights (models × pairs) using precomputed values from pair_model_analysis.csv.
        """
        models = self.get_models()
        pairs = self.get_pairs()
        matrix = np.full((len(models), len(pairs)), np.nan)

        for pair_idx, pair_name in enumerate(pairs):
            from_opioid, to_opioid = pair_name.split(' -> ')
            for model_idx, model in enumerate(models):
                row = self.pair_model_data[
                    (self.pair_model_data['from_opioid'] == from_opioid) &
                    (self.pair_model_data['to_opioid'] == to_opioid) &
                    (self.pair_model_data['model'] == model)
                ]
                if len(row) > 0 and 'weight' in row.columns:
                    value = row['weight'].iloc[0]
                    if pd.notna(value):
                        matrix[model_idx, pair_idx] = value

        return matrix
    
    def compute_range_matrix(self, range_type: str = "tight") -> np.ndarray:
        """
        Compute matrix of proportions within range (models × pairs) using precomputed values from pair_model_analysis.csv.
        Automatically finds the correct column for 'tight' or 'broad' range.
        """
        models = self.get_models()
        pairs = self.get_pairs()
        matrix = np.full((len(models), len(pairs)), np.nan)

        # Find the column name for the range type
        col_name = None
        for col in self.pair_model_data.columns:
            if col.startswith(f"{range_type}("):
                col_name = col
                break
        if col_name is None:
            raise ValueError(f"No column found for range type '{range_type}' in pair_model_analysis.csv")

        for pair_idx, pair_name in enumerate(pairs):
            from_opioid, to_opioid = pair_name.split(' -> ')
            for model_idx, model in enumerate(models):
                row = self.pair_model_data[
                    (self.pair_model_data['from_opioid'] == from_opioid) &
                    (self.pair_model_data['to_opioid'] == to_opioid) &
                    (self.pair_model_data['model'] == model)
                ]
                if len(row) > 0 and col_name in row.columns:
                    value = row[col_name].iloc[0]
                    if pd.notna(value):
                        matrix[model_idx, pair_idx] = value

        return matrix

    # ---------------------- construction helpers --------------------

    @classmethod
    def from_explicit_keys(
        cls,
        experiment_results_csv: str,
        pair_model_csv: str,
        consensus_csv: str,
        keys: Sequence[PairPromptKey],
    ) -> "OpioidPairBatch":
        """
        Build a batch from an explicit sequence of keys.
        You control the order completely.
        """
        return cls(
            experiment_results_csv=experiment_results_csv,
            pair_model_csv=pair_model_csv,
            consensus_csv=consensus_csv,
            keys=list(keys),
        )

    @classmethod
    def from_filter_specs(
        cls,
        experiment_results_csv: str,
        pair_model_csv: str,
        consensus_csv: str,
        filter_specs: Optional[Sequence[Dict[str, Any]]] = None,
        order_by: Optional[str] = "opioid_pair",
    ) -> "OpioidPairBatch":
        """
        Build a batch by filtering pair_model_analysis.csv.

        filter_specs: list of dicts specifying conditions.
          - AND within each dict
          - OR across dicts
          - values can be scalars or iterables

        Example:
            filter_specs = [
                {"ome": True, "include_examples": True},
                {"from_opioid": ["iv_fentanyl", "iv_hydromorphone"]},
            ]

        order_by: optional sorting of the resulting keys.
            One of: None, "from_opioid", "to_opioid", "opioid_pair"
        """
        df = pd.read_csv(pair_model_csv)

        # Work at the (pair, prompt_conditions, ome) level
        group_cols = ["from_opioid", "from_opioid_unit", "to_opioid", "to_opioid_unit", "include_nielson", "include_examples"]
        if "ome" in df.columns:
            group_cols.append("ome")

        combos = df[group_cols].drop_duplicates().copy()

        exclude_self_pairs_flag = False

        if filter_specs:
            mask_total = np.zeros(len(combos), dtype=bool)

            for spec in filter_specs:
                if spec.get("exclude_self_pairs", False):
                    exclude_self_pairs_flag = True

                mask = np.ones(len(combos), dtype=bool)
                for col, val in spec.items():
                    if col == "exclude_self_pairs":
                        continue
                    
                    if col not in combos.columns:
                        raise KeyError(
                            f"Column '{col}' not found in pair_model_analysis.csv. "
                            f"Available columns for filtering: {list(combos.columns)}"
                        )

                    if isinstance(val, (list, tuple, set)):
                        mask &= combos[col].isin(list(val))
                    else:
                        mask &= combos[col] == val

                mask_total |= mask

            combos = combos[mask_total]

        # Add option to exclude self-pairs
        if exclude_self_pairs_flag:
            combos = combos[combos["from_opioid"] != combos["to_opioid"]]

        # Optional ordering
        if order_by is not None:
            if order_by == "opioid_pair":
                combos = combos.assign(
                    opioid_pair=lambda d: d["from_opioid"] + " -> " + d["to_opioid"]
                ).sort_values("opioid_pair")
            elif order_by in combos.columns:
                combos = combos.sort_values(order_by)
            else:
                raise ValueError(
                    f"order_by='{order_by}' not supported. "
                    f"Use None, 'opioid_pair', or one of {list(combos.columns)}."
                )

        if len(combos) == 0:
            raise ValueError(
                f"No combinations found for filter_specs={filter_specs}"
            )

        # Build keys
        keys: List[PairPromptKey] = []
        for _, row in combos.iterrows():
            ome_value = row["ome"] if "ome" in row.index else None
            key = PairPromptKey(
                from_opioid=row["from_opioid"],
                from_opioid_unit=row["from_opioid_unit"],
                to_opioid=row["to_opioid"],
                to_opioid_unit=row["to_opioid_unit"],
                include_nielson=bool(row["include_nielson"]),
                include_examples=bool(row["include_examples"]),
                ome=ome_value if pd.notna(ome_value) else None,
            )
            keys.append(key)

        return cls(
            experiment_results_csv=experiment_results_csv,
            pair_model_csv=pair_model_csv,
            consensus_csv=consensus_csv,
            keys=keys,
        )