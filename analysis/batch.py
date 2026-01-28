from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence, Type
from visualize.visualization_config import get_model_order, normalize_model_name
import numpy as np
import pandas as pd
import warnings

# ------------------- Key dataclasses -------------------

@dataclass(frozen=True)
class PairPromptKey:
    from_opioid: str
    from_opioid_unit: str
    to_opioid: str
    to_opioid_unit: str
    include_nielson: bool
    include_examples: bool
    ome: Optional[bool] = None
    ask_for_units: Optional[bool] = None
    explain_units: Optional[bool] = None

    @property
    def pair_name(self) -> str:
        return f"{self.from_opioid} ({self.from_opioid_unit}) -> {self.to_opioid} ({self.to_opioid_unit})"

@dataclass(frozen=True)
class VignettePromptKey:
    vignette_id: str
    include_nielson: bool
    include_examples: bool
    ome: Optional[bool] = None
    ask_for_units: Optional[bool] = None
    explain_units: Optional[bool] = None

# ------------------- Base Batch class -------------------

class Batch:
    group_cols: List[str] = []
    key_type: Type = None
    item_id_col: str = None  # e.g. 'pair_name' or 'vignette_id'

    def __init__(
        self,
        experiment_results_csv: str,
        variable_model_csv: str,
        consensus_csv: str,
        keys: List[Any],
        model_csv_attr: str = "variable_model_csv",
    ):
        if not keys:
            raise ValueError(f"{self.__class__.__name__} requires at least one key.")
        self.experiment_results_csv = experiment_results_csv
        setattr(self, model_csv_attr, variable_model_csv)
        self.model_csv_attr = model_csv_attr
        self.consensus_csv = consensus_csv
        self._keys = keys
        self._model_df: Optional[pd.DataFrame] = None
        self._experiment_df: Optional[pd.DataFrame] = None
        self._consensus_df: Optional[pd.DataFrame] = None

    @property
    def keys(self) -> List[Any]:
        return self._keys

    @property
    def model_data(self) -> pd.DataFrame:
        if self._model_df is None:
            csv_path = getattr(self, self.model_csv_attr)
            df = pd.read_csv(csv_path)
            self._model_df = self._filter_dataframe(df)
        return self._model_df

    @property
    def experiment_data(self) -> pd.DataFrame:
        if self._experiment_df is None:
            df = pd.read_csv(self.experiment_results_csv)
            self._experiment_df = self._filter_dataframe(df)
        return self._experiment_df

    @property
    def consensus_data(self) -> pd.DataFrame:
        if self._consensus_df is None:
            df = pd.read_csv(self.consensus_csv)
            self._consensus_df = self._filter_dataframe(df)
        return self._consensus_df

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        masks = []
        for key in self._keys:
            mask = np.ones(len(df), dtype=bool)
            for col in self.group_cols:
                val = getattr(key, col, None)
                if val is not None and col in df.columns:
                    mask &= (df[col] == val)
            masks.append(mask)
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask |= mask
            return df[combined_mask].copy()
        return pd.DataFrame()

    @staticmethod
    def add_mean_row_and_col(matrix: np.ndarray) -> np.ndarray:
        col_means = np.nanmean(matrix, axis=1, keepdims=True)
        matrix = np.hstack([matrix, col_means])
        row_means = np.nanmean(matrix, axis=0, keepdims=True)
        matrix = np.vstack([matrix, row_means])
        return matrix

    def get_models(self) -> List[str]:
        raw_models = list(self.model_data['model'].unique())
        normalized_models = [normalize_model_name(m) for m in raw_models]
        ordered_models = get_model_order(normalized_models)
        return ordered_models

    def get_items(self) -> List[str]:
        items = []
        for key in self._keys:
            item_id = getattr(key, self.item_id_col)
            if item_id not in items:
                items.append(item_id)
        return items

    def _get_model_row(self, item_name: str, model: str) -> pd.DataFrame:
        # Subclasses may override for more complex logic
        return self.model_data[
            (self.model_data[self.item_id_col] == item_name) &
            (self.model_data['model'] == model)
        ]

    def compute_weight_matrix(self) -> np.ndarray:
        models = self.get_models()
        items = self.get_items()
        matrix = np.full((len(models), len(items)), np.nan)
        for item_idx, item_name in enumerate(items):
            for model_idx, model in enumerate(models):
                row = self._get_model_row(item_name, model)
                if len(row) > 0 and 'weight' in row.columns:
                    value = row['weight'].iloc[0]
                    if pd.notna(value):
                        matrix[model_idx, item_idx] = value
        return matrix

    def compute_range_matrix(self, range_type: str = "tight") -> np.ndarray:
        models = self.get_models()
        items = self.get_items()
        matrix = np.full((len(models), len(items)), np.nan)
        col_name = None
        for col in self.model_data.columns:
            if col.startswith(f"{range_type}("):
                col_name = col
                break
        if col_name is None:
            raise ValueError(f"No column found for range type '{range_type}' in analysis CSV")
        for item_idx, item_name in enumerate(items):
            for model_idx, model in enumerate(models):
                row = self._get_model_row(item_name, model)
                if len(row) > 0 and col_name in row.columns:
                    value = row[col_name].iloc[0]
                    if pd.notna(value):
                        matrix[model_idx, item_idx] = value
        return matrix

    def get_weight_matrix_with_summary(self) -> np.ndarray:
        matrix = self.compute_weight_matrix()
        return self.add_mean_row_and_col(matrix)

    def get_range_matrix_with_summary(self, range_type: str = "tight") -> np.ndarray:
        matrix = self.compute_range_matrix(range_type)
        return self.add_mean_row_and_col(matrix)

    def get_mean_accuracy_by_model(self, model_order: List[str]) -> (List[float], List[float]):
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

    @classmethod
    def from_explicit_keys(
        cls,
        experiment_results_csv: str,
        variable_model_csv: str,
        consensus_csv: str,
        keys: Sequence[Any],
        model_csv_attr: str = None,
    ):
        if model_csv_attr is None and hasattr(cls, "default_model_csv_attr"):
            model_csv_attr = cls.default_model_csv_attr
        elif model_csv_attr is None:
            model_csv_attr = "variable_model_csv"

        return cls(
            experiment_results_csv=experiment_results_csv,
            variable_model_csv=variable_model_csv,
            consensus_csv=consensus_csv,
            keys=list(keys),
            model_csv_attr=model_csv_attr,
        )

    @classmethod
    def from_filter_specs(
        cls,
        experiment_results_csv: str,
        variable_model_csv: str,
        consensus_csv: str,
        filter_specs: Optional[Sequence[Dict[str, Any]]] = None,
        order_by: Optional[str] = None,
        model_csv_attr: str = None,
    ):
        if model_csv_attr is None and hasattr(cls, "default_model_csv_attr"):
            model_csv_attr = cls.default_model_csv_attr
        elif model_csv_attr is None:
            model_csv_attr = "variable_model_csv"

        df = pd.read_csv(variable_model_csv)
        group_cols = [col for col in cls.group_cols if col in df.columns]
        combos = df[group_cols].drop_duplicates().copy()
        if filter_specs:
            mask_total = np.zeros(len(combos), dtype=bool)
            for spec in filter_specs:
                mask = np.ones(len(combos), dtype=bool)
                for col, val in spec.items():
                    if col not in combos.columns:
                        warnings.warn(f"Filter key '{col}' not found in analysis CSV. Skipping this filter.")
                        continue
                    if isinstance(val, (list, tuple, set)):
                        mask &= combos[col].isin(list(val))
                    else:
                        mask &= combos[col] == val
                mask_total |= mask
            combos = combos[mask_total]
        if order_by and order_by in combos.columns:
            combos = combos.sort_values(order_by)
        elif order_by and order_by == "opioid_pair" and "from_opioid" in combos.columns and "to_opioid" in combos.columns:
            combos = combos.assign(
                opioid_pair=lambda d: d["from_opioid"] + " -> " + d["to_opioid"]
            ).sort_values("opioid_pair")
        if len(combos) == 0:
            raise ValueError(f"No combinations found for filter_specs={filter_specs}")
        keys = []
        for _, row in combos.iterrows():
            key_args = {col: row[col] for col in group_cols if col in row.index}
            if 'ome' in row.index:
                key_args['ome'] = row['ome'] if pd.notna(row['ome']) else None
            keys.append(cls.key_type(**key_args))
        batch = cls(
            experiment_results_csv=experiment_results_csv,
            variable_model_csv=variable_model_csv,
            consensus_csv=consensus_csv,
            keys=keys,
            model_csv_attr=model_csv_attr,
        )
        setattr(batch, model_csv_attr, variable_model_csv)
        return batch

# ------------------- OpioidPairBatch -------------------

class OpioidPairBatch(Batch):
    group_cols = ["from_opioid", "from_opioid_unit", "to_opioid", "to_opioid_unit", "include_nielson", "include_examples", "ome", "ask_for_units", "explain_units"]
    key_type = PairPromptKey
    item_id_col = "pair_name"
    default_model_csv_attr = "pair_model_csv"

    def __init__(
        self,
        experiment_results_csv: str,
        variable_model_csv: str,
        consensus_csv: str,
        keys: List[Any],
        model_csv_attr: str = None,
    ):
        if model_csv_attr is None:
            model_csv_attr = self.default_model_csv_attr
        super().__init__(
            experiment_results_csv=experiment_results_csv,
            variable_model_csv=variable_model_csv,
            consensus_csv=consensus_csv,
            keys=keys,
            model_csv_attr=model_csv_attr,
        )

    def get_items(self) -> List[str]:
        pairs = []
        for key in self._keys:
            pair_name = key.pair_name
            if pair_name not in pairs:
                pairs.append(pair_name)
        return pairs

    def _get_model_row(self, pair_name: str, model: str) -> pd.DataFrame:
        from_opioid, to_opioid = pair_name.split(' -> ')
        return self.model_data[
            (self.model_data['from_opioid'] == from_opioid) &
            (self.model_data['to_opioid'] == to_opioid) &
            (self.model_data['model'] == model)
        ]

    def get_data_for_pair_model(self, pair_name: str, model: str) -> Dict[str, Any]:
        from_opioid, to_opioid = pair_name.split(' -> ')
        matching_key = None
        for key in self._keys:
            if key.from_opioid == from_opioid and key.to_opioid == to_opioid:
                matching_key = key
                break
        if matching_key is None:
            return {}
        pair_model_row = self.model_data[
            (self.model_data['from_opioid'] == from_opioid) &
            (self.model_data['to_opioid'] == to_opioid) &
            (self.model_data['model'] == model) &
            (self.model_data['include_nielson'] == matching_key.include_nielson) &
            (self.model_data['include_examples'] == matching_key.include_examples)
        ]
        exp_rows = self.experiment_data[
            (self.experiment_data['from_opioid'] == from_opioid) &
            (self.experiment_data['to_opioid'] == to_opioid) &
            (self.experiment_data['model'] == model) &
            (self.experiment_data['include_nielson'] == matching_key.include_nielson) &
            (self.experiment_data['include_examples'] == matching_key.include_examples)
        ]
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
    
    @classmethod
    def from_filter_specs(
        cls,
        experiment_results_csv: str,
        variable_model_csv: str,
        consensus_csv: str,
        filter_specs: Optional[Sequence[Dict[str, Any]]] = None,
        order_by: Optional[str] = None,
        model_csv_attr: str = None,
    ):
        if model_csv_attr is None:
            model_csv_attr = cls.default_model_csv_attr

        df = pd.read_csv(variable_model_csv)
        group_cols = [col for col in cls.group_cols if col in df.columns]
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
                        warnings.warn(f"Filter key '{col}' not found in analysis CSV. Skipping this filter.")
                        continue
                    if isinstance(val, (list, tuple, set)):
                        mask &= combos[col].isin(list(val))
                    else:
                        mask &= combos[col] == val
                mask_total |= mask
            combos = combos[mask_total]
        # Exclude self-pairs if requested
        if exclude_self_pairs_flag:
            combos = combos[combos["from_opioid"] != combos["to_opioid"]]
        if order_by and order_by in combos.columns:
            combos = combos.sort_values(order_by)
        elif order_by and order_by == "opioid_pair" and "from_opioid" in combos.columns and "to_opioid" in combos.columns:
            combos = combos.assign(
                opioid_pair=lambda d: d["from_opioid"] + " -> " + d["to_opioid"]
            ).sort_values("opioid_pair")
        if len(combos) == 0:
            raise ValueError(f"No combinations found for filter_specs={filter_specs}")
        keys = []
        for _, row in combos.iterrows():
            key_args = {col: row[col] for col in group_cols if col in row.index}
            if 'ome' in row.index:
                key_args['ome'] = row['ome'] if pd.notna(row['ome']) else None
            keys.append(cls.key_type(**key_args))
        batch = cls(
            experiment_results_csv=experiment_results_csv,
            variable_model_csv=variable_model_csv,
            consensus_csv=consensus_csv,
            keys=keys,
            model_csv_attr=model_csv_attr,
        )
        setattr(batch, model_csv_attr, variable_model_csv)
        return batch

# ------------------- VignetteBatch -------------------

class VignetteBatch(Batch):
    group_cols = ["vignette_id", "include_nielson", "include_examples", "ome", "ask_for_units", "explain_units"]
    key_type = VignettePromptKey
    item_id_col = "vignette_id"
    default_model_csv_attr = "vignette_model_csv"  # <-- Add this line

    def __init__(
        self,
        experiment_results_csv: str,
        variable_model_csv: str,
        consensus_csv: str,
        keys: List[Any],
        model_csv_attr: str = None,
    ):
        if model_csv_attr is None:
            model_csv_attr = self.default_model_csv_attr
        super().__init__(
            experiment_results_csv=experiment_results_csv,
            variable_model_csv=variable_model_csv,
            consensus_csv=consensus_csv,
            keys=keys,
            model_csv_attr=model_csv_attr,
        )

    def get_items(self) -> List[str]:
        vignettes = []
        for key in self._keys:
            if key.vignette_id not in vignettes:
                vignettes.append(key.vignette_id)
        return vignettes

    def _get_model_row(self, vignette_id: str, model: str) -> pd.DataFrame:
        return self.model_data[
            (self.model_data['vignette_id'] == vignette_id) &
            (self.model_data['model'] == model)
        ]