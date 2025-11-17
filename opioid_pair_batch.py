from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence

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


class OpioidPairBatch:
    """
    Selects a batch of PairPromptKey rows to visualize, based on:
    - explicit keys, OR
    - flexible filtering over pair_model_analysis.csv.

    It does *not* load experiment/consensus data itself – those are
    used later by the visualization layer.
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

    # -------------------------- public API ---------------------------

    @property
    def keys(self) -> List[PairPromptKey]:
        """Return the list of keys (rows) in the batch, in order."""
        return self._keys

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
        group_cols = ["from_opioid", "to_opioid", "include_nielson", "include_examples"]
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
                to_opioid=row["to_opioid"],
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
