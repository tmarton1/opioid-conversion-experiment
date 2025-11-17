import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import csv
from scipy import stats


class OpioidPairConsensus:
    """
    Computes log-space consensus coefficients across models for an opioid pair + prompt condition.

    All inference is performed in LOG-COEFFICIENT space.

    Let for model j:
        μ_j     = observed_log_coefficient_mean
        σ_j     = observed_log_coefficient_std
        w_j     = weight (non-negative)
        v_j     = w_j / Σ_k w_k   (normalized weight)

    We compute:
        consensus_log_coefficient = Σ_j v_j μ_j
        consensus_coefficient     = exp(consensus_log_coefficient)  # display only

        weighted_variance_log     = Σ_j v_j (μ_j - μ*)² / (1 - Σ_j v_j²)
        se_consensus_log          = sqrt(weighted_variance_log / M_eff)
        sigma_log_cons²           = Σ_j v_j σ_j² + Σ_j v_j (μ_j - μ*)²
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        from_opioid: str,
        to_opioid: str,
        prompt_conditions: Dict[str, bool],
        analysis_df: pd.DataFrame,
        expected_coefficient: Optional[float] = None,
    ):
        self.from_opioid = from_opioid
        self.to_opioid = to_opioid
        self.prompt_conditions = prompt_conditions
        self.analysis_df = analysis_df
        self.expected_coefficient = expected_coefficient

        # Outputs
        self.normalized_weights = None  # v_j
        self.consensus_log_coefficient = None
        self.consensus_coefficient = None  # display only

        self.effective_n_models = None
        self.weighted_variance_log = None
        self.se_consensus_log = None

        self.t_score_log = None
        self.p_value_log = None
        self.df = None

        # mixture variance in log space
        self.sigma_log_cons_sq = None
        self.sigma_log_cons = None

    # ------------------------------------------------------------------
    def calculate_consensus(self):
        """
        Compute all consensus statistics in LOG space.
        """

        log_means = self.analysis_df["observed_log_coefficient_mean"].values
        log_sds = self.analysis_df["observed_log_coefficient_std"].values
        weights = self.analysis_df["weight"].values

        if len(log_means) == 0:
            raise ValueError(f"No model data found for {self.from_opioid} -> {self.to_opioid}")

        # Filter valid
        valid_mask = (
            (weights > 0)
            & np.isfinite(weights)
            & np.isfinite(log_means)
            & np.isfinite(log_sds)
        )
        if not np.any(valid_mask):
            raise ValueError(f"No valid weighted models for {self.from_opioid} -> {self.to_opioid}")

        log_means = log_means[valid_mask]
        log_sds = log_sds[valid_mask]
        weights = weights[valid_mask]

        # Normalize
        v = weights / np.sum(weights)
        self.normalized_weights = v

        # Consensus log coefficient
        self.consensus_log_coefficient = float(np.sum(v * log_means))
        self.consensus_coefficient = float(np.exp(self.consensus_log_coefficient))

        # Effective number of models
        sum_v_sq = float(np.sum(v ** 2))
        self.effective_n_models = 1.0 / sum_v_sq

        # Between-model variance (unbiased)
        deviations_sq = (log_means - self.consensus_log_coefficient) ** 2
        denom = 1.0 - sum_v_sq
        if denom > 0:
            self.weighted_variance_log = float(np.sum(v * deviations_sq) / denom)
        else:
            self.weighted_variance_log = np.nan

        # Standard error of log consensus
        if np.isfinite(self.weighted_variance_log) and self.effective_n_models > 1:
            self.se_consensus_log = float(
                np.sqrt(self.weighted_variance_log / self.effective_n_models)
            )
        else:
            self.se_consensus_log = np.nan

        # Degrees of freedom
        self.df = self.effective_n_models - 1

        # t-test vs expected (literature)
        if (
            self.expected_coefficient is not None
            and self.expected_coefficient > 0
            and self.se_consensus_log is not None
            and np.isfinite(self.se_consensus_log)
            and self.se_consensus_log > 0
        ):
            log_expected = np.log(self.expected_coefficient)
            self.t_score_log = (
                self.consensus_log_coefficient - log_expected
            ) / self.se_consensus_log
            self.p_value_log = 2 * (
                1 - stats.t.cdf(abs(self.t_score_log), df=self.df)
            )
        else:
            self.t_score_log = np.nan
            self.p_value_log = np.nan

        # Mixture SD for per-response variability in log space
        within = np.sum(v * (log_sds ** 2))
        between = np.sum(v * deviations_sq)

        self.sigma_log_cons_sq = float(within + between)
        self.sigma_log_cons = float(np.sqrt(self.sigma_log_cons_sq))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """
        Convert consensus stats to a dictionary suitable for CSV.
        """

        return {
            "from_opioid": self.from_opioid,
            "to_opioid": self.to_opioid,
            "opioid_pair": f"{self.from_opioid} -> {self.to_opioid}",
            "include_nielson": self.prompt_conditions.get("include_nielson", False),
            "include_examples": self.prompt_conditions.get("include_examples", False),
            "expected_coefficient": self.expected_coefficient,

            # location
            "consensus_log_coefficient": self.consensus_log_coefficient,
            "consensus_coefficient": self.consensus_coefficient,  # display only

            "effective_n_models": self.effective_n_models,
            "weighted_variance_log": self.weighted_variance_log,
            "se_consensus_log": self.se_consensus_log,
            "t_score_log": self.t_score_log,
            "df": self.df,
            "p_value_log": self.p_value_log,

            "sigma_log_cons_sq": self.sigma_log_cons_sq,
            "sigma_log_cons": self.sigma_log_cons,

            "n_models": len(self.analysis_df),
        }

    # ------------------------------------------------------------------
    def append_to_csv(self, csv_file: str):
        """
        Append results to CSV, writing header if needed.
        """
        csv_path = Path(csv_file)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        row = self.to_dict()

        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    @classmethod
    def from_csv_filter(
        cls,
        csv_file: str,
        from_opioid: str,
        to_opioid: str,
        prompt_conditions: Dict[str, bool],
        conversion_service=None,
    ):
        """
        Build a consensus object for a specific pair and prompt condition
        from pair_model_analysis.csv.
        """
        df = pd.read_csv(csv_file)

        mask = (
            (df["from_opioid"] == from_opioid)
            & (df["to_opioid"] == to_opioid)
            & (df["include_nielson"] == prompt_conditions.get("include_nielson", False))
            & (df["include_examples"] == prompt_conditions.get("include_examples", False))
        )
        subset = df[mask].copy()

        if len(subset) == 0:
            raise ValueError(f"No rows for {from_opioid} -> {to_opioid} with {prompt_conditions}")

        expected_coeff = subset["expected_coefficient"].iloc[0]

        return cls(
            from_opioid,
            to_opioid,
            prompt_conditions,
            analysis_df=subset,
            expected_coefficient=expected_coeff,
        )

    # ------------------------------------------------------------------
    @classmethod
    def process_all_combinations(
        cls,
        csv_file: str,
        output_file: str,
        conversion_service=None,
        verbose: bool = True,
    ):
        """
        Build consensus rows for every opioid-pair + prompt condition combination
        found in pair_model_analysis.csv.
        """
        df = pd.read_csv(csv_file)

        groups = df.groupby(
            ["from_opioid", "to_opioid", "include_nielson", "include_examples"]
        )

        out = Path(output_file)
        if out.exists():
            out.unlink()

        results = []
        for (from_opioid, to_opioid, inc_n, inc_ex), group_df in groups:
            prompt = {"include_nielson": inc_n, "include_examples": inc_ex}

            expected_coeff = group_df["expected_coefficient"].iloc[0]

            consensus = cls(
                from_opioid, to_opioid, prompt, group_df, expected_coefficient=expected_coeff
            )
            consensus.calculate_consensus()
            consensus.append_to_csv(output_file)
            results.append(consensus)

        if verbose:
            print(f"Processed {len(groups)} combinations.")
            print(f"Saved consensus to {output_file}")

        return results

    # ------------------------------------------------------------------
    @classmethod
    def process_all_combinations(
        cls,
        csv_file: str,
        output_file: str,
        conversion_service=None,
        verbose: bool = True,
    ):
        """
        Build consensus rows for every opioid-pair + prompt condition combination
        found in pair_model_analysis.csv.

        Pairs with no valid weighted models are skipped.
        """
        df = pd.read_csv(csv_file)

        groups = df.groupby(
            ["from_opioid", "to_opioid", "include_nielson", "include_examples"]
        )

        out = Path(output_file)
        if out.exists():
            out.unlink()

        results = []
        skipped = 0

        for (from_opioid, to_opioid, inc_n, inc_ex), group_df in groups:
            prompt = {"include_nielson": inc_n, "include_examples": inc_ex}

            # If there is literally no row, just skip
            if len(group_df) == 0:
                if verbose:
                    print(f"Skipping {from_opioid} -> {to_opioid}: no rows in analysis_df")
                skipped += 1
                continue

            expected_coeff = group_df["expected_coefficient"].iloc[0]

            consensus = cls(
                from_opioid,
                to_opioid,
                prompt,
                group_df,
                expected_coefficient=expected_coeff,
            )

            try:
                consensus.calculate_consensus()
                consensus.append_to_csv(output_file)
                results.append(consensus)

            except ValueError as e:
                # Specifically skip when there are no valid weighted models
                msg = str(e)
                if "No valid weighted models" in msg or "No model data found" in msg:
                    if verbose:
                        print(f"Skipping {from_opioid} -> {to_opioid}: {msg}")
                    skipped += 1
                    continue
                else:
                    # Unexpected error: re-raise so you notice
                    raise

        if verbose:
            print(f"Processed {len(results)} consensus rows.")
            print(f"Skipped {skipped} pairs with no valid weighted models.")
            print(f"Consensus saved to {output_file}")

        return results
    
    # ------------------------------------------------------------------
    @classmethod
    def update_pair_model_analysis(cls,
                                   pair_model_csv: str,
                                   consensus_csv: str,
                                   output_csv: str,
                                   verbose: bool = True):
        """
        Update pair_model_analysis.csv with consensus metrics from opioid_pair_consensus.csv
        and add per-model normalized weights (v_j) within each pair/prompt group.

        For each model-pair-prompt combination, adds:
        - consensus_log_coefficient
        - consensus_coefficient (exp of log-consensus, for display)
        - se_consensus_log
        - sigma_log_cons
        - t_score_log, p_value_log
        - normalized_weight: v_j = weight_j / sum_k(weight_k) within the same group
        - consensus_center_t: (mu_log_j - mu_log_cons) / sigma_log_cons
        - consensus_expected_z2: (sd_log_j^2 + (mu_log_j - mu_log_cons)^2) / sigma_log_cons^2
        """
        # Read both CSVs
        pair_model_df = pd.read_csv(pair_model_csv)
        consensus_df = pd.read_csv(consensus_csv)

        if verbose:
            print(f"Loaded {len(pair_model_df)} rows from {pair_model_csv}")
            print(f"Loaded {len(consensus_df)} rows from {consensus_csv}")

        # Merge consensus fields onto pair_model_df
        merged = pair_model_df.merge(
            consensus_df[
                [
                    "from_opioid",
                    "to_opioid",
                    "include_nielson",
                    "include_examples",
                    "consensus_log_coefficient",
                    "consensus_coefficient",
                    "weighted_variance_log",
                    "se_consensus_log",
                    "sigma_log_cons",
                    "t_score_log",
                    "df",
                    "p_value_log",
                ]
            ],
            on=["from_opioid", "to_opioid", "include_nielson", "include_examples"],
            how="left",
            suffixes=("", "_consensus"),
        )

        # ---- NEW: compute normalized_weight v_j within each pair/prompt group ----
        merged["normalized_weight"] = 0.0  # default

        group_cols = ["from_opioid", "to_opioid", "include_nielson", "include_examples"]

        for _, idx in merged.groupby(group_cols).groups.items():
            idx = list(idx)
            weights = merged.loc[idx, "weight"].astype(float)

            # valid weights: finite & positive
            valid_mask = weights.apply(lambda w: np.isfinite(w) and (w > 0))
            total = weights[valid_mask].sum()

            if total > 0:
                merged.loc[idx, "normalized_weight"] = np.where(
                    valid_mask, weights / total, 0.0
                )
            else:
                # no valid weights in this group: keep all v_j = 0.0
                merged.loc[idx, "normalized_weight"] = 0.0

        # --- per-model consensus-centered metrics in LOG space ---
        mu_log_j = merged["observed_log_coefficient_mean"].astype(float)
        sd_log_j = merged["observed_log_coefficient_std"].astype(float)
        mu_log_cons = merged["consensus_log_coefficient"].astype(float)
        sigma_log_cons = merged["sigma_log_cons"].astype(float)

        valid_sigma_mask = (sigma_log_cons > 0) & (~sigma_log_cons.isna())

        merged["consensus_center_t"] = np.nan
        merged["consensus_expected_z2"] = np.nan

        delta_log = mu_log_j - mu_log_cons

        merged.loc[valid_sigma_mask, "consensus_center_t"] = (
            delta_log[valid_sigma_mask] / sigma_log_cons[valid_sigma_mask]
        )

        merged.loc[valid_sigma_mask, "consensus_expected_z2"] = (
            sd_log_j[valid_sigma_mask] ** 2 + delta_log[valid_sigma_mask] ** 2
        ) / (sigma_log_cons[valid_sigma_mask] ** 2)

        # Save to output CSV
        merged.to_csv(output_csv, index=False)

        if verbose:
            print(f"Updated analysis saved to: {output_csv}")
            print(
                "Added columns: consensus_log_coefficient, consensus_coefficient, "
                "se_consensus_log, sigma_log_cons, t_score_log, df, p_value_log, "
                "normalized_weight, consensus_center_t, consensus_expected_z2"
            )

        return merged
