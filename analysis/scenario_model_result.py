import pandas as pd
import numpy as np
from pathlib import Path
import csv
from scipy import stats


class ScenarioModelResult:

    @staticmethod
    def _find_range_cols(df: pd.DataFrame):
        tight_cols = [c for c in df.columns if c.startswith("tight(")]
        broad_cols = [c for c in df.columns if c.startswith("broad(")]
        tight_col = tight_cols[0] if tight_cols else None
        broad_col = broad_cols[0] if broad_cols else None
        return tight_col, broad_col

    @staticmethod
    def calculate_scenario_stats(group_df: pd.DataFrame) -> dict:
        """
        group_df is vignette-level rows for a single scenario/model/prompt combination
        (multiple vignette_id values).
        """

        weights = group_df["weight"].astype(float)
        log_means = group_df["observed_log_dose_mean"].astype(float)
        log_stds = group_df["observed_log_dose_std"].astype(float)
        n_obs_unfiltered = group_df["n_observations"].astype(float)

        valid = np.isfinite(weights) & np.isfinite(log_means)
        if not valid.any() or weights[valid].sum() <= 0:
            return {
                "observed_log_dose_mean": np.nan,
                "observed_dose_mean": np.nan,
                "observed_log_dose_within_std": np.nan,
                "observed_log_dose_between_std": np.nan,
                "observed_log_dose_std": np.nan,
                "observed_log_dose_between_cv": np.nan,
                "effective_n_vignettes": np.nan,
                "n_vignettes": len(group_df),
                "total_observations": n_obs_unfiltered.sum(),
            }

        weights = weights[valid]
        log_means = log_means[valid]
        log_stds = log_stds[valid]
        n_obs = n_obs_unfiltered[valid]

        expected_to_dose = (
            group_df["expected_to_dose"].iloc[0]
            if "expected_to_dose" in group_df
            else np.nan
        )

        # ---------------------------------------------------------------------
        # 1. Scenario-level observed log dose mean (weighted)
        # ---------------------------------------------------------------------
        observed_log_dose_mean = np.average(log_means, weights=weights)
        observed_dose_mean = float(np.exp(observed_log_dose_mean))

        # ---------------------------------------------------------------------
        # 2. Within-vignette standard deviation (log-space)
        # ---------------------------------------------------------------------
        within_var = np.average(log_stds**2, weights=weights)
        observed_log_dose_within_std = np.sqrt(within_var)

        # ---------------------------------------------------------------------
        # 3. Between-vignette standard deviation (log-space)
        # ---------------------------------------------------------------------
        between_var = np.average((log_means - observed_log_dose_mean)**2, weights=weights)
        observed_log_dose_between_std = np.sqrt(between_var)

        # ---------------------------------------------------------------------
        # 4. Total standard deviation
        # ---------------------------------------------------------------------
        observed_log_dose_std = np.sqrt(within_var + between_var)

        # ---------------------------------------------------------------------
        # 5. Between-vignette CV (log-space)
        # ---------------------------------------------------------------------
        if observed_log_dose_mean != 0 and np.isfinite(observed_log_dose_between_std):
            observed_log_dose_between_cv = (
                observed_log_dose_between_std / abs(observed_log_dose_mean)
            )
        else:
            observed_log_dose_between_cv = np.nan

        # ---------------------------------------------------------------------
        # 6. Effective number of vignettes
        # ---------------------------------------------------------------------
        effective_n_vignettes = (weights.sum() ** 2) / (weights**2).sum()

        # ---------------------------------------------------------------------
        # 7. Scenario-level tight/broad proportions (weighted by n_obs)
        # ---------------------------------------------------------------------
        tight_col, broad_col = ScenarioModelResult._find_range_cols(group_df)

        if tight_col is not None:
            tight_vals = group_df[tight_col].astype(float).values
            tight_mean = np.average(tight_vals, weights=n_obs_unfiltered)
        else:
            tight_mean = np.nan

        if broad_col is not None:
            broad_vals = group_df[broad_col].astype(float).values
            broad_mean = np.average(broad_vals, weights=n_obs_unfiltered)
        else:
            broad_mean = np.nan

        # ---------------------------------------------------------------------
        # Build dictionary with exact column names requested
        # ---------------------------------------------------------------------
        res = {
            "observed_log_dose_mean": float(observed_log_dose_mean),
            "observed_dose_mean": float(observed_dose_mean),

            "observed_log_dose_within_std": float(observed_log_dose_within_std),
            "observed_log_dose_between_std": float(observed_log_dose_between_std),
            "observed_log_dose_std": float(observed_log_dose_std),

            "observed_log_dose_between_cv": float(observed_log_dose_between_cv),

            "effective_n_vignettes": float(effective_n_vignettes),
            "n_vignettes": int(len(group_df)),
            "total_observations": float(group_df["n_observations"].sum()),
        }

        # Insert the tight/broad column names dynamically as requested
        if tight_col is not None:
            res[tight_col] = float(tight_mean)

        if broad_col is not None:
            res[broad_col] = float(broad_mean)

        return res

    @classmethod
    def process_all_combinations(cls, vignette_results_csv: str, output_file: str, verbose: bool = True):
        df = pd.read_csv(vignette_results_csv)

        groups = df.groupby([
            "scenario_key",
            "from_opioid",
            "from_opioid_unit",
            "to_opioid",
            "to_opioid_unit",
            "multiple_from_opioids",
            "ome",
            "has_vignette_text",
            "model",
            "include_nielson",
            "include_examples",
            "ask_for_units",
            "explain_units",
        ])

        output_path = Path(output_file)
        if output_path.exists():
            output_path.unlink()

        total = len(groups)
        processed = 0
        results_list = []

        for keys, group_df in groups:
            (
                scenario_key,
                from_opioid,
                from_opioid_unit,
                to_opioid,
                to_opioid_unit,
                multiple_from_opioids,
                ome,
                has_vignette_text,
                model,
                include_nielson,
                include_examples,
                ask_for_units,
                explain_units,
            ) = keys

            stats_dict = cls.calculate_scenario_stats(group_df)

            scenario_dict = {
                "scenario_key": scenario_key,
                "from_opioid": from_opioid,
                "from_opioid_unit": from_opioid_unit,
                "to_opioid": to_opioid,
                "to_opioid_unit": to_opioid_unit,
                "multiple_from_opioids": multiple_from_opioids,
                "ome": ome,
                "has_vignette_text": has_vignette_text,
                "model": model,
                "include_nielson": include_nielson,
                "include_examples": include_examples,
                "ask_for_units": ask_for_units,
                "explain_units": explain_units,
                **stats_dict,
            }

            file_exists = output_path.exists() and output_path.stat().st_size > 0
            with open(output_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=scenario_dict.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(scenario_dict)

            results_list.append(scenario_dict)
            processed += 1
            if verbose and processed % 50 == 0:
                print(f"Processed {processed}/{total} scenario groups...")

        if verbose:
            print(f"Completed! Processed {processed}/{total} scenario groups.")
            print(f"Scenario results saved to: {output_file}")

        return results_list
