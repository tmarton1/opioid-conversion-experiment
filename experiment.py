from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from prompt_service import PromptedOpioidConversion, PromptService
from conversion_service import ConversionService
from nielson_data import NIELSON_CONVERSIONS
from opioid import Opioid
import os
from datetime import datetime
from itertools import product 
from matplotlib.ticker import ScalarFormatter
import csv


class CSVWriteError(Exception):
    """Custom exception for errors when writing to the CSV file."""
    pass

class Experiment:
    ID_COUNTER_FILE = "experiment_id_counter.txt"  # File to store the current experiment ID

    @staticmethod
    def _get_next_experiment_id():
        """
        Retrieve the next experiment ID from the counter file. If the file does not exist, start at 1.

        Returns:
            int: The next experiment ID.
        """
        if os.path.exists(Experiment.ID_COUNTER_FILE):
            with open(Experiment.ID_COUNTER_FILE, "r") as f:
                current_id = int(f.read().strip())
        else:
            current_id = 0  # Start at 0 if the file does not exist

        next_id = current_id + 1

        # Update the counter file with the new ID
        with open(Experiment.ID_COUNTER_FILE, "w") as f:
            f.write(str(next_id))

        return next_id

    def __init__(self, opioid_pairs, prompt_conditions, models, repetitions=10, experiment_id=None):
        """
        Initialize the Experiment class.

        Args:
            opioid_pairs (list of tuples): List of (from_opioid, to_opioid) pairs.
            prompt_conditions (list of dict): List of dictionaries specifying prompt conditions.
            repetitions (int): Number of repetitions for each combination of opioid_pairs and prompt_conditions.
            experiment_id (int): Unique identifier for the experiment. If None, a new ID will be generated.
        """
        # Generate an incremental experiment ID if not provided
        self.experiment_id = experiment_id or self._get_next_experiment_id()
        self.opioid_pairs = [
            (from_opioid, Opioid(administration_method="oral", substance="morphine"))
            if isinstance(from_opioid, Opioid) else from_opioid
            for from_opioid in opioid_pairs
        ]
        self.prompt_conditions = prompt_conditions
        self.models = models
        self.repetitions = repetitions
        self.results = []  # To store results of all experiments

        # Generate all combinations of opioid_pairs, prompt_conditions, and models
        self.combinations = list(product(self.opioid_pairs, self.prompt_conditions, self.models))

        # Initialize the PromptService once for the entire experiment
        self.prompt_service = PromptService(ConversionService(NIELSON_CONVERSIONS))

    def run(self):
        """
        Run the experiment for all combinations of opioid_pairs, prompt_conditions, and models, repeated for the specified number of repetitions.
        """
        for (from_opioid, to_opioid), condition, model in self.combinations:
            for _ in range(self.repetitions):
                conversion = PromptedOpioidConversion(
                    from_opioid=from_opioid,
                    to_opioid=to_opioid,
                    include_nielson=condition["include_nielson"],
                    include_examples=condition["include_examples"],
                    model=model
                )
                conversion.generate_prompt(prompt_service=self.prompt_service)
                try:
                    conversion.ask_model()
                    result = {
                        "experiment_id": self.experiment_id,
                        "datetime": datetime.now().isoformat(),  # Add timestamp
                        "from_opioid": f"{from_opioid.administration_method}_{from_opioid.substance}",
                        "from_opioid_unit": from_opioid.unit,
                        "to_opioid": f"{to_opioid.administration_method}_{to_opioid.substance}",
                        "to_opioid_unit": to_opioid.unit,
                        "from_dose": conversion.from_dose,
                        "expected_to_dose": conversion.expected_to_dose,
                        "model": model,
                        "prompt": conversion.prompt,
                        "include_nielson": condition["include_nielson"],
                        "include_examples": condition["include_examples"],
                        "observed_to_dose": conversion.observed_to_dose,
                        "ratio_observed_expected": conversion.ratio_observed_expected,
                        "response": conversion.response
                    }
                    # Save the result directly to the CSV file
                    self.add_result_to_csv(result)
                except CSVWriteError as e:
                    print(f"Error writing result to CSV: {e}")
                except Exception as e:
                    print(f"Error during conversion: {e}")

    def to_dataframe(self):
        """
        Convert the results to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the experiment results.
        """
        return pd.DataFrame(self.results)

    def calculate_aggregates(self):
        """
        Calculate aggregates such as regression, p-values, and proportions.

        Returns:
            dict: A dictionary containing aggregate statistics.
        """
        df = self.to_dataframe()
        aggregates = {}

        # Group by (from_opioid, to_opioid)
        for (from_opioid, to_opioid), group in df.groupby(["from_opioid", "to_opioid"]):
            # Regression of expected dose vs. ratio observed/expected
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                group["expected_to_dose"], group["ratio_observed_expected"]
            )

            # Proportion of ratios inside 0.7 - 1.3
            within_range = group["ratio_observed_expected"].between(0.7, 1.3).mean()

            aggregates[(from_opioid, to_opioid)] = {
                "slope": slope,
                "p_value_slope": p_value,
                "proportion_within_range": within_range
            }

        return aggregates

    def save_to_csv(self, filename="experiment_results.csv"):
        """
        Save the results to a CSV file.

        Args:
            filename (str): The name of the CSV file.
        """
        df = self.to_dataframe()
        try:
            # Check if the file exists
            file_exists = os.path.isfile(filename)

            # Write to the file (append if it exists, create if it doesn't)
            df.to_csv(filename, mode="a" if file_exists else "w", header=not file_exists, index=False)
            if file_exists:
                print(f"Results appended to {filename}")
            else:
                print(f"Results saved to new file: {filename}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def add_result_to_csv(self, result, filename="experiment_results.csv"):
        """
        Append a single result to the CSV file.

        Args:
            result (dict): The result to append to the CSV file.
            filename (str): The name of the CSV file.

        Raises:
            CSVWriteError: If there is an error writing to the CSV file.
        """
        # Define the fieldnames (columns) for the CSV file
        fieldnames = [
            "experiment_id", "datetime", "from_opioid", "from_opioid_unit", "to_opioid", "to_opioid_unit",
            "from_dose", "expected_to_dose", "model", "prompt", "include_nielson",
            "include_examples", "observed_to_dose", "ratio_observed_expected", "response"
        ]

        try:
            # Open the CSV file in append mode
            with open(filename, mode="a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write the header if the file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()

                # Write the result to the CSV file
                writer.writerow(result)
        except Exception as e:
            raise CSVWriteError(f"Failed to write result to CSV file '{filename}': {e}")

    def visualize(self, plots: List[str] = None, title: str = None, ax=None):
        """
        Visualize the results using matplotlib.

        Args:
            plots (List[str], optional): A list of plot types to generate. Options are:
                - "scatter": Scatter plot of Expected vs Observed Dose.
            - "cdf": CDF of Ratio Observed/Expected.
            - "histogram": Histogram of Ratio Observed/Expected.
            If None, all plots will be generated.
            title (str, optional): Custom title for the plot. If None, a default title will be used.
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None, a new figure will be created.
        """
        if plots is None:
            plots = ["scatter", "cdf", "histogram"]

        df = self.to_dataframe()

        if "scatter" in plots:
            # Scatter plot: Expected vs Observed Dose
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))

            cmap = plt.cm.get_cmap("Spectral", 15)  # Use the Spectral colormap with at least 15 colors

            # Assign a unique color to each opioid pair
            unique_pairs = list(df.groupby(["from_opioid", "to_opioid"]).groups.keys())
            pair_colors = {pair: cmap(i) for i, pair in enumerate(unique_pairs)}

            for (from_opioid, to_opioid), group in df.groupby(["from_opioid", "to_opioid"]):
                color = pair_colors[(from_opioid, to_opioid)]
                ax.scatter(
                    group["expected_to_dose"],
                    group["ratio_observed_expected"],
                    label=f"{from_opioid} → {to_opioid}",
                    color=color,
                    alpha=0.6  # Set transparency
                )
            ax.axhline(1.0, color="red", linestyle="--", label="Ideal Ratio (1.0)")
            ax.axhline(0.7, color="orange", linestyle="--", label="Lower Bound (0.7)")
            ax.axhline(1.3, color="orange", linestyle="--", label="Upper Bound (1.3)")
            ax.set_xlabel("Expected Dose")
            ax.set_ylabel("Ratio Observed/Expected")
            ax.set_title(title if title else "Expected Dose vs Ratio Observed/Expected")
            ax.grid()

            # Set y-axis to log scale
            ax.set_yscale("log")

            # Hardcode y-axis tick marks
            y_ticks = [0.1, 0.3, 0.7, 1.0, 1.3, 2, 5, 10, 30]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(tick) for tick in y_ticks])

            # Format the y-axis to show raw values
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(axis="y", which="both", labelsize=10)

            # Move the legend outside the plot if using a single plot
            if ax is None:
                plt.tight_layout()
                plt.show()

        if "cdf" in plots:
            # CDF of Ratio Observed/Expected
            plt.figure(figsize=(5, 3))
            for (from_opioid, to_opioid), group in df.groupby(["from_opioid", "to_opioid"]):
                sorted_ratios = np.sort(group["ratio_observed_expected"])
                cdf = np.arange(len(sorted_ratios)) / float(len(sorted_ratios))
                plt.plot(sorted_ratios, cdf, label=f"{from_opioid} → {to_opioid}")
            plt.axvline(0.7, color="orange", linestyle="--", label="Lower Bound (0.7)")
            plt.axvline(1.3, color="orange", linestyle="--", label="Upper Bound (1.3)")
            plt.xlabel("Ratio Observed/Expected")
            plt.ylabel("CDF")
            plt.title("CDF of Ratio Observed/Expected")
            plt.legend()
            plt.grid()
            plt.show()

        if "histogram" in plots:
            # Histogram: Proportion of Ratios Inside 0.7 - 1.3
            plt.figure(figsize=(5, 3))
            df["ratio_observed_expected"].hist(bins=20, edgecolor="black")
            plt.axvline(1.0, color="red", linestyle="--", label="Ideal Ratio (1.0)")
            plt.axvline(0.7, color="orange", linestyle="--", label="Lower Bound (0.7)")
            plt.axvline(1.3, color="orange", linestyle="--", label="Upper Bound (1.3)")
            plt.xlabel("Ratio Observed/Expected")
            plt.ylabel("Frequency")
            plt.title("Distribution of Ratio Observed/Expected")
            plt.legend()
            plt.grid()
            plt.show()


class ExperimentFromCSVConverter:
    """
    A class to convert experiment results from a CSV file back into an Experiment object.
    """
    
    @staticmethod
    def list_experiment_ids(csv_file):
        """
        List all unique experiment IDs in the CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            list: A list of unique experiment IDs.
        """
        df = pd.read_csv(csv_file)
        return df["experiment_id"].unique().tolist()

    @staticmethod
    def filter_experiments(csv_file, filters=None):
        """
        Filter experiments based on specified fields and return the count of observations for each matching experiment ID.

        Args:
            csv_file (str): Path to the CSV file.
            filters (dict, optional): A dictionary of filters to apply (e.g., {"include_nielson": True, "model": "gpt-4.1-mini"}).

        Returns:
            pd.DataFrame: A DataFrame with experiment IDs and their observation counts.
        """
        df = pd.read_csv(csv_file)

        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                df = df[df[key] == value]

        # Count observations for each experiment_id
        counts = df.groupby("experiment_id").size().reset_index(name="observation_count")
        return counts

    @staticmethod
    def from_csv(csv_file, experiment_id=None, filters=None, prompt_conditions=None):
        """
        Create an Experiment object from a CSV file.

        Args:
            csv_file (str): Path to the CSV file containing experiment results.
            experiment_id (int, optional): The ID of the experiment to recreate. If None, include all experiment IDs.
            filters (dict, optional): Additional filters to apply (e.g., {"model": "gpt-4.1-mini"}).
            prompt_conditions (list of dict, optional): List of dictionaries specifying prompt conditions to filter by. If None, include all prompt conditions.

        Returns:
            Experiment: An Experiment object with the data loaded from the CSV.
        """
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_file)

        # Apply experiment_id filter if provided
        if experiment_id is not None:
            df = df[df["experiment_id"] == experiment_id]

        # Apply additional filters if provided
        if filters:
            for key, value in filters.items():
                df = df[df[key] == value]

        # Apply prompt_conditions filter if provided
        if prompt_conditions:
            # Create a mask to filter rows that match any of the dictionaries in prompt_conditions
            condition_mask = pd.concat(
                [df[list(condition)].eq(pd.Series(condition)).all(axis=1) for condition in prompt_conditions],
                axis=1
            ).any(axis=1)
            df = df[condition_mask]

        # Extract unique opioid pairs from the DataFrame
        opioid_pairs = []
        for _, row in df[["from_opioid", "to_opioid"]].drop_duplicates().iterrows():
            # Parse the from_opioid string
            from_opioid_parts = row["from_opioid"].split("_")
            from_opioid = Opioid(
                administration_method=from_opioid_parts[0],
                substance=from_opioid_parts[1]
            )

            # Parse the to_opioid string
            to_opioid_parts = row["to_opioid"].split("_")
            to_opioid = Opioid(
                administration_method=to_opioid_parts[0],
                substance=to_opioid_parts[1]
            )

            # Append the pair to the opioid_pairs list
            opioid_pairs.append((from_opioid, to_opioid))

        # Extract unique prompt conditions
        unique_conditions = df[["include_nielson", "include_examples"]].drop_duplicates().to_dict(orient="records")

        # Extract unique models
        models = df["model"].unique().tolist()
        
        # Validate consistency of opioid_pairs and prompt_conditions
        pair_condition_map = df.groupby(["from_opioid", "to_opioid"])[["include_nielson", "include_examples"]].apply(
            lambda x: x.drop_duplicates().to_dict(orient="records")
        ).to_dict()
        missing_conditions = {}
        for pair, conditions in pair_condition_map.items():
            missing = [cond for cond in unique_conditions if cond not in conditions]
            if missing:
                missing_conditions[pair] = missing
        covered_conditions = []
        for condition in unique_conditions:
            if all(condition in pair_condition_map[pair] for pair in pair_condition_map):
                covered_conditions.append(condition)
        if missing_conditions:
            error_message = "Inconsistent prompt_conditions in the CSV file:\n"
            error_message += "1. Opioid pairs missing some prompt_conditions:\n"
            for pair, missing in missing_conditions.items():
                error_message += f"   - {pair}: Missing {missing}\n"
            error_message += "2. Prompt_conditions completely covered by all opioid_pairs:\n"
            error_message += f"   - {covered_conditions}\n"
            raise ValueError(error_message)

        # Create an Experiment object
        experiment = Experiment(opioid_pairs, prompt_conditions=unique_conditions, models=models)

        # Populate the results directly from the filtered CSV data
        experiment.results = df.to_dict(orient="records")

        return experiment

    @staticmethod
    def format_experiment_title(model: str, prompt_conditions: dict) -> str:
        """
        Format a human-readable title for an experiment based on model and prompt conditions.
        
        Args:
            model (str): The model name.
            prompt_conditions (dict): Dictionary with keys like 'include_nielson', 'include_examples'.
        
        Returns:
            str: Formatted title string.
        """
        nielson_text = "with Nielson reference" if prompt_conditions.get("include_nielson") else "no Nielson reference"
        examples_text = "with examples" if prompt_conditions.get("include_examples") else "no examples"
        return f"Model: {model} | {nielson_text}, {examples_text}"
    
    @classmethod
    def plot_comparison(cls, csv_file: str, experiment_configs: List[dict], 
                       plot_type: str = "scatter", figsize: tuple = (10, 12),
                       debug: bool = False):
        """
        Plot multiple experiments for comparison.
        
        Args:
            csv_file (str): Path to the CSV file containing experiment results.
            experiment_configs (List[dict]): List of experiment configurations, each containing:
                - model (str): The model name to filter by
                - prompt_conditions (dict): The prompt conditions to filter by
                - experiment_id (int): The experiment ID to filter by
            plot_type (str): Type of plot to generate (e.g., "scatter", "cdf", "histogram"). Default is "scatter".
            figsize (tuple): Figure size (width, height). Default is (10, 12).
            debug (bool): Whether to print debug information. Default is False.
        
        Returns:
            tuple: (fig, axes) The matplotlib figure and axes objects.
        """
        import matplotlib.pyplot as plt
        
        n_experiments = len(experiment_configs)
        fig, axes = plt.subplots(n_experiments, 1, figsize=figsize)
        
        # Ensure axes is always iterable (handle single subplot case)
        if n_experiments == 1:
            axes = [axes]
        
        for i, exp_config in enumerate(experiment_configs):
            if debug:
                print(f"Plotting experiment {i} on axes[{i}]")
                print(f"Filters: {exp_config['prompt_conditions']}")
                print(f"Axes object: {axes[i]}")
            
            # Retrieve the experiment
            experiment = cls.from_csv(
                csv_file,
                experiment_id=exp_config["experiment_id"],
                filters={"model": exp_config["model"]},
                prompt_conditions=[exp_config["prompt_conditions"]]
            )
            
            # Generate title
            title = cls.format_experiment_title(
                exp_config["model"], 
                exp_config["prompt_conditions"]
            )
            
            # Visualize on the respective axis
            experiment.visualize(plots=[plot_type], title=title, ax=axes[i])
            
            # Add legend
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        return fig, axes