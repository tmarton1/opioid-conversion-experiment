import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from prompt_service import PromptedOpioidConversion, PromptService
from conversion_service import ConversionService
from nielson_data import NIELSON_CONVERSIONS
from opioid import Opioid
import os
from itertools import product  # For generating combinations of opioid_pairs and prompt_conditions


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

    def __init__(self, opioid_pairs, prompt_conditions, repetitions=10, experiment_id=None):
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
        self.repetitions = repetitions
        self.results = []  # To store results of all experiments

        # Generate all combinations of opioid_pairs and prompt_conditions
        self.combinations = list(product(opioid_pairs, prompt_conditions))

        # Initialize the PromptService once for the entire experiment
        self.prompt_service = PromptService(ConversionService(NIELSON_CONVERSIONS))

    def run(self):
        """
        Run the experiment for all combinations of opioid_pairs and prompt_conditions, repeated for the specified number of repetitions.
        """
        for (from_opioid, to_opioid), condition in self.combinations:
            for _ in range(self.repetitions):
                conversion = PromptedOpioidConversion(
                    from_opioid=from_opioid,
                    to_opioid=to_opioid,
                    include_nielson=condition["include_nielson"],
                    include_examples=condition["include_examples"]
                )
                conversion.generate_prompt(prompt_service=self.prompt_service)
                try:
                    conversion.ask_model()
                    self.results.append({
                        "experiment_id": self.experiment_id,  # Add experiment ID to each result
                        "from_opioid": f"{from_opioid.administration_method}_{from_opioid.substance}",
                        "from_opioid_unit": from_opioid.unit,
                        "to_opioid": f"{to_opioid.administration_method}_{to_opioid.substance}",
                        "to_opioid_unit": to_opioid.unit,
                        "from_dose": conversion.from_dose,
                        "expected_to_dose": conversion.expected_to_dose,
                        "model": conversion.model,
                        "prompt": conversion.prompt,
                        "include_nielson": condition["include_nielson"],
                        "include_examples": condition["include_examples"],
                        "observed_to_dose": conversion.observed_to_dose,
                        "ratio_observed_expected": conversion.ratio_observed_expected,
                        "response": conversion.response
                    })
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

    def visualize(self):
        """
        Visualize the results using matplotlib.
        """
        df = self.to_dataframe()

        # Scatter plot: Expected vs Observed Dose
        plt.figure(figsize=(5, 5))
        for (from_opioid, to_opioid), group in df.groupby(["from_opioid", "to_opioid"]):
            plt.scatter(group["expected_to_dose"], group["ratio_observed_expected"], label=f"{from_opioid} → {to_opioid}")
        plt.axhline(1.0, color="red", linestyle="--", label="Ideal Ratio (1.0)")
        plt.axhline(0.7, color="orange", linestyle="--", label="Lower Bound (0.7)")
        plt.axhline(1.3, color="orange", linestyle="--", label="Upper Bound (1.3)")
        plt.xlabel("Expected Dose")
        plt.ylabel("Ratio Observed/Expected")
        plt.title("Expected Dose vs Ratio Observed/Expected")
        plt.legend()
        plt.grid()
        plt.show()

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

        Returns:
            Experiment: An Experiment object with the data loaded from the CSV.
        """
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_file)

        # Apply filters
        if experiment_id:
            df = df[df["experiment_id"] == experiment_id]
        if filters:
            for key, value in filters.items():
                df = df[df[key] == value]

        if prompt_conditions:
            # Ensure all rows match the provided prompt_conditions
            for condition in prompt_conditions:
                for key, value in condition.items():
                    df = df[df[key] == value]

        # Extract unique opioid pairs and prompt conditions from the DataFrame
        opioid_pairs = df[["from_opioid", "to_opioid"]].drop_duplicates().to_records(index=False)
        prompt_conditions = df[["include_nielson", "include_examples"]].drop_duplicates().to_dict(orient="records")

        # Create a mapping of opioid_pairs to their associated prompt_conditions
        pair_condition_map = df.groupby(["from_opioid", "to_opioid"])[["include_nielson", "include_examples"]].apply(
            lambda x: x.drop_duplicates().to_dict(orient="records")
        )

        # Check for missing prompt_conditions for each opioid pair
        missing_conditions = {}
        for pair, conditions in pair_condition_map.items():
            missing = [cond for cond in prompt_conditions if cond not in conditions]
            if missing:
                missing_conditions[pair] = missing

        # Check for prompt_conditions that are completely covered by all opioid_pairs
        covered_conditions = []
        for condition in prompt_conditions:
            if all(condition in pair_condition_map[pair] for pair in pair_condition_map):
                covered_conditions.append(condition)

        # Raise an error if there are inconsistencies
        if missing_conditions:
            error_message = "Inconsistent prompt_conditions in the CSV file:\n"
            error_message += "1. Opioid pairs missing some prompt_conditions:\n"
            for pair, missing in missing_conditions.items():
                error_message += f"   - {pair}: Missing {missing}\n"
            error_message += "2. Prompt_conditions completely covered by all opioid_pairs:\n"
            error_message += f"   - {covered_conditions}\n"
            raise ValueError(error_message)

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

        # Create an Experiment object
        experiment = Experiment(opioid_pairs)

        # Populate the results directly from the CSV data
        experiment.results = df.to_dict(orient="records")

        return experiment