import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from prompt_service import PromptedOpioidConversion, PromptService
from conversion_service import ConversionService
from nielson_data import NIELSON_CONVERSIONS
from opioid import Opioid


class Experiment:
    def __init__(self, opioid_pairs, repetitions=10, include_nielson=True, include_examples=False):
        """
        Initialize the Experiment class.

        Args:
            opioid_pairs (list of tuples): List of (from_opioid, to_opioid) pairs or a list of from_opioids (defaulting to morphine as to_opioid).
            repetitions (int): Number of repetitions for each pair.
            include_nielson (bool): Whether to include Nielson coefficients in the prompt.
            include_examples (bool): Whether to include examples in the prompt.
        """
        # If only a list of from_opioids is provided, default to morphine as the to_opioid
        self.opioid_pairs = [
            (from_opioid, Opioid(administration_method="oral", substance="morphine"))
            if isinstance(from_opioid, Opioid) else from_opioid
            for from_opioid in opioid_pairs
        ]
        self.repetitions = repetitions
        self.include_nielson = include_nielson
        self.include_examples = include_examples
        self.results = []  # To store results of all experiments

        # Initialize the PromptService once for the entire experiment
        self.prompt_service = PromptService(ConversionService(NIELSON_CONVERSIONS))


    def run(self):
        """
        Run the experiment for all opioid pairs and repetitions.
        """
        for from_opioid, to_opioid in self.opioid_pairs:
            for _ in range(self.repetitions):
                conversion = PromptedOpioidConversion(
                    from_opioid=from_opioid,
                    to_opioid=to_opioid,
                    include_nielson=self.include_nielson,
                    include_examples=self.include_examples
                )
                conversion.generate_prompt(prompt_service=self.prompt_service)
                try:
                    conversion.ask_model()
                    self.results.append({
                        "from_opioid": f"{from_opioid.administration_method}_{from_opioid.substance}",
                        "from_opioid_unit": from_opioid.unit,
                        "to_opioid": f"{to_opioid.administration_method}_{to_opioid.substance}",
                        "to_opioid_unit": to_opioid.unit,
                        "from_dose": conversion.from_dose,
                        "expected_to_dose": conversion.expected_to_dose,
                        "model": conversion.model,
                        "prompt": conversion.prompt,
                        "include_nielson": conversion.include_nielson,
                        "include_examples": conversion.include_examples,
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
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

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
    def from_csv(csv_file):
        """
        Create an Experiment object from a CSV file.

        Args:
            csv_file (str): Path to the CSV file containing experiment results.

        Returns:
            Experiment: An Experiment object with the data loaded from the CSV.
        """
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_file)

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