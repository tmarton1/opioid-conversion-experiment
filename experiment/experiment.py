from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from experiment.prompt_service import PromptedOpioidConversion, PromptService
from common.conversion_service import ConversionService
from resources.nielsen_data import NIELSEN_CONVERSIONS
from common.opioid import Opioid
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

    def __init__(self, opioid_pairs, prompt_conditions, models, repetitions=10, experiment_id=None, filename="experiment_results.csv"):
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
        self.prompt_service = PromptService(ConversionService(NIELSEN_CONVERSIONS))
        self.filename = filename

    def run(self):
        """
        Run the experiment for all combinations of opioid_pairs, prompt_conditions, and models, repeated for the specified number of repetitions.
        """
        for (from_opioid, to_opioid), condition, model in self.combinations:
            for _ in range(self.repetitions):
                conversion = PromptedOpioidConversion(
                    from_opioid=from_opioid,
                    to_opioid=to_opioid,
                    include_nielsen=condition["include_nielsen"],
                    include_examples=condition["include_examples"],
                    model=model,
                    ask_for_units=condition.get("ask_for_units", False),
                    explain_units=condition.get("explain_units", False)
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
                        "include_nielsen": condition["include_nielsen"],
                        "include_examples": condition["include_examples"],
                        "ask_for_units": condition.get("ask_for_units", False),
                        "explain_units": condition.get("explain_units", False),
                        "observed_to_dose": conversion.observed_to_dose,
                        "ratio_observed_expected": conversion.ratio_observed_expected,
                        "response": conversion.response
                    }
                    # Save the result directly to the CSV file
                    self.add_result_to_csv(result, filename=self.filename)
                except CSVWriteError as e:
                    print(f"Error writing result to CSV: {e}")
                except Exception as e:
                    print(f"Error during conversion: {e}")

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
            "from_dose", "expected_to_dose", "model", "prompt", "include_nielsen",
            "include_examples", "ask_for_units", "explain_units", "observed_to_dose", "ratio_observed_expected", "response"
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