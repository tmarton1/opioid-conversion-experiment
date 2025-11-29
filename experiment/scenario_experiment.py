from experiment.experiment import Experiment
from datetime import datetime
from common.conversion_service import ConversionService
from resources.nielson_data import NIELSON_CONVERSIONS
from experiment.prompt_service import PromptService
import csv

class CSVWriteError(Exception):
    """Custom exception for errors when writing to the CSV file."""
    pass

class ScenarioExperiment(Experiment):
    def __init__(self, scenarios, prompt_conditions, models, repetitions=10, experiment_id=None, filename="scenario_experiment_results.csv"):
        self.experiment_id = experiment_id or self._get_next_experiment_id()
        self.scenarios = scenarios  # List of ScenarioPromptedConversion objects
        self.prompt_conditions = prompt_conditions
        self.models = models
        self.repetitions = repetitions
        self.results = []
        self.filename = filename
        self.combinations = [
            (scenario, condition, model)
            for scenario in self.scenarios
            for condition in self.prompt_conditions
            for model in self.models
        ]
        self.prompt_service = PromptService(ConversionService(NIELSON_CONVERSIONS))

    def run(self):
        for scenario_conversion, condition, model in self.combinations:
            for _ in range(self.repetitions):
                scenario_conversion.model = model
                scenario_conversion.include_nielson = condition["include_nielson"]
                scenario_conversion.include_examples = condition["include_examples"]
                scenario_conversion.generate_prompt(self.prompt_service)

                from_opioids_doses = getattr(scenario_conversion, "from_opioids_doses", None)
                if from_opioids_doses and len(from_opioids_doses) > 0:
                    from_opioid_names = [f"{op.administration_method}_{op.substance}" for op, _ in from_opioids_doses]
                    from_opioid_units = [op.unit for op, _ in from_opioids_doses]
                    from_doses = [dose for _, dose in from_opioids_doses]
                    multiple_from_opioids = len(from_opioids_doses) > 1
                else:
                    # Fallback to single from_opioid attributes
                    op = getattr(scenario_conversion, "from_opioid", None)
                    dose = getattr(scenario_conversion, "from_dose", None)
                    from_opioid_names = [f"{op.administration_method}_{op.substance}"] if op else [""]
                    from_opioid_units = [op.unit] if op else [""]
                    from_doses = [dose] if dose is not None else [""]
                    multiple_from_opioids = False

                try:
                    scenario_conversion.ask_model()
                    result = {
                        "experiment_id": self.experiment_id,
                        "datetime": datetime.now().isoformat(),
                        "vignette_id": scenario_conversion.vignette_id,
                        "from_opioid": ",".join(from_opioid_names),
                        "from_opioid_unit": ",".join(from_opioid_units),
                        "to_opioid": f"{scenario_conversion.to_opioid.administration_method}_{scenario_conversion.to_opioid.substance}",
                        "to_opioid_unit": scenario_conversion.to_opioid.unit,
                        "from_dose": ",".join(str(d) for d in from_doses),
                        "expected_to_dose": scenario_conversion.expected_to_dose,
                        "model": model,
                        "prompt": scenario_conversion.prompt,
                        "include_nielson": condition["include_nielson"],
                        "include_examples": condition["include_examples"],
                        "observed_to_dose": scenario_conversion.observed_to_dose,
                        "ratio_observed_expected": scenario_conversion.ratio_observed_expected,
                        "response": scenario_conversion.response,
                        "multiple_from_opioids": multiple_from_opioids
                    }
                    self.add_result_to_csv(result, filename=self.filename)
                except Exception as e:
                    print(f"Error during scenario conversion: {e}")
                
    def add_result_to_csv(self, result, filename="scenario_experiment_results.csv"):
        fieldnames = [
            "experiment_id", "datetime", "vignette_id", "from_opioid", "from_opioid_unit",
            "to_opioid", "to_opioid_unit", "from_dose", "expected_to_dose", "model", "prompt",
            "include_nielson", "include_examples", "observed_to_dose", "ratio_observed_expected",
            "response", "multiple_from_opioids"
        ]
        if filename is None:
            filename = self.filename
        try:
            with open(filename, mode="a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(result)
        except Exception as e:
            raise CSVWriteError(f"Failed to write result to CSV file '{filename}': {e}")