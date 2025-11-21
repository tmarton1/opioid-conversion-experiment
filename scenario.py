from opioid import Opioid
from prompt_service import PromptedOpioidConversion, PromptService
import re

class Scenario:
    def __init__(self, scenario_dict):
        self.vignette_id = scenario_dict.get("vignette_id")
        self.text = scenario_dict["text"]
        self.from_opioid = scenario_dict["from_opioid"]
        self.to_opioid = scenario_dict["to_opioid"]
        self.dose_distribution = scenario_dict.get("dose_distribution")
        self.dose_placeholder = scenario_dict.get("dose_placeholder", "{{dose}}")
        self.derived_values = scenario_dict.get("derived_values", {})
        self.output_unit = scenario_dict.get("output_unit")
        self.output_conversion = scenario_dict.get("output_conversion")

    def sample_dose(self):
        import random
        if isinstance(self.dose_distribution, dict):
            return {k: random.choice(v) for k, v in self.dose_distribution.items()}
        else:
            # For single value, use the placeholder name as key
            key = re.sub(r"[{}]", "", self.dose_placeholder)
            return {key: random.choice(self.dose_distribution)}

    def fill_text(self, sampled_dose):
        text = self.text
        # Compute derived values
        derived = {}
        for k, expr in self.derived_values.items():
            # Safe eval: only allow variables from sampled_dose
            derived[k] = eval(expr, {}, sampled_dose)
        # Replace all placeholders
        all_values = {**sampled_dose, **derived}
        for k, v in all_values.items():
            text = text.replace(f"{{{{{k}}}}}", str(v))
        return text

class ScenarioPromptedConversion(PromptedOpioidConversion):
    def __init__(
        self,
        scenario: Scenario,
        model="gpt-4.1-mini",
        include_nielson=False,
        include_examples=False,
        conversion_service=None
    ):
        sampled_dose_dict = scenario.sample_dose()
        sampled_dose = sampled_dose_dict["dose"]
        scenario_text = scenario.fill_text(sampled_dose_dict)
        from_opioid = Opioid(**scenario.from_opioid)
        to_opioid = Opioid(**scenario.to_opioid)
        super().__init__(
            from_opioid=from_opioid,
            to_opioid=to_opioid,
            from_dose=sampled_dose,
            model=model,
            include_nielson=include_nielson,
            include_examples=include_examples,
            conversion_service=conversion_service
        )
        self.scenario_text = scenario_text
        self.vignette_id = scenario.vignette_id
        self.output_unit = getattr(scenario, "output_unit", to_opioid.unit)
        output_conversion = getattr(scenario, "output_conversion", None)
        if output_conversion:
            self.output_dose = eval(output_conversion, {}, {"expected_to_dose": self.expected_to_dose})
            self.expected_to_dose = self.output_dose
            self.to_opioid.unit = self.output_unit
        else:
            self.output_dose = self.expected_to_dose

    def generate_prompt(self, prompt_service: PromptService):
        self.prompt = prompt_service.generate_prompt(
            from_opioid=self.from_opioid,
            to_opioid=self.to_opioid,
            from_dose=self.from_dose,
            include_nielson=self.include_nielson,
            include_examples=self.include_examples,
            scenario_text=self.scenario_text
        )
        return self.prompt