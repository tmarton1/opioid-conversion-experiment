from opioid import Opioid
from nielson_data import NIELSON_CONVERSIONS

class PromptService:
    def __init__(self, conversion_service):
        """
        Initialize the PromptService with a ConversionService instance.

        Args:
            conversion_service (ConversionService): The service used for dose calculations.
        """
        self.conversion_service = conversion_service

    def generate_prompt(
        self,
        from_opioid: Opioid,
        to_opioid: Opioid,
        from_dose: float,
        include_nielson: bool = True,
        include_examples: bool = True
    ) -> str:
        """
        Generate a prompt for the model based on the given inputs.

        Args:
            from_opioid (Opioid): The source opioid.
            to_opioid (Opioid): The target opioid.
            dose_from (float): The dose of the `from_opioid`.
            include_nielson (bool): Whether to include Nielson conversion information in the prompt.
            include_examples (bool): Whether to include example conversions in the prompt.

        Returns:
            str: The generated prompt.
        """
        prompt = f"Convert {from_dose:.2f} {from_opioid.unit} of {from_opioid.substance} ({from_opioid.administration_method}) to {to_opioid.unit} of {to_opioid.substance} ({to_opioid.administration_method}).\n" \
                 f"How many {to_opioid.unit} of {to_opioid.substance} ({to_opioid.administration_method}) is equivalent to {from_dose:.2f} {from_opioid.unit} of {from_opioid.substance} ({from_opioid.administration_method})?\n\n"
        
        from_conversion = NIELSON_CONVERSIONS.get(f"{from_opioid.administration_method}_{from_opioid.substance}", None)
        to_conversion = NIELSON_CONVERSIONS.get(f"{to_opioid.administration_method}_{to_opioid.substance}", None)
        
        if include_nielson:
            if from_conversion and to_conversion:
                prompt += f"Use the following conversion factor(s):\n"
                prompt += f"{from_conversion['conversion_factor']} OME: 1 {from_opioid.unit} of {from_opioid.substance} ({from_opioid.administration_method}) is equivalent to {from_conversion['conversion_factor']} mg/day morphine (oral).\n"
                if to_opioid != Opioid(administration_method="oral", substance="morphine"):
                    prompt += f"{to_conversion['conversion_factor']} OME: 1 {to_opioid.unit} of {to_opioid.substance} ({to_opioid.administration_method}) is equivalent to {to_conversion['conversion_factor']} mg/day morphine (oral).\n\n"

        if include_examples:
            prompt += (
                f"{from_conversion['conversion_factor']} OME for {from_opioid.substance} ({from_opioid.administration_method}) means 10 mg/day of morphine (oral) is equivalent to {self.conversion_service.calculate_expected_dose(10, Opioid(administration_method='oral', substance='morphine'), from_opioid):.2f} {from_opioid.unit} of {from_opioid.substance} ({from_opioid.administration_method}).\n"
                f"And 120 mg/day of morphine (oral) is equivalent to {self.conversion_service.calculate_expected_dose(120, Opioid(administration_method='oral', substance='morphine'), from_opioid):.2f} {from_opioid.unit} of {from_opioid.substance} ({from_opioid.administration_method}).\n")
            if to_opioid != Opioid(administration_method="oral", substance="morphine"):
                prompt += (
                    f"{to_conversion['conversion_factor']} OME for {to_opioid.substance} ({to_opioid.administration_method}) means 10 mg/day of morphine (oral) is equivalent to {self.conversion_service.calculate_expected_dose(10, Opioid(administration_method='oral', substance='morphine'), to_opioid):.2f} {to_opioid.unit} of {to_opioid.substance} ({to_opioid.administration_method}).\n"
                    f"And 120 mg/day of morphine (oral) is equivalent to {self.conversion_service.calculate_expected_dose(120, Opioid(administration_method='oral', substance='morphine'), to_opioid):.2f} {to_opioid.unit} of {to_opioid.substance} ({to_opioid.administration_method})."
                )
        
        return prompt