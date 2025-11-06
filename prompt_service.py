from opioid import Opioid
from nielson_data import NIELSON_CONVERSIONS
from dataclasses import dataclass, field
from conversion_service import ConversionService
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

@dataclass
class PromptedOpioidConversion:
    from_opioid: Opioid
    to_opioid: Opioid = field(default_factory=lambda: Opioid(administration_method="oral", substance="morphine"))
    from_dose: float = None
    expected_to_dose: float = None
    include_nielson: bool = True
    include_examples: bool = True
    prompt: str = None
    model: str = "gpt-4.1-mini"
    observed_to_dose: float = None
    ratio_observed_expected: float = None
    response: str = None

    # Class-level attribute for the OpenAI client
    client = None

    @classmethod
    def initialize_client(cls):
        """
        Initialize the OpenAI client as a class-level attribute.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        cls.client = OpenAI()

    def __post_init__(self, conversion_service=None):
        """
        Post-initialization to handle default values and random dose generation.

        Args:
            conversion_service (ConversionService, optional): The ConversionService instance to use for generating a random dose.
        """
        if conversion_service is None:
            conversion_service = ConversionService(NIELSON_CONVERSIONS)

        if self.from_dose is None:
            self.from_dose = round(conversion_service.generate_random_dose(from_opioid=self.from_opioid), 2)

        if conversion_service is not None:
            self.expected_to_dose = round(conversion_service.calculate_expected_dose(
                dose_from=self.from_dose,
                from_opioid=self.from_opioid,
                to_opioid=self.to_opioid
            ), 2)

    def generate_prompt(self, prompt_service):
        """
        Generate a prompt using the provided PromptService instance.

        Args:
            prompt_service (PromptService): The PromptService instance to use for generating the prompt.

        Returns:
            str: The generated prompt.
        """
        if not prompt_service:
            raise ValueError("A valid PromptService instance must be provided.")

        self.prompt = prompt_service.generate_prompt(
            from_opioid=self.from_opioid,
            to_opioid=self.to_opioid,
            from_dose=self.from_dose,
            include_nielson=self.include_nielson,
            include_examples=self.include_examples
        )
        return self.prompt
    
    def ask_model(self):
        """
        Query the OpenAI model with the generated prompt.

        Returns:
            float: The observed dose predicted by the model.
        """
        if not self.prompt:
            raise ValueError("Prompt has not been generated. Call `generate_prompt` first.")

        if not PromptedOpioidConversion.client:
            raise ValueError("OpenAI client has not been initialized. Call `PromptedOpioidConversion.initialize_client()` first.")

        # Query the model
        response = PromptedOpioidConversion.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt}]
        )

        # Store the raw text of the response
        self.response = response.choices[0].message.content.strip()

        # Extract the observed dose from the model's response
        try:
            # Ensure the response matches the expected format
            if self.response.startswith("{") and self.response.endswith("}") and "calculated dose" in self.response:
                # Extract the value after "calculated dose"
                match = re.search(r"'calculated dose'\s*:\s*([\d.]+)", self.response)
                if match:
                    self.observed_to_dose = float(match.group(1))
                else:
                    raise ValueError("The response does not contain a valid numeric value for 'calculated dose'.")
            else:
                raise ValueError("The response does not match the expected format: {'calculated dose': XX}.")
        except Exception as e:
            raise ValueError(f"Error parsing the model response. Response: {self.response}") from e

        # Calculate the ratio of observed to expected dose
        if self.expected_to_dose:
            self.ratio_observed_expected = round(self.observed_to_dose / self.expected_to_dose, 2)

        return self.observed_to_dose


class PromptService:
    def __init__(self, conversion_service, model_name="gpt-4.1-mini"):
        """
        Initialize the PromptService with a ConversionService instance.

        Args:
            conversion_service (ConversionService): The service used for dose calculations.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.conversion_service = conversion_service
        self.model_name = model_name
        self.client = OpenAI()

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

        prompt += "\nExpress the response in the format {'calculated dose': value}.  Do not include any additional text in your response, only the dictionary."
        return prompt