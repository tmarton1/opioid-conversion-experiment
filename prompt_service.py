from opioid import Opioid
from nielson_data import NIELSON_CONVERSIONS
from dataclasses import dataclass, field
from conversion_service import ConversionService
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import google.generativeai as genai

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
    anthropic_client = None
    groq_client = None

    @classmethod
    def initialize_client(cls):
        """
        Initialize all API clients.
        """
        if cls.client is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            cls.client = OpenAI(api_key=api_key)
        
        if cls.anthropic_client is None and os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            cls.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        if cls.groq_client is None and os.getenv("GROQ_API_KEY"):
            from groq import Groq
            cls.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        if os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    
    def _extract_numeric_value(self, response_text: str) -> float:
        """
        Extract a numeric value from the model's response.
        Handles various formats including dictionaries, plain numbers, and text.
        """
        # Try to extract from dictionary format: {'calculated dose': value}
        dict_match = re.search(r"['\"]calculated dose['\"]:\s*([\d.]+)", response_text)
        if dict_match:
            return float(dict_match.group(1))
        
        # Try to extract any number from the response
        number_match = re.search(r"(\d+\.?\d*)", response_text)
        if number_match:
            return float(number_match.group(1))
        
        raise ValueError(f"Could not extract numeric value from response: {response_text}")
    
    def ask_model(self):
        """Query the model and return the observed dose."""
        if self.model.startswith("gpt"):
            # OpenAI
            response = self.client.chat.completions.create(  # Changed from openai_client to client
                model=self.model,
                messages=[{"role": "user", "content": self.prompt}],
                max_tokens=50,
                temperature=0.0
            )
            response_text = response.choices[0].message.content.strip()
            
        elif self.model.startswith("claude"):
            # Anthropic
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0.0,
                messages=[{"role": "user", "content": self.prompt}]
            )
            response_text = response.content[0].text.strip()
            
        elif self.model.startswith("llama"):
            # Meta via Groq
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt}],
                max_tokens=50,
                temperature=0.0
            )
            response_text = response.choices[0].message.content.strip()
            
        elif self.model.startswith("models/gemini"):
            # Google Gemini with safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            gemini_model = genai.GenerativeModel(self.model)
            response = gemini_model.generate_content(
                self.prompt,
                safety_settings=safety_settings,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.0
                )
            )
            
            # Check if response has valid content
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.text.strip()
            else:
                # Get detailed error information
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                finish_reason_name = {
                    0: "FINISH_REASON_UNSPECIFIED",
                    1: "STOP (normal completion)",
                    2: "MAX_TOKENS (ran out of tokens)",
                    3: "SAFETY (blocked by safety filters)",
                    4: "RECITATION (blocked due to recitation)",
                    5: "OTHER"
                }.get(finish_reason, f"UNKNOWN({finish_reason})")
                
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else []
                safety_info = "\n".join([f"  - {rating.category}: {rating.probability}" for rating in safety_ratings])
                
                prompt_feedback = response.prompt_feedback
                
                error_msg = f"""Gemini failed to generate response:
Finish Reason: {finish_reason_name}
Safety Ratings:
{safety_info if safety_info else '  (none)'}
Prompt Feedback: {prompt_feedback}
Prompt (first 200 chars): {self.prompt[:200]}..."""
                
                raise ValueError(error_msg)
        
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
        # Extract numeric value
        self.observed_to_dose = self._extract_numeric_value(response_text)
        self.ratio_observed_expected = self.observed_to_dose / self.expected_to_dose
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