from opioid import Opioid
from nielson_data import NIELSON_CONVERSIONS
from dataclasses import dataclass, field
from conversion_service import ConversionService
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
from dotenv import load_dotenv
import os
import re
import google.generativeai as genai


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

    # Class-level clients (initialized once)
    openai_client = None
    openrouter_client = None
    anthropic_client = None
    groq_client = None
    google_configured = False


    @classmethod
    def initialize_client(cls):
        """Initialize all API clients once at the class level."""
        # OpenAI direct client
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            cls.openai_client = OpenAI(api_key=openai_key)
        
        # OpenRouter client
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        if openrouter_key:
            headers = {}
            if os.getenv("OPENROUTER_SITE_URL"):
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
            if os.getenv("OPENROUTER_APP_NAME"):
                headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")
            cls.openrouter_client = OpenAI(
                api_key=openrouter_key,
                base_url=openrouter_base,
                default_headers=headers or None
            )
        
        # Anthropic client (optional - only if API key exists)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            cls.anthropic_client = Anthropic(api_key=anthropic_key)
        
        # Groq client (optional - only if API key exists)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            cls.groq_client = Groq(api_key=groq_key)
        
        # Google client (optional - only if API key exists)
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            genai.configure(api_key=google_key)
            cls.google_configured = True

    def __init__(self, from_opioid, to_opioid=None, from_dose=None, model="gpt-4.1-mini",
                 include_nielson=False, include_examples=False, conversion_service=None):
        """
        Initialize a PromptedOpioidConversion instance.
        
        Args:
            from_opioid (Opioid): The source opioid.
            to_opioid (Opioid, optional): The target opioid. Defaults to oral morphine.
            from_dose (float, optional): The dose to convert from. If None, a random dose is generated.
            model (str): The model to use for conversion.
            include_nielson (bool): Whether to include Nielson conversion data in the prompt.
            include_examples (bool): Whether to include examples in the prompt.
            conversion_service (ConversionService, optional): Service for dose calculations.
        """
        # Store opioid information
        self.from_opioid = from_opioid
        self.to_opioid = to_opioid if to_opioid else Opioid(
            administration_method="oral",
            substance="morphine",
            unit="mg/day"
        )
        
        # Store model and prompt settings
        self.model = model
        self.include_nielson = include_nielson
        self.include_examples = include_examples
        
        # Initialize conversion service if not provided
        if not conversion_service:
            from nielson_data import NIELSON_CONVERSIONS
            conversion_service = ConversionService(NIELSON_CONVERSIONS)
        
        # Generate random dose if not provided
        if from_dose is None:
            self.from_dose = conversion_service.generate_random_dose(from_opioid)
        else:
            self.from_dose = from_dose
        
        # Calculate expected dose
        self.expected_to_dose = conversion_service.calculate_expected_dose(
            self.from_dose,
            self.from_opioid,
            self.to_opioid
        )
        
        # Initialize result variables
        self.prompt = None
        self.observed_to_dose = None
        self.ratio_observed_expected = None

    @staticmethod
    def _is_openrouter_model(model_id):
        """Check if a model ID should be routed through OpenRouter."""
        openrouter_prefixes = [
            "openai/", "anthropic/", "mistralai/", "google/",
            "cohere/", "ai21/", "qwen/", "meta-llama/", "deepseek/"
        ]
        return any(model_id.startswith(prefix) for prefix in openrouter_prefixes)

    def ask_model(self):
        """Query the specified model with the generated prompt and extract the calculated dose."""
        if not self.prompt:
            raise ValueError("Prompt must be generated before querying the model.")
        
        # Determine which client to use
        if self._is_openrouter_model(self.model):
            response_text = self._query_openrouter()
        elif self.model.startswith("claude-"):
            response_text = self._query_anthropic()
        elif self.model.startswith("llama-"):
            response_text = self._query_groq()
        elif self.model.startswith("models/gemini-") or self.model.startswith("gemini-"):
            response_text = self._query_google()
        else:
            # Default to OpenAI direct
            response_text = self._query_openai()
        
        # Extract the calculated dose
        self.observed_to_dose = self._extract_calculated_dose(response_text)
        self.ratio_observed_expected = (
            self.observed_to_dose / self.expected_to_dose if self.expected_to_dose else None
        )
        return self.observed_to_dose

    def _query_openrouter(self):
        """Query a model via OpenRouter."""
        if not self.openrouter_client:
            raise ValueError("OpenRouter client not initialized. Call initialize_client() first.")
        
        try:
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error for model {self.model}: {e}")

    def _query_openai(self):
        """Query OpenAI models directly."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Call initialize_client() first.")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _query_anthropic(self):
        """Query Anthropic Claude models directly."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Call initialize_client() first.")
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.0,
                messages=[{"role": "user", "content": self.prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def _query_groq(self):
        """Query Groq models."""
        if not self.groq_client:
            raise ValueError("Groq client not initialized. Call initialize_client() first.")
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}")

    def _query_google(self):
        """Query Google Gemini models."""
        if not self.google_configured:
            raise ValueError("Google client not configured. Call initialize_client() first.")
        
        try:
            # Remove "models/" prefix if present
            model_name = self.model.replace("models/", "")
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                self.prompt,
                safety_settings=safety_settings,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.0
                )
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}")


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
    
    def _extract_calculated_dose(self, response_text: str) -> float:
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

        prompt += "\nExpress the response in the format {'calculated dose': value}.  Do not include any additional text in your response, only the dictionary."
        return prompt