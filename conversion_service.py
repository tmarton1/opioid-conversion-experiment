from dataclasses import dataclass, field
from opioid import Opioid
from typing import Dict, Tuple
import numpy as np

class ConversionService:
    def __init__(self, nielson_conversions: Dict[str, float]):
        self.nielson_conversions = nielson_conversions

    def generate_random_dose(self, from_opioid: Opioid, dose_range: Tuple[float, float] = (10, 120)) -> float:
        """
        Generate a random dose for the given opioid within the specified range.

        Args:
            from_opioid (Opioid): The opioid for which the dose is being generated.
            dose_range (Tuple[float, float]): The range (min, max) for the dose in oral morphine equivalents.

        Returns:
            float: The generated dose in oral morphine equivalents, adjusted for the opioid.
        """
        min_dose, max_dose = dose_range
        oral_morphine_dose = np.exp(np.random.uniform(np.log(min_dose), np.log(max_dose)))
        opioid_key = f"{from_opioid.administration_method}_{from_opioid.substance}"

        if opioid_key not in self.nielson_conversions:
            raise KeyError(f"Opioid key '{opioid_key}' not found in Nielson conversions.")
        
        conversion_factor = self.nielson_conversions[opioid_key]['conversion_factor']
        return oral_morphine_dose / conversion_factor
    
    def calculate_expected_dose(self, dose_from: float, from_opioid: Opioid, to_opioid: Opioid) -> float:
        """
        Calculate the expected dose of the target opioid based on the dose of the source opioid.

        Args:
            dose_from (float): The dose of the source opioid.
            from_opioid (Opioid): The source opioid.
            to_opioid (Opioid): The target opioid.

        Returns:
            float: The expected dose of the target opioid.
        """
        from_key = f"{from_opioid.administration_method}_{from_opioid.substance}"
        to_key = f"{to_opioid.administration_method}_{to_opioid.substance}"

        if from_key not in self.nielson_conversions or to_key not in self.nielson_conversions:
            raise KeyError(f"Opioid keys '{from_key}' or '{to_key}' not found in Nielson conversions.")
        
        from_conversion = self.nielson_conversions[from_key]["conversion_factor"]
        to_conversion = self.nielson_conversions[to_key]["conversion_factor"]
        nielson_unit = self.nielson_conversions[to_key]["unit"]

        # Convert dose_from to oral morphine equivalent, then to target opioid
        oral_morphine_equivalent = dose_from * from_conversion
        expected_dose_to = oral_morphine_equivalent / to_conversion

            # Unit correction logic
        if to_opioid.unit != nielson_unit:
            # Example: mg/day to mg/4hr
            if nielson_unit == "mg/day" and to_opioid.unit == "mg/4hr":
                expected_dose_to = expected_dose_to / 6
            elif nielson_unit == "mg/day" and to_opioid.unit == "mg/hour":
                expected_dose_to = expected_dose_to / 24
            # Add more conversion rules here as needed
            else:
                raise ValueError(f"Unit conversion from '{nielson_unit}' to '{to_opioid.unit}' is not implemented.")

        return expected_dose_to
    
    def calculate_expected_dose_multiple(self, from_opioids_doses: List[Tuple[Opioid, float]], to_opioid: Opioid) -> float:
        """
        Convert multiple opioid doses to a single target opioid dose.

        Args:
            from_opioids_doses (List[Tuple[Opioid, float]]): List of (Opioid, dose) pairs.
            to_opioid (Opioid): The target opioid.

        Returns:
            float: The expected dose of the target opioid.
        """
        total_ome = 0.0
        for opioid, dose in from_opioids_doses:
            # Convert each opioid dose to oral morphine equivalent (OME)
            ome = self.calculate_expected_dose(dose, opioid, Opioid(administration_method="oral", substance="morphine"))
            total_ome += ome

        # Convert total OME to target opioid
        expected_dose_to = self.calculate_expected_dose(
            total_ome,
            Opioid(administration_method="oral", substance="morphine"),
            to_opioid
        )
        return expected_dose_to