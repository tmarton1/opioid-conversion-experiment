from dataclasses import dataclass, field
from common.opioid import Opioid
from typing import Dict, Tuple
import numpy as np

class ConversionService:
    def __init__(self, nielsen_conversions: Dict[str, float]):
        self.nielsen_conversions = nielsen_conversions

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

        if opioid_key not in self.nielsen_conversions:
            raise KeyError(f"Opioid key '{opioid_key}' not found in Nielsen conversions.")
        
        conversion_factor = self.nielsen_conversions[opioid_key]['conversion_factor']
        nielsen_unit = self.nielsen_conversions[opioid_key]['unit']

        dose_in_nielsen_unit = oral_morphine_dose / conversion_factor

        # Convert to requested unit if needed
        dose_in_requested_unit = dose_in_nielsen_unit
        if from_opioid.unit != nielsen_unit:
            if nielsen_unit == "mg/day" and from_opioid.unit == "mg/hour":
                dose_in_requested_unit = dose_in_nielsen_unit / 24
            elif nielsen_unit == "mg/day" and from_opioid.unit in ["mg/4hr", "mg/4 hours"]:
                dose_in_requested_unit = dose_in_nielsen_unit / 6
            else:
                raise ValueError(f"Unit conversion from '{nielsen_unit}' to '{from_opioid.unit}' is not implemented.")

        return dose_in_requested_unit
    
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

        if from_key not in self.nielsen_conversions or to_key not in self.nielsen_conversions:
            raise KeyError(f"Opioid keys '{from_key}' or '{to_key}' not found in Nielsen conversions.")
        
        from_conversion = self.nielsen_conversions[from_key]["conversion_factor"]
        to_conversion = self.nielsen_conversions[to_key]["conversion_factor"]
        nielsen_from_unit = self.nielsen_conversions[from_key]["unit"]
        nielsen_to_unit = self.nielsen_conversions[to_key]["unit"]

        # --- Unit correction for from_opioid ---
        dose_from_nielsen = dose_from
        if from_opioid.unit != nielsen_from_unit:
            # Example: mg/hour to mg/day
            if from_opioid.unit == "mg/hour" and nielsen_from_unit == "mg/day":
                dose_from_nielsen = dose_from * 24
            elif from_opioid.unit == "mg/4hr" and nielsen_from_unit == "mg/day":
                dose_from_nielsen = dose_from * 6
            elif from_opioid.unit == "mg/4 hours" and nielsen_from_unit == "mg/day":
                dose_from_nielsen = dose_from * 6
            else:
                raise ValueError(f"Unit conversion from '{from_opioid.unit}' to '{nielsen_from_unit}' is not implemented.")


        # Convert dose_from to oral morphine equivalent, then to target opioid
        oral_morphine_equivalent = dose_from_nielsen * from_conversion
        expected_dose_to = oral_morphine_equivalent / to_conversion

        dose_from_nielsen = dose_from
        if to_opioid.unit != nielsen_to_unit:
            # Example: mg/day to mg/4hr
            if nielsen_to_unit == "mg/day" and to_opioid.unit == "mg/4hr":
                expected_dose_to = expected_dose_to / 6
            elif nielsen_to_unit == "mg/day" and to_opioid.unit == "mg/4 hours":
                expected_dose_to = expected_dose_to / 6
            elif nielsen_to_unit == "mg/day" and to_opioid.unit == "mg/hour":
                expected_dose_to = expected_dose_to / 24
            # Add more conversion rules here as needed
            else:
                raise ValueError(f"Unit conversion from '{nielsen_to_unit}' to '{to_opioid.unit}' is not implemented.")

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
            ome = self.calculate_expected_dose(dose, opioid, Opioid(administration_method="oral", substance="morphine", unit="mg/day"))
            total_ome += ome

        # Convert total OME to target opioid
        expected_dose_to = self.calculate_expected_dose(
            total_ome,
            Opioid(administration_method="oral", substance="morphine", unit="mg/day"),
            to_opioid
        )
        return expected_dose_to