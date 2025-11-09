from typing import List, Tuple, Union
from dataclasses import dataclass
from itertools import combinations
from nielson_data import NIELSON_CONVERSIONS

@dataclass
class Opioid:
    administration_method: str
    substance: str
    unit: str = None

    def __post_init__(self):
        key = f"{self.administration_method}_{self.substance}"
        if key in NIELSON_CONVERSIONS:
            self.unit = NIELSON_CONVERSIONS[key]["unit"]
        else:
            raise KeyError(f"Opioid key '{key}' not found in Nielson conversions.")
        
class OpioidPairGenerator:
    def __init__(self, priority_logic):
        """
        Initialize the OpioidPairGenerator with a priority logic.

        Args:
            priority_logic (function): A function that takes a priority value and returns True or False.
        """
        self.priority_logic = priority_logic

    def generate_opioid_list_or_pairs(self, pairs: bool = True) -> Union[List[Opioid],List[Tuple[Opioid, Opioid]]]:
        """
        Generate a list of unique opioid pairs based on the priority logic.

        Returns:
            List[Tuple[Opioid, Opioid]]: A list of unique opioid pairs.
        """
        # Filter opioids based on the priority logic
        filtered_opioids = [
            key for key, data in NIELSON_CONVERSIONS.items()
            if "priority" in data and self.priority_logic(data["priority"])
        ]

        # Create Opioid objects for the filtered opioids
        opioid_objects = [
            Opioid(
                administration_method=key.split("_")[0],
                substance=key.split("_")[1]
            )
            for key in filtered_opioids
        ]

        if pairs:
            # Generate unique combinations of opioid pairs
            opioid_pairs = list(combinations(opioid_objects, 2))
            return opioid_pairs
        else:
            # Return the list of individual opioids
            return opioid_objects
