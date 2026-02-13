from typing import List, Tuple, Union
from dataclasses import dataclass
from itertools import permutations, combinations
from resources.nielsen_data import NIELSEN_CONVERSIONS

@dataclass
class Opioid:
    administration_method: str
    substance: str
    unit: str = None

    def __post_init__(self):
        key = f"{self.administration_method}_{self.substance}"
        if self.unit is None:
            if key in NIELSEN_CONVERSIONS:
                self.unit = NIELSEN_CONVERSIONS[key]["unit"]
            else:
                raise KeyError(f"Opioid key '{key}' not found in Nielsen conversions.")
        
    def to_str(self):
        return f"{self.administration_method}_{self.substance}"
        
class OpioidPairGenerator:
    def __init__(self, priority_logic):
        """
        Initialize the OpioidPairGenerator with a priority logic.

        Args:
            priority_logic (function): A function that takes a priority value and returns True or False.
        """
        self.priority_logic = priority_logic

    def generate_opioid_list_or_pairs(self, pairs: bool = True, use_permutations: bool = False) -> Union[List[Opioid],List[Tuple[Opioid, Opioid]]]:
        """
        Generate a list of unique opioid pairs based on the priority logic.

        Returns:
            List[Tuple[Opioid, Opioid]]: A list of unique opioid pairs.
        """
        # Filter opioids based on the priority logic
        filtered_opioids = [
            key for key, data in NIELSEN_CONVERSIONS.items()
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
            if use_permutations:
                # Generate ordered pairs (permutations)
                opioid_pairs = list(permutations(opioid_objects, 2))
            else:
                # Generate unique combinations of opioid pairs
                opioid_pairs = list(combinations(opioid_objects, 2))
            return opioid_pairs
        else:
            # Return the list of individual opioids
            return opioid_objects
        
def dict_to_opioid(d):
    """
    Create an Opioid object from a dict, ignoring extra fields.
    """
    allowed_keys = {"administration_method", "substance", "unit"}
    filtered = {k: v for k, v in d.items() if k in allowed_keys}
    return Opioid(**filtered)

def opioid_list_to_str(opioids):
    """
    Convert a single Opioid or a list of Opioid objects to a comma-separated string.
    """
    if isinstance(opioids, list):
        return ",".join([o.to_str() for o in opioids])
    return opioids.to_str()
