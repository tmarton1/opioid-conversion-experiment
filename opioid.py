from typing import Dict
from dataclasses import dataclass
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