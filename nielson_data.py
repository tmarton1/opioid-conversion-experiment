import json
from pathlib import Path

def load_nielson_conversions(json_path: str):
    with open(json_path, "r") as file:
        data = json.load(file)
    return {
        key: {
            "conversion_factor": value["conversion_factor"],
            "unit": value.get("unit", "mg/day")
        }
        for key, value in data.items()
    }

# Load Nielson conversions globally
NIELSON_CONVERSIONS = load_nielson_conversions(Path("nielson_conversions.json"))