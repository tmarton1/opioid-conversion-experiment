import json
from pathlib import Path

def load_scenarios(json_path: str):
    with open(json_path, "r") as file:
        data = json.load(file)
    return {
        key: value
        for key, value in data.items()
    }

# Load scenarios globally
SCENARIO_VIGNETTES = load_scenarios(Path("vignettes.json"))