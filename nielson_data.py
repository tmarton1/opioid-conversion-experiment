import json
import pandas as pd
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

class NielsonDataCSVConverter:
    def __init__(self, json_file, csv_file):
        """
        Initialize the NielsonDataConverter with file paths.

        Args:
            json_file (str): Path to the JSON file.
            csv_file (str): Path to the output CSV file.
        """
        self.json_file = json_file
        self.csv_file = csv_file

    def load_json(self):
        """
        Load the JSON file into a dictionary.

        Returns:
            dict: The loaded JSON data.
        """
        with open(self.json_file, "r") as f:
            data = json.load(f)
        return data

    def convert_to_csv(self):
        """
        Convert the JSON data to a CSV file.
        """
        # Load the JSON data
        data = self.load_json()

        # Convert the JSON data into a DataFrame
        df = pd.DataFrame.from_dict(data, orient="index")

        # Save the DataFrame to a CSV file
        df.to_csv(self.csv_file, index_label="key")

        print(f"JSON data has been successfully converted to {self.csv_file}")