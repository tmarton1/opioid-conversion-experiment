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

class NielsonDataPriorityUpdater:
    def __init__(self, json_file):
        """
        Initialize the NielsonDataPriorityUpdater with the JSON file path.

        Args:
            json_file (str): Path to the JSON file.
        """
        self.json_file = json_file

    def load_json(self):
        """
        Load the JSON file into a dictionary.

        Returns:
            dict: The loaded JSON data.
        """
        with open(self.json_file, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, data):
        """
        Save the updated data back to the JSON file.

        Args:
            data (dict): The updated JSON data.
        """
        with open(self.json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Updated JSON data has been saved to {self.json_file}")

    def update_priorities(self, priority_dict):
        """
        Update the JSON data with a new 'priority' field based on the input dictionary.

        Args:
            priority_dict (dict): A dictionary where keys are priority levels and values are lists of opioid keys.

        Example:
            priority_dict = {1: ['iv_fentanyl', 'oral_morphine'], 2: ['oral_hydromorphone']}
        """
        # Load the JSON data
        data = self.load_json()

        # Flatten the priority dictionary into a mapping of opioid keys to priority levels
        priority_mapping = {}
        for priority, opioids in priority_dict.items():
            for opioid in opioids:
                priority_mapping[opioid] = priority

        # Update each entry in the JSON data with the 'priority' field
        for key in data:
            data[key]["priority"] = priority_mapping.get(key, None)

        # Save the updated JSON data
        self.save_json(data)