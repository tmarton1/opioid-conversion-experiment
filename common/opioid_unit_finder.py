import pandas as pd
from typing import Optional

class OpioidUnitFinder:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def get_unit(self, opioid_col: str, opioid_name: str, unit_col: str) -> Optional[str]:
        filtered = self.df[self.df[opioid_col] == opioid_name]
        units = filtered[unit_col].dropna().unique()
        if len(units) == 1:
            return units[0]
        elif len(units) > 1:
            print(f"Warning: Multiple units found for {opioid_name}: {units}. Using first.")
            return units[0]
        else:
            print(f"Warning: No unit found for {opioid_name}.")
            return None