import pandas as pd

def add_ome_column(input_csv: str, output_csv: str = None):
    """
    Add an 'ome' column to experiment results indicating if oral morphine is involved.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (defaults to input_csv if None)
    """
    if output_csv is None:
        output_csv = input_csv
    
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    # Check if oral morphine is involved in either from or to opioid
    df['ome'] = (
        (df['from_opioid'] == 'oral_morphine') |
        (df['to_opioid'] == 'oral_morphine')
    )
    
    # Save back to CSV
    df.to_csv(output_csv, index=False)
    print(f"Added 'ome' column to {output_csv}")