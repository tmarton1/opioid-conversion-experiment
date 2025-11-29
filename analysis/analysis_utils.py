import pandas as pd
from analysis.prefix import harmonize_model_names
from common.conversion_service import ConversionService
from resources.nielson_data import NIELSON_CONVERSIONS
from analysis.opioid_pair_model_result import OpioidPairModelResult
from analysis.opioid_pair_consensus import OpioidPairConsensus

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

def processing_pipeline(
    input_csv,
    output_csv,
    variable_model_csv,
    consensus_csv,
    mode="pair",  # "pair" or "vignette"
    verbose=True
):

    add_ome_column(input_csv=input_csv, output_csv=output_csv)
    harmonize_model_names(output_csv)

    if mode == "pair":

        conversion_service = ConversionService(NIELSON_CONVERSIONS)

        OpioidPairModelResult.process_all_combinations(
            csv_file=output_csv,
            output_file=variable_model_csv,
            conversion_service=conversion_service,
            groupby_fields=["from_opioid", "from_opioid_unit", "to_opioid", "to_opioid_unit", "model", "include_nielson", "include_examples", "explain_units"],
            verbose=verbose
        )
        OpioidPairConsensus.process_all_combinations(
            csv_file=variable_model_csv,
            output_file=consensus_csv,
            conversion_service=conversion_service,
            groupby_fields=["from_opioid", "from_opioid_unit", "to_opioid", "to_opioid_unit", "include_nielson", "include_examples", "explain_units"],
            verbose=verbose
        )
        OpioidPairConsensus.update_pair_model_analysis(
            pair_model_csv=variable_model_csv,
            consensus_csv=consensus_csv,
            output_csv=variable_model_csv,
            verbose=verbose
        )
    elif mode == "vignette":
        from analysis.vignette_model_result import VignetteModelResult, VignetteModelConsensus

        VignetteModelResult.process_all_combinations(
            csv_file=output_csv,
            output_file=variable_model_csv,
            verbose=verbose
        )
        VignetteModelConsensus.process_all_combinations(
            csv_file=variable_model_csv,
            output_file=consensus_csv,
            verbose=verbose
        )
    else:
        raise ValueError("mode must be 'pair' or 'vignette'")