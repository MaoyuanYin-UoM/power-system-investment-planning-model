"""
Extract specified data values from Excel files.

A utility function for reading data from multiple Excel files where each sheet contains
a two-column structure: names in the first column and corresponding values in the second column.
"""

import pandas as pd
from typing import List, Union
import os


def extract_data_from_excel_files(file_paths: List[str],
                                  sheet_name: str,
                                  target_names: List[str]) -> List[List[Union[float, None]]]:
    """
    Extract specified data values from multiple Excel files.

    Args:
        file_paths: List of file paths to Excel (.xlsx) files
        sheet_name: Name of the sheet to read from each file
        target_names: List of names to search for in the first column

    Returns:
        List of lists where each inner list contains the values corresponding to
        target_names for each file, in the same order as target_names.
        Returns None for missing values.

    Raises:
        FileNotFoundError: If any file in file_paths doesn't exist
        ValueError: If sheet_name doesn't exist in any file
    """
    all_results = []

    for file_path in file_paths:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read the specified sheet without header
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

            # Ensure we have at least 2 columns
            if df.shape[1] < 2:
                raise ValueError(f"Sheet '{sheet_name}' in '{file_path}' must have at least 2 columns")

            # Create name-to-value mapping from first two columns
            # Convert first column to string and strip whitespace for robust matching
            name_value_dict = {}
            for idx, row in df.iterrows():
                name = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                value = row.iloc[1] if pd.notna(row.iloc[1]) else None

                # Convert to numeric if possible
                if value is not None:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        # Keep original value if conversion fails
                        pass

                name_value_dict[name] = value

            # Extract values for target names in the specified order
            file_results = []
            for target_name in target_names:
                value = name_value_dict.get(target_name, None)
                file_results.append(value)

            all_results.append(file_results)

        except Exception as e:
            raise ValueError(f"Error processing file '{file_path}', sheet '{sheet_name}': {str(e)}")

    return all_results


# Example usage and test function
def test_extract_data_function():
    """Test function to demonstrate usage."""
    results_dir = '../Optimization_Results/Investment_Model/dgmax100_hrdncost2e5_hrdnshift15_normalprob0.99_rep4_scn214_allgustspeed-5'

    filenames = ['results_network_29BusGB-Kearsley_4_ws_resilience_threshold_inf_20251006_172934.xlsx',
                  # 'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_8.00e3_20251006_182453.xlsx',
                  # 'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_7.00e3_20251006_182710.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_6.00e3_20251006_190016.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_5.00e3_20251007_050230.xlsx',
                  # 'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_4.00e3_20251006_183643.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_3.00e3_20251007_044249.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_2.00e3_20251007_053512.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_1.00e3_20251007_061853.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_8.00e2_20251006_192209.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_6.00e2_20251006_194454.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_4.00e2_20251006_200254.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_2.00e2_20251006_202744.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_1.00e2_20251006_203925.xlsx',
                  'results_network_29BusGB-Kearsley_4_ws_resilience_threshold_0.00e0_20251006_204733.xlsx',
                  ]

    file_paths = [os.path.join(results_dir, filename) for filename in filenames]

    sheet_name = 'Meta'

    target_names = ['resilience_metric_threshold',
                    'ws_exp_total_eens_relprob_dn',
                    'total_investment_cost',
                    'investment_cost_line_hardening',
                    'investment_cost_dg',
                    'investment_cost_ess',
                    ]

    try:
        results = extract_data_from_excel_files(file_paths, sheet_name, target_names)
        print("Extraction successful:")
        for i, file_path in enumerate(file_paths):
            print(f"  {file_path}: {results[i]}")
    except Exception as e:
        print(f"Error: {e}")

    # print("Test function ready. Uncomment example code to test with actual files.")


if __name__ == "__main__":
    test_extract_data_function()