import json
import random
from pathlib import Path


def load_scenario_library(file_path):
    """Load JSON scenario library from file."""
    with open(file_path, 'r') as f:
        scenario_library = json.load(f)
    return scenario_library


def modify_bch_ttr(scenario_library, lower_bound, upper_bound, seed=None):
    """
    Modify all 'bch_ttr' fields in the scenario library by resampling
    each value with new bounds.

    Parameters:
    -----------
    scenario_library : dict
        The loaded scenario library dictionary
    lower_bound : int
        Lower bound for resampling (inclusive)
    upper_bound : int
        Upper bound for resampling (inclusive)
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    bch_ttr_found = False

    # Access scenarios dictionary
    scenarios = scenario_library["scenarios"]

    # Iterate through all scenarios
    for scenario_key, scenario_data in scenarios.items():
        # Iterate through each event in the scenario
        for event_data in scenario_data['events']:
            if 'bch_ttr' in event_data:
                # Get the length of the current bch_ttr list
                num_values = len(event_data['bch_ttr'])
                # Resample with new bounds
                event_data['bch_ttr'] = [random.randint(lower_bound, upper_bound)
                                         for _ in range(num_values)]
                bch_ttr_found = True

    if not bch_ttr_found:
        print("Warning: 'bch_ttr' not found in expected location. Please check the scenario library structure.")

    return scenario_library


def save_scenario_library(scenario_library, file_path, save_copy=False, output_dir=None):
    """
    Save the modified scenario library.

    Parameters:
    -----------
    scenario_library : dict
        The modified scenario library dictionary
    file_path : str or Path
        Original file path
    save_copy : bool
        If True, save as a new file with '_ttr_modified.json' suffix
        If False, overwrite the original file
    output_dir : str or Path, optional
        Directory to save the copy (only used if save_copy=True)
        If None, saves in the same directory as the original file
    """
    file_path = Path(file_path)

    if save_copy:
        # Create new filename with suffix
        new_filename = file_path.stem + '_ttr_modified.json'

        if output_dir is not None:
            output_path = Path(output_dir) / new_filename
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = file_path.parent / new_filename
    else:
        output_path = file_path

    # Save the modified scenario library
    with open(output_path, 'w') as f:
        json.dump(scenario_library, f, indent=2)

    print(f"Modified scenario library saved to: {output_path}")


def main():
    """Main function to modify bch_ttr values in scenario library."""

    # ========== Configuration ==========
    # Path to the scenario library JSON file
    file_path = "../Scenario_Database/Scenarios_Libraries/Representatives_from_Convergence_Based/rep_scn2_interval_from214scn_29BusGB-Kearsley_29GB_seed10000_beta0.050.json"

    # New bounds for bch_ttr resampling
    lower_bound = 24
    upper_bound = 48

    # Save options
    save_copy = True  # Set to False to modify the original file in-place
    output_dir = None  # Set to a directory path to save copy there, or None for same directory

    # Random seed for reproducibility (set to None for random sampling)
    seed = 10000
    # ===================================

    print(f"Loading scenario library from: {file_path}")
    scenario_library = load_scenario_library(file_path)

    print(f"Modifying 'bch_ttr' fields with bounds [{lower_bound}, {upper_bound}]...")
    scenario_library = modify_bch_ttr(scenario_library, lower_bound, upper_bound, seed=seed)

    print("Saving modified scenario library...")
    save_scenario_library(scenario_library, file_path, save_copy=save_copy, output_dir=output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
