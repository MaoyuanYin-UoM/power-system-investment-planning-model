"""
This script processes DFES data for scenario tree construction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class DFESScenario:
    """Represents a single DFES scenario"""
    name: str
    description: str
    # Data by year and location
    demand_data: pd.DataFrame  # MW by year and BSP/PS
    dg_capacity_data: pd.DataFrame  # MW by year and BSP/PS
    storage_capacity_data: pd.DataFrame  # MWh by year and BSP/PS
    ev_uptake_data: pd.DataFrame  # Number of EVs by year and BSP/PS


class DFESDataProcessor:
    """
    Process DFES 2024 Excel data for scenario tree construction.
    """

    # DFES 2024 scenario names mapping
    SCENARIO_NAMES = {
        'BV': 'Best View',
        'CF': 'Counterfactual',
        'HE': 'Hydrogen Evolution',
        'EE': 'Electric Engagement',
        'HT': 'Holistic Transition',
        'AD': 'Accelerated Decarbonisation'
    }

    # Sheet names in DFES workbook
    SHEET_MAPPING = {
        'demand': 'Maximum_Demand',
        'dg': 'Distributed_Generation_Capacity',
        'storage': 'Storage_Capacity',
        'ev': 'EV_Uptake'
    }

    # Updated data structure parameters for the Excel file with bus ID columns
    SCENARIO_DATA_STRUCTURE = {
        'scenario_name_row': 0,  # Row 0 contains scenario names
        'header_row': 1,  # Row 1 contains year headers and explanatory text
        'first_data_row': 3,  # Row 3 is where actual data starts (row 2 is empty)
        'location_name_col': 0,  # Column A contains location names
        'bus_name_col': 1,  # Column B contains bus names
        'bus_id_col': 2,  # Column C contains bus IDs
        'columns_per_scenario': 29,  # Each scenario takes 29 columns (1 explanatory + 28 years)
        'years_per_scenario': 28,  # 2024-2051 = 28 years
        'scenario_positions': {  # Starting column for each scenario (shifted by 2 due to new columns)
            'BV': 3,  # Best View (was 1, now 3)
            'CF': 32,  # Counterfactual (was 30, now 32)
            'HE': 61,  # Hydrogen Evolution (was 59, now 61)
            'EE': 90,  # Electric Engagement (was 88, now 90)
            'HT': 119,  # Holistic Transition (was 117, now 119)
            'AD': 148  # Accelerated Decarbonisation (was 146, now 148)
        }
    }

    def __init__(self, file_path: str):
        """
        Initialize DFES data processor.

        Args:
            file_path: Path to DFES Excel workbook
        """
        self.file_path = file_path
        self.scenarios = {}

        # Load all data on initialization
        print(f"Loading DFES data from: {file_path}")
        self._load_all_data()

    def _load_all_data(self):
        """Load all DFES data from the workbook."""
        # Read data for each type
        demand_data = self._read_sheet_data('demand')
        dg_data = self._read_sheet_data('dg')
        storage_data = self._read_sheet_data('storage')
        ev_data = self._read_sheet_data('ev')

        # Create scenario objects
        for scenario_code in self.SCENARIO_NAMES.keys():
            self.scenarios[scenario_code] = DFESScenario(
                name=scenario_code,
                description=self.SCENARIO_NAMES[scenario_code],
                demand_data=demand_data.get(scenario_code, pd.DataFrame()),
                dg_capacity_data=dg_data.get(scenario_code, pd.DataFrame()),
                storage_capacity_data=storage_data.get(scenario_code, pd.DataFrame()),
                ev_uptake_data=ev_data.get(scenario_code, pd.DataFrame())
            )

        print(f"✓ Loaded {len(self.scenarios)} DFES scenarios")

    def _read_sheet_data(self, data_type: str) -> Dict[str, pd.DataFrame]:
        """
        Read and parse data for all scenarios from a specific sheet.
        Now includes bus ID information.

        Args:
            data_type: Type of data ('demand', 'dg', 'storage', 'ev')

        Returns:
            Dictionary mapping scenario codes to DataFrames with bus IDs as index
        """
        sheet_name = self.SHEET_MAPPING.get(data_type)
        if not sheet_name:
            print(f"Unknown data type: {data_type}")
            return {}

        try:
            # Read entire sheet without any header interpretation
            df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None)
            print(f"  Reading {data_type} data from sheet '{sheet_name}'...")

            scenario_data = {}
            structure = self.SCENARIO_DATA_STRUCTURE

            # Extract years from the header row (columns after bus ID column for first scenario)
            header_row_idx = structure['header_row']
            first_scenario_start = structure['scenario_positions']['BV']
            years = []

            for year_offset in range(structure['years_per_scenario']):
                col_idx = first_scenario_start + 1 + year_offset  # +1 to skip explanatory text
                if col_idx < df.shape[1]:
                    year_value = df.iloc[header_row_idx, col_idx]
                    try:
                        year = int(float(year_value))
                        if 2020 <= year <= 2060:  # Reasonable year range
                            years.append(year)
                    except (ValueError, TypeError):
                        print(f"Warning: Could not parse year from column {col_idx}: {year_value}")
                        break

            if not years:
                print(f"Error: No valid years found in {sheet_name}")
                return {}

            # Extract location names, bus names, and bus IDs from the data rows
            locations = []
            bus_names = []
            bus_ids = []
            first_data_row = structure['first_data_row']

            location_col = structure['location_name_col']
            bus_name_col = structure['bus_name_col']
            bus_id_col = structure['bus_id_col']

            for row_idx in range(first_data_row, df.shape[0]):
                # Location name
                location = df.iloc[row_idx, location_col]
                if pd.notna(location) and str(location).strip():
                    locations.append(str(location).strip())
                else:
                    break  # Stop when we hit empty location names

                # Bus name
                bus_name = df.iloc[row_idx, bus_name_col]
                bus_names.append(str(bus_name).strip() if pd.notna(bus_name) else "")

                # Bus ID
                bus_id = df.iloc[row_idx, bus_id_col]
                try:
                    bus_ids.append(int(bus_id) if pd.notna(bus_id) else None)
                except (ValueError, TypeError):
                    bus_ids.append(None)

            if not locations:
                print(f"Error: No locations found in {sheet_name}")
                return {}

            # Create multi-level index with location, bus_name, and bus_id
            index_data = list(zip(locations, bus_names, bus_ids))

            # Extract data for each scenario
            for scenario_code, scenario_col in structure['scenario_positions'].items():
                try:
                    # Data starts at scenario_col + 1 (skip explanatory text)
                    data_start_col = scenario_col + 1

                    # Extract data for this scenario
                    scenario_matrix = []
                    for loc_idx in range(len(locations)):
                        row_idx = first_data_row + loc_idx
                        row_data = []

                        for year_idx in range(len(years)):
                            col_idx = data_start_col + year_idx
                            if row_idx < df.shape[0] and col_idx < df.shape[1]:
                                value = df.iloc[row_idx, col_idx]
                                try:
                                    numeric_value = float(value) if pd.notna(value) else np.nan
                                    row_data.append(numeric_value)
                                except (ValueError, TypeError):
                                    row_data.append(np.nan)
                            else:
                                row_data.append(np.nan)

                        scenario_matrix.append(row_data)

                    # Create DataFrame with multi-level index
                    if scenario_matrix:
                        # Create DataFrame with bus IDs as primary index for easier mapping
                        valid_indices = [(i, bus_id) for i, (loc, bus_name, bus_id) in enumerate(index_data)
                                         if bus_id is not None]

                        if valid_indices:
                            # Create a clean DataFrame indexed by bus_id
                            clean_data = []
                            clean_bus_ids = []

                            for i, bus_id in valid_indices:
                                clean_data.append(scenario_matrix[i])
                                clean_bus_ids.append(bus_id)

                            scenario_df = pd.DataFrame(
                                data=clean_data,
                                index=clean_bus_ids,
                                columns=years
                            )
                            scenario_df.index.name = 'bus_id'
                            scenario_data[scenario_code] = scenario_df

                            # Validation
                            non_nan_count = scenario_df.count().sum()
                            total_values = scenario_df.size
                            print(
                                f"  ✓ {scenario_code}: {scenario_df.shape[0]} buses × {scenario_df.shape[1]} years ({non_nan_count}/{total_values} valid values)")

                except Exception as e:
                    print(f"Error processing scenario {scenario_code} in {sheet_name}: {e}")
                    continue

            print(f"  ✓ Completed {data_type} data processing for {len(scenario_data)} scenarios")
            return scenario_data

        except Exception as e:
            print(f"Error reading {sheet_name}: {e}")
            return {}

    def get_scenario_state_data(self, scenario_code: str, year: int,
                                bus_mapping: Dict[int, int]) -> Dict:
        """
        Get system state data for a specific DFES scenario and year.
        Now uses direct bus ID mapping instead of location strings.

        Args:
            scenario_code: DFES scenario code ('BV', 'CF', etc.)
            year: Year to extract data for
            bus_mapping: Mapping from Kearsley bus IDs to integrated network bus IDs

        Returns:
            Dictionary with state data mapped to network bus IDs
        """
        if scenario_code not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_code}")

        scenario = self.scenarios[scenario_code]
        state_data = {}

        # Get demand factor (relative to base year)
        if not scenario.demand_data.empty and year in scenario.demand_data.columns:
            base_year = min(scenario.demand_data.columns)
            demand_factor_data = {}

            for kearsley_bus_id in scenario.demand_data.index:
                if kearsley_bus_id in bus_mapping:
                    network_bus_id = bus_mapping[kearsley_bus_id]
                    current_demand = scenario.demand_data.loc[kearsley_bus_id, year]
                    base_demand = scenario.demand_data.loc[kearsley_bus_id, base_year]

                    if pd.notna(current_demand) and pd.notna(base_demand) and base_demand > 0:
                        demand_factor = current_demand / base_demand
                    else:
                        demand_factor = 1.0

                    demand_factor_data[network_bus_id] = demand_factor

            state_data['demand_factor'] = demand_factor_data

        # Get DG capacity data
        if not scenario.dg_capacity_data.empty and year in scenario.dg_capacity_data.columns:
            dg_data = {}
            for kearsley_bus_id in scenario.dg_capacity_data.index:
                if kearsley_bus_id in bus_mapping:
                    network_bus_id = bus_mapping[kearsley_bus_id]
                    capacity = scenario.dg_capacity_data.loc[kearsley_bus_id, year]
                    dg_data[network_bus_id] = float(capacity) if pd.notna(capacity) else 0.0
            state_data['dg_capacity'] = dg_data

        # Get storage capacity data
        if not scenario.storage_capacity_data.empty and year in scenario.storage_capacity_data.columns:
            storage_data = {}
            for kearsley_bus_id in scenario.storage_capacity_data.index:
                if kearsley_bus_id in bus_mapping:
                    network_bus_id = bus_mapping[kearsley_bus_id]
                    capacity = scenario.storage_capacity_data.loc[kearsley_bus_id, year]
                    storage_data[network_bus_id] = float(capacity) if pd.notna(capacity) else 0.0
            state_data['storage_capacity'] = storage_data

        # Get EV uptake data
        if not scenario.ev_uptake_data.empty and year in scenario.ev_uptake_data.columns:
            ev_data = {}
            for kearsley_bus_id in scenario.ev_uptake_data.index:
                if kearsley_bus_id in bus_mapping:
                    network_bus_id = bus_mapping[kearsley_bus_id]
                    uptake = scenario.ev_uptake_data.loc[kearsley_bus_id, year]
                    ev_data[network_bus_id] = float(uptake) if pd.notna(uptake) else 0.0
            state_data['ev_uptake'] = ev_data

        return state_data

    def get_initial_state_2025(self, bus_mapping: Dict[int, int]) -> Dict:
        """
        Get initial state data for 2025 (base year) across all scenarios.
        Uses the first available scenario (BV) for baseline.
        """
        return self.get_scenario_state_data('BV', 2025, bus_mapping)

    def create_scenario_branches(self, stages: List[int], method: str = 'clustering') -> Dict:
        """
        Create scenario branches for the tree based on DFES data.

        This is a placeholder that needs to be implemented based on the specific
        branching strategy (clustering, selection, interpolation, fan).

        For now, returns a simple fan structure with all 6 DFES scenarios.
        """
        if method == 'fan':
            # Fan tree structure - all scenarios branch from root
            return {
                'structure_type': 'fan',
                'branch_names': list(self.SCENARIO_NAMES.keys()),
                'transition_probabilities': {
                    'stage_0_to_1': {code: 1.0 / 6 for code in self.SCENARIO_NAMES.keys()}
                }
            }
        else:
            # Placeholder for other methods
            raise NotImplementedError(f"Branching method '{method}' not yet implemented")

    def set_fan_tree_probabilities(self, custom_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Set and normalize custom probabilities for fan tree structure.

        Args:
            custom_probabilities: Dictionary of scenario code to probability

        Returns:
            Normalized probabilities
        """
        # Ensure all scenarios have probabilities
        probs = {}
        for code in self.SCENARIO_NAMES.keys():
            probs[code] = custom_probabilities.get(code, 0.0)

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            # Equal probabilities if none provided
            probs = {k: 1.0 / len(self.SCENARIO_NAMES) for k in self.SCENARIO_NAMES.keys()}

        return probs
