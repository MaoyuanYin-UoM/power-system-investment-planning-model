"""
This script processes DFES data for scenario tree construction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
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
    Supports fan tree structure (direct use of 6 DFES scenarios)
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

    # Sheet names in DFES workbook (the format aligns with the file 'dfes_main_workbook_kearsley_gsp_group.xlsm')
    SHEET_MAPPING = {
        'demand': 'Maximum_Demand',
        'dg': 'Distributed_Generation_Capacity',
        'storage': 'Storage_Capacity',
        'ev': 'EV_Uptake'
    }

    # Data structure parameters
    SCENARIO_DATA_STRUCTURE = {
        'scenario_name_col': 2,  # Column C
        'location_name_col': 1,  # Column B
        'first_year_col': 3,  # Column D (2024)
        'header_offset': 1,  # Headers are 1 row below scenario name
        'data_offset': 3,  # Data starts 3 rows below scenario name
        'locations_per_scenario': 24  # Based on Kearsley GSP group
    }

    def __init__(self, file_path: str):
        """Initialize with DFES Excel file path"""
        self.file_path = file_path
        self.scenarios: Dict[str, DFESScenario] = {}
        self._load_data()

    def _load_data(self):
        """Load all DFES data from Excel file"""
        # Process each data type
        demand_data = self._read_sheet_data('demand')
        dg_data = self._read_sheet_data('dg')
        storage_data = self._read_sheet_data('storage')
        ev_data = self._read_sheet_data('ev')

        # Create scenario objects
        for scenario_code, scenario_name in self.SCENARIO_NAMES.items():
            self.scenarios[scenario_code] = DFESScenario(
                name=scenario_name,
                description=f"DFES 2024 {scenario_name} scenario",
                demand_data=demand_data.get(scenario_code, pd.DataFrame()),
                dg_capacity_data=dg_data.get(scenario_code, pd.DataFrame()),
                storage_capacity_data=storage_data.get(scenario_code, pd.DataFrame()),
                ev_uptake_data=ev_data.get(scenario_code, pd.DataFrame())
            )

    def _read_sheet_data(self, data_type: str) -> Dict[str, pd.DataFrame]:
        """Read data from a specific sheet type"""
        sheet_name = self.SHEET_MAPPING.get(data_type)

        try:
            # Read entire sheet
            df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None)

            scenario_data = {}

            # Find scenario sections
            scenario_rows = {}
            for idx, row in df.iterrows():
                cell_value = str(row[self.SCENARIO_DATA_STRUCTURE['scenario_name_col']])
                for code, name in self.SCENARIO_NAMES.items():
                    if name in cell_value:
                        scenario_rows[code] = idx
                        break

            # Extract data for each scenario
            for code, start_row in scenario_rows.items():
                # Extract header row (years)
                header_row = start_row + self.SCENARIO_DATA_STRUCTURE['header_offset']
                years = df.iloc[header_row, self.SCENARIO_DATA_STRUCTURE['first_year_col']:].dropna().astype(int).tolist()

                # Extract data rows
                data_start = start_row + self.SCENARIO_DATA_STRUCTURE['data_offset']
                data_end = data_start + self.SCENARIO_DATA_STRUCTURE['locations_per_scenario']

                # Get location names
                locations = df.iloc[data_start:data_end, self.SCENARIO_DATA_STRUCTURE['location_name_col']].tolist()

                # Get data values
                values = df.iloc[data_start:data_end, self.SCENARIO_DATA_STRUCTURE['first_year_col']:self.SCENARIO_DATA_STRUCTURE['first_year_col'] + len(years)]

                # Create DataFrame with locations as index and years as columns
                scenario_df = pd.DataFrame(values.values, index=locations, columns=years)
                scenario_df = scenario_df.apply(pd.to_numeric, errors='coerce')

                scenario_data[code] = scenario_df

        except Exception as e:
            print(f"Error reading {sheet_name}: {e}")
            return {}

        return scenario_data

    def create_fan_tree_branches(self, stages: List[int],
                                 custom_probabilities: Optional[Dict[str, float]] = None) -> Dict:
        """
        Create fan tree structure using all 6 DFES scenarios directly.

        Args:
            stages: Years for decision stages [2025, 2035, 2050]
            custom_probabilities: Optional custom probabilities for scenarios
                                 If None, uses equal probabilities

        Returns:
            Dictionary with branching structure and probabilities
        """
        branch_names = list(self.SCENARIO_NAMES.keys())

        # Set probabilities
        if custom_probabilities:
            # Validate and normalize custom probabilities
            if set(custom_probabilities.keys()) != set(branch_names):
                missing = set(branch_names) - set(custom_probabilities.keys())
                raise ValueError(f"Missing probabilities for scenarios: {missing}")

            total = sum(custom_probabilities.values())
            base_probs = {k: v/total for k, v in custom_probabilities.items()}
        else:
            # Default equal probabilities
            base_probs = {code: 1.0/6 for code in branch_names}

        # For fan structure, only first transition matters
        transition_probs = {
            'stage_0_to_1': base_probs
        }

        # Add identity transitions for subsequent stages
        for i in range(1, len(stages) - 1):
            stage_key = f"stage_{i}_to_{i + 1}"
            # Each scenario continues to itself with probability 1
            transition_probs[stage_key] = {
                code: {code: 1.0} for code in branch_names
            }

        branches = {
            'stages': stages,
            'branches_per_stage': 6,  # 6 scenarios
            'branch_names': branch_names,
            'dfes_mapping': {code: [code] for code in branch_names},  # Direct mapping
            'transition_probabilities': transition_probs,
            'structure_type': 'fan'
        }

        return branches

    def get_initial_state_2025(self, bus_mapping: Dict[str, List[int]]) -> Dict:
        """
        Get initial system state for 2025 (base year).

        Returns averaged values across all scenarios for 2025.
        """
        initial_data = {
            'demand': {},
            'dg_capacity': {},
            'storage_capacity': {},
            'ev_uptake': {}
        }

        # Average across all scenarios for 2025
        for location in bus_mapping.keys():
            demand_values = []
            dg_values = []
            storage_values = []
            ev_values = []

            for scenario in self.scenarios.values():
                if location in scenario.demand_data.index and 2025 in scenario.demand_data.columns:
                    demand_values.append(scenario.demand_data.loc[location, 2025])

                if location in scenario.dg_capacity_data.index and 2025 in scenario.dg_capacity_data.columns:
                    dg_values.append(scenario.dg_capacity_data.loc[location, 2025])

                if location in scenario.storage_capacity_data.index and 2025 in scenario.storage_capacity_data.columns:
                    storage_values.append(scenario.storage_capacity_data.loc[location, 2025])

                if location in scenario.ev_uptake_data.index and 2025 in scenario.ev_uptake_data.columns:
                    ev_values.append(scenario.ev_uptake_data.loc[location, 2025])

            # Calculate averages
            initial_data['demand'][location] = np.mean(demand_values) if demand_values else 0.0
            initial_data['dg_capacity'][location] = np.mean(dg_values) if dg_values else 0.0
            initial_data['storage_capacity'][location] = np.mean(storage_values) if storage_values else 0.0
            initial_data['ev_uptake'][location] = np.mean(ev_values) if ev_values else 0.0

        return initial_data

    def get_scenario_state_data(self, scenario_code: str, year: int,
                                bus_mapping: Dict[str, List[int]]) -> Dict:
        """
        Get system state data for a specific DFES scenario and year.
        """
        state_data = {
            'demand_factor': {},
            'dg_capacity': {},
            'storage_capacity': {},
            'ev_uptake': {}
        }

        scenario = self.scenarios.get(scenario_code)
        if not scenario:
            raise ValueError(f"Unknown scenario code: {scenario_code}")

        # Extract data for the specified year
        for location, buses in bus_mapping.items():
            if location in scenario.demand_data.index and year in scenario.demand_data.columns:
                # Get 2025 baseline for demand factor calculation
                demand_2025 = scenario.demand_data.loc[location, 2025]
                demand_year = scenario.demand_data.loc[location, year]

                for bus in buses:
                    state_data['demand_factor'][bus] = demand_year / demand_2025 if demand_2025 > 0 else 1.0

                    if location in scenario.dg_capacity_data.index:
                        state_data['dg_capacity'][bus] = scenario.dg_capacity_data.loc[location, year]

                    if location in scenario.storage_capacity_data.index:
                        state_data['storage_capacity'][bus] = scenario.storage_capacity_data.loc[location, year]

                    if location in scenario.ev_uptake_data.index:
                        state_data['ev_uptake'][bus] = scenario.ev_uptake_data.loc[location, year]

        return state_data

    def save_scenarios_to_json(self, output_path: str):
        """Save scenario data to JSON for inspection"""
        output_data = {}

        for code, scenario in self.scenarios.items():
            output_data[code] = {
                'name': scenario.name,
                'description': scenario.description,
                'years': list(scenario.demand_data.columns) if not scenario.demand_data.empty else [],
                'locations': list(scenario.demand_data.index) if not scenario.demand_data.empty else []
            }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)