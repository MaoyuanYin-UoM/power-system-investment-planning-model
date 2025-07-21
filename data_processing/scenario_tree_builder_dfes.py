"""
scenario_tree_builder_dfes.py - Enhanced to support both general and fan tree structures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from core.scenario_tree import ScenarioTree, SystemState, ScenarioNode
from data_processing.dfes_data_processor import DFESDataProcessor


class DFESScenarioTreeBuilder:
    """
    Build scenario trees from DFES data with various branching strategies.

    Supports:
    1. General tree structures with branching at multiple stages
    2. Fan tree structures (two-stage recourse) with 6 parallel scenarios
    """

    def __init__(self, network, dfes_processor: DFESDataProcessor):
        """
        Initialize builder with network and DFES data.

        Args:
            network: NetworkClass instance
            dfes_processor: Initialized DFESDataProcessor
        """
        self.network = network
        self.buses = network.data.net.bus
        self.dfes = dfes_processor

        # Map network buses to DFES locations (BSP/PS)
        self.bus_to_location = self._create_bus_location_mapping()

    def _create_bus_location_mapping(self) -> Dict[int, str]:
        """[Previous implementation remains the same]"""
        mapping = {}
        for bus in self.buses:
            if bus in self.network.data.net.bus_dn:
                mapping[bus] = "KEARSLEY 33/11KV"
            else:
                mapping[bus] = "KEARSLEY GSP"

        return mapping

    def build_tree_from_dfes(self,
                             stages: List[int] = [2025, 2035, 2050],
                             method: str = 'clustering',
                             transition_model: str = 'markov',
                             custom_probabilities: Optional[Dict[str, float]] = None) -> ScenarioTree:
        """
        Build scenario tree from DFES data.

        Args:
            stages: Decision years
            method: Branching method ('clustering', 'selection', 'interpolation', 'fan')
            transition_model: How to model transitions (for non-fan structures)
                            - 'markov': Markovian (memoryless) transitions
                            - 'persistent': Higher probability of staying in same pathway
                            - 'converging': Scenarios converge over time
            custom_probabilities: Optional custom probabilities (mainly for fan structure)

        Returns:
            Populated ScenarioTree
        """

        # Create tree structure
        tree = ScenarioTree(stages, self.buses)

        # Get branching configuration
        branch_config = self.dfes.create_scenario_branches(stages, method)

        # Update root node with 2025 data
        self._update_root_node(tree, stages[0])

        # Build tree based on structure type
        if branch_config.get('structure_type') == 'fan':
            # Fan tree structure
            self._build_fan_tree(tree, branch_config, custom_probabilities)
        else:
            # General tree structure
            self._build_tree_recursive(tree, 0, branch_config, transition_model)

        return tree

    def _update_root_node(self, tree: ScenarioTree, base_year: int):
        """[Previous implementation remains the same]"""
        root = tree.nodes[0]

        # Get 2025 baseline data (common across all scenarios)
        initial_data = self.dfes.get_initial_state_2025(self.bus_to_location)

        # Create system state with actual 2025 values
        state = SystemState(
            demand_factor={},
            DG_capacity={},
            BESS_capacity={},
            EV_uptake={}
        )

        # Populate state based on DFES data
        for bus in self.buses:
            location = self.bus_to_location.get(bus, "DEFAULT")

            state.demand_factor[bus] = 1.0
            state.DG_capacity[bus] = initial_data['dg_capacity'].get(location, 0.0)
            state.BESS_capacity[bus] = initial_data['storage_capacity'].get(location, 0.0)
            state.EV_uptake[bus] = initial_data['ev_uptake'].get(location, 0.0)

        root.state = state
        root.year = base_year

    def _build_fan_tree(self, tree: ScenarioTree, branch_config: Dict,
                        custom_probabilities: Optional[Dict[str, float]] = None):
        """
        Build fan tree structure with 6 parallel scenarios.

        Each DFES scenario becomes a separate branch from the root.
        """

        # Get scenario codes
        scenario_codes = branch_config['branch_names']

        # Get probabilities
        if custom_probabilities:
            # Normalize custom probabilities
            probs = self.dfes.set_fan_tree_probabilities(custom_probabilities)
        else:
            # Use default from branch config
            probs = branch_config['transition_probabilities']['stage_0_to_1']

        # Create a branch for each DFES scenario
        for scenario_code in scenario_codes:
            probability = probs[scenario_code]
            self._create_scenario_path(tree, scenario_code, probability)

    def _create_scenario_path(self, tree: ScenarioTree, scenario_code: str,
                              initial_probability: float):
        """
        Create a single path for one DFES scenario in a fan tree.
        """
        parent_id = 0  # Start from root

        # Create nodes for each stage after root
        for stage_idx in range(1, tree.num_stages):
            year = tree.stages[stage_idx]

            # Get state data directly from DFES scenario
            state_data = self.dfes.get_scenario_state_data(
                scenario_code, year, self.bus_to_location
            )

            # Create system state
            state = SystemState(
                demand_factor=state_data['demand_factor'],
                DG_capacity=state_data['dg_capacity'],
                BESS_capacity=state_data['storage_capacity'],
                EV_uptake=state_data['ev_uptake']
            )

            # Transition probability
            # - First transition: use scenario probability
            # - Subsequent transitions: 1.0 (no branching)
            trans_prob = initial_probability if stage_idx == 1 else 1.0

            # Add node to tree
            node_id = tree.add_node(parent_id, state, trans_prob)

            # Store scenario name in first-stage node for reference
            if stage_idx == 1:
                tree.nodes[node_id].scenario_name = scenario_code

            # Update parent for next iteration
            parent_id = node_id

    def _build_tree_recursive(self, tree: ScenarioTree,
                              parent_id: int,
                              branch_config: Dict,
                              transition_model: str):
        """[Previous implementation remains the same - for general trees]"""
        parent = tree.nodes[parent_id]

        if parent.stage >= tree.num_stages - 1:
            return  # Reached final stage

        # Get transition probabilities for this stage
        stage_key = f"stage_{parent.stage}_to_{parent.stage + 1}"

        if transition_model == 'markov':
            transition_probs = branch_config['transition_probabilities'][stage_key]
        elif transition_model == 'persistent':
            transition_probs = self._get_persistent_transitions(
                parent, branch_config['transition_probabilities'][stage_key]
            )
        elif transition_model == 'converging':
            transition_probs = self._get_converging_transitions(
                parent, branch_config['transition_probabilities'][stage_key]
            )

        # Create child nodes for each branch
        for branch in branch_config['branch_names']:
            prob = transition_probs[branch]

            if prob < 0.01:
                continue

            # Create system state for this branch and stage
            next_year = tree.stages[parent.stage + 1]
            state = self._create_branch_state(
                parent.state, parent.year, next_year,
                branch, branch_config
            )

            # Add node to tree
            child_id = tree.add_node(parent_id, state, prob)

            # Recurse
            self._build_tree_recursive(
                tree, child_id, branch_config, transition_model
            )

    def _create_branch_state(self, parent_state: SystemState,
                             from_year: int, to_year: int,
                             branch: str, branch_config: Dict) -> SystemState:
        """[Previous implementation remains the same]"""
        new_state = SystemState(
            demand_factor={},
            DG_capacity={},
            BESS_capacity={},
            EV_uptake={}
        )

        # Get DFES scenarios associated with this branch
        if 'dfes_mapping' in branch_config:
            dfes_scenarios = branch_config['dfes_mapping'][branch]
        elif 'interpolation_weights' in branch_config:
            weights = branch_config['interpolation_weights'][branch]
            dfes_scenarios = list(weights.keys())
        else:
            dfes_scenarios = [branch]

        # Process each bus
        for bus in self.buses:
            location = self.bus_to_location.get(bus, "DEFAULT")

            # Calculate values based on DFES data
            if len(dfes_scenarios) == 1:
                # Single scenario
                scenario = dfes_scenarios[0]
                new_state.demand_factor[bus] = self._get_demand_factor(
                    scenario, location, from_year, to_year
                )
                new_state.DG_capacity[bus] = self._get_dg_capacity(
                    scenario, location, to_year
                )
                new_state.BESS_capacity[bus] = self._get_storage_capacity(
                    scenario, location, to_year
                )
                new_state.EV_uptake[bus] = self._get_ev_uptake(
                    scenario, location, to_year
                )
            else:
                # Multiple scenarios - interpolate or average
                if 'interpolation_weights' in branch_config:
                    weights = branch_config['interpolation_weights'][branch]
                else:
                    weights = {s: 1.0 / len(dfes_scenarios) for s in dfes_scenarios}

                # Weighted average
                new_state.demand_factor[bus] = sum(
                    weights[s] * self._get_demand_factor(s, location, from_year, to_year)
                    for s in dfes_scenarios
                )
                new_state.DG_capacity[bus] = sum(
                    weights[s] * self._get_dg_capacity(s, location, to_year)
                    for s in dfes_scenarios
                )
                new_state.BESS_capacity[bus] = sum(
                    weights[s] * self._get_storage_capacity(s, location, to_year)
                    for s in dfes_scenarios
                )
                new_state.EV_uptake[bus] = sum(
                    weights[s] * self._get_ev_uptake(s, location, to_year)
                    for s in dfes_scenarios
                )

        return new_state

    # All other methods implementation continues below
    def _get_demand_factor(self, scenario: str, location: str,
                           from_year: int, to_year: int) -> float:
        """Get demand growth factor from DFES data."""
        # Updated for DFES 2024 scenarios
        growth_rates = {
            'CF': 0.005,  # Counterfactual - 0.5% annual growth
            'BV': 0.010,  # Best View - 1.0% annual growth
            'HE': 0.012,  # Hydrogen Evolution - 1.2% annual growth
            'HT': 0.015,  # Holistic Transition - 1.5% annual growth
            'EE': 0.020,  # Electric Engagement - 2.0% annual growth
            'AD': 0.025  # Accelerated Decarbonisation - 2.5% annual growth
        }

        annual_rate = growth_rates.get(scenario, 0.015)
        years = to_year - from_year

        return (1 + annual_rate) ** years

    def _get_dg_capacity(self, scenario: str, location: str, year: int) -> float:
        """Get DG capacity from DFES data for specific year."""
        # Updated placeholder values for DFES 2024 scenarios (MW)
        dg_projections = {
            'CF': {2025: 5, 2035: 10, 2050: 20},  # Minimal growth
            'BV': {2025: 5, 2035: 25, 2050: 60},  # Balanced growth
            'HE': {2025: 5, 2035: 20, 2050: 50},  # Moderate growth (hydrogen focus)
            'HT': {2025: 5, 2035: 30, 2050: 70},  # Holistic approach
            'EE': {2025: 5, 2035: 35, 2050: 90},  # High electrification
            'AD': {2025: 5, 2035: 40, 2050: 120}  # Accelerated deployment
        }

        return dg_projections.get(scenario, {}).get(year, 0.0)

    def _get_storage_capacity(self, scenario: str, location: str, year: int) -> float:
        """Get storage capacity from DFES data."""
        # Updated placeholder values for DFES 2024 scenarios (MWh)
        storage_projections = {
            'CF': {2025: 2, 2035: 5, 2050: 10},  # Minimal storage
            'BV': {2025: 2, 2035: 12, 2050: 30},  # Balanced deployment
            'HE': {2025: 2, 2035: 10, 2050: 25},  # Moderate (hydrogen storage)
            'HT': {2025: 2, 2035: 15, 2050: 40},  # Integrated approach
            'EE': {2025: 2, 2035: 20, 2050: 55},  # High for electrification
            'AD': {2025: 2, 2035: 25, 2050: 80}  # Rapid deployment
        }

        return storage_projections.get(scenario, {}).get(year, 0.0)

    def _get_ev_uptake(self, scenario: str, location: str, year: int) -> float:
        """Get EV charging demand from DFES data."""
        # Updated for DFES 2024 scenarios
        # Convert number of EVs to MW demand
        # Assume 7kW average charging per EV, diversity factor of 0.2
        ev_numbers = {
            'CF': {2025: 100, 2035: 500, 2050: 2000},  # Slow uptake
            'BV': {2025: 100, 2035: 1200, 2050: 6000},  # Balanced uptake
            'HE': {2025: 100, 2035: 800, 2050: 4000},  # Moderate (hydrogen cars)
            'HT': {2025: 100, 2035: 1500, 2050: 7000},  # Good uptake
            'EE': {2025: 100, 2035: 2000, 2050: 10000},  # Very high uptake
            'AD': {2025: 100, 2035: 2500, 2050: 12000}  # Aggressive uptake
        }

        num_evs = ev_numbers.get(scenario, {}).get(year, 0)
        return num_evs * 7 * 0.2 / 1000  # Convert to MW

    def _get_persistent_transitions(self, parent: ScenarioNode,
                                    base_probs: Dict[str, float]) -> Dict[str, float]:
        """[Previous implementation]"""
        adjusted_probs = base_probs.copy()

        parent_branch = self._determine_node_branch(parent)

        if parent_branch:
            persistence_factor = 1.5

            for branch in adjusted_probs:
                if branch == parent_branch:
                    adjusted_probs[branch] *= persistence_factor
                else:
                    adjusted_probs[branch] *= 0.75

            # Normalize
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

        return adjusted_probs

    def _get_converging_transitions(self, parent: ScenarioNode,
                                    base_probs: Dict[str, float]) -> Dict[str, float]:
        """[Previous implementation]"""
        adjusted_probs = base_probs.copy()

        convergence_factor = 1 + 0.3 * parent.stage

        if 'Medium' in adjusted_probs:
            adjusted_probs['Medium'] *= convergence_factor

            for branch in ['Low', 'High']:
                if branch in adjusted_probs:
                    adjusted_probs[branch] /= (convergence_factor ** 0.5)

        # Normalize
        total = sum(adjusted_probs.values())
        adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

        return adjusted_probs

    def _determine_node_branch(self, node: ScenarioNode) -> Optional[str]:
        """[Previous implementation]"""
        if not node.state:
            return None

        avg_dg = np.mean(list(node.state.DG_capacity.values()))

        if avg_dg < 30:
            return 'Low'
        elif avg_dg < 70:
            return 'Medium'
        else:
            return 'High'