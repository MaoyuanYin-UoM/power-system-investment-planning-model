# scenario_tree_builder_dfes.py - Updated version with consistent bus ID mapping

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

        # Create bus mapping from Kearsley IDs to integrated network IDs
        self.bus_mapping = self._create_bus_location_mapping()

    def _create_bus_location_mapping(self) -> Dict[int, int]:
        """
        Create mapping from Kearsley bus IDs to integrated network bus IDs.

        Returns:
            Dictionary mapping {kearsley_bus_id: integrated_network_bus_id}
        """
        # Get the offset used in network_factory.py
        # This is max(transmission_bus_ids)
        transmission_buses = [b for b in self.network.data.net.bus
                              if self.network.data.net.bus_level.get(b, 'T') == 'T']
        offset = max(transmission_buses)

        # Create mapping: kearsley_id -> integrated_id
        kearsley_to_integrated = {}

        # Get all distribution buses from integrated network
        distribution_buses = [b for b in self.network.data.net.bus
                              if self.network.data.net.bus_level.get(b, 'T') == 'D']

        # Map each distribution bus back to its original Kearsley ID
        for integrated_bus_id in distribution_buses:
            original_kearsley_id = integrated_bus_id - offset
            kearsley_to_integrated[original_kearsley_id] = integrated_bus_id

        return kearsley_to_integrated

    def build_tree_from_dfes(self,
                             stages: List[int] = [2025, 2030, 2035, 2040, 2045, 2050],
                             investment_stages: List[int] = [2030, 2040],  # NEW PARAMETER
                             method: str = 'fan',
                             transition_model: str = 'markov',
                             custom_probabilities: Optional[Dict[str, float]] = None,
                             windstorm_scenarios_per_investment: int = 4) -> ScenarioTree:  # NEW PARAMETER
        """
        Build scenario tree from DFES data with investment and operational nodes.

        Args:
            stages: All time stages in the tree
            investment_stages: Stages where investment decisions are made
            method: Tree construction method ('fan', 'clustering', etc.)
            transition_model: Probability transition model
            custom_probabilities: Custom branch probabilities
            windstorm_scenarios_per_investment: Number of windstorm scenarios at each investment node
        """
        # Create tree with investment stages info
        tree = ScenarioTree(stages, list(self.buses), investment_stages)

        # Initialize root with weighted average of all scenarios
        if custom_probabilities:
            probs = self.dfes.set_fan_tree_probabilities(custom_probabilities)
        else:
            probs = {code: 1 / 6 for code in ['BV', 'CF', 'HE', 'EE', 'HT', 'AD']}

        self._update_root_node_weighted(tree, stages[0], probs)

        # Build tree structure based on method
        if method == 'fan':
            self._build_fan_tree_with_stages(tree, custom_probabilities,
                                             investment_stages,
                                             windstorm_scenarios_per_investment)
        else:
            # Other methods...
            pass

        return tree

    def _update_root_node_weighted(self, tree: ScenarioTree, base_year: int,
                                   scenario_probabilities: Dict[str, float]):
        """
        Update root node with weighted average of 2025 data across all DFES scenarios.

        Args:
            tree: ScenarioTree object
            base_year: Base year (2025)
            scenario_probabilities: Dictionary of scenario probabilities
        """
        root = tree.nodes[0]

        # Initialize weighted state
        weighted_state = SystemState(
            demand_factor={b: 0.0 for b in self.buses},
            dg_capacity={b: 0.0 for b in self.buses},
            storage_capacity={b: 0.0 for b in self.buses},
            ev_uptake={b: 0.0 for b in self.buses}
        )

        # Calculate weighted average across all scenarios
        for scenario_code, probability in scenario_probabilities.items():
            scenario_data = self.dfes.get_scenario_state_data(
                scenario_code, base_year, self.bus_mapping
            )

            # Accumulate weighted values
            if 'demand_factor' in scenario_data:
                for bus_id, value in scenario_data['demand_factor'].items():
                    weighted_state.demand_factor[bus_id] += value * probability

            if 'dg_capacity' in scenario_data:
                for bus_id, value in scenario_data['dg_capacity'].items():
                    weighted_state.dg_capacity[bus_id] += value * probability

            if 'storage_capacity' in scenario_data:  # Note: storage_capacity from DFES
                for bus_id, value in scenario_data['storage_capacity'].items():
                    weighted_state.storage_capacity[bus_id] += value * probability

            if 'ev_uptake' in scenario_data:
                for bus_id, value in scenario_data['ev_uptake'].items():
                    weighted_state.ev_uptake[bus_id] += value * probability

        # Set missing buses to default values
        for bus in self.buses:
            if bus not in weighted_state.demand_factor:
                weighted_state.demand_factor[bus] = 1.0
            if bus not in weighted_state.dg_capacity:
                weighted_state.dg_capacity[bus] = 0.0
            if bus not in weighted_state.storage_capacity:
                weighted_state.storage_capacity[bus] = 0.0
            if bus not in weighted_state.ev_uptake:
                weighted_state.ev_uptake[bus] = 0.0

        root.state = weighted_state
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

    def _build_fan_tree_with_stages(self, tree: ScenarioTree,
                                    custom_probabilities: Optional[Dict[str, float]],
                                    investment_stages: List[int],
                                    windstorm_scenarios_per_investment: int):
        """Build fan tree with investment and operational stages."""

        # Get scenario codes and probabilities
        scenario_codes = ['BV', 'CF', 'HE', 'EE', 'HT', 'AD']

        if custom_probabilities:
            probs = self.dfes.set_fan_tree_probabilities(custom_probabilities)
        else:
            probs = {code: 1 / 6 for code in scenario_codes}

        # Create a branch for each DFES scenario
        for scenario_code in scenario_codes:
            probability = probs[scenario_code]
            self._create_scenario_path_with_stages(tree, scenario_code, probability,
                                                   investment_stages,
                                                   windstorm_scenarios_per_investment)

    def _create_scenario_path_with_stages(self, tree: ScenarioTree,
                                          scenario_code: str,
                                          initial_probability: float,
                                          investment_stages: List[int],
                                          windstorm_scenarios_per_investment: int):
        """Create a single path with investment and operational nodes."""
        parent_id = 0  # Start from root

        # Create nodes for each stage after root
        for stage_idx in range(1, tree.num_stages):
            year = tree.stages[stage_idx]

            # Get state data from DFES
            state_data = self.dfes.get_scenario_state_data(
                scenario_code, year, self.bus_mapping
            )

            # Create system state
            state = SystemState(
                demand_factor={b: 1.0 for b in self.buses},
                dg_capacity={b: 0.0 for b in self.buses},
                storage_capacity={b: 0.0 for b in self.buses},
                ev_uptake={b: 0.0 for b in self.buses}
            )

            # Update with DFES data
            for attr in ['demand_factor', 'dg_capacity', 'storage_capacity', 'ev_uptake']:
                if attr in state_data:
                    getattr(state, attr).update(state_data[attr])

            # Determine node type
            stage_type = 'investment' if year in investment_stages else 'operational'

            # Transition probability
            trans_prob = initial_probability if stage_idx == 1 else 1.0

            # Add node to tree
            node_id = tree.add_node(parent_id, state, trans_prob,
                                    stage_type=stage_type,
                                    dfes_scenario=scenario_code)

            # If this is an investment node, prepare for embedded scenarios
            if stage_type == 'investment':
                node = tree.nodes[node_id]
                # Embedded scenarios will be added in a separate step
                node.operational_scenarios = []

            # Store scenario name in first-stage node for reference
            if stage_idx == 1:
                tree.nodes[node_id].scenario_name = scenario_code

            # Update parent for next iteration
            parent_id = node_id

    def _build_tree_recursive(self, tree: ScenarioTree,
                              parent_id: int,
                              branch_config: Dict,
                              transition_model: str):
        """Build general tree structure recursively."""
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
        """Create system state for a branch using direct bus ID mapping."""

        # Get DFES scenarios associated with this branch
        if 'dfes_mapping' in branch_config:
            dfes_scenarios = branch_config['dfes_mapping'][branch]
        elif 'interpolation_weights' in branch_config:
            weights = branch_config['interpolation_weights'][branch]
            dfes_scenarios = list(weights.keys())
        else:
            dfes_scenarios = [branch]

        # Initialize new state with default values for all buses
        new_state = SystemState(
            demand_factor={b: 1.0 for b in self.buses},
            dg_capacity={b: 0.0 for b in self.buses},
            storage_capacity={b: 0.0 for b in self.buses},
            ev_uptake={b: 0.0 for b in self.buses}
        )

        # Calculate values based on DFES data
        if len(dfes_scenarios) == 1:
            # Single scenario - get data directly
            scenario = dfes_scenarios[0]
            scenario_data = self.dfes.get_scenario_state_data(
                scenario, to_year, self.bus_mapping
            )

            # Update state with DFES data where available
            for attr in ['demand_factor', 'dg_capacity', 'storage_capacity', 'ev_uptake']:
                if attr in scenario_data:
                    new_state.__dict__[attr].update(scenario_data[attr])
        else:
            # Multiple scenarios - interpolate or average
            if 'interpolation_weights' in branch_config:
                weights = branch_config['interpolation_weights'][branch]
            else:
                weights = {s: 1.0 / len(dfes_scenarios) for s in dfes_scenarios}

            # Weighted average across scenarios
            weighted_data = {
                'demand_factor': {},
                'dg_capacity': {},
                'storage_capacity': {},
                'ev_uptake': {}
            }

            for scenario in dfes_scenarios:
                scenario_data = self.dfes.get_scenario_state_data(
                    scenario, to_year, self.bus_mapping
                )
                weight = weights[scenario]

                # Accumulate weighted values
                for attr in weighted_data:
                    if attr in scenario_data:
                        for bus_id, value in scenario_data[attr].items():
                            if bus_id not in weighted_data[attr]:
                                weighted_data[attr][bus_id] = 0.0
                            weighted_data[attr][bus_id] += weight * value

            # Update state with weighted data
            for attr in weighted_data:
                new_state.__dict__[attr].update(weighted_data[attr])

        # For transmission buses (no DFES data), inherit from parent or use growth factor
        for bus in self.buses:
            if bus not in new_state.demand_factor or new_state.demand_factor[bus] == 1.0:
                # This is likely a transmission bus - apply simple growth
                if parent_state and bus in parent_state.demand_factor:
                    # Apply default growth rate
                    years_elapsed = to_year - from_year
                    growth_rate = 0.01  # 1% annual growth for transmission
                    new_state.demand_factor[bus] = parent_state.demand_factor[bus] * (1 + growth_rate) ** years_elapsed

        return new_state

    def _get_persistent_transitions(self, parent: ScenarioNode,
                                    base_probs: Dict[str, float]) -> Dict[str, float]:
        """Adjust transition probabilities for persistent model."""
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
        """Adjust transition probabilities for converging model."""
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
        """Determine which branch a node belongs to based on its state."""
        if not node.state:
            return None

        # Use average DG capacity across all buses as indicator
        avg_dg = np.mean(list(node.state.dg_capacity.values()))

        if avg_dg < 30:
            return 'Low'
        elif avg_dg < 70:
            return 'Medium'
        else:
            return 'High'