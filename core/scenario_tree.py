"""
This script construct the scenario tree structure for multi-stage stochastic optimisation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json


@dataclass
class SystemState:
    """
    Represents the full system state at a particular node in the scenario tree.
    This includes all uncertain parameters that evolve over time:
    --> i.e., bus specific data at distribution level:
    1) Demand
    2) DG capacity
    3) BESS capacity
    4) EV uptake (i.e., number of EVs)

    """
    # Demand levels at each bus (MW) - indexed by bus number
    demand_factor: Dict[int, float]  # Multiplicative factor on base demand

    # DG capacity at each bus (MW) - indexed by bus number
    dg_capacity: Dict[int, float]

    # BESS capacity at each bus (MWh) - indexed by bus number
    bess_capacity: Dict[int, float]

    # EV charging demand at each bus (MW) - indexed by bus number
    ev_uptake: Dict[int, float]

    # Additional parameters can be added here
    renewable_penetration: float = 0.0  # System-wide renewable percentage

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'demand_factor': self.demand_factor,
            'dg_capacity': self.dg_capacity,
            'bess_capacity': self.bess_capacity,
            'ev_uptake': self.ev_uptake,
            'renewable_penetration': self.renewable_penetration,
        }

    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ScenarioNode:
    """
    Represents a single node in the scenario tree.
    """
    node_id: int
    stage: int  # Time stage (0 = root, 1 = stage 1, etc.)
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    # Probability of reaching this node from its parent
    transition_probability: float = 1.0

    # Cumulative probability from root
    cumulative_probability: float = 1.0

    # System state at this node
    state: SystemState = None

    # Time information
    year: int = 0  # Years from start of planning horizon

    def add_child(self, child_id: int):
        """Add a child node"""
        self.children_ids.append(child_id)

    def is_leaf(self):
        """Check if this is a leaf node"""
        return len(self.children_ids) == 0

    def is_root(self):
        """Check if this is the root node"""
        return self.parent_id is None


class ScenarioTree:
    """
    Multi-stage scenario tree structure for representing uncertainty evolution.
    """

    def __init__(self, stages: List[int], buses: List[int]):
        """
        Initialize scenario tree.

        Args:
            stages: List of years where decisions are made [0, 5, 10, 15, 20]
            buses: List of bus numbers in the network
        """
        self.stages = stages
        self.num_stages = len(stages)
        self.buses = buses
        self.nodes: Dict[int, ScenarioNode] = {}
        self.next_node_id = 0

        # Create root node
        self._create_root_node()

    def _create_root_node(self):
        """Create the root node with base system state"""
        base_state = SystemState(
            demand_factor={b: 1.0 for b in self.buses},
            dg_capacity={b: 0.0 for b in self.buses},
            bess_capacity={b: 0.0 for b in self.buses},
            ev_uptake={b: 0.0 for b in self.buses}
        )

        root = ScenarioNode(
            node_id=0,
            stage=0,
            parent_id=None,
            state=base_state,
            year=self.stages[0]
        )

        self.nodes[0] = root
        self.next_node_id = 1

    def add_node(self, parent_id: int, state: SystemState,
                 transition_probability: float) -> int:
        """
        Add a new node to the scenario tree.

        Returns:
            node_id of the newly created node
        """
        parent = self.nodes[parent_id]
        stage = parent.stage + 1

        if stage >= self.num_stages:
            raise ValueError("Cannot add node beyond final stage")

        node = ScenarioNode(
            node_id=self.next_node_id,
            stage=stage,
            parent_id=parent_id,
            state=state,
            transition_probability=transition_probability,
            cumulative_probability=parent.cumulative_probability * transition_probability,
            year=self.stages[stage]
        )

        self.nodes[self.next_node_id] = node
        parent.add_child(self.next_node_id)

        self.next_node_id += 1
        return node.node_id

    def get_stage_nodes(self, stage: int) -> List[ScenarioNode]:
        """Get all nodes at a particular stage"""
        return [node for node in self.nodes.values() if node.stage == stage]

    def get_leaf_nodes(self) -> List[ScenarioNode]:
        """Get all leaf nodes (scenarios)"""
        return [node for node in self.nodes.values() if node.is_leaf()]

    def get_path_to_root(self, node_id: int) -> List[ScenarioNode]:
        """Get path from a node back to root"""
        path = []
        current = self.nodes[node_id]

        while current is not None:
            path.append(current)
            if current.parent_id is not None:
                current = self.nodes[current.parent_id]
            else:
                break

        return list(reversed(path))

    def get_scenarios(self) -> List[Tuple[List[ScenarioNode], float]]:
        """
        Get all root-to-leaf paths (scenarios) with their probabilities.

        Returns:
            List of (path, probability) tuples
        """
        scenarios = []
        for leaf in self.get_leaf_nodes():
            path = self.get_path_to_root(leaf.node_id)
            scenarios.append((path, leaf.cumulative_probability))
        return scenarios

    def save_to_json(self, filepath: str):
        """Save scenario tree to JSON file"""
        data = {
            'stages': self.stages,
            'buses': self.buses,
            'nodes': {
                str(node_id): {
                    'node_id': node.node_id,
                    'stage': node.stage,
                    'parent_id': node.parent_id,
                    'children_ids': node.children_ids,
                    'transition_probability': node.transition_probability,
                    'cumulative_probability': node.cumulative_probability,
                    'year': node.year,
                    'state': node.state.to_dict() if node.state else None
                }
                for node_id, node in self.nodes.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def is_fan_tree(self) -> bool:
        """
        Check if this is a fan tree (two-stage recourse structure).

        A fan tree has:
        - Only one branching point at the root
        - All other nodes have at most one child
        """
        for node_id, node in self.nodes.items():
            if node_id == 0:  # Root can have multiple children
                continue
            if len(node.children_ids) > 1:
                return False
        return True

    def get_scenario_paths(self) -> Dict[str, List[ScenarioNode]]:
        """
        For fan trees, get each scenario as a separate path.
        Works for general trees too (returns all root-to-leaf paths).

        Returns:
            Dictionary mapping path names to their node sequences
        """
        paths = {}

        if self.is_fan_tree():
            # For fan trees, use scenario names if available
            first_stage_nodes = self.get_stage_nodes(1)

            for i, start_node in enumerate(first_stage_nodes):
                # Get path from root to leaf
                leaf_nodes = [n for n in self.nodes.values()
                              if n.is_leaf() and self._has_ancestor(n.node_id, start_node.node_id)]

                if leaf_nodes:
                    leaf = leaf_nodes[0]  # Should be only one in fan tree
                    path = self.get_path_to_root(leaf.node_id)

                    # Use scenario name if available
                    scenario_name = getattr(start_node, 'scenario_name', f'Scenario_{i + 1}')
                    paths[scenario_name] = path
        else:
            # For general trees, enumerate all paths
            for i, (path, prob) in enumerate(self.get_scenarios()):
                paths[f'Path_{i + 1}'] = path

        return paths

    def _has_ancestor(self, node_id: int, ancestor_id: int) -> bool:
        """Check if a node has a specific ancestor."""
        current = self.nodes[node_id]
        while current.parent_id is not None:
            if current.parent_id == ancestor_id:
                return True
            current = self.nodes[current.parent_id]
        return node_id == ancestor_id  # Check if same node

    def get_tree_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about the tree structure.
        """
        stats = {
            'num_nodes': len(self.nodes),
            'num_stages': self.num_stages,
            'num_scenarios': len(self.get_leaf_nodes()),
            'is_fan_tree': self.is_fan_tree(),
            'stages': self.stages,
            'buses': len(self.buses)
        }

        # Stage-wise statistics
        stage_stats = []
        for stage in range(self.num_stages):
            nodes = self.get_stage_nodes(stage)
            stage_info = {
                'stage': stage,
                'year': self.stages[stage],
                'num_nodes': len(nodes),
                'cumulative_prob_sum': sum(n.cumulative_probability for n in nodes)
            }
            stage_stats.append(stage_info)

        stats['stage_statistics'] = stage_stats

        # Branching factor
        if self.is_fan_tree():
            stats['branching_type'] = 'fan'
            stats['num_branches'] = len(self.get_stage_nodes(1))
        else:
            stats['branching_type'] = 'general'
            # Calculate average branching factor
            total_branches = sum(len(node.children_ids) for node in self.nodes.values()
                                 if node.children_ids)
            parent_nodes = sum(1 for node in self.nodes.values() if node.children_ids)
            stats['avg_branching_factor'] = total_branches / parent_nodes if parent_nodes > 0 else 0

        return stats

    def validate_probabilities(self, tolerance: float = 1e-6) -> Dict[str, any]:
        """
        Validate probability consistency throughout the tree.

        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check stage-wise probability sums
        for stage in range(self.num_stages):
            nodes = self.get_stage_nodes(stage)
            prob_sum = sum(n.cumulative_probability for n in nodes)

            if abs(prob_sum - 1.0) > tolerance:
                results['valid'] = False
                results['errors'].append(
                    f"Stage {stage}: Cumulative probabilities sum to {prob_sum:.6f}, not 1.0"
                )

        # Check parent-child transition probabilities
        for node in self.nodes.values():
            if node.children_ids:
                child_prob_sum = sum(
                    self.nodes[child_id].transition_probability
                    for child_id in node.children_ids
                )
                if abs(child_prob_sum - 1.0) > tolerance:
                    results['valid'] = False
                    results['errors'].append(
                        f"Node {node.node_id}: Children transition probabilities sum to {child_prob_sum:.6f}"
                    )

        # Special check for fan trees
        if self.is_fan_tree():
            # Check that all non-root, non-first-stage transitions are 1.0
            for stage in range(2, self.num_stages):
                nodes = self.get_stage_nodes(stage)
                for node in nodes:
                    if node.transition_probability != 1.0:
                        results['warnings'].append(
                            f"Fan tree node {node.node_id} has transition probability {node.transition_probability}, expected 1.0"
                        )

        return results

    @classmethod
    def load_from_json(cls, filepath: str):
        """Load scenario tree from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tree = cls(stages=data['stages'], buses=data['buses'])
        tree.nodes = {}

        for node_id_str, node_data in data['nodes'].items():
            node = ScenarioNode(
                node_id=node_data['node_id'],
                stage=node_data['stage'],
                parent_id=node_data['parent_id'],
                children_ids=node_data['children_ids'],
                transition_probability=node_data['transition_probability'],
                cumulative_probability=node_data['cumulative_probability'],
                year=node_data['year']
            )

            if node_data['state']:
                node.state = SystemState.from_dict(node_data['state'])

            tree.nodes[int(node_id_str)] = node

        tree.next_node_id = max(tree.nodes.keys()) + 1
        return tree


class ScenarioTreeBuilder:
    """
    Builder class for constructing scenario trees based on uncertainty specifications.
    """

    def __init__(self, network):
        """
        Initialize builder with network data.

        Args:
            network: NetworkClass instance
        """
        self.network = network
        self.buses = network.data.net.bus

    def build_tree(self, stages: List[int], uncertainty_config: Dict) -> ScenarioTree:
        """
        Build a scenario tree based on uncertainty configuration.

        Args:
            stages: List of decision years [0, 5, 10, 15, 20]
            uncertainty_config: Configuration for uncertainty branching
                {
                    'demand_growth': {
                        'scenarios': ['low', 'medium', 'high'],
                        'probabilities': [0.3, 0.5, 0.2],
                        'annual_rates': [0.01, 0.02, 0.03]
                    },
                    'dg_adoption': {
                        'scenarios': ['slow', 'moderate', 'fast'],
                        'probabilities': [0.3, 0.4, 0.3],
                        'capacities_by_year': {...}  # MW per bus
                    },
                    # ... other uncertainties
                }

        Returns:
            Constructed ScenarioTree
        """
        tree = ScenarioTree(stages, self.buses)

        # Build tree recursively from root
        self._build_tree_recursive(tree, 0, uncertainty_config)

        return tree

    def _build_tree_recursive(self, tree: ScenarioTree, parent_id: int,
                              uncertainty_config: Dict):
        """Recursively build the scenario tree"""
        parent = tree.nodes[parent_id]

        if parent.stage >= tree.num_stages - 1:
            return  # Reached final stage

        # Generate child scenarios based on uncertainty realizations
        scenarios = self._generate_scenarios(parent, uncertainty_config)

        for scenario_state, prob in scenarios:
            child_id = tree.add_node(parent_id, scenario_state, prob)
            self._build_tree_recursive(tree, child_id, uncertainty_config)

    def _generate_scenarios(self, parent: ScenarioNode,
                            uncertainty_config: Dict) -> List[Tuple[SystemState, float]]:
        """
        Generate child scenarios for a parent node.

        This is where you implement the logic for how uncertainties branch.
        For simplicity, this example creates a 3-branch tree at each stage.
        """
        scenarios = []

        # Example: Create low/medium/high demand growth scenarios
        demand_config = uncertainty_config.get('demand_growth', {})

        for i, (scenario, prob) in enumerate(zip(
                demand_config.get('scenarios', ['low', 'medium', 'high']),
                demand_config.get('probabilities', [0.3, 0.5, 0.2])
        )):
            # Calculate new system state
            new_state = self._evolve_state(
                parent.state,
                parent.year,
                parent.year + (self.network.data.stages[parent.stage + 1] -
                               self.network.data.stages[parent.stage]),
                scenario,
                uncertainty_config
            )

            scenarios.append((new_state, prob))

        return scenarios

    def _evolve_state(self, current_state: SystemState, from_year: int,
                      to_year: int, scenario: str,
                      uncertainty_config: Dict) -> SystemState:
        """
        Evolve system state from one time period to another under a scenario.

        This is where you implement the actual uncertainty realization logic.
        """
        years_elapsed = to_year - from_year

        # Example evolution logic
        new_state = SystemState(
            demand_factor={},
            dg_capacity={},
            bess_capacity={},
            ev_uptake={}
        )

        # Evolve demand
        demand_growth = uncertainty_config['demand_growth']['annual_rates'][
            uncertainty_config['demand_growth']['scenarios'].index(scenario)
        ]

        for bus in self.buses:
            new_state.demand_factor[bus] = (
                    current_state.demand_factor[bus] *
                    (1 + demand_growth) ** years_elapsed
            )

        # Add DG capacity based on scenario
        # (This is simplified - you'd have more sophisticated logic)
        dg_growth_factor = {'slow': 0.5, 'moderate': 1.0, 'fast': 2.0}
        base_dg_growth = 10.0  # MW per stage

        for bus in self.buses:
            new_state.dg_capacity[bus] = (
                    current_state.dg_capacity[bus] +
                    base_dg_growth * dg_growth_factor.get(scenario, 1.0)
            )

        # Similar logic for BESS and EV...
        new_state.bess_capacity = current_state.bess_capacity.copy()
        new_state.ev_uptake = current_state.ev_uptake.copy()

        return new_state