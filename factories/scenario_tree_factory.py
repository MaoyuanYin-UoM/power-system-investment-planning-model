"""
scenario_tree_factory.py - Factory pattern for creating scenario trees
"""

from typing import Dict, List, Optional
from pathlib import Path
from core.scenario_tree import ScenarioTree, SystemState
from data_processing.dfes_data_processor import DFESDataProcessor
from data_processing.scenario_tree_builder_dfes import DFESScenarioTreeBuilder

# Predefined configurations for common scenario trees
SCENARIO_TREE_CONFIGS = {
    'simple_3stage': {
        'description': 'Simple 3-stage tree with Low/Medium/High branches',
        'stages': [2025, 2035, 2050],
        'branching_method': 'clustering',
        'transition_model': 'markov',
        'base_probabilities': {'Low': 0.25, 'Medium': 0.5, 'High': 0.25}
    },

    'uk_net_zero': {
        'description': 'UK Net Zero pathway aligned tree',
        'stages': [2025, 2035, 2050],
        'branching_method': 'clustering',
        'transition_model': 'persistent',
        'base_probabilities': {'Low': 0.2, 'Medium': 0.6, 'High': 0.2}
    },

    'detailed_5stage': {
        'description': 'Detailed 5-stage tree for long-term planning',
        'stages': [2025, 2030, 2035, 2040, 2050],
        'branching_method': 'interpolation',
        'transition_model': 'converging',
        'base_probabilities': {'Low': 0.3, 'Medium': 0.5, 'High': 0.2}
    },

    'conservative': {
        'description': 'Conservative tree with higher weight on pessimistic scenarios',
        'stages': [2025, 2035, 2050],
        'branching_method': 'selection',
        'transition_model': 'persistent',
        'base_probabilities': {'CF': 0.4, 'BV': 0.4, 'AD': 0.2}
    },

    'optimistic': {
        'description': 'Optimistic tree for ambitious decarbonization planning',
        'stages': [2025, 2035, 2050],
        'branching_method': 'clustering',
        'transition_model': 'markov',
        'base_probabilities': {'Low': 0.15, 'Medium': 0.45, 'High': 0.4}
    },

    'fan_equal': {
        'description': 'Fan tree with all 6 DFES scenarios (equal probabilities)',
        'stages': [2025, 2030, 2035, 2040, 2050],
        'branching_method': 'fan',
        'transition_model': None,
        'base_probabilities': {'BV': 1 / 6, 'CF': 1 / 6, 'HE': 1 / 6, 'EE': 1 / 6, 'HT': 1 / 6, 'AD': 1 / 6}
    },

    'fan_net_zero': {
        'description': 'Fan tree with net-zero weighted probabilities',
        'stages': [2025, 2030, 2035, 2040, 2050],
        'branching_method': 'fan',
        'transition_model': None,
        'base_probabilities': {'BV': 0.20, 'CF': 0.10, 'HE': 0.15, 'EE': 0.20, 'HT': 0.15, 'AD': 0.20}
    }
}


def make_scenario_tree(preset: str,
                       network,
                       dfes_file: Optional[str] = None) -> ScenarioTree:
    """
    Create a scenario tree using a predefined configuration.

    Args:
        preset: Name of the preset configuration
        network: NetworkClass instance
        dfes_file: Path to DFES Excel file (if None, uses default)

    Returns:
        Populated ScenarioTree
    """

    if preset not in SCENARIO_TREE_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available presets: {list(SCENARIO_TREE_CONFIGS.keys())}")

    config = SCENARIO_TREE_CONFIGS[preset]

    # Use default DFES file if not provided
    if dfes_file is None:
        dfes_file = 'Input_Data/dfes_2024_main_workbook_kearsley_gsp_group.xlsx'

    # Check if file exists
    if not Path(dfes_file).exists():
        raise FileNotFoundError(f"DFES file not found: {dfes_file}")

    # Create tree using configuration
    if config['branching_method'] == 'fan':
        # Fan tree with custom probabilities if specified
        return create_tree_from_config(
            network=network,
            dfes_file=dfes_file,
            stages=config['stages'],
            branching_method=config['branching_method'],
            transition_model=None,
            custom_probabilities=config.get('base_probabilities')
        )
    else:
        # General tree structures
        return create_tree_from_config(
            network=network,
            dfes_file=dfes_file,
            stages=config['stages'],
            branching_method=config['branching_method'],
            transition_model=config['transition_model']
        )


def create_tree_from_config(network,
                            dfes_file: str,
                            stages: List[int],
                            branching_method: str = 'clustering',
                            transition_model: Optional[str] = 'markov',
                            custom_probabilities: Optional[Dict[str, float]] = None) -> ScenarioTree:
    """
    Create a scenario tree from configuration parameters.

    Args:
        network: NetworkClass instance
        dfes_file: Path to DFES Excel file
        stages: List of decision years
        branching_method: Method for creating branches
        transition_model: Transition probability model (None for fan trees)
        custom_probabilities: Optional custom branch probabilities

    Returns:
        Populated ScenarioTree
    """

    # Initialize DFES processor
    dfes_processor = DFESDataProcessor(dfes_file)

    # Create tree builder
    builder = DFESScenarioTreeBuilder(network, dfes_processor)

    # Build tree
    if branching_method == 'fan':
        # Fan tree doesn't use transition model
        tree = builder.build_tree_from_dfes(
            stages=stages,
            method=branching_method,
            custom_probabilities=custom_probabilities
        )
    else:
        # General tree structures
        tree = builder.build_tree_from_dfes(
            stages=stages,
            method=branching_method,
            transition_model=transition_model if transition_model else 'markov',
            custom_probabilities=custom_probabilities
        )

    return tree


def create_simple_test_tree(network, stages: List[int] = [2025, 2035, 2050]) -> ScenarioTree:
    """
    Create a simple test scenario tree without DFES data.

    Useful for testing and debugging.
    """

    tree = ScenarioTree(stages, network.data.net.bus)

    # Manually build a simple tree
    # Root node is already created
    root = tree.nodes[0]

    # Update root with simple initial state
    root.state = SystemState(
        demand_factor={b: 1.0 for b in tree.buses},
        DG_capacity={b: 5.0 for b in tree.buses},  # 5 MW initial DG
        BESS_capacity={b: 2.0 for b in tree.buses},  # 2 MWh initial storage
        EV_uptake={b: 0.5 for b in tree.buses}  # 0.5 MW initial EV demand
    )

    # Add first stage branches (2025 -> 2035)
    branches_stage1 = [
        ('Low', 0.3, {'demand_growth': 0.1, 'dg_growth': 10, 'storage_growth': 5}),
        ('Medium', 0.5, {'demand_growth': 0.2, 'dg_growth': 20, 'storage_growth': 10}),
        ('High', 0.2, {'demand_growth': 0.3, 'dg_growth': 40, 'storage_growth': 20})
    ]

    for branch_name, prob, growth_params in branches_stage1:
        state = SystemState(
            demand_factor={b: 1.0 + growth_params['demand_growth'] for b in tree.buses},
            DG_capacity={b: 5.0 + growth_params['dg_growth'] for b in tree.buses},
            BESS_capacity={b: 2.0 + growth_params['storage_growth'] for b in tree.buses},
            EV_uptake={b: 2.0 for b in tree.buses}
        )

        node_id = tree.add_node(0, state, prob)

        # Add second stage branches (2035 -> 2050) if applicable
        if len(stages) > 2:
            branches_stage2 = [
                ('Low', 0.4, {'demand_growth': 0.1, 'dg_growth': 20, 'storage_growth': 10}),
                ('Medium', 0.4, {'demand_growth': 0.2, 'dg_growth': 40, 'storage_growth': 30}),
                ('High', 0.2, {'demand_growth': 0.3, 'dg_growth': 80, 'storage_growth': 50})
            ]

            for branch2_name, prob2, growth2 in branches_stage2:
                state2 = SystemState(
                    demand_factor={b: state.demand_factor[b] * (1 + growth2['demand_growth'])
                                   for b in tree.buses},
                    DG_capacity={b: state.DG_capacity[b] + growth2['dg_growth']
                                 for b in tree.buses},
                    BESS_capacity={b: state.BESS_capacity[b] + growth2['storage_growth']
                                   for b in tree.buses},
                    EV_uptake={b: 5.0 for b in tree.buses}
                )

                tree.add_node(node_id, state2, prob2)

    return tree


def list_available_presets() -> Dict[str, str]:
    """
    List all available scenario tree presets.

    Returns:
        Dictionary of preset names and descriptions
    """
    return {name: config['description']
            for name, config in SCENARIO_TREE_CONFIGS.items()}


def validate_tree(tree: ScenarioTree) -> Dict[str, any]:
    """
    Validate a scenario tree for common issues.

    Returns:
        Dictionary with validation results
    """

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    # Check probability sum at each stage
    for stage in range(tree.num_stages - 1):
        nodes = tree.get_stage_nodes(stage)

        for node in nodes:
            # Sum of children probabilities
            children_prob_sum = sum(
                tree.nodes[child_id].transition_probability
                for child_id in node.children_ids
            )

            if abs(children_prob_sum - 1.0) > 1e-6:
                results['errors'].append(
                    f"Node {node.node_id}: Children probabilities sum to {children_prob_sum:.4f}, not 1.0"
                )
                results['valid'] = False

    # Check leaf probabilities
    leaf_prob_sum = sum(node.cumulative_probability
                        for node in tree.get_leaf_nodes())

    if abs(leaf_prob_sum - 1.0) > 1e-6:
        results['errors'].append(
            f"Leaf node probabilities sum to {leaf_prob_sum:.4f}, not 1.0"
        )
        results['valid'] = False

    # Check for isolated nodes
    for node_id, node in tree.nodes.items():
        if node_id != 0 and node.parent_id not in tree.nodes:
            results['errors'].append(
                f"Node {node_id} has invalid parent {node.parent_id}"
            )
            results['valid'] = False

    # Collect statistics
    results['statistics'] = {
        'num_nodes': len(tree.nodes),
        'num_stages': tree.num_stages,
        'num_scenarios': len(tree.get_leaf_nodes()),
        'stages': tree.stages
    }

    # Warnings for unusual configurations
    if len(tree.get_leaf_nodes()) > 100:
        results['warnings'].append(
            f"Large number of scenarios ({len(tree.get_leaf_nodes())}). "
            "Consider scenario reduction for computational efficiency."
        )

    return results


# Example usage in docstring
"""
Example usage:

    from factories.network_factory import make_network
    from factories.scenario_tree_factory import make_scenario_tree, list_available_presets

    # List available presets
    presets = list_available_presets()
    print("Available scenario tree presets:")
    for name, description in presets.items():
        print(f"  {name}: {description}")

    # Create network
    network = make_network('29_bus_GB_transmission_network_with_Kearsley_GSP_group')

    # Create scenario tree using preset
    tree = make_scenario_tree('uk_net_zero', network)

    # Or create with custom configuration
    tree = create_tree_from_config(
        network=network,
        dfes_file='path/to/dfes_data.xlsx',
        stages=[2025, 2035, 2050],
        branching_method='clustering',
        transition_model='persistent'
    )

    # Validate tree
    validation = validate_tree(tree)
    if validation['valid']:
        print("Scenario tree is valid!")
    else:
        print("Errors found:", validation['errors'])
"""