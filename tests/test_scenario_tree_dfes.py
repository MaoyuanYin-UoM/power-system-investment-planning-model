"""
Simple test script for DFES data reading and scenario tree construction
"""

from data_processing.dfes_data_processor import DFESDataProcessor
from data_processing.scenario_tree_builder_dfes import DFESScenarioTreeBuilder
from factories.network_factory import make_network
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
import os


def main():
    # 1. Load DFES data
    print("Loading DFES data...")
    dfes_processor = DFESDataProcessor("../Input_Data/DFES_Projections/dfes_2024_main_workbook_kearsley_gsp_group.xlsx")

    # 2. Load network
    print("\nLoading network...")
    network = make_network("29_bus_GB_transmission_network_with_Kearsley_GSP_group")
    print(f"✓ Loaded network with {len(network.data.net.bus)} buses")

    # 3. Build scenario tree (using existing processor - no duplicate loading)
    print("\nBuilding scenario tree...")
    builder = DFESScenarioTreeBuilder(network, dfes_processor)

    # Test the bus mapping functionality
    print("\nTesting bus mapping...")
    bus_mapping = builder._create_bus_location_mapping()
    print(f"✓ Created bus mapping for {len(bus_mapping)} Kearsley buses")
    print(f"Sample mapping: {dict(list(bus_mapping.items())[:5])}")  # Show first 5 mappings

    # Build the tree
    tree = builder.build_tree_from_dfes(
        stages=[2025, 2030, 2035, 2040, 2045, 2050],  # All 6 stages
        investment_stages=[2030, 2040],  # Investment decision years
        method='fan',
        custom_probabilities={'BV': 0.10, 'CF': 0.10, 'HE': 0.20, 'EE': 0.20, 'HT': 0.20, 'AD': 0.20},
        windstorm_scenarios_per_investment=4  # Will be implemented later
    )
    print(f"✓ Built scenario tree with {len(tree.nodes)} nodes")

    # 4. Test root node state data
    print("\nTesting root node state...")
    root_state = tree.nodes[0].state
    if root_state:
        print(f"✓ Root has demand factors for {len(root_state.demand_factor)} buses")
        print(f"✓ Root has DG capacity for {len(root_state.dg_capacity)} buses")
        print(f"✓ Root has storage capacity for {len(root_state.storage_capacity)} buses")
        print(f"✓ Root has EV uptake for {len(root_state.ev_uptake)} buses")
    else:
        print("⚠ Root node has no state data")

    # 5. Print tree structure
    print("\nScenario Tree Structure:")
    print(f"Stages: {tree.stages}")
    print(f"Number of scenarios: {tree.get_num_scenarios()}")

    # Print each node's basic info
    for node_id, node in tree.nodes.items():
        if node.parent_id is None:
            print(f"\nRoot (Node {node_id}): Year {node.year}")
        else:
            parent_year = tree.nodes[node.parent_id].year if node.parent_id else "N/A"
            scenario_name = getattr(node, 'scenario_name', 'Unknown')
            print(f"Node {node_id}: Year {node.year}, Parent {node.parent_id} (Year {parent_year}), "
                  f"Scenario: {scenario_name}, Prob={node.transition_probability:.2f}")

    # 6. Verify leaf node probabilities sum to 1
    leaf_prob_sum = sum(node.cumulative_probability for node in tree.nodes.values() if node.is_leaf())
    print(f"\nLeaf probabilities sum: {leaf_prob_sum:.4f} (should be 1.0)")

    # 7. Visualization of scenario tree
    # # Create hierarchical visualization
    # visualize_tree_hierarchical(tree, investment_stages=[2035, 2050])
    #
    # # Create simplified visualization with ellipsis
    # visualize_tree_with_ellipsis(tree, investment_stages=[2035, 2050])

    # 8. output scenario tree into an excel file
    # export_scenario_tree_to_excel(tree, "../Scenario_Tree_Results/test_scenario_tree_export.xlsx")
    tree.to_json(filepath="../Scenario_Tree_Results/test_scenario_tree_export.json")

    print("\n✓ Test completed.")


# def visualize_tree_hierarchical(tree, investment_stages=[2035, 2050], figsize=(12, 10)):
#     """
#     Create a hierarchical visualization of the scenario tree similar to Liu et al. 2018.
#
#     Args:
#         tree: ScenarioTree object
#         investment_stages: List of years where investment decisions are enabled
#         figsize: Figure size (width, height)
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     import networkx as nx
#     from matplotlib.patches import FancyBboxPatch
#
#     # Create directed graph
#     G = nx.DiGraph()
#
#     # Add edges
#     for node_id, node in tree.nodes.items():
#         if node.parent_id is not None:
#             G.add_edge(node.parent_id, node_id)
#
#     # Create hierarchical positions
#     pos = {}
#
#     # Get nodes by stage
#     nodes_by_stage = {}
#     for node_id, node in tree.nodes.items():
#         # Find stage index based on node's year
#         stage_idx = None
#         for i, year in enumerate(tree.stages):
#             if node.year == year:
#                 stage_idx = i
#                 break
#
#         if stage_idx is not None:
#             if stage_idx not in nodes_by_stage:
#                 nodes_by_stage[stage_idx] = []
#             nodes_by_stage[stage_idx].append(node_id)
#
#     # Position nodes hierarchically
#     y_spacing = 2.0  # Vertical spacing between stages
#
#     for stage_idx, node_ids in nodes_by_stage.items():
#         y = -stage_idx * y_spacing  # Negative to put root at top
#
#         if stage_idx == 0:
#             # Root node - center at top
#             pos[node_ids[0]] = (0, y)
#         else:
#             # Fan out nodes horizontally
#             n_nodes = len(node_ids)
#             if n_nodes == 1:
#                 x_positions = [0]
#             else:
#                 # Spread nodes evenly
#                 total_width = 8.0  # Adjust for desired spread
#                 x_positions = [-total_width / 2 + i * total_width / (n_nodes - 1) for i in range(n_nodes)]
#
#             # Sort nodes by scenario name for consistent ordering
#             sorted_nodes = sorted(node_ids, key=lambda nid: getattr(tree.nodes[nid], 'scenario_name', str(nid)))
#
#             for i, node_id in enumerate(sorted_nodes):
#                 pos[node_id] = (x_positions[i], y)
#
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
#
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
#                            arrowsize=15, arrowstyle='->', width=1.5, alpha=0.7)
#
#     # Draw nodes with different shapes based on investment capability
#     for node_id, (x, y) in pos.items():
#         node = tree.nodes[node_id]
#
#         # Determine if this is an investment stage
#         is_investment_node = node.year in investment_stages and node_id != 0
#
#         # Node properties
#         if is_investment_node:
#             # Circle for investment nodes
#             node_shape = plt.Circle((x, y), 0.4, color='lightblue',
#                                     edgecolor='darkblue', linewidth=2)
#         else:
#             # Square for non-investment nodes
#             node_shape = FancyBboxPatch((x - 0.4, y - 0.4), 0.8, 0.8,
#                                         boxstyle="round,pad=0.1",
#                                         facecolor='lightgray',
#                                         edgecolor='darkgray',
#                                         linewidth=2)
#
#         ax.add_patch(node_shape)
#
#         # Add node labels
#         if node_id == 0:
#             label = f"Root\n{node.year}"
#         else:
#             scenario_name = getattr(node, 'scenario_name', f'S{node_id}')
#             label = f"{scenario_name}\n{node.year}"
#
#         ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
#
#         # Add probability labels on edges (except root)
#         if node.parent_id is not None and node.transition_probability < 1.0:
#             parent_pos = pos[node.parent_id]
#             # Position probability label slightly offset from edge midpoint
#             prob_x = (x + parent_pos[0]) / 2
#             prob_y = (y + parent_pos[1]) / 2 + 0.2
#             ax.text(prob_x, prob_y, f'{node.transition_probability:.2f}',
#                     ha='center', va='bottom', fontsize=8,
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
#
#     # Add stage labels on the left
#     for stage_idx in nodes_by_stage:
#         y = -stage_idx * y_spacing
#         if stage_idx > 0:
#             ax.text(-6, y, f'Stage {stage_idx}', ha='center', va='center',
#                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
#                                            facecolor='lightyellow', alpha=0.7))
#
#     # Add investment period indicators
#     investment_period = 1
#     for stage_idx in range(1, len(tree.stages)):
#         year = tree.stages[stage_idx]
#         if year in investment_stages:
#             y = -stage_idx * y_spacing
#             ax.axhline(y=y + 0.6, color='red', linestyle='--', alpha=0.3, linewidth=1)
#             ax.text(-7.5, y, f'Investment\nPeriod {investment_period}',
#                     ha='center', va='center', fontsize=9, color='red',
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
#             investment_period += 1
#
#     # Create legend
#     circle_patch = mpatches.Circle((0, 0), 0.1, color='lightblue',
#                                    edgecolor='darkblue', linewidth=2)
#     square_patch = mpatches.FancyBboxPatch((0, 0), 0.2, 0.2,
#                                            boxstyle="round,pad=0.02",
#                                            facecolor='lightgray',
#                                            edgecolor='darkgray',
#                                            linewidth=2)
#
#     ax.legend([circle_patch, square_patch],
#               ['Investment Decision Node', 'Operational Node'],
#               loc='upper right', fontsize=10)
#
#     # Set title
#     ax.set_title('Multi-Stage Stochastic Planning: Scenario Tree Structure\n' +
#                  f'(Fan Tree with {len(tree.get_scenarios())} DFES Scenarios)',
#                  fontsize=14, fontweight='bold', pad=20)
#
#     # Set axis properties
#     ax.set_xlim(-8, 8)
#     ax.set_ylim(-len(tree.stages) * y_spacing - 1, 1)
#     ax.axis('off')
#
#     # Add grid for clarity
#     ax.grid(True, alpha=0.1)
#
#     plt.tight_layout()
#
#     # Save figure
#     import os
#     save_dir = "Images_and_Plots/scenario_tree"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "scenario_tree_hierarchical.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"\n✓ Saved hierarchical tree visualization to {save_path}")
#
#     # Show the plot
#     plt.show()
#
#     return fig, ax
#
#
# def visualize_tree_with_ellipsis(tree, investment_stages=[2035, 2050],
#                                  show_intermediate_years=False, figsize=(12, 8)):
#     """
#     Create a simplified visualization with ellipsis for intermediate years.
#     Similar to Liu et al. 2018 Figure 1.
#
#     Args:
#         tree: ScenarioTree object
#         investment_stages: List of years where investment decisions are enabled
#         show_intermediate_years: Whether to show non-investment years
#         figsize: Figure size
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     import networkx as nx
#     from matplotlib.patches import FancyBboxPatch, Ellipse
#
#     # If not showing intermediate years, only show root and investment stages
#     if not show_intermediate_years:
#         stages_to_show = [0] + [i for i, year in enumerate(tree.stages) if year in investment_stages]
#     else:
#         stages_to_show = list(range(len(tree.stages)))
#
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
#
#     # Position calculations
#     y_spacing = 3.0
#     x_spread = 10.0
#
#     # Get DFES scenarios (assuming fan tree)
#     scenarios = ['BV', 'CF', 'HE', 'EE', 'HT', 'AD']
#     n_scenarios = len(scenarios)
#
#     # Draw scenario branches
#     for i, scenario in enumerate(scenarios):
#         x_pos = -x_spread / 2 + i * x_spread / (n_scenarios - 1)
#
#         # Draw line from root to bottom
#         ax.plot([0, x_pos], [0, -y_spacing], 'gray', linewidth=1.5, alpha=0.7)
#
#         # Continue line down with ellipsis if needed
#         current_y = -y_spacing
#
#         for stage_idx in range(1, len(investment_stages)):
#             # Add ellipsis
#             ellipsis_y = current_y - y_spacing / 2
#             for j in range(3):
#                 ax.plot(x_pos, ellipsis_y - j * 0.2, 'ko', markersize=3)
#
#             # Draw to next investment stage
#             next_y = current_y - y_spacing
#             ax.plot([x_pos, x_pos], [current_y - y_spacing / 3, next_y],
#                     'gray', linewidth=1.5, alpha=0.7)
#             current_y = next_y
#
#     # Draw nodes
#     # Root node
#     root_circle = plt.Circle((0, 0), 0.5, color='lightgreen',
#                              edgecolor='darkgreen', linewidth=2)
#     ax.add_patch(root_circle)
#     ax.text(0, 0, f'Root\n{tree.stages[0]}', ha='center', va='center',
#             fontsize=10, fontweight='bold')
#
#     # Investment nodes for each scenario
#     for stage_idx, year in enumerate(investment_stages):
#         y_pos = -(stage_idx + 1) * y_spacing
#
#         for i, scenario in enumerate(scenarios):
#             x_pos = -x_spread / 2 + i * x_spread / (n_scenarios - 1)
#
#             # Investment node (circle)
#             circle = plt.Circle((x_pos, y_pos), 0.4, color='lightblue',
#                                 edgecolor='darkblue', linewidth=2)
#             ax.add_patch(circle)
#
#             # Label
#             ax.text(x_pos, y_pos, f'{scenario}\n{year}',
#                     ha='center', va='center', fontsize=9)
#
#             # Probability label (only for first stage)
#             if stage_idx == 0:
#                 # Get probability from tree
#                 prob = 1 / 6  # Default equal probability
#                 for node_id, node in tree.nodes.items():
#                     if hasattr(node, 'scenario_name') and node.scenario_name == scenario:
#                         prob = node.transition_probability
#                         break
#
#                 ax.text(x_pos / 2, -y_spacing / 2, f'{prob:.2f}',
#                         ha='center', va='center', fontsize=8,
#                         bbox=dict(boxstyle="round,pad=0.2",
#                                   facecolor='white', alpha=0.8))
#
#     # Add stage labels
#     ax.text(-7, 0, 'Stage 0', ha='center', va='center', fontsize=10,
#             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
#
#     for i, year in enumerate(investment_stages):
#         y_pos = -(i + 1) * y_spacing
#         ax.text(-7, y_pos, f'Stage {i + 1}', ha='center', va='center', fontsize=10,
#                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
#
#         # Investment period label
#         ax.text(-8.5, y_pos, f'Investment\nPeriod {i + 1}',
#                 ha='center', va='center', fontsize=9, color='red')
#
#     # Title
#     ax.set_title('Multi-Stage Stochastic Investment Planning\n' +
#                  'Scenario Tree Structure with Investment Periods',
#                  fontsize=14, fontweight='bold', pad=20)
#
#     # Legend
#     circle_patch = mpatches.Circle((0, 0), 0.1, color='lightblue',
#                                    edgecolor='darkblue', linewidth=2)
#     ax.legend([circle_patch], ['Investment Decision Node'],
#               loc='upper right', fontsize=10)
#
#     # Axis settings
#     ax.set_xlim(-9, 6)
#     ax.set_ylim(-len(investment_stages) * y_spacing - 1, 1)
#     ax.axis('off')
#
#     plt.tight_layout()
#
#     # Save
#     import os
#     save_dir = "Images_and_Plots/scenario_tree"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "scenario_tree_simplified.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"\n✓ Saved simplified tree visualization to {save_path}")
#
#     # Show the plot
#     plt.show()
#
#     return fig, ax


def export_scenario_tree_to_excel(tree, filepath="../Scenario_Tree_Results/test_scenario_tree_export.xlsx"):
    """
    Export scenario tree structure and data to Excel spreadsheet.

    Args:
        tree: ScenarioTree object
        filepath: Output file path
    """

    with ExcelWriter(filepath, engine='openpyxl') as writer:

        # 1. Tree Structure Sheet
        tree_data = []
        for node_id, node in tree.nodes.items():
            tree_data.append({
                'Node_ID': node_id,
                'Stage': node.stage,
                'Year': node.year,
                'Parent_ID': node.parent_id if node.parent_id is not None else 'Root',
                'Children_IDs': str(node.children_ids) if node.children_ids else 'Leaf',
                'Stage_Type': node.stage_type,
                'DFES_Scenario': node.dfes_scenario if node.dfes_scenario else 'N/A',
                'Transition_Prob': node.transition_probability,
                'Cumulative_Prob': node.cumulative_probability,
                'Has_Embedded_Scenarios': len(node.embedded_scenarios) > 0
            })

        df_structure = pd.DataFrame(tree_data)
        df_structure.to_excel(writer, sheet_name='Tree_Structure', index=False)

        # 2. Scenario Paths Sheet
        scenario_data = []
        scenarios = tree.get_scenarios()
        for i, (path, prob) in enumerate(scenarios):
            path_str = " → ".join([f"{n.year}({n.dfes_scenario or 'Root'})" for n in path])
            scenario_data.append({
                'Scenario_ID': i + 1,
                'Path': path_str,
                'Probability': prob,
                'DFES_Branch': path[1].dfes_scenario if len(path) > 1 else 'N/A'
            })

        df_scenarios = pd.DataFrame(scenario_data)
        df_scenarios.to_excel(writer, sheet_name='Scenario_Paths', index=False)

        # 3. State Data Sheets (one for each parameter)
        # Sample 5 buses for display (or all if less than 5)
        sample_buses = sorted(list(tree.buses))[:5]

        # Demand Factors
        demand_data = []
        for node_id, node in tree.nodes.items():
            if node.state:
                row = {
                    'Node_ID': node_id,
                    'Year': node.year,
                    'DFES_Scenario': node.dfes_scenario or 'Root',
                    'Stage_Type': node.stage_type
                }
                for bus in sample_buses:
                    row[f'Bus_{bus}'] = node.state.demand_factor.get(bus, 0.0)
                demand_data.append(row)

        df_demand = pd.DataFrame(demand_data)
        df_demand.to_excel(writer, sheet_name='Demand_Factors_Sample', index=False)

        # DG Capacity
        dg_data = []
        for node_id, node in tree.nodes.items():
            if node.state:
                row = {
                    'Node_ID': node_id,
                    'Year': node.year,
                    'DFES_Scenario': node.dfes_scenario or 'Root',
                    'Stage_Type': node.stage_type
                }
                for bus in sample_buses:
                    row[f'Bus_{bus}_MW'] = node.state.dg_capacity.get(bus, 0.0)
                dg_data.append(row)

        df_dg = pd.DataFrame(dg_data)
        df_dg.to_excel(writer, sheet_name='DG_Capacity_Sample', index=False)

        # BESS Capacity
        bess_data = []
        for node_id, node in tree.nodes.items():
            if node.state:
                row = {
                    'Node_ID': node_id,
                    'Year': node.year,
                    'DFES_Scenario': node.dfes_scenario or 'Root',
                    'Stage_Type': node.stage_type
                }
                for bus in sample_buses:
                    row[f'Bus_{bus}_MWh'] = node.state.storage_capacity.get(bus, 0.0)
                bess_data.append(row)

        df_bess = pd.DataFrame(bess_data)
        df_bess.to_excel(writer, sheet_name='BESS_Capacity_Sample', index=False)

        # 4. Summary Statistics Sheet
        summary_data = {
            'Metric': [
                'Total_Nodes',
                'Number_of_Stages',
                'Number_of_Scenarios',
                'Investment_Stages',
                'Operational_Stages',
                'Investment_Nodes',
                'Operational_Nodes',
                'DFES_Scenarios_Used'
            ],
            'Value': [
                len(tree.nodes),
                tree.num_stages,
                tree.get_num_scenarios(),
                str(tree.investment_stages),
                str([y for y in tree.stages if y not in tree.investment_stages]),
                len([n for n in tree.nodes.values() if n.stage_type == 'investment']),
                len([n for n in tree.nodes.values() if n.stage_type == 'operational']),
                str(list(set(n.dfes_scenario for n in tree.nodes.values() if n.dfes_scenario)))
            ]
        }

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        # 5. Investment Nodes Detail
        inv_nodes_data = []
        for node in tree.nodes.values():
            if node.stage_type == 'investment':
                inv_nodes_data.append({
                    'Node_ID': node.node_id,
                    'Year': node.year,
                    'DFES_Scenario': node.dfes_scenario,
                    'Parent_Year': tree.nodes[node.parent_id].year if node.parent_id is not None else 'N/A',
                    'Cumulative_Probability': node.cumulative_probability,
                    'Embedded_Scenarios_Count': len(node.embedded_scenarios)
                })

        if inv_nodes_data:
            df_inv_nodes = pd.DataFrame(inv_nodes_data)
            df_inv_nodes.to_excel(writer, sheet_name='Investment_Nodes', index=False)

    print(f"\n✓ Scenario tree exported to: {filepath}")
    print(f"  - Contains {len(tree.nodes)} nodes across {tree.num_stages} stages")
    print(f"  - {tree.get_num_scenarios()} scenarios total")
    print(f"  - Investment stages: {tree.investment_stages}")

    return filepath


if __name__ == "__main__":
    main()