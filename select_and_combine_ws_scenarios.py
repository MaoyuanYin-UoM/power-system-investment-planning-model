from scenario_generation_model import *
from visualization import *


# ========================================
# Generate windstorm scenarios and visualize them
# (i.e., generate single scenarios using "generate_ws_scenario" and then )
# ========================================

# network_preset = "29_bus_GB_transmission_network_with_Kearsley_GSP_group"
# windstorm_preset = "windstorm_GB_transmission_network"
#
# # candidate_seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
# # candidate_seeds = [115]
# candidate_seeds = range(101, 161)
#
# generated_files = []
#
# for seed in candidate_seeds:
#     print(f"\nGenerating scenario with seed {seed}...")
#     generate_ws_scenarios(
#         num_ws_prd=[1],  # Single scenario with single windstorm event
#         seed=seed,
#         out_dir="Scenario_Results/Full_Windstorm_Scenarios_Single",
#         network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
#         windstorm_preset="windstorm_GB_transmission_network"
#     )
#
#     # Construct the filename to add to our list
#     # - shorten the network and windstorm preset name
#     if network_preset == "29_bus_GB_transmission_network_with_Kearsley_GSP_group":
#         network_alias = "29BusGB-KearsleyGSPGroup"
#
#     if windstorm_preset == "windstorm_GB_transmission_network":
#         windstorm_alias = "GB"
#
#     file_path = f"Scenario_Results/Full_Windstorm_Scenarios_Single/1_full_scenarios_network_{network_alias}_windstorm_{windstorm_alias}_year_seed_{seed}.json"
#     generated_files.append(file_path)
#
#
# visualize_bch_and_ws_contour(network_name = network_preset,
#                                  windstorm_name = windstorm_preset,
#                                  label_buses = True,
#                                  label_fontsize = 8)
#
# for file_path in generated_files:
#     visualize_all_windstorm_events(file_path)


# --> Manually selected effective scenarios:
# [104, 112, 116, 149, 152, 157]





# # ========================================
# # Combine selected single scenarios into a .json file with extracted windstorm windows using "combine_selected_scenarios"
# # ========================================

# Step 1. Extract the windstorm window from specified full scenarios
network_preset = "29_bus_GB_transmission_network_with_Kearsley_GSP_group"
windstorm_preset = "windstorm_GB_transmission_network"

# Specify selected seeds
selected_seeds = [112, 149, 152, 166, 177, 198]

full_files = []
extracted_files = []

for seed in selected_seeds:
    # Construct file path for full scenario file
    if network_preset == "29_bus_GB_transmission_network_with_Kearsley_GSP_group":
        network_alias = "29BusGB-KearsleyGSPGroup"
    if windstorm_preset == "windstorm_GB_transmission_network":
        windstorm_alias = "GB"

    full_file_path = f"Scenario_Results/Full_Windstorm_Scenarios_Single/1_full_scenarios_network_{network_alias}_windstorm_{windstorm_alias}_year_seed_{seed}.json"
    full_files.append(full_file_path)

    # Create the file path for extracted scenario file
    extracted_file_path = f"Scenario_Results/Extracted_Windstorm_Scenarios_Single/1_ws_scenarios_network_{network_alias}_windstorm_{windstorm_alias}_year_seed_{seed}.json"

    # Extract the windstorm wind from the full scenario
    extract_ws_scenarios(full_file_path, extracted_file_path)
    extracted_files.append(extracted_file_path)


# Step 2. Combine the extracted scenarios
# Create the file path for the combined scenario file
combined_extracted_file = f"Scenario_Results/Extracted_Windstorm_Scenarios/{len(selected_seeds)}_selected_ws_scenarios_network_{network_alias}_windstorm_{windstorm_alias}_year_seeds_{selected_seeds}.json"

combined_file = combine_extracted_scenarios(
    scenario_files=extracted_files,
    out_file=combined_extracted_file
)
