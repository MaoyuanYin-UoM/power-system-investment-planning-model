from visualization import *
from data_processing.scenario_generation_for_multi_stage_model import *

# wcon = WindConfig()
# ws = WindClass(wcon)
#
# ncon = NetConfig()
# net = NetworkClass(ncon)
#
# icon = InvestmentConfig()
# inv = InvestmentClass(icon)

# print(net.data.net.demand_profile_active)
# print(len(net.data.net.demand_profile_active))
# print(len(net.data.net.demand_profile_active[0]))

# print(inv.piecewise_linearize_fragility(ws, num_pieces=10))


# visualize_fragility_curve(wcon)
# visualize_bch_and_ws_contour(network_name='matpower_case22', windstorm_name='windstorm_1_matpower_case22')
#
# file_path = "Scenario_Database/all_scenarios_month.json"
# scenario_number = 1
# event_number = 1
# visualize_windstorm_event(file_path, scenario_number, event_number)


# run_full_year_dc_opf()

# visualize_all_windstorm_events()


# # ==================================
# # Test DC and AC power flow on the matpower 22-bus case
# net = make_network('matpower_case22')
#
# ws = make_windstorm('windstorm_1_matpower_case22')
# # model = net.build_dc_opf_model()
# model = net.build_linearized_ac_opf_model()
# net.solve_linearized_ac_opf(model)


# generate_ws_scenarios(network_preset='matpower_case22', windstorm_preset='windstorm_1_matpower_case22')
# extract_ws_scenarios('Scenario_Database/all_full_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json',
#                      'Scenario_Database/all_ws_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json')


# visualize_windstorm_event('Scenario_Database/all_ws_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json', 1, 1)
# visualize_all_windstorm_events(file_path='Scenario_Database/all_ws_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json')


# net = make_network('GB_transmission_network')
# model = net.build_dc_opf_model()
# results = net.solve_dc_opf(model, write_xlsx=True)

# net = make_network('GB_transmission_network')
# print("BRANCH LIST (bch):", net.data.net.bch)
# print("NUMBER OF BUSES   :", net.data.net.bus)
# islanded = net.find_islanded_buses()
# print(islanded)
# visualize_bch_and_ws_contour(network_name='GB_transmission_network', windstorm_name='windstorm_GB_transmission_network')


# net = make_network('Manchester_distribution_network_Kearsley')
# model = net.build_linearized_ac_opf_model()
# results = net.solve_linearized_ac_opf(model, write_xlsx=True)


# net = make_network('GB_transmission_network_with_Kearsley_GSP_group')
# model = net.build_combined_dc_linearized_ac_opf_model()
# results = net.solve_combined_dc_linearized_ac_opf(model, write_xlsx=True)


# net = make_network('29_bus_GB_transmission_network_with_Kearsley_GSP_group')
# model = net.build_combined_dc_linearized_ac_opf_model()
# results = net.solve_combined_dc_linearized_ac_opf(model, write_xlsx=True)


# net = make_network('GB_Transmission_Network_29_Bus')
# model = net.build_dc_opf_model()
# results = net.solve_dc_opf(model, write_xlsx=True)

# visualize_bch_and_ws_contour(network_name='GB_transmission_network',
#                              windstorm_name='windstorm_GB_transmission_network')
# visualize_bch_and_ws_contour(network_name='Manchester_distribution_network_Kearsley',
#                              windstorm_name='windstorm_GB_transmission_network')
# visualize_bch_and_ws_contour(network_name='GB_transmission_network_with_Kearsley_GSP_group',
#                              windstorm_name='windstorm_GB_transmission_network')














# ========================================
# Generate scenarios using the old method
# (i.e., generate multiple scenarios at once without control for each scenario)
# ========================================
# num_ws_prd = [1, 1, 1, 1, 1]
# seed = 102
# generate_ws_scenarios(num_ws_prd=num_ws_prd,
#                       seed=seed,
#                       network_preset='29_bus_GB_transmission_network_with_Kearsley_GSP_group',
#                       windstorm_preset='windstorm_GB_transmission_network')
#
# extract_ws_scenarios(full_file=f'Scenario_Database/Full_Windstorm_Scenarios/{len(num_ws_prd)}_full_scenarios_29_bus_GB_transmission_network_with_Kearsley_GSP_group_windstorm_GB_transmission_network_year_seed_{seed}.json',
#                      out_file=f'Scenario_Database/Extracted_Windstorm_Scenarios/{len(num_ws_prd)}_ws_scenarios_GB29-Kearsley_network_seed_{seed}.json')
#
# for i in range(len(num_ws_prd)):
#     for j in range(num_ws_prd[i]):
#         visualize_windstorm_event(file_path=f'Scenario_Database/Extracted_Windstorm_Scenarios/{len(num_ws_prd)}_ws_scenarios_GB29-Kearsley_network_seed_{seed}.json',
#                                   scenario_number=i, event_number=j)




# ========================================
# Generate windstorm scenarios and visualize them
# (i.e., generate single scenarios using "generate_ws_scenario" and then )
# ========================================

# network_preset = "29_bus_GB_transmission_network_with_Kearsley_GSP_group"
# windstorm_preset = "windstorm_GB_transmission_network"
#
# # candidate_seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
# # candidate_seeds = [115]
# candidate_seeds = range(161, 201)
#
# generated_files = []
#
# for seed in candidate_seeds:
#     print(f"\nGenerating scenario with seed {seed}...")
#     generate_ws_scenarios(
#         num_ws_prd=[1],  # Single scenario with single windstorm event
#         seed=seed,
#         out_dir="Scenario_Database/Full_Windstorm_Scenarios_Single",
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
#     file_path = f"Scenario_Database/Full_Windstorm_Scenarios_Single/1_full_scenarios_network_{network_alias}_windstorm_{windstorm_alias}_year_seed_{seed}.json"
#     generated_files.append(file_path)
#
#
# # visualize_bch_and_ws_contour(network_name = network_preset,
# #                                  windstorm_name = windstorm_preset,
# #                                  label_buses = True,
# #                                  label_fontsize = 8)
#
# for file_path in generated_files:
#     visualize_all_windstorm_events(file_path)


# --> Manually selected effective scenarios:
# [112, 149, 152, 166, 177, 198]


# ========================================
# Combine selected single scenarios into a .json file with extracted windstorm windows using "combine_selected_scenarios"
# ========================================

# network_preset = "29_bus_GB_transmission_network_with_Kearsley_GSP_group"
# windstorm_preset = "windstorm_GB_transmission_network"
#
# # Your selected seeds
# selected_seeds = [112, 149, 152, 166, 177, 198]
#
# # Construct file paths for selected scenarios
# selected_files = []
# for seed in selected_seeds:
#     if network_preset == "29_bus_GB_transmission_network_with_Kearsley_GSP_group":
#         network_alias = "29BusGB-KearsleyGSPGroup"
#     if windstorm_preset == "windstorm_GB_transmission_network":
#         windstorm_alias = "GB"
#
#     file_path = f"Scenario_Database/Full_Windstorm_Scenarios_Single/1_full_scenarios_network_{network_alias}_windstorm_{windstorm_alias}_year_seed_{seed}.json"
#     selected_files.append(file_path)
#
# # Combine the selected scenarios
# print("=== Combining selected scenarios ===")
# combined_file = combine_extracted_scenarios(
#     scenario_files=selected_files,
#     out_file=f"Scenario_Database/Full_Windstorm_Scenarios/{len(selected_seeds)}_selected_effective_scenarios.json"
# )
#
# # Extract windstorm windows for use in the investment model
# print("\n=== Extracting windstorm windows ===")
# extracted_file = "Scenario_Database/Extracted_Windstorm_Scenarios/6_ws_selected_effective_scenarios.json"
# extract_ws_scenarios(combined_file, extracted_file)
#
# print(f"\n=== Extract and combine scenarios completed! ===")
# print(f"Combined full scenarios: {combined_file}")
# print(f"Extracted windstorm scenarios: {extracted_file}")
# print(f"Seeds used: {selected_seeds}")








# =================================================================================

# inv = InvestmentClass()
# model = inv.build_investment_model()
# results = inv.solve_investment_model(model, write_lp=False, write_result=True,
#                                      result_path='Optimization_Results/Investment_Model/results_selected_variable.csv'
#                                      )



# visualize_network_bch(network_name="GB_Transmission_Network_29_Bus")
# net = make_network('29_bus_GB_transmission_network_with_Kearsley_GSP_group')
# ax  = visualize_network_bch('29_bus_GB_transmission_network_with_Kearsley_GSP_group')

# visualize_bch_hrdn_and_fail(results_xlsx="")

# net = make_network('GB_Transmission_Network_29_Bus')
# model = net.build_dc_opf_model()
# net.solve_dc_opf(model, write_xlsx=False)

# net = make_network('Manchester_distribution_network_Kearsley')
# model = net.build_linearized_ac_opf_model()
# results = net.solve_linearized_ac_opf(model, write_xlsx=False)

# net = make_network('29_bus_GB_transmission_network_with_Kearsley_GSP_group')
# model = net.build_combined_dc_linearized_ac_opf_model()
# results = net.solve_combined_dc_linearized_ac_opf(model, write_xlsx=True)



# visualize_network_bch(network_name='29_bus_GB_transmission_network_with_Kearsley_GSP_group')

# visualize_bch_and_ws_contour(network_name='29_bus_GB_transmission_network_with_Kearsley_GSP_group',
#                              windstorm_name='windstorm_GB_transmission_network',
#                              label_buses=True
#                              )

# visualize_bch_and_ws_contour(network_name='GB_transmission_network',
#                              windstorm_name='windstorm_GB_transmission_network',
#                              label_buses=True
#                              )



# import pyomo.environ as pyo
# from pyomo.opt import SolverFactory
#
# # Create a simple test model
# model = pyo.ConcreteModel()
# model.x = pyo.Var(within=pyo.Binary)
# model.y = pyo.Var(within=pyo.Binary)
# model.obj = pyo.Objective(expr=model.x + 2*model.y, sense=pyo.maximize)
# model.con = pyo.Constraint(expr=model.x + model.y <= 1)
#
# # Test solver
# solver_name = 'cbc'  # or 'glpk'
# solver = SolverFactory(solver_name)
# if solver.available():
#     print(f"{solver_name} is available!")
#     results = solver.solve(model)
#     print(f"x = {pyo.value(model.x)}, y = {pyo.value(model.y)}")
# else:
#     print(f"{solver_name} is NOT available")


# visualize_network_bch(network_name="29_bus_GB_transmission_network_with_Kearsley_GSP_group")

# visualize_bch_and_ws_contour(network_name="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
#                              windstorm_name="windstorm_GB_transmission_network",
#                              show_windstorm_contours=False,
#                              label_buses=False,
#                              zoomed_distribution=True,
#                              zoom_border=0.03,
#                              tn_linewidth=2,
#                              dn_linewidth=2.8,
#                              title="Topology for Kearsley GSP group",
#                              )


wcon = make_windstorm("windstorm_GB_transmission_network")
# visualize_fragility_curve(wcon)

visualize_fragility_curve_shift(
    wcon,
    hardening_levels=[20],
    colors=['blue', 'green'],
    show_arrow=True,
    save_path="../Images_and_Plots/fragility_curve_shift.png",
    title="Fragility Curve Shift",
    show_textbox=False,
    title_fontsize=16,
    axis_label_fontsize=14,
    axis_tick_fontsize=12,
    legend_fontsize=12,
    arrow_text_fontsize=12,
)
#
# # 2. Show just one hardening level for clarity
# visualize_fragility_curve_shift(
#     wcon,
#     hardening_levels=[20],  # Just show 20 m/s shift
#     colors=['blue', 'red'],
#     title="Line Hardening: 20 m/s Rightward Shift"
# )

# # 3. Side-by-side comparison
# visualize_fragility_curve_comparison(
#     wcon,
#     hardening_amount=20,
#     save_path="Images_and_Plots/fragility_comparison.png"
# )


# visualize_investment_vs_resilience(excel_file="Optimization_Results/Investment_Model/RM_expected_total_EENS_dn/4_ws_scenario_[112, 152, 166, 198]/data_for_plot.xlsx")

# visualize_investment_pareto_front(
#     excel_file="Optimization_Results/Investment_Model/RM_expected_total_EENS_dn/4_ws_scenario_[112, 152, 166, 198]_updated_(2)/data_for_plot.xlsx",
#     show_extreme_points=False,
#     show_stats=False,
#     show_feasible_region=False,
# )


# =======================================================
# Visualize line hardening at different resilience metric thresholds
# =======================================================
# file_paths = [
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_inf_20250624_115353.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_1.40e4_20250624_102808.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_1.30e4_20250624_181302.xlsx",
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_1.20e4_20250624_182312.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_1.10e4_20250624_000410.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_1.00e4_20250624_002429.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_9.00e3_20250624_004752.xlsx",
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_8.00e3_20250624_011149.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_7.00e3_20250624_024441.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_6.00e3_20250624_030723.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_5.00e3_20250624_040635.xlsx",
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_4.00e3_20250624_043700.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_3.00e3_20250624_135145.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_2.00e3_20250624_142239.xlsx",
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_1.00e3_20250624_150536.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_8.00e2_20250624_155225.xlsx",
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_6.00e2_20250624_162707.xlsx",
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_4.00e2_20250624_164202.xlsx",
#     # "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_3.00e2_20250624_164855.xlsx",
#     "Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_2.80e2_20250624_165422.xlsx",
#
# ]
#
# resilience_metric_values = [
#     "inf",
#     "12.00 (GWh)",
#     "8.00 (GWh)",
#     "4.00 (GWh)",
#     "1.00 (GWh)",
#     "0.60 (GWh)",
#     "0.40 (GWh)",
#     "0.28 (GWh)",
# ]
#
# for i in range(len(file_paths)):
#     visualize_bch_hrdn(results_xlsx=file_paths[i],
#                        cmap_name='viridis',
#                        colorbar_limits=(0, 25),
#                        zoomed_distribution=True,
#                        zoom_border=0.02,
#                        title="Line Hardening Visualization - Resilience Metric = " + resilience_metric_values[i],
#                        title_fontsize = 16,
#                        label_fontsize = 14,
#                        tick_fontsize = 12,
#                        stats_fontsize = 12,
#                        colorbar_label_fontsize = 14,
#                        colorbar_tick_fontsize = 12,
#                        )