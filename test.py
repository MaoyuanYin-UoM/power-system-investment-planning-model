import json
from config import *
from utils import *
from visualization import *
from windstorm import *
from network import *
from investment_model import *
from compute_baseline_yearly_cost import *
from scenario_generation_model import *
from network_factory import make_network

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
# file_path = "Scenario_Results/all_scenarios_month.json"
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
# extract_ws_scenarios('Scenario_Results/all_full_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json',
#                      'Scenario_Results/all_ws_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json')


# visualize_windstorm_event('Scenario_Results/all_ws_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json', 1, 1)
# visualize_all_windstorm_events(file_path='Scenario_Results/all_ws_scenarios_matpower_case22_windstorm_1_matpower_case22_year.json')


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


# visualize_bch_and_ws_contour(network_name='GB_transmission_network',
#                              windstorm_name='windstorm_GB_transmission_network')
# visualize_bch_and_ws_contour(network_name='Manchester_distribution_network_Kearsley',
#                              windstorm_name='windstorm_GB_transmission_network')
# visualize_bch_and_ws_contour(network_name='GB_transmission_network_with_Kearsley_GSP_group',
#                              windstorm_name='windstorm_GB_transmission_network')



num_ws_prd = [1]
seed = 112
generate_ws_scenarios(num_ws_prd=num_ws_prd,
                      seed=seed,
                      network_preset='29_bus_GB_transmission_network_with_Kearsley_GSP_group',
                      windstorm_preset='windstorm_GB_transmission_network')

extract_ws_scenarios(full_file=f'Scenario_Results/Full_Scenarios/{len(num_ws_prd)}_full_scenarios_29_bus_GB_transmission_network_with_Kearsley_GSP_group_windstorm_GB_transmission_network_year_seed_{seed}.json',
                     out_file=f'Scenario_Results/Extracted_Scenarios/{len(num_ws_prd)}_ws_scenarios_GB29-Kearsley_network_seed_{seed}.json')

for i in range(len(num_ws_prd)):
    for j in range(num_ws_prd[i]):
        visualize_windstorm_event(file_path=f'Scenario_Results/Extracted_Scenarios/{len(num_ws_prd)}_ws_scenarios_GB29-Kearsley_network_seed_{seed}.json',
                                  scenario_number=i, event_number=j)





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

