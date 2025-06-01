
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


net = make_network('UK_transmission_network')
model = net.build_dc_opf_model()
results = net.solve_dc_opf(model)

# net = make_network('UK_transmission_network')
# print("BRANCH LIST (bch):", net.data.net.bch)
# print("NUMBER OF BUSES   :", net.data.net.bus)
# islanded = net.find_islanded_buses()
# print(islanded)
# visualize_bch_and_ws_contour(network_name='UK_transmission_network', windstorm_name='windstorm_UK_transmission_network')