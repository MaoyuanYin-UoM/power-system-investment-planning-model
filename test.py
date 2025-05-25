
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


# ==================================
# Test DC and AC power flow on the matpower 22-bus case
net = make_network('matpower_case22')

ws = make_windstorm('windstorm_1_matpower_case22')
# model = net.build_dc_opf_model()
model = net.build_linearized_ac_opf_model()
net.solve_linearized_ac_opf(model)

