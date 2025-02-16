

from config import *
from utils import *
from visualization import *
from windstorm import *
from network_linear import *
from investment_model import *


wcon = WindConfig()
ws = WindClass(wcon)

ncon = NetConfig()
net = NetworkClass(ncon)
#
# icon = InvestmentConfig()
# inv = InvestmentClass(icon)

# print(net.data.net.demand_profile_active)
# print(len(net.data.net.demand_profile_active))
# print(len(net.data.net.demand_profile_active[0]))

# print(inv.piecewise_linearize_fragility(ws, num_pieces=10))



# visualize_fragility_curve(wcon)
# visualize_bch_and_ws_contour()
#
file_path = "Scenario_Results/all_scenarios_month.json"
scenario_number = 1
event_number = 1
visualize_windstorm_event(file_path, scenario_number, event_number)