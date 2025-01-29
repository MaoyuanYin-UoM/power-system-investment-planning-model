

from config import *
from utils import *
from visualization import *
from windstorm import *
from network_linear import *


wcon = WindConfig()
ws = WindClass(wcon)

ncon = NetConfig()
net = NetworkClass(ncon)

# visualize_fragility_curve(wcon)
visualize_bch_and_ws_contour()

file_path = "Results/all_scenarios.json"
scenario_number = 1
event_number = 1
visualize_windstorm_event(file_path, scenario_number, event_number)