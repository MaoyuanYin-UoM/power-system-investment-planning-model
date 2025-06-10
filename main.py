

from config import *
from utils import *
from visualization import *
from windstorm import *
from network import *
from investment_model import *

path_ws_scenarios = "Scenario_Results/Extracted_Scenarios/5_ws_scenarios_UK-Kearsley_network_seed_102.json"

inv = InvestmentClass()
model = inv.build_investment_model(path_all_ws_scenarios=path_ws_scenarios)
results = inv.solve_investment_model(model, write_lp=False, write_result=True,
                                     result_path='Optimization_Results/Investment_Model/results_selected_variable.csv'
                                     )