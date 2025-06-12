

from config import *
from utils import *
from visualization import *
from windstorm import *
from network import *
from investment_model import *

path_ws_scenarios = "Scenario_Results/Extracted_Scenarios/1_ws_scenarios_GB29-Kearsley_network_seed_104.json"

inv = InvestmentClass()
model = inv.build_investment_model(path_all_ws_scenarios=path_ws_scenarios)
results = inv.solve_investment_model(model, write_lp=False, write_result=True,
                                     result_path='Optimization_Results/Investment_Model/results_selected_variable.csv'
                                     )
#
# for (sc, l, t) in model.Set_slt_lines:
#     if model.fail_applies[sc, l, t].value > 0.5:
#         print(f"Line {l} at t={t}: gust={model.gust_speed[sc,t].value}, "
#               f"shifted={model.shifted_gust_speed[sc,l,t].value}, "
#               f"fail_prob={model.fail_prob[sc,l,t].value}")
#
# for (sc, l, t) in model.Set_slt_lines:
#     if model.repair_applies[sc, l, t].value > 0.5:
#         print(f"Repair: Line {l} at time {t}")
