from config import *
from utils import *
from visualization import *
from windstorm import *
from network import *
from investment_model import *

# path_ws_scenarios = "Scenario_Results/Extracted_Scenarios/1_ws_scenarios_GB29-Kearsley_network_seed_104.json"
#
# inv = InvestmentClass()
# model = inv.build_investment_model(path_all_ws_scenarios=path_ws_scenarios,
#                                    resilience_level_threshold=None)
# results = inv.solve_investment_model(model, write_lp=False, write_result=True,
#                                      # result_path='Optimization_Results/Investment_Model/results_selected_variable.csv',
#                                      mip_gap=5e-3,
#                                      time_limit=300
#                                      )

# =====================
# Loop with different windstorm scenarios

# paths = ["Scenario_Results/Extracted_Scenarios/1_ws_scenarios_GB29-Kearsley_network_seed_104.json",
#          "Scenario_Results/Extracted_Scenarios/5_ws_scenarios_GB29-Kearsley_network_seed_101.json",
#          "Scenario_Results/Extracted_Scenarios/5_ws_scenarios_GB29-Kearsley_network_seed_102.json",
#          ]
#
# for path in paths:
#     inv = InvestmentClass()
#     model = inv.build_investment_model(path_all_ws_scenarios=path,
#                                        resilience_level_threshold=None)
#     results = inv.solve_investment_model(model, write_lp=False, write_result=True,
#                                          # result_path='Optimization_Results/Investment_Model/results_selected_variable.csv',
#                                          mip_gap=1e-2,
#                                          time_limit=300
#                                          )

# =====================
# Loop with different resilience level thresholds

# Seeds with windstorms passing the Kearsley group:
# --> for 1-ws scenarios, seed=112
# --> for 5-ws scenarios, seed=104
path_ws_scenarios = "Scenario_Results/Extracted_Scenarios/1_ws_scenarios_GB29-Kearsley_network_seed_112.json"

# resilience_thresholds = [
#     None,
#     1e9,
#     9e8,
#     8e8,
#     7e8,
#     6e8,
#     5e8,
#     4e8,
#     3e8,
#     2.5e8,
#     2.3e8
# ]

resilience_thresholds = [
    None,
    2.5e4,
    2.4e4,
]

for resilience_threshold in resilience_thresholds:
    inv = InvestmentClass()
    model = inv.build_investment_model(path_all_ws_scenarios=path_ws_scenarios,
                                       resilience_level_threshold=resilience_threshold)
    results = inv.solve_investment_model(model, write_lp=False, write_result=True,
                                         # result_path='Optimization_Results/Investment_Model/results_selected_variable.csv',
                                         mip_gap=5e-3,
                                         time_limit=300
                                         )
