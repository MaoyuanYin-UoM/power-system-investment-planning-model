from config import *
from utils import *
from visualization import *
from windstorm import *
from network import *
from investment_model import *

# path_ws_scenarios = "Scenario_Results/Extracted_Windstorm_Scenarios/1_ws_scenarios_GB29-Kearsley_network_seed_104.json"
#
# inv = InvestmentClass()
# model = inv.build_investment_model(path_all_ws_scenarios=path_ws_scenarios,
#                                    resilience_metric_threshold=None)
# results = inv.solve_investment_model(model, write_lp=False, write_result=True,
#                                      # result_path='Optimization_Results/Investment_Model/results_selected_variable.csv',
#                                      mip_gap=5e-3,
#                                      time_limit=300
#                                      )

# =====================
# Loop with different windstorm scenarios

# paths = ["Scenario_Results/Extracted_Windstorm_Scenarios/1_ws_scenarios_GB29-Kearsley_network_seed_104.json",
#          "Scenario_Results/Extracted_Windstorm_Scenarios/5_ws_scenarios_GB29-Kearsley_network_seed_101.json",
#          "Scenario_Results/Extracted_Windstorm_Scenarios/5_ws_scenarios_GB29-Kearsley_network_seed_102.json",
#          ]
#
# for path in paths:
#     inv = InvestmentClass()
#     model = inv.build_investment_model(path_all_ws_scenarios=path,
#                                        resilience_metric_threshold=None)
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
path_ws_scenarios = "Scenario_Results/Extracted_Windstorm_Scenarios/4_selected_ws_scenarios_network_29BusGB-KearsleyGSPGroup_windstorm_GB_year_seeds_[112, 152, 166, 198].json"
path_normal_scenario = "Scenario_Results/Normal_Scenarios/normal_operation_scenario_network_29BusGB-KearsleyGSPGroup_8760hrs.json"

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

resilience_metric_thresholds = [
    # None,
    # 1.3e4,
    # 1.2e4,
    # 1.1e4,
    # 1e4,
    # 9e3,
    # 8e3,
    7e3,
    # 6e3,
    # 5e3,
    # 4e3,
    # 3e3,
    # 2e3,
    # 1e3,
    # 5e2,
    # 0
]

for resilience_metric_threshold in resilience_metric_thresholds:
    inv = InvestmentClass()
    model = inv.build_investment_model(path_all_ws_scenarios=path_ws_scenarios,
                                       normal_scenario_prob = 0.99,
                                       resilience_metric_threshold=resilience_metric_threshold
                                       )
    results = inv.solve_investment_model(model, write_lp=False, write_result=True,
                                         # result_path='Optimization_Results/Investment_Model/results_selected_variable.csv',
                                         mip_gap=5e-4,
                                         time_limit=1800
                                         )
