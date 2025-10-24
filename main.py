# =====================
# Version Selection
# =====================
MODEL_VERSION = "old_no_ess"  # Options: "new", "old", "old_no_ess"
# - "new": Binary hardening + DG + ESS (investment_model_two_stage.py)
# - "old": Continuous hardening with ESS operations (investment_model_two_stage_linehrdn_only.py)
# - "old_no_ess": Continuous hardening without ESS (investment_model_two_stage_linehrdn_only_without_ess.py)

# Import all three versions
from core.investment_model_two_stage import InvestmentClass as InvestmentClassNew
from core.investment_model_two_stage_linehrdn_only import InvestmentClassOld
from core.investment_model_two_stage_linehrdn_only_without_ess import InvestmentClassOld as InvestmentClassOldNoESS

# Select which version to use
version_map = {
    "new": InvestmentClassNew,
    "old": InvestmentClassOld,
    "old_no_ess": InvestmentClassOldNoESS
}

if MODEL_VERSION not in version_map:
    raise ValueError(f"Invalid MODEL_VERSION: {MODEL_VERSION}. Must be one of {list(version_map.keys())}")

InvestmentClass = version_map[MODEL_VERSION]

version_descriptions = {
    "new": "NEW VERSION (binary hardening + DG + ESS)",
    "old": "OLD VERSION (continuous hardening with ESS)",
    "old_no_ess": "OLD VERSION (continuous hardening without ESS)"
}

print(f"\n{'='*60}")
print(f"Using: {version_descriptions[MODEL_VERSION]}")
print(f"{'='*60}\n")

# =====================
# Loop with different resilience level thresholds

# Seeds with windstorms passing the Kearsley group:
# --> for 1-ws scenarios, seed=112
# --> for 5-ws scenarios, seed=104
path_ws_scenario_library = "Scenario_Database/Scenarios_Libraries/Representatives_from_Convergence_Based/rep_scn2_interval_from214scn_29BusGB-Kearsley_29GB_seed10000_beta0.050.json"
# path_normal_scenario = "Scenario_Database/Scenarios_for_Old_Two_Stage_Model/Normal_Scenarios/normal_operation_scenario_network_29BusGB-KearsleyGSPGroup_8760hrs.json"

resilience_metric_thresholds = [
    None,
    # 1.75e4,
    # 1.7e4,
    # 1.6e4,
    # 1.5e4,
    # 1.4e4,
    # 1.25e4,
    # 1.3e4,
    # 1.2e4,
    # 1.1e4,
    # 1e4,
    # 9e3,
    # 8e3,
    # 7.5e3,
    # 7e3,
    # 6e3,
    # 5e3,
    # 4e3,
    # 3.8e3,
    # 3.6e3,
    # 3.4e3,
    # 3.2e3,
    # 3e3,
    # 2.5e3,
    # 2e3,
    # 1.5e3,
    # 1.2e3,
    # 1e3,
    # 9e2,
    # 8e2,
    # 7e2,
    # 6e2,
    # 5e2,
    # 4e2,
    # 3e2,
    # 2.8e2,
    # 2.6e2,
    # 2.5e2,
    # 2.4e2,
    # 2.2e2,
    # 2e2,
    # 1e2,
    0
]

additional_notes = """
OLD_VERSION_INVESTMENT_MODEL = True;
bch_hrdn_limits = [0.0, 100.0];
mip_gap = 1e-8,
numeric_focus = 2,
mip_focus = 2,
mip_focus = 2,
method = 1,
heuristics = 0.15,
cuts = 2,
presolve = 2,
"""

# additional_notes = """
# dg_install_capacity_max=10
# hrdn_cost_rate=1e6
# fixed_hrdn_shift=15
# """

for resilience_metric_threshold in resilience_metric_thresholds:
    inv = InvestmentClass()
    model = inv.build_investment_model(path_ws_scenario_library=path_ws_scenario_library,
                                       include_normal_scenario=True,
                                       normal_scenario_prob=0.99,
                                       resilience_metric_threshold=resilience_metric_threshold
                                       )
    results = inv.solve_investment_model(model,
                                         write_lp=False,
                                         write_result=True,
                                         solver_name='gurobi',
                                         result_path=None,
                                         mip_gap=1e-8,
                                         mip_gap_abs=1e3,
                                         time_limit=36000,
                                         numeric_focus=2,
                                         mip_focus=2,
                                         method=1,
                                         heuristics=0.15,
                                         cuts=2,
                                         presolve=2,
                                         additional_notes=additional_notes,
                                         print_gap_callback=True,
                                         gap_print_interval=10,
                                         )


# path_ws_scenario_library_2 = "Scenario_Database/Scenarios_Libraries/Clustered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15_eens_k1.json"
# resilience_metric_thresholds_2 = [
#     None,
#     # 8e2,
#     7e2,
#     6e2,
#     5e2,
#     4e2,
#     3e2,
#     2.8e2,
#     2.6e2,
#     # 2.5e2,
#     2.4e2,
#     2.2e2,
#     # 2e2,
# ]
#
# for resilience_metric_threshold in resilience_metric_thresholds_2:
#     inv = InvestmentClass()
#     model = inv.build_investment_model(path_ws_scenario_library=path_ws_scenario_library_2,
#                                        include_normal_scenario=True,
#                                        normal_scenario_prob=0.999,
#                                        resilience_metric_threshold=resilience_metric_threshold
#                                        )
#     results_2 = inv.solve_investment_model(model, write_lp=False, write_result=True,
#                                          solver_name='gurobi',
#                                          result_path=None,
#                                          mip_gap=1e-9,
#                                          mip_gap_abs=1e4,
#                                          time_limit=10800
#                                          )

# =====================
# Test comment lines to check auto-accept mode
# This is a redundant line for testing purposes
# =====================