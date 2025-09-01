# This script is to test power network features (e.g., network modelling and DC power flows)

import json

from core.config import NetConfig
from core.network import NetworkClass
from factories.network_factory import make_network


# ------------------------------------------------------------------
# Test method 'build_combined_dc_linearized_ac_opf_model'
# ------------------------------------------------------------------

# net = make_network("29_bus_GB_transmission_network_with_Kearsley_GSP_group")
#
# model = net.build_combined_dc_linearized_ac_opf_model(timesteps=list(range(8760)))
# results = net.solve_combined_dc_linearized_ac_opf(model,
#                                                   solver='gurobi',
#                                                   mip_gap=1e-4,
#                                                   mip_gap_abs=1e3,
#                                                   time_limit=10800,
#                                                   )


# ------------------------------------------------------------------
# Test method 'build_combined_opf_model_under_ws_scenarios'
# ------------------------------------------------------------------

# Test a full scenario library
net = make_network("29_bus_GB_transmission_network_with_Kearsley_GSP_group")

print("Loading windstorm scenario library...")
# ws_library_path = "../Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15.json"
ws_library_path = "../Scenario_Database/Scenarios_Libraries/Clustered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15_eens_k1.json"
with open(ws_library_path, 'r') as f:
    ws_library = json.load(f)

scenarios = ws_library.get("scenarios", {})
scenario_ids = list(scenarios.keys())

print(f"Found {len(scenario_ids)} scenarios in library")

eens_results = {}

for scenario_id in scenario_ids:
    print(f"\n{'=' * 60}")
    print(f"Testing scenario: {scenario_id}")
    print(f"{'=' * 60}")

    scenario_data = scenarios[scenario_id]

    # Check scenario has events
    num_events = len(scenario_data.get("events", []))
    print(f"Number of windstorm events: {num_events}")

    if num_events > 0:
        # Print first event details
        first_event = scenario_data["events"][0]
        print(f"First event: Hour {first_event['bgn_hr']}, Duration {first_event['duration']}h")
        print(f"Max wind speed: {max(first_event['gust_speed']):.1f} m/s")

    # Build the OPF model
    print("\nBuilding OPF model...")
    model = net.build_combined_opf_model_under_ws_scenarios(
        single_ws_scenario=scenario_data,
        scenario_probability=1.0,
    )

    # Check model was created
    print(f"Model created with {len(model.Set_bus)} buses, {len(model.Set_bch)} branches")
    print(f"Windstorm window: {model.num_timesteps} timesteps")

    # Solve the model
    print("\nSolving OPF model...")
    eens = net.solve_combined_opf_model_under_ws_scenarios(
        model=model,
        solver_name="gurobi",
        mip_gap=1e-5,
        time_limit=60
    )

    eens_results[scenario_id] = eens
    print(f"✓ SUCCESS: EENS = {eens:.4f} MWh")


print(f"\n{'=' * 60}")
print("SUMMARY OF RESULTS")
print(f"{'=' * 60}")

for scenario_id, eens in eens_results.items():
    if eens is not None:
        print(f"{scenario_id}: {eens:.4f} MWh")
    else:
        print(f"{scenario_id}: FAILED")

# Check if results are reasonable
successful_results = [e for e in eens_results.values() if e is not None]
if successful_results:
    print(f"\nEENS range: {min(successful_results):.4f} - {max(successful_results):.4f} MWh")
    print(f"Average EENS: {sum(successful_results) / len(successful_results):.4f} MWh")


# Test a single scenario
# net = make_network("29_bus_GB_transmission_network_with_Kearsley_GSP_group")
#
# print("Loading windstorm scenario library...")
# ws_library_path = "../Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15.json"
# with open(ws_library_path, 'r') as f:
#     ws_library = json.load(f)
#
# scenarios = ws_library.get("scenarios", {})
# scenario_ids = list(scenarios.keys())
#
# print(f"Found {len(scenario_ids)} scenarios in library")
#
# eens_results = {}
#
# scenario_id = 'ws_0278'
#
# if scenario_id in scenario_ids:
#     print(f"\n{'=' * 60}")
#     print(f"Testing scenario: {scenario_id}")
#     print(f"{'=' * 60}")
#
#     scenario_data = scenarios[scenario_id]
#
#     # Check scenario has events
#     num_events = len(scenario_data.get("events", []))
#     print(f"Number of windstorm events: {num_events}")
#
#     if num_events > 0:
#         # Print first event details
#         first_event = scenario_data["events"][0]
#         print(f"First event: Hour {first_event['bgn_hr']}, Duration {first_event['duration']}h")
#         print(f"Max wind speed: {max(first_event['gust_speed']):.1f} m/s")
#
#     # Build the OPF model
#     print("\nBuilding OPF model...")
#     model = net.build_combined_opf_model_under_ws_scenarios(
#         single_ws_scenario=scenario_data,
#         scenario_probability=1.0,
#     )
#
#     # Check model was created
#     print(f"Model created with {len(model.Set_bus)} buses, {len(model.Set_bch)} branches")
#     print(f"Windstorm window: {model.num_timesteps} timesteps")
#
#     # Solve the model
#     print("\nSolving OPF model...")
#     eens = net.solve_combined_opf_model_under_ws_scenarios(
#         model=model,
#         solver_name="gurobi",
#         mip_gap=1e-5,
#         time_limit=60
#     )
#
#     print(f"✓ SUCCESS: EENS = {eens:.4f} MWh")
