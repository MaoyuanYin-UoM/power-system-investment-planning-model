# This script contains functions to generate baseline scenarios and windstorm scenarios

import numpy as np
import json
import os
from utils import set_random_seed
from factories.network_factory import make_network
from factories.windstorm_factory import make_windstorm


def generate_ws_scenarios(num_ws_prd, seed=None, out_dir="Scenario_Database/Full_Windstorm_Scenarios",
                          network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
                          windstorm_preset="windstorm_GB_transmission_network"):
    """
    Generate and save full windstorm scenarios.

    Note: 1) This function generates multiple scenarios at once into a .json file, without control to each single
             scenario.
          2) Please use 'generate_single_ws_scenario' and then 'combine_selected_scenarios' if you want manually select
          each single scenario.
    """
    # todo: to be updated to incorporate the new gust speed, radius, translation speed model
    # Optional reproducibility
    if seed is not None:
        set_random_seed(seed)

    # Initialize
    net = make_network(network_preset)
    ws = make_windstorm(windstorm_preset)

    if num_ws_prd is not None:
        ws.data.MC.num_prds = len(num_ws_prd)
        ws.MC.WS.num_ws_prd = list(num_ws_prd)
        ws.MC.WS.num_ws_total = sum(num_ws_prd)

        # re-draw any arrays that depend on the total number of events
        max_v, min_v = ws._get_lim_max_v_ws(), ws._get_lim_min_v_ws()
        lim_lng_ws = ws._get_lim_lng_ws()
        ws.MC.WS.lim_v_ws_all = [
            [np.random.uniform(*max_v), np.random.uniform(*min_v)]
            for _ in range(ws.MC.WS.num_ws_total)
        ]
        ws.MC.WS.lng = [
            np.random.randint(*lim_lng_ws)
            for _ in range(ws.MC.WS.num_ws_total)
        ]

    ws.crt_bgn_hr()
    ws.init_ws_path0()

    if not num_ws_prd:
        num_ws_prd = ws.MC.WS.num_ws_prd

    num_bch = len(net.data.net.bch)
    num_hrs_prd = ws._get_num_hrs_prd()

    all_results = []

    # loop over each simulation
    global_evt = 0
    for prd, n_evt in enumerate(num_ws_prd):
        net.set_gis_data()
        bch_gis_bgn = net._get_bch_gis_bgn()
        bch_gis_end = net._get_bch_gis_end()

        sim = {"simulation_id": prd+1, "events": []}

        bch_rand_nums = np.random.rand(num_bch, num_hrs_prd)
        flgs_impacted_bch = np.zeros((num_bch, num_hrs_prd), dtype=int)

        start_lon, start_lat, end_lon, end_lat = ws.init_ws_path(num_ws_prd[prd])

        # loop over each windstorm events in a simulation
        for i in range(n_evt):
            ts = int(ws.MC.WS.bgn_hrs_ws_prd[prd][i])
            lng_ws = ws._get_lng_ws()[global_evt]
            path_ws = ws.crt_ws_path(start_lon[i], start_lat[i], end_lon[i], end_lat[i], lng_ws)
            radius_ws = ws.crt_ws_radius(lng_ws)
            v_ws = ws.crt_ws_v(ws._get_lim_v_ws_all()[global_evt], lng_ws)

            for t in range(lng_ws):
                epicentre = path_ws[t]
                flgs_impacted_bch[:, ts+t] = np.array(
                    ws.compare_circle(epicentre, radius_ws[t], bch_gis_bgn, bch_gis_end, num_bch), dtype=int
                )

            sim["events"].append({
                "event_id": i+1,
                "bgn_hr": ts,
                "duration": lng_ws,
                "epicentre": path_ws,
                "radius": radius_ws.tolist(),
                "gust_speed": v_ws
            })

            global_evt += 1

        ttr_min, ttr_max = ws.data.WS.event.ttr
        bch_ttr = np.random.randint(ttr_min, ttr_max, size=num_bch)

        sim["bch_rand_nums"] = bch_rand_nums.tolist()
        sim["flgs_impacted_bch"] = flgs_impacted_bch.tolist()
        sim["bch_ttr"] = bch_ttr.tolist()

        all_results.append(sim)


    # assemble metadata + scenarios into one dict
    output = {
        "metadata": {
            "scenario_type": "windstorm",
            "seed":seed,
            "network_preset": network_preset,
            "windstorm_preset": windstorm_preset,
            "number_of_ws_simulations": len(num_ws_prd),
            "period_type": ws.data.MC.lng_prd
        },

        "scenarios": all_results
    }


    # write out
    os.makedirs(out_dir, exist_ok=True)
    # to avoid the file name length exceeding Windows's limit, shortened aliases are used
    if network_preset == "29_bus_GB_transmission_network_with_Kearsley_GSP_group":
        network_alias = "29BusGB-KearsleyGSPGroup"

    if windstorm_preset == "windstorm_GB_transmission_network":
        windstorm_alias = "GB"

    file_name = (
        f"{len(num_ws_prd)}_full_scenarios_"
        f"network_{network_alias}_windstorm_{windstorm_alias}_{ws.data.MC.lng_prd}_seed_{seed}.json"
    )
    path = os.path.join(out_dir, file_name)
    with open(path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Windstorm scenarios saved to {path}")


def combine_extracted_scenarios(scenario_files, out_file, scenario_probabilities=None):
    """
    Combine manually selected extracted windstorm scenarios into one multi-scenario file.
    """
    all_ws_scenarios = []
    combined_metadata = {}
    all_seeds = []  # Track all seeds

    for i, file in enumerate(scenario_files):
        with open(file, "r") as f:
            data = json.load(f)

        # Extract metadata and scenarios
        if "metadata" in data:
            file_metadata = data["metadata"]
            scenarios = data.get("ws_scenarios", data.get("scenarios", []))

            # Collect seed information
            if "seed" in file_metadata:
                seed = file_metadata["seed"]
                # Handle both single seed and list of seeds
                if isinstance(seed, list):
                    all_seeds.extend(seed)
                else:
                    all_seeds.append(seed)
        else:
            file_metadata = {}
            scenarios = data

        # Update metadata (use first file's metadata as base)
        if i == 0:
            combined_metadata = file_metadata.copy()
            # Remove seed from base metadata as we'll add combined seeds later
            if "seed" in combined_metadata:
                del combined_metadata["seed"]

        # Update simulation IDs to ensure uniqueness
        for j, scn in enumerate(scenarios):
            scn["simulation_id"] = len(all_ws_scenarios) + j + 1

        all_ws_scenarios.extend(scenarios)

    # Update combined metadata
    combined_metadata.update({
        "type": "windstorm",
        "seed": all_seeds if len(all_seeds) > 1 else (all_seeds[0] if all_seeds else None),
        # List if multiple, single if one
        "number_of_ws_simulations": len(all_ws_scenarios),
        "combined_from_files": scenario_files
    })

    # Create output structure
    output = {
        "metadata": combined_metadata,
        "ws_scenarios": all_ws_scenarios
    }

    # Add scenario probabilities if provided
    if scenario_probabilities:
        output["scenario_probabilities"] = scenario_probabilities

    # Write to file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Combined {len(all_ws_scenarios)} windstorm scenarios saved to {out_file}")
    return out_file


def extract_ws_scenarios(
    full_file: str = "Scenario_Database/Full_Windstorm_Scenarios/all_full_scenarios_year.json",
    out_file: str  = "Scenario_Database/Extracted_Windstorm_Scenarios/all_ws_scenarios_year.json"
):
    """
    Extract each windstorm window (storm duration + all additional repair hours)
    from the full-year scenarios JSON and save to a new JSON.
    Note: this works well when the each full scenario contains only 1 windstorm event
    """
    # 1) Load data from full scenarios
    with open(full_file, "r") as f:
        data = json.load(f)

    # support both the old version (a list without metadata field) or the new version (a dict with metadata field)
    # of the full_scenarios .json file
    if isinstance(data, dict) and "scenarios" in data:
        meta = data.get("metadata", {})
        all_full = data["scenarios"]
    else:
        meta = {}
        all_full = data

    # 2) Extract data
    all_ws = []
    for sim in all_full:
        sim_id = sim["simulation_id"]
        events = sim["events"]
        # Determine window bounds
        starts   = [event["bgn_hr"] for event in events]
        durations= [len(event["gust_speed"]) for event in events]
        ttr_list = sim["bch_ttr"]
        # earliest start, max duration, max repair tail
        start_hr    = min(starts)
        max_dur     = max(durations)
        max_ttr     = max(ttr_list)
        end_hr      = start_hr + max_dur + max_ttr - 1

        # 2) Slice the time-series arrays
        rand_arr = np.array(sim["bch_rand_nums"])       # shape: [branches x full_hours]
        flgs_arr = np.array(sim["flgs_impacted_bch"])   # same shape
        # Python slicing: start index = start_hr-1, end index = end_hr (exclusive)
        s_idx = start_hr - 1
        e_idx = end_hr
        rand_trim = rand_arr[:, s_idx:e_idx]
        flgs_trim = flgs_arr[:, s_idx:e_idx]

        # 3) Build extreme-scenario dict
        sim_ws = {
            "simulation_id":      sim_id,
            "events":             events,
            "bch_rand_nums":      rand_trim.tolist(),
            "flgs_impacted_bch":  flgs_trim.tolist(),
            "bch_ttr":            sim["bch_ttr"]
        }
        all_ws.append(sim_ws)

        # 4) Write out (keeping the metadata field)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Update metadata to include type
        if "type" not in meta:
            meta["type"] = "windstorm"

        output = {
            "metadata": meta,
            "ws_scenarios": all_ws
        }

        with open(out_file, "w") as f:
            json.dump(output, f, indent=4)

        print(f"Extracted windstorm scenarios saved to {out_file}")


def generate_normal_operation_scenario(
        duration_hours=8760,  # Full year by default
        start_hour=1,
        out_dir="Scenario_Database/Normal_Scenarios",
        network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group"
):
    """
    Generate a normal operation scenario (no windstorm events).

    Args:
        duration_hours: Duration of normal operation in hours (8760 for full year)
        start_hour: Starting hour for the scenario
        out_dir: Output directory for the scenario file
        network_preset: Network configuration preset
    """
    # Initialize network to get branch count
    net = make_network(network_preset)
    num_bch = len(net.data.net.bch)

    # Create scenario structure
    normal_scenario = {
        "simulation_id": "normal",
        "events": [{
            "event_id": 1,
            "bgn_hr": start_hour,
            "duration": duration_hours,
            "epicentre": [],  # No windstorm epicentre
            "radius": [],  # No windstorm radius
            "gust_speed": [0] * duration_hours  # Zero gust speed
        }],
        "bch_rand_nums": [[1.0] * duration_hours for _ in range(num_bch)],  # All 1.0 (no random failures)
        "flgs_impacted_bch": [[0] * duration_hours for _ in range(num_bch)],  # No branches impacted
        "bch_ttr": [0] * num_bch  # No repair time needed
    }

    # Create output structure
    output = {
        "metadata": {
            "scenario_type": "normal_operation",
            "network_preset": network_preset,
            "number_of_scenarios": 1,
            "period_type": "year",
            "duration_hours": duration_hours
        },
        "scenarios": [normal_scenario]
    }

    # Write to file
    os.makedirs(out_dir, exist_ok=True)

    # Shortened alias for network name
    network_alias = "29BusGB-KearsleyGSPGroup" if network_preset == "29_bus_GB_transmission_network_with_Kearsley_GSP_group" else network_preset

    file_name = f"normal_operation_scenario_network_{network_alias}_{duration_hours}hrs.json"
    path = os.path.join(out_dir, file_name)

    with open(path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Normal operation scenario saved to {path}")
    return path