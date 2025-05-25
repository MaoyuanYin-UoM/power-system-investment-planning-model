# This script contains functions to generate baseline scenarios and windstorm scenarios

import numpy as np
import json
import os
from config import WindConfig, NetConfig
from utils import set_random_seed
from windstorm import WindClass
from network import NetworkClass


def generate_ws_scenarios(seed=None, out_dir="Scenario_Results"):
    """
    Generate and save full windstorm scenarios.
    """
    # Optional reproducibility
    if seed is not None:
        set_random_seed(seed)

    # Initialize
    wcon = WindConfig()
    ws = WindClass(wcon)
    ncon = NetConfig()
    net = NetworkClass(ncon)

    ws.crt_bgn_hr()
    ws.init_ws_path0()

    num_ws_prd = ws.MC.WS.num_ws_prd
    num_bch = len(net.data.net.bch)
    num_hrs_prd = ws._get_num_hrs_prd()

    all_results = []

    for prd in range(len(num_ws_prd)):
        net.set_gis_data()
        bch_gis_bgn = net._get_bch_gis_bgn()
        bch_gis_end = net._get_bch_gis_end()

        sim = {"simulation_id": prd+1, "events": []}

        bch_rand_nums = np.random.rand(num_bch, num_hrs_prd)
        flgs_impacted_bch = np.zeros((num_bch, num_hrs_prd), dtype=int)

        start_lon, start_lat, end_lon, end_lat = ws.init_ws_path(num_ws_prd[prd])

        for i in range(num_ws_prd[prd]):
            ts = int(ws.MC.WS.bgn_hrs_ws_prd[prd][i])
            lng_ws = ws._get_lng_ws()[i]
            path_ws = ws.crt_ws_path(start_lon[i], start_lat[i], end_lon[i], end_lat[i], lng_ws)
            radius_ws = ws.crt_ws_radius(lng_ws)
            v_ws = ws.crt_ws_v(ws._get_lim_v_ws_all()[i], lng_ws)

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

        ttr_min, ttr_max = wcon.data.WS.event.ttr
        bch_ttr = np.random.randint(ttr_min, ttr_max, size=num_bch)

        sim["bch_rand_nums"] = bch_rand_nums.tolist()
        sim["flgs_impacted_bch"] = flgs_impacted_bch.tolist()
        sim["bch_ttr"] = bch_ttr.tolist()

        all_results.append(sim)

    os.makedirs(out_dir, exist_ok=True)
    fname = f"all_full_scenarios_{wcon.data.MC.lng_prd}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Windstorm scenarios saved to {path}")



def extract_ws_scenarios(
    full_file: str = "Scenario_Results/all_full_scenarios_year.json",
    out_file: str  = "Scenario_Results/all_ws_scenarios_year.json"
):
    """
    Extract each windstorm window (storm duration + all additional repair hours)
    from the full-year scenarios JSON and save to a new JSON.
    Note: this works well when the each full scenario contains only 1 windstorm event
    """
    # 1) Load full-year scenarios
    with open(full_file, "r") as f:
        all_full = json.load(f)

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

    # 4) Write out
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_ws, f, indent=4)
    print(f"Extracted windstorm scenarios saved to {out_file}")