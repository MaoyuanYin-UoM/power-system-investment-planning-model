# This script is to test the windstorm generation features

import numpy as np
import json
import os
from config import *
from utils import *
from visualization import *
from windstorm import *
from network_linear import *


wcon = WindConfig()
ws = WindClass(wcon)

ncon = NetConfig()
net = NetworkClass(ncon)

ws.crt_bgn_hr()
ws.init_ws_path0()

num_ws_prd = ws.MC.WS.num_ws_prd
num_ws_total = ws._get_num_ws_total()

# Create a list to store all results
all_results = []

for prd in range(len(num_ws_prd)):  # loop over each simulation

    # prepare parameters:
    net.set_gis_data()
    bch_gis_bgn = net._get_bch_gis_bgn()
    bch_gis_end = net._get_bch_gis_end()
    num_bch = len(net.data.net.bch)

    num_hrs_prd = ws._get_num_hrs_prd()

    # Initialize storage for each simulation
    sim_results = {
        "simulation_id": prd + 1,
        "events": []

    }

    # sample random numbers
    bch_rand_nums = np.random.rand(num_bch, num_hrs_prd)
    # initialize storage
    flgs_impacted_bch = np.zeros((num_bch, num_hrs_prd), dtype=bool)

    # Initialise paths for all windstorm events in this simulation
    start_lon, start_lat, end_lon, end_lat = ws.init_ws_path(num_ws_prd[prd])

    for i in range(num_ws_prd[prd]):  # loop over each windstorm event in this simulation
        # set current timestep to the beginning hour of the current windstorm event
        ts = ws.MC.WS.bgn_hrs_ws_prd[prd][i]
        ts = int(ts)

        # create windstorm path (in an hourly basis)
        lng_ws = ws._get_lng_ws()[i]
        path_ws = ws.crt_ws_path(start_lon[i], start_lat[i], end_lon[i], end_lat[i], lng_ws)  # obtain windstorm path

        # create windstorm radius
        radius_ws = ws.crt_ws_radius(lng_ws)

        # create the windstorm gust speeds for each path
        lim_v_ws = ws._get_lim_v_ws_all()[i]
        v_ws = ws.crt_ws_v(lim_v_ws, lng_ws)  # obtain gust speeds at each timestep during the windstorm

        duration = ws.MC.WS.lng[i]
        for t in range(duration):  # during the i-th windstorm, loop over each timestep t
            epicentre = path_ws[t]
            gust_speed = v_ws[t]
            radius = radius_ws[t]
            # determine if any branch is impacted by the windstorm at this timestep
            flgs_impacted_bch[:, ts+t] = ws.compare_circle(epicentre, radius, bch_gis_bgn, bch_gis_end, num_bch)
            # store the epicentre and gust speed at this timestep


        # Store event results
        sim_results["events"].append({
            "event_id": i + 1,
            "epicentre": path_ws,
            "radius": radius_ws.tolist(),  # note radius_ws is a numpy array so that it needs be converted to a list
            "gust_speed": v_ws
        })

    # append
    sim_results["flgs_impacted_bch"] = flgs_impacted_bch.tolist()
    sim_results["bch_rand_nums"] = bch_rand_nums.tolist()

    # Append simulation results to all results
    all_results.append(sim_results)



    # Save the results:
    file_name = "Results/all_scenarios.json"
    with open(file_name, "w") as f:
        json.dump(all_results, f, indent=4)  # Save in a .JSON file with formatting
    print(f"All results saved to {file_name}")









# visualize_ws_contour(wcon)

