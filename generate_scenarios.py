# This script is to test the windstorm generation features

import numpy as np
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

for prd in range(len(num_ws_prd)):  # loop over each simulation

    # prepare parameters:
    rad_ws = ws.data.WS.radius
    net.set_gis_data()
    bch_gis_bgn = net._get_bch_gis_bgn()
    bch_gis_end = net._get_bch_gis_end()
    num_bch = len(net.data.net.bch)

    num_hrs_prd = ws._get_num_hrs_prd()
    flgs_bch_status = np.ones((num_bch, num_hrs_prd), dtype=bool)  # initialise an array to store branch status data

    ts = 0  # initialise timestep to 0

    # Initialise paths for all windstorm events in this simulation
    start_lon, start_lat, end_lon, end_lat = ws.init_ws_path(num_ws_prd[prd])

    for i in range(num_ws_prd[prd]):  # loop over each windstorm event in this simulation
        # set current timestep to the beginning hour of the current windstorm event
        ts = ws.MC.WS.bgn_hrs_ws_prd[prd][i]

        # create windstorm path (in an hourly basis)
        lng_ws = ws._get_lng_ws()[i]
        path_ws = ws.crt_ws_path(start_lon[i], start_lat[i], end_lon[i], end_lon[i], lng_ws)  # obtain windstorm path

        # create the windstorm gust speeds for each path
        lim_v_ws = ws._get_lim_v_ws_all()[i]
        v_ws = ws.crt_ws_v(lim_v_ws, lng_ws)  # obtain gust speeds at each timestep during the windstorm

        duration = ws.MC.WS.lng[i]
        for t in range(duration):  # during the i-th windstorm, loop over each timestep t
            epicentre = path_ws[t]
            wind_speed = v_ws[t]
            # determine if any branch is impacted by the windstorm
            flgs_impacted_bch = ws.compare_circle(epicentre, rad_ws, bch_gis_bgn, bch_gis_end, num_bch)
            # if a branch is impacted, sample if it fails (from fragility curve)
            flgs_bch_status = ws.sample_bch_failure(ts+t, flgs_bch_status, flgs_impacted_bch, wind_speed)
            # if a branch fails, sample the time to repair










# visualize_ws_contour(wcon)

