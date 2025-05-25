from config import NetConfig
from network import NetworkClass


def make_network(name: str) -> NetworkClass:
    # build the right config object
    ncon = NetConfig()  # initialize the 'ncon' objective as the default

    if name == 'UK_transmission_network':
        """
        This loads the UK transmission network from a prepared spread sheet
        Note the network model data is only suitable for a DC power flow
        """
        import numpy as np
        import pandas as pd

        # 1) Load all the relevant sheets once
        net_data = pd.ExcelFile("Input_Data\GB_Network\GBNetwork_New.xlsx")

        # - bus data:
        bus_df = net_data.parse("bus", header=0, dtype={"bus": int})
        # - grid export point (i.e., slack bus):
        ext_df = net_data.parse("ext_grid", header=0, dtype={"bus": int})
        # - load data:
        load_df = net_data.parse("load", header=0, dtype={"bus": int, "P_MW": float, "Q_MVar": float})
        # - generator data:
        sgen_df = net_data.parse("sgen", header=0, dtype={"bus": int, "P_MW": float, "Q_MVar": float})
        # - generation cost data
        gen_cost_df = net_data.parse("poly_cost", header=0, index_col=0)
        # - line data:
        line_df = net_data.parse("line", header=0, dtype={"from_bus": int, "to_bus": int,
                                                     "R_ohm_per_km": float, "X_ohm_per_km": float,
                                                     "max_i_ka": float})
        # - transformer data:
        trafo_df = net_data.parse("trafo", header=0, dtype={"hv_bus": int, "lv_bus": int})

        imp_df = net_data.parse("impedance", header=0)  # if you have additional impedance data
        bus_geo_df = net_data.parse("bus_geodata", header=0, dtype={"bus": int, "lon": float, "lat": float})
        line_geo_df = net_data.parse("line_geodata", header=0,
                                dtype={"branch": int, "from_lon": float, "from_lat": float,
                                       "to_lon": float, "to_lat": float})

        # 2) Populate the NetConfig fields
        # - bus list & slack bus
        ncon.data.net.bus = list(range(1, len(bus_df["name"].tolist()) + 1))
        # assume the first external‐grid row is our slack
        ncon.data.net.slack_bus = int(ext_df["bus"].iloc[0])

        # - demand
        # reindex so Pd_max/Qd_max line up with ncon.data.net.bus ordering
        load_indexed = load_df.set_index("bus")
        ncon.data.net.Pd_max = load_indexed.reindex(ncon.data.net.bus)["p_mw"].fillna(0).tolist()
        ncon.data.net.Qd_max = load_indexed.reindex(ncon.data.net.bus)["q_mvar"].fillna(0).tolist()

        # - generators
        ncon.data.net.gen = sgen_df["bus"].tolist()
        ncon.data.net.Pg_max = sgen_df["p_mw"].tolist()
        ncon.data.net.Pg_min = [0] * len(sgen_df["bus"].tolist())
        ncon.data.net.Qg_max = [0] * len(sgen_df["bus"].tolist())  # Assume no reactive power generation (DC power flow)
        ncon.data.net.Qg_min = [0] * len(sgen_df["bus"].tolist())
        ncon.data.net.gen_cost_coef = [
            gen_cost_df.loc[idx, ["cp2_eur_per_mw2", "cp1_eur_per_mw", "cp0_eur"]].values.tolist()
            for idx in ncon.data.net.gen
        ]

        # - branches (including transformers)
        # -- lines (i.e., in which from_bus and to_bus are at same voltage level):
        ncon.data.net.bch = line_df[["from_bus", "to_bus"]].values.tolist()
        ncon.data.net.bch_R = (line_df["length_km"] * line_df["r_ohm_per_km"]).tolist()
        ncon.data.net.bch_X = (line_df["length_km"] * line_df["x_ohm_per_km"]).tolist()

        # compute Pmax and Smax
        bus_v = bus_df.set_index("bus")["vn_kv"]  # index 'vn_kv' by 'bus'
        V_kv = line_df["from_bus"].map(bus_v)  # find the voltage level of the line from the voltage of its 'from_bus'
        I_ka = line_df["max_i_ka"]
        S_max = np.sqrt(3) * V_kv * I_ka
        ncon.data.net.bch_Smax = S_max.tolist()
        ncon.data.net.bch_Pmax = S_max.tolist()  # Assumes power factor = 1

        # -- transformers:
        tr_bch = trafo_df[["hv_bus", "lv_bus"]].values.tolist()

        r_pu = trafo_df["vkr_percent"] / 100.0  # pu resistance
        tr_R = r_pu.tolist()

        z_pu = trafo_df["vk_percent"] / 100.0  # pu impedance
        x_pu = np.sqrt(z_pu ** 2 - r_pu ** 2)  # compute pu reactance
        tr_X = x_pu.tolist()

        tr_Smax = trafo_df["sn_mva"]
        tr_Pmax = trafo_df["sn_mva"]

        # -- append transformer data to the branch data:
        ncon.data.net.bch += tr_bch
        ncon.data.net.bch_R += tr_R
        ncon.data.net.bch_X += tr_X
        ncon.data.net.bch_Smax += tr_Smax
        ncon.data.net.bch_Pmax += tr_Pmax

        # — geographic coordinates for plotting/impacts
        bus_geo = bus_geo_df.set_index("bus").reindex(ncon.data.net.bus)
        ncon.data.net.bus_lon = bus_geo["lon"].tolist()
        ncon.data.net.bus_lat = bus_geo["lat"].tolist()
        ncon.data.net.bch_gis_bgn = list(zip(line_geo_df["from_lon"], line_geo_df["from_lat"]))
        ncon.data.net.bch_gis_end = list(zip(line_geo_df["to_lon"], line_geo_df["to_lat"]))

        # 3) Base values:
        ncon.data.net.base_MVA = 100.0
        ncon.data.net.base_kV = 400.0

        # 4) Other parameters that need to be specified (not from the network data file)
        ncon.data.net.Pc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed active power at bus per MW
        ncon.data.net.Qc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed reactive power at bus per MW


    if name == 'matpower_case22':
        """
        This network model is compatible with the linearized AC power flow
        """
        # 1) Base values
        ncon.data.net.base_MVA = 1.0  # mpc.baseMVA
        ncon.data.net.base_kV = 11.0  # mpc.bus(:,BASE_KV)

        # 2) Bus data
        ncon.data.net.bus = list(range(1, 23))  # bus indices
        ncon.data.net.slack_bus = 1  # slack bus
        ncon.data.net.theta_limits = [-0.5, 0.5]

        # voltage limits (limits of V^2)
        V1 = 1.0 ** 2
        ncon.data.net.V_min = [V1] + [0.9 ** 2] * 21
        ncon.data.net.V_max = [V1] + [1.1 ** 2] * 21

        # max active/reactive demand (in MW / MVAr)
        ncon.data.net.Pd_max = [
            0.0,
            16.78 / 1000, 16.78 / 1000, 33.8 / 1000, 14.56 / 1000, 10.49 / 1000,
            8.821 / 1000, 14.35 / 1000, 19.31 / 1000, 14.35 / 1000, 16.27 / 1000,
            16.27 / 1000, 82.13 / 1000, 34.71 / 1000, 34.71 / 1000, 80.31 / 1000,
            49.62 / 1000, 49.62 / 1000, 43.77 / 1000, 37.32 / 1000, 37.32 / 1000,
            31.02 / 1000
        ]
        ncon.data.net.Qd_max = [
            0.0,
            20.91 / 1000, 20.91 / 1000, 37.32 / 1000, 12.52 / 1000, 14.21 / 1000,
            11.66 / 1000, 18.59 / 1000, 25.87 / 1000, 18.59 / 1000, 19.48 / 1000,
            19.48 / 1000, 71.65 / 1000, 30.12 / 1000, 30.12 / 1000, 70.12 / 1000,
            47.82 / 1000, 47.82 / 1000, 38.93 / 1000, 35.96 / 1000, 35.96 / 1000,
            29.36 / 1000
        ]

        ncon.data.net.profile_Pd = None
        ncon.data.net.profile_Qd = None

        # ncon.data.net.bus_lon = [0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 7, 7, 7]
        # ncon.data.net.bus_lat = [2, 2, 1, 2, 1, 0, -1, -2, 2, 1, 2, 1, 0, -1, -2, -1, 0, 1, 0, 0, -1, -2]

        ncon.data.net.bus_lon = [0, 1 / 2, 1 / 2, 2 / 2, 2 / 2, 2 / 2, 2 / 2, 2 / 2, 3 / 2, 3 / 2, 4 / 2, 4 / 2, 4 / 2,
                                 4 / 2, 4 / 2, 5 / 2, 5 / 2, 5 / 2, 6 / 2, 7 / 2, 7 / 2, 7 / 2]
        ncon.data.net.bus_lat = [2 / 2, 2 / 2, 1 / 2, 2 / 2, 1 / 2, 0 / 2, -1 / 2, -2 / 2, 2 / 2, 1 / 2, 2 / 2, 1 / 2,
                                 0 / 2, -1 / 2, -2 / 2, -1 / 2, 0 / 2, 1 / 2, 0 / 2, 0 / 2, -1 / 2, -2 / 2]

        ncon.data.net.all_bus_coords_in_tuple = None
        ncon.data.net.bch_gis_bgn = None
        ncon.data.net.bch_gis_end = None

        ncon.data.net.Pc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed active power at bus per MW
        ncon.data.net.Qc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed reactive power at bus per MW

        # 3) Branch data (p.u.)
        #    orig. Ohm→p.u. by dividing by Zbase=(11kV)^2/1MVA=121 Ω
        ncon.data.net.bch = [
            [1, 2], [2, 3], [2, 4], [4, 5], [4, 9],
            [5, 6], [6, 7], [6, 8], [9, 10], [9, 11],
            [11, 12], [11, 13], [13, 14], [14, 15], [14, 16],
            [16, 17], [17, 18], [17, 19], [19, 20], [20, 21],
            [20, 22]
        ]
        ncon.data.net.bch_R = [
            0.3664 / 121, 0.0547 / 121, 0.5416 / 121, 0.1930 / 121, 0.7431 / 121,
            1.3110 / 121, 0.0598 / 121, 0.2905 / 121, 0.0547 / 121, 0.6750 / 121,
            0.0547 / 121, 0.3942 / 121, 1.0460 / 121, 0.0220 / 121, 0.0547 / 121,
            0.3212 / 121, 0.0949 / 121, 0.5740 / 121, 0.1292 / 121, 0.0871 / 121,
            0.5329 / 121
        ]
        ncon.data.net.bch_X = [
            0.1807 / 121, 0.0282 / 121, 0.2789 / 121, 0.0990 / 121, 0.3827 / 121,
            0.6752 / 121, 0.0308 / 121, 0.1496 / 121, 0.0282 / 121, 0.3481 / 121,
            0.0282 / 121, 0.2030 / 121, 0.5388 / 121, 0.0116 / 121, 0.0282 / 121,
            0.1654 / 121, 0.0488 / 121, 0.2959 / 121, 0.0660 / 121, 0.0450 / 121,
            0.2744 / 121
        ]

        # thermal (apparent‐power) limits [MW] – set to 1 MVA per branch by default
        # note that no branch power rating values are specified, 1 MVA is large enough that is guaranteed never to bind
        ncon.data.net.bch_Smax = [1] * len(ncon.data.net.bch)
        ncon.data.net.bch_Pmax = [1] * len(ncon.data.net.bch)  # similarly set all Pmax to 1 to avoid binding

        # 4) Generator data
        ncon.data.net.gen = [1]
        ncon.data.net.Pg_max = [10]
        ncon.data.net.Pg_min = [0]
        ncon.data.net.Qg_max = [10]
        ncon.data.net.Qg_min = [-10]
        ncon.data.net.gen_cost_coef = [[0, 20]]


    else:
        raise ValueError(f"Unknown network preset '{name}'")

    return NetworkClass(ncon)
