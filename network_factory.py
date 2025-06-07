from config import NetConfig
from network import NetworkClass


def make_network(name: str) -> NetworkClass:
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # build the right config object
    ncon = NetConfig()  # initialize the 'ncon' objective as the default

    if name == 'UK_transmission_network':
        """
        This loads the UK transmission network from a prepared spread sheet
        Note: 1) the network model data is only suitable for a DC power flow
              2) the network model read from the .xlsx file has inherently some islanded buses:
              --> [224, 225, 248, 437, 438, 439, 440] (1-based indexing)
              As all these buses have zero load and gen, they are kept in the model and won't affect DC power flow.
        """

        # 1) Load all the relevant sheets once
        net_data = pd.ExcelFile(r"Input_Data\GB_Network\GBNetwork_New.xlsx")

        # - bus data:
        bus_df = net_data.parse(
            "bus",
            header=0,
            index_col=0,  # <— use the first unnamed column as the bus ID index column
            dtype={"vn_kv": float}
        )
        bus_df.index = bus_df.index + 1  # convert 0-based indexing into 1-based indexing
        bus_df.index.name = "bus"
        # - grid export point (i.e., slack bus):
        ext_df = net_data.parse("ext_grid", header=0, dtype={"bus": int})
        ext_df["bus"] += 1  # convert to 1-based indexing (and same for below)
        # - load data:
        load_df = net_data.parse("load", header=0, dtype={"bus": int, "p_mw": float, "q_mvar": float})
        load_df["bus"] += 1
        # - generator data:
        sgen_df = net_data.parse("sgen", header=0, dtype={"bus": int, "p_mw": float, "q_mvar": float})
        sgen_df["bus"] += 1
        # - generation cost data
        gen_cost_df = net_data.parse("poly_cost", header=0, index_col=0)
        # - line data:
        line_df = net_data.parse("line", header=0, dtype={"from_bus": int, "to_bus": int,
                                                     "r_ohm_per_km": float, "x_ohm_per_km": float,
                                                     "max_i_ka": float})
        line_df["from_bus"] += 1
        line_df["to_bus"] += 1
        # - transformer data:
        trafo_df = net_data.parse("trafo", header=0, dtype={"hv_bus": int, "lv_bus": int})
        trafo_df["hv_bus"] += 1
        trafo_df["lv_bus"] += 1

        imp_df = net_data.parse("impedance", header=0)  # if you have additional impedance data
        bus_geo_df = net_data.parse(
            "bus_geodata",
            header=0,
            index_col=0,
            dtype={"lon": float, "lat": float}
        )
        bus_geo_df.index = bus_geo_df.index + 1

        line_geo_df = net_data.parse("line_geodata", header=0,
                                dtype={"branch": int, "from_lon": float, "from_lat": float,
                                       "to_lon": float, "to_lat": float})

        # 2) Populate the NetConfig fields
        # - bus list & slack bus
        ncon.data.net.bus = bus_df.index.tolist()
        # assume the first external‐grid row is our slack
        ncon.data.net.slack_bus = int(ext_df["bus"].iloc[0])

        # - demand
        # merge loads at same buses to avoid index error
        load_agg = (
            load_df
            .groupby("bus", as_index=True)[["p_mw", "q_mvar"]]
            .sum()
        )
        # for missing buses, fill the value with 0
        load_ordered = load_agg.reindex(ncon.data.net.bus, fill_value=0)
        ncon.data.net.Pd_max = load_ordered["p_mw"].tolist()
        ncon.data.net.Qd_max = load_ordered["q_mvar"].tolist()

        # - generators
        ncon.data.net.gen = sgen_df["bus"].tolist()
        ncon.data.net.Pg_max = sgen_df["p_mw"].tolist()
        ncon.data.net.Pg_min = [0] * len(sgen_df["bus"].tolist())
        ncon.data.net.Qg_max = [0] * len(sgen_df["bus"].tolist())  # Assume no reactive power generation (DC power flow)
        ncon.data.net.Qg_min = [0] * len(sgen_df["bus"].tolist())

        ncon.data.net.gen_cost_coef = gen_cost_df[
            ["cp2_eur_per_mw2", "cp1_eur_per_mw", "cp0_eur"]
        ].values.tolist()

        # - branches (including transformers)
        # -- lines (i.e., in which from_bus and to_bus are at same voltage level):
        ncon.data.net.bch = line_df[["from_bus", "to_bus"]].values.tolist()
        ncon.data.net.bch_R = (line_df["length_km"] * line_df["r_ohm_per_km"]).tolist()
        ncon.data.net.bch_X = (line_df["length_km"] * line_df["x_ohm_per_km"]).tolist()

        # compute Pmax and Smax
        bus_v = bus_df["vn_kv"]  # note 'vn_kv' is already indexed by the bus indices
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

        tr_Smax = trafo_df["sn_mva"].tolist()
        tr_Pmax = trafo_df["sn_mva"].tolist()

        # -- append transformer data to the branch data:
        ncon.data.net.bch += tr_bch
        ncon.data.net.bch_R += tr_R
        ncon.data.net.bch_X += tr_X
        ncon.data.net.bch_Smax += tr_Smax
        ncon.data.net.bch_Pmax += tr_Pmax

        # — geographic coordinates for plotting/impacts
        bus_geo = bus_geo_df.reindex(ncon.data.net.bus)
        ncon.data.net.bus_lon = bus_geo["x"].tolist()
        ncon.data.net.bus_lat = bus_geo["y"].tolist()
        # ncon.data.net.bch_gis_bgn = list(zip(line_geo_df["from_lon"], line_geo_df["from_lat"]))
        # ncon.data.net.bch_gis_end = list(zip(line_geo_df["to_lon"], line_geo_df["to_lat"]))

        # 3) Base values:
        ncon.data.net.base_MVA = 100.0
        ncon.data.net.base_kV = 400.0

        # 4) Other parameters that need to be specified (not from the network data file)
        ncon.data.net.Pc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed active power at bus per MW
        ncon.data.net.Qc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed reactive power at bus per MW

        ncon.data.net.Pext_cost = 50


    elif name == 'Manchester_distribution_network_kearsley':
        """
        This loads the distribution network at GSP, BSP, PS level, corresponding to 'Kearsley' GSP group.
        
        This model is compatible with linearized AC power flow
        """

        # -----------------------------------------------------------
        # 1.  Read the workbook
        # -----------------------------------------------------------
        wb_path = Path(r"Input_Data/Manchester_Distribution_Network/Kearsley_GSP_group_only.xlsx")
        wb = pd.ExcelFile(wb_path)

        df_base = wb.parse("base_values")
        df_bus = wb.parse("bus")
        df_bch = wb.parse("line&trafo")
        df_load = wb.parse("load")
        df_gen = wb.parse("gen")

        # -----------------------------------------------------------
        # 2.  Base values
        # -----------------------------------------------------------
        ncon.data.net.base_MVA = float(
            df_base.loc[df_base["Name"] == "Base_MVA", "Value"]
            .fillna(100).iloc[0]
        )

        ncon.data.net.base_kV = None  # cosmetic for this model

        # -----------------------------------------------------------
        # 3.  Bus data ------------------------------------------------
        # -----------------------------------------------------------
        ncon.data.net.bus = df_bus["Bus ID"].astype(int).tolist()

        # slack bus (assume exactly one row has If Slack == 1)
        ncon.data.net.slack_bus = int(df_bus.loc[df_bus["If Slack"] == 1,
        "Bus ID"].iloc[0])

        # θ limits (taken from the first row – all rows identical in your file)
        ncon.data.net.theta_limits = [
            float(df_bus["V_angle_min"].iloc[0]),
            float(df_bus["V_angle_max"].iloc[0])
        ]

        # voltage-magnitude limits (square them - LinDistFlow expects V²)
        ncon.data.net.V_min = (df_bus["V_min (pu)"] ** 2).tolist()
        ncon.data.net.V_max = (df_bus["V_max (pu)"] ** 2).tolist()

        # GEO coordinates
        ncon.data.net.bus_lon = df_bus["Geo_lon"].tolist()
        ncon.data.net.bus_lat = df_bus["Geo_lat"].tolist()

        # -----------------------------------------------------------
        # 4.  Branch (line & transformer) data ----------------------
        # -----------------------------------------------------------
        ncon.data.net.bch = (
            df_bch[["From_bus ID", "To_bus ID"]]
            .astype(int).values.tolist()
        )

        # R & X are already in p.u. in the workbook
        ncon.data.net.bch_R = df_bch["Resistance (pu)"].astype(float).tolist()
        ncon.data.net.bch_X = df_bch["Reactance (pu)"].astype(float).tolist()

        # Apparent- and active-power limits
        ncon.data.net.bch_Smax = df_bch["S_max"].astype(float).tolist()
        ncon.data.net.bch_Pmax = df_bch["P_max"].astype(float).tolist()

        # -----------------------------------------------------------
        # 5.  Load data ---------------------------------------------
        # -----------------------------------------------------------
        # Build Pd/Qd vectors aligned with the master bus list
        load_agg = (
            df_load.set_index("Bus ID")[["Pd_max", "Qd_max"]]
            .astype(float)
            .groupby(level=0).sum()  # just in case duplicates
            .reindex(ncon.data.net.bus)  # fill missing with 0
            .fillna(0)
        )
        ncon.data.net.Pd_max = load_agg["Pd_max"].tolist()
        ncon.data.net.Qd_max = load_agg["Qd_max"].tolist()

        # profiles will be filled later by NetworkClass.set_scaled_profile…
        ncon.data.net.profile_Pd = None
        ncon.data.net.profile_Qd = None

        # -----------------------------------------------------------
        # 6.  Generator data ----------------------------------------
        # -----------------------------------------------------------
        ncon.data.net.gen = df_gen["Bus ID"].astype(int).tolist()
        ncon.data.net.Pg_max = df_gen["Pg_max"].astype(float).tolist()
        ncon.data.net.Pg_min = df_gen["Pg_min"].astype(float).tolist()
        ncon.data.net.Qg_max = df_gen["Qg_max"].astype(float).tolist()
        ncon.data.net.Qg_min = df_gen["Qg_min"].astype(float).tolist()

        # generation cost coefficients  (quadratic - linear)
        ncon.data.net.gen_cost_coef = (
            df_gen[["gen_cost_coef_2", "gen_cost_coef_1"]]
            .fillna(0).values.tolist()
        )

        # -----------------------------------------------------------
        # 7.  Cost-related parameters ---------------------------
        # -----------------------------------------------------------
        ncon.data.net.Pc_cost = [1000] * len(ncon.data.net.bus)  # active load shedding cost
        ncon.data.net.Qc_cost = [1000] * len(ncon.data.net.bus)  # reactive load shedding cost
        ncon.data.net.Pext_cost = 10  # grid active import cost
        ncon.data.net.Qext_cost = 10  # grid reactive import cost

        # placeholders – will be set by NetworkClass.set_gis_data()
        ncon.data.net.all_bus_coords_in_tuple = None
        ncon.data.net.bch_gis_bgn = None
        ncon.data.net.bch_gis_end = None



    elif name == 'matpower_case22':
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


