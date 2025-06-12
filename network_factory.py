from config import NetConfig
from network import NetworkClass


def make_network(name: str) -> NetworkClass:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import copy

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
        net_data = pd.ExcelFile(r"Input_Data/GB_Network_full\GBNetwork_New.xlsx")

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

        # 2) Define base values
        base_MVA = 100
        base_kV = 400

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

        # treat the load values from the dataset as maximum demands
        ncon.data.net.Pd_max = load_ordered["p_mw"].tolist()
        ncon.data.net.Qd_max = load_ordered["q_mvar"].tolist()

        # assume that minimum demand values are not provided
        ncon.data.net.Pd_min = None
        ncon.data.net.Qd_min = None

        # - generators
        ncon.data.net.gen = sgen_df["bus"].tolist()
        ncon.data.net.Pg_max = sgen_df["p_mw"].tolist()
        ncon.data.net.Pg_min = [0] * len(sgen_df["bus"].tolist())
        ncon.data.net.Qg_max = [0] * len(sgen_df["bus"].tolist())  # Assume no reactive power generation (DC power flow)
        ncon.data.net.Qg_min = [0] * len(sgen_df["bus"].tolist())

        ncon.data.net.gen_cost_coef = gen_cost_df[
            ["cp0_eur", "cp1_eur_per_mw"]
        ].values.tolist()

        # - branches (including transformers)
        # -- lines (i.e., in which from_bus and to_bus are at same voltage level):
        ncon.data.net.bch = line_df[["from_bus", "to_bus"]].values.tolist()

        bus_v_kv = bus_df["vn_kv"]  # get bus voltage levels
        line_v_kv = line_df["from_bus"].map(bus_v_kv)  # get branch voltage levels
        Z_base_ohm = (line_v_kv ** 2) / base_MVA

        # compute impedance R, X in p.u. values
        ncon.data.net.bch_R = (line_df["length_km"] * line_df["r_ohm_per_km"] / Z_base_ohm).tolist()
        ncon.data.net.bch_X = (line_df["length_km"] * line_df["x_ohm_per_km"] / Z_base_ohm).tolist()

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

        # set branch length and type (1=line, 0=transformer)
        ncon.data.net.bch_length_km = (
                line_df["length_km"].tolist()  # real transmission‐line lengths
                + [0.0] * len(trafo_df)  # transformers → 0 km
        )
        ncon.data.net.bch_type = ([1] * len(line_df) + [0] * len(trafo_df))

        # — geographic coordinates for plotting/impacts
        bus_geo = bus_geo_df.reindex(ncon.data.net.bus)
        ncon.data.net.bus_lon = bus_geo["x"].tolist()
        ncon.data.net.bus_lat = bus_geo["y"].tolist()
        # ncon.data.net.bch_gis_bgn = list(zip(line_geo_df["from_lon"], line_geo_df["from_lat"]))
        # ncon.data.net.bch_gis_end = list(zip(line_geo_df["to_lon"], line_geo_df["to_lat"]))

        # 3) Base values:
        ncon.data.net.base_MVA = base_MVA
        ncon.data.net.base_kV = base_kV

        # 4) Other parameters that need to be specified (not from the network data file)
        ncon.data.net.Pc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed active power at bus per MW
        ncon.data.net.Qc_cost = [100] * len(ncon.data.net.bus)  # cost of curtailed reactive power at bus per MW

        ncon.data.net.Pimp_cost = 50
        ncon.data.net.Pexp_cost = 0  # remuneration for exporting electricity to distribution network
        ncon.data.net.Qimp_cost = 50
        ncon.data.net.Qexp_cost = 0

        # 5) Fragility data:
        num_bch = len(ncon.data.net.bch)
        ncon.data.frg.mu = [3.8] * num_bch
        ncon.data.frg.sigma = [0.122] * num_bch
        ncon.data.frg.thrd_1 = [20] * num_bch
        ncon.data.frg.thrd_2 = [90] * num_bch
        ncon.data.frg.shift_f = [0.0] * num_bch

        net = NetworkClass(ncon)
        net.name = name
        return net


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

        # angle limits (taken from the first row – all rows identical in your file)
        ncon.data.net.theta_limits = [
            float(df_bus["V_angle_min (rad)"].iloc[0]),
            float(df_bus["V_angle_max (rad)"].iloc[0])
        ]

        # voltage-magnitude limits
        ncon.data.net.V_min = (df_bus["V_min (pu)"]).tolist()
        ncon.data.net.V_max = (df_bus["V_max (pu)"]).tolist()

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

        # BRANCH TYPE: 1=line, 0=transformer
        ncon.data.net.bch_type = (
            df_bch["Type (1 for line, 0 for transformer)"].astype(int).tolist())

        # R & X are already in p.u. in the workbook
        ncon.data.net.bch_R = df_bch["Resistance (pu)"].astype(float).tolist()
        ncon.data.net.bch_X = df_bch["Reactance (pu)"].astype(float).tolist()

        # Apparent- and active-power limits
        ncon.data.net.bch_Smax = df_bch["S_max (MW)"].astype(float).tolist()
        ncon.data.net.bch_Pmax = df_bch["P_max (MW)"].astype(float).tolist()

        # branch length (will be used for branch hardening)
        ncon.data.net.bch_length_km = df_bch["Length (km)"].astype(float).tolist()

        # -----------------------------------------------------------
        # 5.  Load data ---------------------------------------------
        # -----------------------------------------------------------
        # Build Pd/Qd vectors aligned with the master bus list
        load_agg = (
            df_load.set_index("Bus ID")[["Pd_max", "Pd_min", "Qd_max", "Qd_min"]]
            .astype(float)
            .groupby(level=0).sum()  # just in case duplicates
            .reindex(ncon.data.net.bus)  # fill missing with 0
            .fillna(0)
        )
        ncon.data.net.Pd_max = load_agg["Pd_max"].tolist()
        ncon.data.net.Pd_min = load_agg["Pd_min"].tolist()
        ncon.data.net.Qd_max = load_agg["Qd_max"].tolist()
        ncon.data.net.Qd_min = load_agg["Qd_min"].tolist()

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
            df_gen[["gen_cost_coef_0", "gen_cost_coef_1"]]
            .fillna(0).values.tolist()
        )

        # -----------------------------------------------------------
        # 7.  Cost-related parameters ---------------------------
        # -----------------------------------------------------------
        ncon.data.net.Pc_cost = [1e3] * len(ncon.data.net.bus)  # active load shedding cost
        ncon.data.net.Qc_cost = [1e3] * len(ncon.data.net.bus)  # reactive load shedding cost

        ncon.data.net.Pimp_cost = 50  # active power importing cost
        ncon.data.net.Pexp_cost = 0  # active power exporting remuneration
        ncon.data.net.Qimp_cost = 50  # reactive power importing cost
        ncon.data.net.Qexp_cost = 0  # reactive power exporting remuneration

        # placeholders – will be set by NetworkClass.set_gis_data()
        ncon.data.net.all_bus_coords_in_tuple = None
        ncon.data.net.bch_gis_bgn = None
        ncon.data.net.bch_gis_end = None

        # -----------------------------------------------------------
        # 8.  Fragility data ---------------------------
        # -----------------------------------------------------------
        num_bch = len(ncon.data.net.bch)
        ncon.data.frg.mu = [3.8] * num_bch
        ncon.data.frg.sigma = [0.122] * num_bch
        ncon.data.frg.thrd_1 = [20] * num_bch
        ncon.data.frg.thrd_2 = [90] * num_bch
        ncon.data.frg.shift_f = [0.0] * num_bch

        net = NetworkClass(ncon)
        net.name = name
        return net


    elif name == 'UK_transmission_network_with_kearsley_GSP_group':
        """
        A composite network model:
          • UK transmission network (compatible DC power flow)
          • Manchester ‘Kearsley’ distribution group (compatible with linearized AC power flow)

        TSO-DSO coupling point:
          – bus 184 (UK)  ←→  bus 1 (Kearsley) via a “zero-impedance” link
        
        Slack bus:
          – identical to the UK‐transmission slack bus (Bus 184)
          
        Note:
          – 'ncon.data.net.bus_level' and 'ncon.data.net.branch_level' specify the buses and branches at transmission or
            distribution level 
        """

        # 1. Load the two networks exactly as already defined
        uk_net = make_network('UK_transmission_network')  # UK transmission network
        dn_net = make_network('Manchester_distribution_network_kearsley')  # Manchester distribution network

        uk = uk_net.data.net  # shorthand
        dn = dn_net.data.net

        # 2. Create a fresh NetConfig initialized with the transmission network model
        ncon = NetConfig()
        ncon.data.net = copy.deepcopy(uk)  # start from the UK template

        # Tag the buses and branches from uk_net as transmission level
        ncon.data.net.bus_level = {b: 'T' for b in uk.bus}
        ncon.data.net.branch_level = {i + 1: 'T' for i in range(len(uk.bch))}

        # 3. Re-index the distribution buses so we have unique IDs (simply shift them by max(uk.bus))
        offset = max(uk.bus)
        dn_bus_map = {b: b + offset for b in dn.bus}  # distribution bus index: old->new

        # helper to remap a bus index or a [from,to] pair
        def _map_bus(b):
            return dn_bus_map[b]

        def _map_pair(pair):
            return [_map_bus(pair[0]), _map_bus(pair[1])]

        # 4.  Append the distribution network data to ncon
        # 4.1) Append bus data
        new_dn_buses = [_map_bus(b) for b in dn.bus]
        ncon.data.net.bus.extend(new_dn_buses)

        # ensure V_min and V_max exist on the copied UK net before extending (assign default values if not exist)
        if getattr(ncon.data.net, 'V_min', None) is None:
            ncon.data.net.V_min = [0.95] * len(uk.bus)
        if getattr(ncon.data.net, 'V_max', None) is None:
            ncon.data.net.V_max = [1.05] * len(uk.bus)

        ncon.data.net.V_min.extend(dn.V_min)
        ncon.data.net.V_max.extend(dn.V_max)
        ncon.data.net.Pd_max.extend(dn.Pd_max)
        ncon.data.net.Qd_max.extend(dn.Qd_max)
        ncon.data.net.Pc_cost.extend(dn.Pc_cost)
        ncon.data.net.Qc_cost.extend(dn.Qc_cost)

        if dn.bus_lon is not None:
            ncon.data.net.bus_lon.extend(dn.bus_lon)
            ncon.data.net.bus_lat.extend(dn.bus_lat)

        # tag the buses as distribution level
        ncon.data.net.bus_level.update({b: 'D' for b in new_dn_buses})

        # 4.2) Append branch data (lines & transformers)
        dn_pairs_mapped = [_map_pair(p) for p in dn.bch]
        ncon.data.net.bch.extend(dn_pairs_mapped)

        ncon.data.net.bch_type = list(uk.bch_type) + list(dn.bch_type)

        ncon.data.net.bch_R.extend(dn.bch_R)
        ncon.data.net.bch_X.extend(dn.bch_X)
        ncon.data.net.bch_Smax.extend(dn.bch_Smax)
        ncon.data.net.bch_Pmax.extend(dn.bch_Pmax)
        ncon.data.net.bch_length_km.extend(dn.bch_length_km)

        # 4.3) Add a zero-impedance coupling branch between bus-184 (UK) and remapped bus-1 (DN)
        #      (inserted exactly after the transmission branches)
        idx_cpl = len(uk.bch)
        ncon.data.net.bch.insert(idx_cpl, [184, dn_bus_map[1]])
        ncon.data.net.bch_R.insert(len(uk.bch), 0)
        ncon.data.net.bch_X.insert(len(uk.bch), 0.0001)
        ncon.data.net.bch_Smax.insert(len(uk.bch), 1e6)
        ncon.data.net.bch_Pmax.insert(len(uk.bch), 1e6)
        ncon.data.net.bch_length_km.insert(len(uk.bch), 0.0)  # padding value

        ncon.data.net.bch_type.insert(idx_cpl, 0)  # treat this virtual coupling branch as non-hardenable
        ncon.data.net.branch_level[idx_cpl] = 'T-D'  # tag it as 'T-D' which will present in both tn and dn level

        # 4.4) Rebuild the 'branch_level' (ensure indices are correct after inserting the coupling branch)
        num_tn = len(uk.bch)
        num_dn = len(dn.bch)
        ncon.data.net.branch_level = {}
        for i in range(1, num_tn + 1):
            ncon.data.net.branch_level[i] = 'T'  # pure TN
        ncon.data.net.branch_level[num_tn + 1] = 'T-D'  # the tn-dn coupling link
        for k in range(num_dn):
            ncon.data.net.branch_level[num_tn + 2 + k] = 'D'  # pure DN

        # 4.5) Append gen data
        ncon.data.net.gen.extend([_map_bus(b) for b in dn.gen])
        ncon.data.net.Pg_max.extend(dn.Pg_max)
        ncon.data.net.Pg_min.extend(dn.Pg_min)
        ncon.data.net.Qg_max.extend(dn.Qg_max)
        ncon.data.net.Qg_min.extend(dn.Qg_min)
        ncon.data.net.gen_cost_coef.extend(dn.gen_cost_coef)

        # 4.6) Merge the fragility data
        ncon.data.frg.mu = list(uk_net.data.frg.mu) + list(dn_net.data.frg.mu)
        ncon.data.frg.sigma = list(uk_net.data.frg.sigma) + list(dn_net.data.frg.sigma)
        ncon.data.frg.thrd_1 = list(uk_net.data.frg.thrd_1) + list(dn_net.data.frg.thrd_1)
        ncon.data.frg.thrd_2 = list(uk_net.data.frg.thrd_2) + list(dn_net.data.frg.thrd_2)
        ncon.data.frg.shift_f = list(uk_net.data.frg.shift_f) + list(dn_net.data.frg.shift_f)

        # 5. Set slack & base values (same to UK transmission network)
        ncon.data.net.slack_bus = uk.slack_bus
        ncon.data.net.base_MVA = uk.base_MVA
        ncon.data.net.base_kV = uk.base_kV

        # 6. Specify additional data that are needed when building the investment model
        # 6.1) line hardening cost
        ncon.data.cost_rate_hrdn = 1e2  # hardening cost (£) per unit length (km) of the line and per unit amount (m/s)
        # that the fragility curve is shifted
        ncon.data.cost_bch_hrdn = [
            ncon.data.cost_rate_hrdn * length if ncon.data.net.bch_type[i] == 1 else 0.0
            for i, length in enumerate(ncon.data.net.bch_length_km)
        ]  # cost per unit fragility curve shift
        # Note: only distribution line hardening is considered

        # 6.2) line repair cost
        rep_rate_tn = 1e2  # repair cost (£) per unit length (km) of line at transmission level
        rep_rate_dn = 1e2  # repair cost (£) per unit length (km) of line at distribution level
        ncon.data.cost_bch_rep = [
            rep_rate_dn * length if ncon.data.net.branch_level[i + 1] == 'D' else rep_rate_tn * length
            for i, length in enumerate(ncon.data.net.bch_length_km)
        ]  # Note: the line repair at both transmission and distribution level are considered

        # 6.3) line hardening limits and budget
        ncon.data.bch_hrdn_limits = [0.0, 30.0]  # in m/s
        ncon.data.budget_bch_hrdn = 1e8  # in £

        net = NetworkClass(ncon)
        net.name = name
        return net


    elif name == 'GB_Transmission_Network_29_Bus':
        """
        29-bus simplified GB transmission system (DC-OPF compatible)

        """

        # -----------------------------------------------------------
        # 1.  Read workbook
        # -----------------------------------------------------------
        wb_path = Path(r"Input_Data/GB_Network_29bus/GB_Transmission_Network_29_Bus.xlsx")
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
            df_base.loc[df_base["Name"] == "Base_MVA", "Value"].fillna(100).iloc[0]
        )
        ncon.data.net.base_kV = float(
            df_base.loc[df_base["Name"] == "Base_kV", "Value"].fillna(400).iloc[0]
        )

        # -----------------------------------------------------------
        # 3.  Bus data
        # -----------------------------------------------------------
        ncon.data.net.bus = df_bus["Bus ID"].astype(int).tolist()

        # slack bus – unique row where “If Slack” == 1
        ncon.data.net.slack_bus = int(df_bus.loc[df_bus["If Slack"] == 1, "Bus ID"].iloc[0])

        # angle limits (rad) – identical for all rows
        ncon.data.net.theta_limits = [
            float(df_bus["V_angle_min (rad)"].iloc[0]),
            float(df_bus["V_angle_max (rad)"].iloc[0]),
        ]

        # voltage magnitude limits (pu²) – only cosmetic for DC OPF
        ncon.data.net.V_min = (df_bus["V_min (pu)"] ** 2).tolist()
        ncon.data.net.V_max = (df_bus["V_max (pu)"] ** 2).tolist()

        # GEO coordinates (optional but helpful for plotting / wind-storm overlay)
        ncon.data.net.bus_lon = df_bus["Geo_lon"].tolist()
        ncon.data.net.bus_lat = df_bus["Geo_lat"].tolist()

        # -----------------------------------------------------------
        # 4.  Branch data
        # -----------------------------------------------------------
        ncon.data.net.bch = (
            df_bch[["From_bus ID", "To_bus ID"]]
            .astype(int).values.tolist()
        )

        ncon.data.net.bch_type = df_bch["Type (1 for line, 0 for transformer)"].astype(int).tolist()

        # R & X already given in p.u.
        ncon.data.net.bch_R = df_bch["Resistance (pu)"].astype(float).tolist()
        ncon.data.net.bch_X = df_bch["Reactance (pu)"].astype(float).tolist()

        # Apparent- and active-power limits
        ncon.data.net.bch_Smax = df_bch["S_max (MW)"].astype(float).tolist()
        ncon.data.net.bch_Pmax = df_bch["P_max (MW)"].astype(float).tolist()

        # Line length (km) – used for hardening / repair cost
        ncon.data.net.bch_length_km = df_bch["Length (km)"].astype(float).tolist()

        # -----------------------------------------------------------
        # 5.  Load data
        # -----------------------------------------------------------
        load_agg = (
            df_load.set_index("Bus ID")[["Pd_max", "Pd_min", "Qd_max", "Qd_min"]]
            .astype(float)
            .groupby(level=0).sum()  # merge duplicates
            .reindex(ncon.data.net.bus)  # fill missing with 0
            .fillna(0)
        )
        ncon.data.net.Pd_max = load_agg["Pd_max"].tolist()
        ncon.data.net.Pd_min = load_agg["Pd_min"].tolist()
        ncon.data.net.Qd_max = load_agg["Qd_max"].tolist()
        ncon.data.net.Qd_min = load_agg["Qd_min"].tolist()

        # profiles will be filled later by NetworkClass.set_scaled_profile_for_buses()
        ncon.data.net.profile_Pd = None
        ncon.data.net.profile_Qd = None

        # -----------------------------------------------------------
        # 6.  Generator data
        # -----------------------------------------------------------
        ncon.data.net.gen = df_gen["Bus ID"].astype(int).tolist()
        ncon.data.net.Pg_max = df_gen["Pg_max"].astype(float).tolist()
        ncon.data.net.Pg_min = df_gen["Pg_min"].astype(float).tolist()
        ncon.data.net.Qg_max = df_gen["Qg_max"].astype(float).tolist()
        ncon.data.net.Qg_min = df_gen["Qg_min"].astype(float).tolist()

        # cost coefficients  (quadratic  + linear)
        ncon.data.net.gen_cost_coef = (
            df_gen[["gen_cost_coef_0", "gen_cost_coef_1"]]
            .fillna(0).values.tolist()
        )

        # -----------------------------------------------------------
        # 7.  Economic parameters
        # -----------------------------------------------------------
        ncon.data.net.Pc_cost = [100] * len(ncon.data.net.bus)  # load-shedding penalty £/MW
        ncon.data.net.Pimp_cost = 50  # grid import £/MW
        ncon.data.net.Pexp_cost = 0  # export remuneration £/MW

        # No reactive-power modelling – set dummies
        ncon.data.net.Qc_cost = [0] * len(ncon.data.net.bus)
        ncon.data.net.Qimp_cost = 50
        ncon.data.net.Qexp_cost = 0

        # -----------------------------------------------------------
        # 8.  Fragility data (uniform defaults)
        # -----------------------------------------------------------
        num_bch = len(ncon.data.net.bch)
        ncon.data.frg.mu = [3.8] * num_bch
        ncon.data.frg.sigma = [0.122] * num_bch
        ncon.data.frg.thrd_1 = [20] * num_bch
        ncon.data.frg.thrd_2 = [90] * num_bch
        ncon.data.frg.shift_f = [0.0] * num_bch

        # -----------------------------------------------------------
        # 9.  Instantiate & label the network
        # -----------------------------------------------------------
        net = NetworkClass(ncon)
        net.name = name
        return net



    elif name == '29_bus_GB_transmission_network_with_kearsley_GSP_group':
        """
        Composite test system
            • 29-bus simplified GB transmission network (DC-OPF compatible)
            • ‘Kearsley’ distribution group (linearised AC compatible)

        Coupling (TSO–DSO) link
            – GB-bus 21  ←→  Kearsley-bus 1
              modelled as a near-zero-impedance branch inserted between the
              TN and DN graphs.

        Slack bus
            – identical to the slack defined in the 29-bus GB template

        """

        import copy

        # ------------------------------------------------------------------
        # 1.  Load the two component networks
        # ------------------------------------------------------------------
        gb_net = make_network('GB_Transmission_Network_29_Bus')  # 29-bus TN
        dn_net = make_network('Manchester_distribution_network_kearsley')  # Kearsley DN

        gb = gb_net.data.net
        dn = dn_net.data.net

        # ------------------------------------------------------------------
        # 2.  Start a fresh NetConfig seeded with the transmission template
        # ------------------------------------------------------------------
        ncon = NetConfig()
        ncon.data.net = copy.deepcopy(gb)  # clone the GB TN as the base

        # Tag existing objects as transmission level
        ncon.data.net.bus_level = {b: 'T' for b in gb.bus}
        ncon.data.net.branch_level = {i + 1: 'T' for i in range(len(gb.bch))}

        # ------------------------------------------------------------------
        # 3.  Re-index the DN buses so IDs are unique
        # ------------------------------------------------------------------
        offset = max(gb.bus)  # 29
        dn_bus_map = {b: b + offset for b in dn.bus}

        def _m(b):  # map a single bus

            return dn_bus_map[b]

        def _pair(p):  # map a branch [from, to]

            return [_m(p[0]), _m(p[1])]

        # ------------------------------------------------------------------
        # 4.  Append the DN data
        # ------------------------------------------------------------------
        # 4.1  Buses and their attributes

        new_dn_buses = [_m(b) for b in dn.bus]
        ncon.data.net.bus.extend(new_dn_buses)
        ncon.data.net.V_min.extend(dn.V_min)
        ncon.data.net.V_max.extend(dn.V_max)
        ncon.data.net.Pd_max.extend(dn.Pd_max)
        ncon.data.net.Qd_max.extend(dn.Qd_max)
        ncon.data.net.Pc_cost.extend(dn.Pc_cost)
        ncon.data.net.Qc_cost.extend(dn.Qc_cost)
        ncon.data.net.Pd_min.extend(dn.Pd_min)
        ncon.data.net.Qd_min.extend(dn.Qd_min)

        if dn.bus_lon is not None:  # GIS coords
            ncon.data.net.bus_lon.extend(dn.bus_lon)
            ncon.data.net.bus_lat.extend(dn.bus_lat)

        # tag
        ncon.data.net.bus_level.update({b: 'D' for b in new_dn_buses})

        # 4.2  Branches
        ncon.data.net.bch.extend([_pair(p) for p in dn.bch])
        ncon.data.net.bch_type = list(gb.bch_type) + list(dn.bch_type)
        ncon.data.net.bch_R.extend(dn.bch_R)
        ncon.data.net.bch_X.extend(dn.bch_X)
        ncon.data.net.bch_Smax.extend(dn.bch_Smax)
        ncon.data.net.bch_Pmax.extend(dn.bch_Pmax)
        ncon.data.net.bch_length_km.extend(dn.bch_length_km)

        # 4.3  Insert the TN–DN coupling branch (after the TN list)
        idx_cpl = len(gb.bch)  # position in the master list
        ncon.data.net.bch.insert(idx_cpl, [21, _m(1)])  # 21 ←→ Kearsley-bus 1
        ncon.data.net.bch_R.insert(idx_cpl, 0.0)
        ncon.data.net.bch_X.insert(idx_cpl, 0.0001)
        ncon.data.net.bch_Smax.insert(idx_cpl, 1e6)
        ncon.data.net.bch_Pmax.insert(idx_cpl, 1e6)
        ncon.data.net.bch_length_km.insert(idx_cpl, 0.0)
        ncon.data.net.bch_type.insert(idx_cpl, 0)  # treat as transformer

        # 4.4  Re-build branch-level tags
        num_tn = len(gb.bch)
        num_dn = len(dn.bch)
        ncon.data.net.branch_level = {}

        for i in range(1, num_tn + 1):
            ncon.data.net.branch_level[i] = 'T'

        ncon.data.net.branch_level[num_tn + 1] = 'T-D'  # the coupling link

        for k in range(num_dn):
            ncon.data.net.branch_level[num_tn + 2 + k] = 'D'

        # 4.5  Generators
        ncon.data.net.gen.extend([_m(b) for b in dn.gen])
        ncon.data.net.Pg_max.extend(dn.Pg_max)
        ncon.data.net.Pg_min.extend(dn.Pg_min)
        ncon.data.net.Qg_max.extend(dn.Qg_max)
        ncon.data.net.Qg_min.extend(dn.Qg_min)
        ncon.data.net.gen_cost_coef.extend(dn.gen_cost_coef)

        # 4.6  Fragility vectors
        ncon.data.frg.mu = list(gb_net.data.frg.mu) + list(dn_net.data.frg.mu)
        ncon.data.frg.sigma = list(gb_net.data.frg.sigma) + list(dn_net.data.frg.sigma)
        ncon.data.frg.thrd_1 = list(gb_net.data.frg.thrd_1) + list(dn_net.data.frg.thrd_1)
        ncon.data.frg.thrd_2 = list(gb_net.data.frg.thrd_2) + list(dn_net.data.frg.thrd_2)
        ncon.data.frg.shift_f = list(gb_net.data.frg.shift_f) + list(dn_net.data.frg.shift_f)

        # ------------------------------------------------------------------
        # 5.  Slack and base values
        # ------------------------------------------------------------------
        ncon.data.net.slack_bus = gb.slack_bus
        ncon.data.net.base_MVA = gb.base_MVA
        ncon.data.net.base_kV = gb.base_kV

        # ------------------------------------------------------------------
        # 6. Specify additional data that are needed when building the investment model
        # ------------------------------------------------------------------
        # 6.1) line hardening cost
        ncon.data.cost_rate_hrdn = 1e2  # hardening cost (£) per unit length (km) of the line and per unit amount (m/s)
        # that the fragility curve is shifted
        ncon.data.cost_bch_hrdn = [
            ncon.data.cost_rate_hrdn * length if ncon.data.net.bch_type[i] == 1 else 0.0
            for i, length in enumerate(ncon.data.net.bch_length_km)
        ]  # cost per unit fragility curve shift
        # Note: only distribution line hardening is considered

        # 6.2) line repair cost
        rep_rate_tn = 1e2  # repair cost (£) per unit length (km) of line at transmission level
        rep_rate_dn = 1e2  # repair cost (£) per unit length (km) of line at distribution level
        ncon.data.cost_bch_rep = [
            rep_rate_dn * length if ncon.data.net.branch_level[i + 1] == 'D' else rep_rate_tn * length
            for i, length in enumerate(ncon.data.net.bch_length_km)
        ]  # Note: the line repair at both transmission and distribution level are considered

        # 6.3) line hardening limits and budget
        ncon.data.bch_hrdn_limits = [0.0, 30.0]  # in m/s
        ncon.data.budget_bch_hrdn = 1e8  # in £


        net = NetworkClass(ncon)
        net.name = name

        return net



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

        # active/reactive demand (in MW / MVAr) data
        ncon.data.net.Pd_max = [
            0.0,
            16.78 / 1000, 16.78 / 1000, 33.8 / 1000, 14.56 / 1000, 10.49 / 1000,
            8.821 / 1000, 14.35 / 1000, 19.31 / 1000, 14.35 / 1000, 16.27 / 1000,
            16.27 / 1000, 82.13 / 1000, 34.71 / 1000, 34.71 / 1000, 80.31 / 1000,
            49.62 / 1000, 49.62 / 1000, 43.77 / 1000, 37.32 / 1000, 37.32 / 1000,
            31.02 / 1000
        ]
        ncon.data.net.Pd_min = [0] * len(ncon.data.net.Pd_max)

        ncon.data.net.Qd_max = [
            0.0,
            20.91 / 1000, 20.91 / 1000, 37.32 / 1000, 12.52 / 1000, 14.21 / 1000,
            11.66 / 1000, 18.59 / 1000, 25.87 / 1000, 18.59 / 1000, 19.48 / 1000,
            19.48 / 1000, 71.65 / 1000, 30.12 / 1000, 30.12 / 1000, 70.12 / 1000,
            47.82 / 1000, 47.82 / 1000, 38.93 / 1000, 35.96 / 1000, 35.96 / 1000,
            29.36 / 1000
        ]
        ncon.data.net.Qd_min = [0] * len(ncon.data.net.Qd_max)

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

        net = NetworkClass(ncon)
        net.name = name
        return net


    else:
        raise ValueError(f"Unknown network preset '{name}'")
