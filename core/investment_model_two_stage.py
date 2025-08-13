import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.stats import lognorm
from core.config import InvestmentConfig
from factories.network_factory import make_network

from pathlib import Path
from datetime import datetime
import math
import json
import os

from factories.windstorm_factory import make_windstorm


class Object(object):
    pass


class InvestmentClass():

    def __init__(self, obj=None):
        # Get default values from InvestmentConfig
        if obj is None:
            obj = InvestmentConfig()

        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def build_investment_model(self,
                               network_name: str = "29_bus_GB_transmission_network_with_Kearsley_GSP_group",
                               windstorm_name: str = "windstorm_GB_transmission_network",
                               path_ws_scenario_library: str = "Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_100scn_s10000_filt_b10_h2_buf15.json",
                               include_normal_scenario: bool = True,
                               normal_scenario_prob: float = 0.99,
                               use_representative_days: bool = True,
                               representative_days: list = None,
                               resilience_metric_threshold: float = None,
                               invetment_budget: float = None,
                               solver_for_normal_opf: str = 'gurobi'):
        """
        Build a Pyomo MILP model for resilience enhancement investment planning (line hardening)
        against windstorms, using the form of stochastic programming over multiple scenarios.


        * 1st-stage:   line-hardening shift  Δv_l  ( identical for TN & DN )
        * 2nd-stage:   scenario-wise DC OPF on TN  +  Linearized AC OPF on DN
                       with windstorm-driven stochastic failures / repairs.
        """

        # ------------------------------------------------------------------
        # 0. Preliminaries
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Building Investment Planning Model...")
        print("=" * 60)

        self.network_name = network_name
        self.windstorm_name = windstorm_name

        net = make_network(self.network_name)
        ws = make_windstorm(self.windstorm_name)

        # 0.1 Compute Normal Operation Costs (if requested)
        normal_operation_opf_results = None

        if include_normal_scenario:
            print("\n" + "=" * 60)
            print("Computing Normal Operation Costs...")
            if use_representative_days:
                days = representative_days or [15, 105, 195, 285]
                print(f"Using representative days: {days}")
            else:
                print("Using full year (8760 hours)")
            print("=" * 60)

            # Build normal scenario OPF model
            normal_model, scale_factor = self.build_normal_scenario_opf_model(
                network_name=network_name,
                use_representative_days=use_representative_days,
                representative_days=representative_days
            )

            # Solve it
            normal_operation_opf_results = self.solve_normal_scenario_opf_model(
                normal_model,
                solver=solver_for_normal_opf,
                print_summary=True
            )

            # Extract timesteps before deleting model
            normal_scenario_hours = list(normal_model.Set_ts.data())

            # Clean up
            del normal_model
            print("Normal operation cost computation completed.\n")

        else:
            # Define timesteps when no normal scenario
            if use_representative_days:
                days = representative_days or [15, 105, 195, 285]
                timesteps = []
                for day in days:
                    start_hour = (day - 1) * 24 + 1  # 1-based absolute hours
                    timesteps.extend(range(start_hour, start_hour + 24))
                scale_factor = 365 / len(days)
            else:
                timesteps = list(range(1, 8761))  # 1..8760
                scale_factor = 1.0

        # 0.2 Read Windstorm Scenarios
        with open(path_ws_scenario_library) as f:
            data = json.load(f)

        if "metadata" in data:
            metadata = data["metadata"]
        else:
            raise ValueError("JSON does not contain 'metadata'")

        # Handle both old and new format
        if "scenarios" in data:
            # New library format: convert dictionary to list
            ws_scenarios = []
            for scenario_id, scenario_data in sorted(data["scenarios"].items()):
                # Add simulation_id for compatibility
                scenario_data["simulation_id"] = int(scenario_id.split("_")[1]) + 1  # ws_0000 → 1
                ws_scenarios.append(scenario_data)
        else:
            raise ValueError("JSON does not contain 'scenarios'")

        # Store windstorm infor into metadata
        self.meta = Object()
        self.meta.ws_seed = metadata.get("seed", metadata.get("base_seed", None))
        self.meta.n_ws_scenarios = metadata.get("number_of_ws_simulations",
                                                metadata.get("num_scenarios", len(ws_scenarios)))
        self.meta.library_type = metadata.get("library_type", "windstorm_scenarios")

        # Store normal scenario info into metadata
        if include_normal_scenario:
            self.meta.normal_scenario_included = True
            self.meta.normal_scenario_prob = normal_scenario_prob
            self.meta.normal_operation_opf_results = normal_operation_opf_results
        else:
            self.meta.normal_scenario_included = False
            self.meta.normal_scenario_prob = 0
            self.meta.normal_operation_opf_results = None

        # Only windstorm scenarios are in our scenario set
        Set_scn = [sim["simulation_id"] for sim in ws_scenarios]

        # Probability calculation
        if include_normal_scenario:
            # Windstorm scenarios share (1 - normal_prob)
            ws_total_prob = 1 - normal_scenario_prob
            scn_prob = {sc: ws_total_prob / len(Set_scn) for sc in Set_scn}
        else:
            # Only windstorm scenarios
            scn_prob = {sc: 1 / len(Set_scn) for sc in Set_scn}

        # ------------------------------------------------------------------
        # 1. Index sets -- tn for transmission level network, dn for distribution level network
        # ------------------------------------------------------------------
        print("\nCreating index sets...")

        # 1.1) Single sets
        # - Basic sets
        Set_bus = net.data.net.bus[:]
        Set_bch = list(range(1, len(net.data.net.bch) + 1))
        Set_gen = list(range(1, len(net.data.net.gen) + 1))
        Set_ess = list(range(1, len(net.data.net.ess) + 1))

        # - Separate TN and DN
        Set_bus_tn = [b for b in Set_bus if net.data.net.bus_level[b] == 'T']
        Set_bus_dn = [b for b in Set_bus if net.data.net.bus_level[b] == 'D']
        Set_bch_tn = [l for l in Set_bch if net.data.net.branch_level[l] == 'T' or
                      net.data.net.branch_level[l] == 'T-D']
        Set_bch_dn = [l for l in Set_bch if net.data.net.branch_level[l] == 'D']

        # - Further separate hardenable branches from branch set
        Set_bch_tn_lines = [l for l in Set_bch_tn
                            if net.data.net.bch_type[l - 1] == 1]  # hardenable branches (i.e., lines) at tn level

        Set_bch_dn_lines = [l for l in Set_bch_dn
                            if net.data.net.bch_type[l - 1] == 1]  # hardenable branches (i.e., lines) at dn level

        Set_bch_lines = Set_bch_tn_lines + Set_bch_dn_lines  # all hardenable branches

        Set_bch_tn_hrdn = [l for l in Set_bch_tn_lines
                           if net.data.net.bch_hardenable[l - 1] == 1]  # tn lines that are hardenable

        Set_bch_dn_hrdn = [l for l in Set_bch_dn_lines
                           if net.data.net.bch_hardenable[l - 1] == 1]  # dn lines that are hardenable

        Set_bch_hrdn_lines = Set_bch_tn_hrdn + Set_bch_dn_hrdn  # all lines that are hardenable

        # - Existing generators (note we have subset 'renewable' and 'non-renewable')
        Set_gen_exst = Set_gen

        has_renewables = (hasattr(net.data.net, 'profile_Pg_renewable') and
                          net.data.net.profile_Pg_renewable is not None and
                          hasattr(net.data.net, 'gen_type'))

        # Classify generators by type (for renewable handling)
        if has_renewables:
            Set_gen_exst_ren = []  # renewable existing generators
            Set_gen_exst_nren = []  # non-renewable existing generators

            for g in Set_gen:  # Note: using Set_gen, not model.Set_gen yet
                gen_idx = g - 1
                gen_bus = net.data.net.gen[gen_idx]

                if gen_bus in Set_bus_dn and net.data.net.gen_type[gen_idx] in ['wind', 'pv']:
                    Set_gen_exst_ren.append(g)
                else:
                    Set_gen_exst_nren.append(g)
        else:
            Set_gen_exst_ren = []
            Set_gen_exst_nren = Set_gen_exst  # All generators are non-renewable

        # - Existing ESS
        Set_ess_exst = Set_ess

        # Set of buses where DG can be installed
        Set_bus_dg_available = [b for b in Set_bus if net.data.net.dg_installation_availability[b - 1] == 1]
        Set_bus_ess_available = [b for b in Set_bus if net.data.net.ess_installation_availability[b - 1] == 1]

        # - New generators
        # Create virtual generator IDs for new DG installations (one per available bus)
        new_gen_to_bus_map = {}
        Set_gen_new_nren = []  # In accordance with network_factory.py, new DGs are all gas type (i.e., non-renewbale)

        for idx, b in enumerate(Set_bus_dg_available, start=1):
            Set_gen_new_nren.append(idx)
            new_gen_to_bus_map[idx] = b

        # Renewable new generators (empty for now, for future extension)
        Set_gen_new_ren = []  # Currently no renewable DG installations planned

        # Combined set of all new generators
        Set_gen_new = Set_gen_new_nren + Set_gen_new_ren

        # - New ESS
        # Create virtual ESS IDs for new ESS installations (one per available bus)
        new_ess_to_bus_map = {}
        Set_ess_new = []

        for idx, b in enumerate(Set_bus_ess_available, start=1):
            Set_ess_new.append(idx)
            new_ess_to_bus_map[idx] = b

        # scenario-specific timestep sets (absolute hour indices from the year)
        Set_ts_scn = {}

        for sim in ws_scenarios:
            sc_id = sim["simulation_id"]

            # Collect all hours affected by any windstorm event
            affected_hours = set()

            if "events" in sim:  # New format
                events = sim["events"]
                for event in events:
                    bgn_hr = event["bgn_hr"]  # Absolute hour in year (1-8760)
                    duration = event["duration"]

                    # Get max TTR for this event
                    if "bch_ttr" in event:
                        max_ttr = max(event["bch_ttr"])
                    else:
                        max_ttr = 0

                    # Add all affected hours (event + repair window)
                    for hr in range(bgn_hr, bgn_hr + duration + max_ttr):
                        if 1 <= hr <= 8760:  # Ensure within year bounds
                            affected_hours.add(hr)

            # Convert to sorted list
            Set_ts_scn[sc_id] = sorted(list(affected_hours))

            # If no events, include at least hour 1 to avoid empty set issues
            if not Set_ts_scn[sc_id]:
                Set_ts_scn[sc_id] = [1]

        # 1.2) Tuple sets:
        #  - st: scenario, timestep
        #  - sbt: scenario, bus, timestep
        #  - slt: scenario, branch, timestep
        #  - sgt: scenario, gen, timestep
        #  - set: scenario, ess, timestep
        Set_st = [(sc, t) for sc in Set_scn for t in Set_ts_scn[sc]]

        Set_sbt = [(sc, b, t) for sc in Set_scn
                   for b in net.data.net.bus
                   for t in Set_ts_scn[sc]]

        Set_sbt_tn = [(sc, b, t) for (sc, b, t) in Set_sbt if b in Set_bus_tn]

        Set_sbt_dn = [(sc, b, t) for (sc, b, t) in Set_sbt if b in Set_bus_dn]

        Set_slt_tn = [(sc, l, t) for sc in Set_scn
                      for l in Set_bch_tn
                      for t in Set_ts_scn[sc]]

        Set_slt_dn = [(sc, l, t) for sc in Set_scn
                      for l in Set_bch_dn
                      for t in Set_ts_scn[sc]]

        Set_slt_tn_lines = [(sc, l, t) for sc in Set_scn
                            for l in Set_bch_tn_lines
                            for t in Set_ts_scn[sc]]

        Set_slt_dn_lines = [(sc, l, t) for sc in Set_scn
                            for l in Set_bch_dn_lines
                            for t in Set_ts_scn[sc]]

        Set_slt_lines = Set_slt_tn_lines + Set_slt_dn_lines

        Set_sgt_exst = [(sc, g, t) for sc in Set_scn
                        for g in Set_gen_exst
                        for t in Set_ts_scn[sc]]

        Set_sgt_exst_ren = [(sc, g, t) for sc in Set_scn
                            for g in Set_gen_exst_ren
                            for t in Set_ts_scn[sc]]

        Set_sgt_exst_nren = [(sc, g, t) for sc in Set_scn
                             for g in Set_gen_exst_nren
                             for t in Set_ts_scn[sc]]

        Set_sgt_dn_exst = [
            (sc, g, t)
            for (sc, g, t) in Set_sgt_exst
            if net.data.net.bus_level[net.data.net.gen[g - 1]] == 'D'
        ]

        Set_sgt_new = [(sc, g, t) for sc in Set_scn
                       for g in Set_gen_new
                       for t in Set_ts_scn[sc]]

        Set_sgt_new_ren = [(sc, g, t) for sc in Set_scn
                           for g in Set_gen_new_ren
                           for t in Set_ts_scn[sc]]

        Set_sgt_new_nren = [(sc, g, t) for sc in Set_scn
                            for g in Set_gen_new_nren
                            for t in Set_ts_scn[sc]]

        Set_sgt_dn_new = [(sc, g, t) for sc in Set_scn
                          for g in Set_gen_new
                          for t in Set_ts_scn[sc]
                          if new_gen_to_bus_map[g] in Set_bus_dn]

        Set_set_exst = [(sc, e, t) for sc in Set_scn
                        for e in Set_ess_exst
                        for t in Set_ts_scn[sc]]

        Set_set_new = [(sc, e, t) for sc in Set_scn
                       for e in Set_ess_new
                       for t in Set_ts_scn[sc]]

        # ------------------------------------------------------------------
        # 2. Initialize Pyomo sets
        # ------------------------------------------------------------------
        print("Initializing Pyomo model and sets...")

        model = pyo.ConcreteModel()

        model.Set_scn = pyo.Set(initialize=Set_scn)
        model.Set_bus = pyo.Set(initialize=Set_bus)
        model.Set_bus_tn = pyo.Set(initialize=Set_bus_tn)
        model.Set_bus_dn = pyo.Set(initialize=Set_bus_dn)
        model.Set_bch = pyo.Set(initialize=Set_bch)
        model.Set_bch_tn = pyo.Set(initialize=Set_bch_tn)
        model.Set_bch_dn = pyo.Set(initialize=Set_bch_dn)
        model.Set_bch_tn_lines = pyo.Set(initialize=Set_bch_tn_lines)
        model.Set_bch_dn_lines = pyo.Set(initialize=Set_bch_dn_lines)
        model.Set_bch_lines = pyo.Set(initialize=Set_bch_lines)
        model.Set_bch_hrdn_lines = pyo.Set(initialize=Set_bch_hrdn_lines)
        model.Set_gen = pyo.Set(initialize=Set_gen)
        model.Set_gen_exst = pyo.Set(initialize=Set_gen_exst)
        model.Set_gen_exst_ren = pyo.Set(initialize=Set_gen_exst_ren)
        model.Set_gen_exst_nren = pyo.Set(initialize=Set_gen_exst_nren)
        model.Set_gen_new = pyo.Set(initialize=Set_gen_new)
        model.Set_gen_new_ren = pyo.Set(initialize=Set_gen_new_ren)
        model.Set_gen_new_nren = pyo.Set(initialize=Set_gen_new_nren)
        model.Set_ess = pyo.Set(initialize=Set_ess)
        model.Set_ess_exst = pyo.Set(initialize=Set_ess_exst)
        model.Set_ess_new = pyo.Set(initialize=Set_ess_new)
        model.Set_bus_dg_available = pyo.Set(initialize=Set_bus_dg_available)
        model.Set_bus_ess_available = pyo.Set(initialize=Set_bus_ess_available)

        model.Set_ts_scn = {sc: pyo.Set(initialize=Set_ts_scn[sc])
                            for sc in Set_scn}

        model.Set_st = pyo.Set(initialize=Set_st, dimen=2)
        model.Set_sbt = pyo.Set(initialize=Set_sbt, dimen=3)
        model.Set_sbt_dn = pyo.Set(initialize=Set_sbt_dn, dimen=3)
        model.Set_sbt_tn = pyo.Set(initialize=Set_sbt_tn, dimen=3)
        model.Set_slt_tn = pyo.Set(initialize=Set_slt_tn, dimen=3)
        model.Set_slt_dn = pyo.Set(initialize=Set_slt_dn, dimen=3)
        model.Set_slt_tn_lines = pyo.Set(initialize=Set_slt_tn_lines, dimen=3)
        model.Set_slt_dn_lines = pyo.Set(initialize=Set_slt_dn_lines, dimen=3)
        model.Set_slt_lines = pyo.Set(initialize=Set_slt_lines, dimen=3)

        # Generator tuple sets - EXISTING
        model.Set_sgt_exst = pyo.Set(initialize=Set_sgt_exst, dimen=3)
        model.Set_sgt_exst_ren = pyo.Set(initialize=Set_sgt_exst_ren, dimen=3)
        model.Set_sgt_exst_nren = pyo.Set(initialize=Set_sgt_exst_nren, dimen=3)
        model.Set_sgt_dn_exst = pyo.Set(initialize=Set_sgt_dn_exst, dimen=3)

        # Generator tuple sets - NEW
        model.Set_sgt_new = pyo.Set(initialize=Set_sgt_new, dimen=3)
        model.Set_sgt_new_ren = pyo.Set(initialize=Set_sgt_new_ren, dimen=3)  # Empty is OK
        model.Set_sgt_new_nren = pyo.Set(initialize=Set_sgt_new_nren, dimen=3)
        model.Set_sgt_dn_new = pyo.Set(initialize=Set_sgt_dn_new, dimen=3)

        # ESS tuple sets
        model.Set_set_exst = pyo.Set(initialize=Set_set_exst, dimen=3)
        model.Set_set_new = pyo.Set(initialize=Set_set_new, dimen=3)


        # ------------------------------------------------------------------
        # 3. Parameters
        # ------------------------------------------------------------------
        print("Setting up model parameters...")

        # 3.1) Resilience level threshold
        model.resilience_metric_threshold = pyo.Param(
            initialize=(resilience_metric_threshold
                        if resilience_metric_threshold is not None
                        else float('inf')),
            mutable=False
        )

        # 3.2) Scenario probability
        model.scn_prob = pyo.Param(model.Set_scn, initialize=scn_prob)

        # 3.3) Network static data
        # - Base value
        model.base_MVA = pyo.Param(initialize=net.data.net.base_MVA)

        # 3.4) Demand profiles (Pd, Qd) using absolute hours
        model.Pd = pyo.Param(model.Set_sbt,
                             initialize=lambda m, sc, b, t: net.data.net.profile_Pd[b - 1][t - 1])

        model.Qd = pyo.Param(model.Set_sbt_dn,
                             initialize=lambda m, sc, b, t: net.data.net.profile_Qd[b - 1][t - 1],
                             default=0.0)

        # 3.5) Voltage limits (squared)
        model.V2_min = pyo.Param(model.Set_bus_dn,
                                 initialize={b: net.data.net.V_min[net.data.net.bus.index(b)] ** 2
                                             for b in Set_bus_dn})
        model.V2_max = pyo.Param(model.Set_bus_dn,
                                 initialize={b: net.data.net.V_max[net.data.net.bus.index(b)] ** 2
                                             for b in Set_bus_dn})

        # 3.6) Branch constants
        model.B_tn = pyo.Param(model.Set_bch_tn,
                               initialize={l: 1.0 / net.data.net.bch_X[l - 1] for l in model.Set_bch_tn})
        model.Pmax_tn = pyo.Param(model.Set_bch_tn,
                                  initialize={l: net.data.net.bch_Pmax[l - 1] for l in model.Set_bch_tn})
        model.R_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: net.data.net.bch_R[l - 1] for l in model.Set_bch_dn})
        model.X_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: net.data.net.bch_X[l - 1] for l in model.Set_bch_dn})
        model.Smax_dn = pyo.Param(model.Set_bch_dn,
                                  initialize={l: net.data.net.bch_Smax[l - 1] for l in model.Set_bch_dn})

        # 3.7) Existing Generator Parameters
        # - Generation cost coefficients (c0 + c1·P)
        coef_len = len(net.data.net.gen_cost_coef[0])
        model.gen_cost_coef_exst = pyo.Param(
            model.Set_gen_exst, range(coef_len),
            initialize={(g, c): net.data.net.gen_cost_coef[g - 1][c]
                        for g in Set_gen_exst for c in range(coef_len)}
        )

        # - Generation limits
        model.Pg_max_exst = pyo.Param(
            model.Set_gen_exst,
            initialize={g: net.data.net.Pg_max_exst[g - 1] for g in Set_gen_exst}
        )
        model.Pg_min_exst = pyo.Param(
            model.Set_gen_exst,
            initialize={g: net.data.net.Pg_min_exst[g - 1] for g in Set_gen_exst}
        )
        model.Qg_max_exst = pyo.Param(
            model.Set_gen_exst,
            initialize={g: net.data.net.Qg_max_exst[g - 1] for g in Set_gen_exst}
        )
        model.Qg_min_exst = pyo.Param(
            model.Set_gen_exst,
            initialize={g: net.data.net.Qg_min_exst[g - 1] for g in Set_gen_exst}
        )

        # - Renewable generation availability (time-varying)
        if has_renewables and model.Set_gen_exst_ren:
            # Get unique windstorm hours across all scenarios
            windstorm_hours = set()
            for sc in Set_scn:
                windstorm_hours.update(Set_ts_scn[sc])
            windstorm_hours = sorted(list(windstorm_hours))

            # Renewable availability for windstorm hours only
            renewable_availability = {}
            for g in model.Set_gen_exst_ren:
                gen_idx = g - 1
                profile = net.data.net.profile_Pg_renewable[gen_idx]
                for hr in windstorm_hours:
                    renewable_availability[(g, hr)] = \
                        profile[hr - 1] if profile else net.data.net.Pg_max_exst[gen_idx]

            model.Pg_renewable_avail = pyo.Param(
                model.Set_gen_exst_ren, windstorm_hours,
                initialize=renewable_availability,
                mutable=False
            )

        # 3.8) New (installed) generator parameters
        if Set_gen_new_nren:
            # - Generation cost for new DGs
            model.gen_cost_coef_new = pyo.Param(
                model.Set_gen_new_nren, range(2),  # Assuming [c0, c1] format
                initialize={(g, c): net.data.net.new_dg_gen_cost_coef[new_gen_to_bus_map[g] - 1][c]
                            for g in Set_gen_new_nren for c in range(2)}
            )

            # - Reactive power ratio for new DGs
            model.new_dg_q_p_ratio = pyo.Param(model.Set_bus_dg_available,
                                               initialize={b: net.data.net.new_dg_q_p_ratio[b - 1]
                                                           for b in Set_bus_dg_available},
                                               within=pyo.NonNegativeReals)

            model.dg_install_capacity_min = pyo.Param(model.Set_gen_new_nren,
                                                      initialize={g: net.data.net.dg_install_capacity_min[
                                                          new_gen_to_bus_map[g] - 1]
                                                                  for g in Set_gen_new_nren})
            model.dg_install_capacity_max = pyo.Param(model.Set_gen_new_nren,
                                                      initialize={g: net.data.net.dg_install_capacity_max[
                                                          new_gen_to_bus_map[g] - 1]
                                                                  for g in Set_gen_new_nren})

        # 3.9) Existing ESS Parameters
        if Set_ess_exst:
            model.Pess_max_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Pess_max_exst[e - 1] for e in Set_ess_exst})
            model.Pess_min_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Pess_min_exst[e - 1] for e in Set_ess_exst})
            model.Eess_max_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Eess_max_exst[e - 1] for e in Set_ess_exst})
            model.Eess_min_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Eess_min_exst[e - 1] for e in Set_ess_exst})
            model.SOC_max_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.SOC_max_exst[e - 1] for e in Set_ess_exst})
            model.SOC_min_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.SOC_min_exst[e - 1] for e in Set_ess_exst})
            model.eff_ch_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.eff_ch_exst[e - 1] for e in Set_ess_exst})
            model.eff_dis_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.eff_dis_exst[e - 1] for e in Set_ess_exst})
            model.initial_SOC_exst = pyo.Param(model.Set_ess_exst,
                                           initialize={e: net.data.net.initial_SOC_exst[e - 1] for e in Set_ess_exst})

        # 3.10) New ESS Parameters
        if Set_ess_new:
            model.ess_install_capacity_min = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.ess_install_capacity_min[new_ess_to_bus_map[e] - 1]
                            for e in Set_ess_new})

            model.ess_install_capacity_max = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.ess_install_capacity_max[new_ess_to_bus_map[e] - 1]
                            for e in Set_ess_new})

            model.ess_power_energy_ratio_new = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.ess_power_energy_ratio_new[new_ess_to_bus_map[e] - 1]
                            for e in Set_ess_new})

            model.SOC_max_new = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.SOC_max_new[new_ess_to_bus_map[e] - 1] for e in Set_ess_new})

            model.SOC_min_new = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.SOC_min_new[new_ess_to_bus_map[e] - 1] for e in Set_ess_new})

            model.eff_ch_new = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.eff_ch_new[new_ess_to_bus_map[e] - 1] for e in Set_ess_new})

            model.eff_dis_new = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.eff_dis_new[new_ess_to_bus_map[e] - 1] for e in Set_ess_new})

            model.initial_SOC_new = pyo.Param(model.Set_ess_new,
                initialize={e: net.data.net.initial_SOC_new[new_ess_to_bus_map[e] - 1] for e in Set_ess_new})

        # 3.11) Cost-related parameters
        # - load shedding (curtailment), grid import/export, repair, hardening costs
        model.Pc_cost = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                  initialize={b: net.data.net.Pc_cost[net.data.net.bus.index(b)]
                                              for b in net.data.net.bus})
        model.Qc_cost = pyo.Param(model.Set_bus_dn,
                                  initialize={b: net.data.net.Qc_cost[net.data.net.bus.index(b)]
                                              for b in Set_bus_dn})
        model.Pimp_cost = pyo.Param(initialize=net.data.net.Pimp_cost)
        model.Pexp_cost = pyo.Param(initialize=net.data.net.Pexp_cost)

        model.rep_cost = pyo.Param(model.Set_bch_tn | model.Set_bch_dn,
                                   initialize={l: net.data.net.repair_cost[l - 1]
                                               for l in range(1, len(net.data.net.bch) + 1)},
                                   within=pyo.NonNegativeReals)
        model.line_hrdn_cost = pyo.Param(model.Set_bch_tn | model.Set_bch_dn,
                                         initialize={l: net.data.net.hrdn_cost[l - 1]
                                                     for l in range(1, len(net.data.net.bch) + 1)},
                                         within=pyo.NonNegativeReals)
        model.dg_install_cost = pyo.Param(model.Set_gen_new_nren,
                                          initialize={g: net.data.net.dg_install_cost[new_gen_to_bus_map[g] - 1]
                                                      for g in Set_gen_new_nren})
        model.ess_install_cost = pyo.Param(model.Set_ess_new,
                                           initialize={e: net.data.net.ess_install_cost[new_ess_to_bus_map[e] - 1]
                                                       for e in Set_ess_new})
        model.investment_budget = pyo.Param(initialize=net.data.investment_budget
                                                       if net.data.investment_budget else 1e10)

        # 3.12) Other parameters
        # Asset lifetime (the number of years that operational cost will be computed based on)
        model.asset_lifetime = pyo.Param(
            initialize=25,  # Adjust as needed
            mutable=False,
            doc="Asset lifetime in years"
        )

        # Discount rate (currently it is assumed to be fixed across the asset lifetime)
        model.discount_rate = pyo.Param(
            initialize=0.05,  # 5% discount rate
            mutable=False,
            doc="Annual discount rate (constant)"
        )

        # timestep length:
        model.dt = pyo.Param(initialize=1.0, mutable=False)  # Hourly timestep

        # Big-M for binary constraints
        model.BigM = pyo.Param(initialize=1e4, mutable=False)

        # 3.13) windstorm stochastic data  (gust, rand, impact, ttr)
        gust_dict = {}
        rand_dict = {}
        impact_dict = {}
        ttr_dict = {}

        for sim in ws_scenarios:
            sc = sim["simulation_id"]

            # Initialize default values for all hours in Set_ts_scn
            for t in Set_ts_scn[sc]:
                gust_dict[(sc, t)] = 0  # No wind - 2D indexing!

            for l in range(1, len(net.data.net.bch) + 1):
                for t in Set_ts_scn[sc]:
                    rand_dict[(sc, l, t)] = 1.0  # No failure (rand > any threshold)
                    impact_dict[(sc, l, t)] = 0  # Not impacted

            # Overwrite with actual event data where events occur
            for event in sim.get("events", []):
                bgn_hr = event["bgn_hr"]
                duration = event["duration"]

                for hr_in_event in range(duration):
                    abs_hr = bgn_hr + hr_in_event  # Absolute hour

                    if abs_hr in Set_ts_scn[sc]:  # Should always be true
                        # Gust speed (same for all branches) - 2D indexing!
                        if isinstance(event["gust_speed"], list):
                            gust_dict[(sc, abs_hr)] = event["gust_speed"][hr_in_event]
                        else:
                            gust_dict[(sc, abs_hr)] = event["gust_speed"]

                        # Update all branches for this hour
                        for l in range(1, len(net.data.net.bch) + 1):
                            l_idx = l - 1  # Convert to 0-based for array indexing

                            # Branch-specific data
                            if "bch_rand_nums" in event:
                                rand_dict[(sc, l, abs_hr)] = event["bch_rand_nums"][l_idx][hr_in_event]

                            if "flgs_impacted_bch" in event:
                                impact_dict[(sc, l, abs_hr)] = event["flgs_impacted_bch"][l_idx][hr_in_event]

                # TTR (max across all events for each branch)
                if "bch_ttr" in event:
                    for l in range(1, len(net.data.net.bch) + 1):
                        key = (sc, l)
                        current_ttr = ttr_dict.get(key, 0)
                        ttr_dict[key] = max(current_ttr, event["bch_ttr"][l - 1])

        model.gust_speed = pyo.Param(model.Set_st,
                                     initialize=gust_dict)
        model.rand_num = pyo.Param(model.Set_slt_tn | model.Set_slt_dn,
                                   initialize=rand_dict)
        model.impacted_branches = pyo.Param(model.Set_slt_tn | model.Set_slt_dn,
                                            initialize=impact_dict, within=pyo.Binary)
        model.branch_ttr = pyo.Param(model.Set_scn, model.Set_bch_tn | model.Set_bch_dn,
                                     initialize=ttr_dict)

        # Create a parameter to store the absolute starting hour of each ws scenario
        # (useful for defining certain constraints later)
        first_timestep_dict = {sc: min(Set_ts_scn[sc]) for sc in Set_scn}
        model.first_timestep = pyo.Param(model.Set_scn,
                                         initialize=first_timestep_dict,
                                         doc="First timestep for each ws scenario")


        # ------------------------------------------------------------------
        # 4. Variables
        # ------------------------------------------------------------------
        print("Defining decision variables...")

        # 4.1) First-stage decision variables:
        # - Line hardening (continuous, m/s)
        model.line_hrdn = pyo.Var(model.Set_bch_hrdn_lines,
                                  bounds=(net.data.bch_hrdn_limits[0], net.data.bch_hrdn_limits[1]),
                                  within=pyo.NonNegativeReals)

        # - (Gas) DG installation capacity (continuous, MW)
        model.dg_install_capacity = pyo.Var(
            model.Set_gen_new_nren,
            within=pyo.NonNegativeReals,
            bounds=lambda model, g: (model.dg_install_capacity_min[g],
                                     model.dg_install_capacity_max[g])
        )

        # - ESS installation power capacity (continuous, MW)
        # note: as the power-to-energy ratio for installed ESS is fixed, determining the power capacity is enough.
        model.ess_install_capacity = pyo.Var(
            model.Set_ess_new,
            within=pyo.NonNegativeReals,
            bounds=lambda model, e: (model.ess_install_capacity_min[e],
                                     model.ess_install_capacity_max[e])
        )

        # 4.2) Second-stage recourse variables  (indexed by scenario)
        # Before defining variables, first check if the model include reactive power demand
        temp_qd_dict = {}  # need to build Qd_dict early
        for sim in ws_scenarios:
            sc = sim["simulation_id"]
            for abs_hr in Set_ts_scn[sc]:  # Use absolute hour directly
                for b in Set_bus_dn:
                    temp_qd_dict[(sc, b, abs_hr)] = net.data.net.profile_Qd[b - 1][abs_hr - 1]

        has_reactive_demand = any(v > 0 for v in temp_qd_dict.values())
        print(f"Reactive demand detected: {has_reactive_demand}")

        # - Generation of existing generators
        # from existing generators:
        model.Pg_exst = pyo.Var(
            model.Set_sgt_exst,
            within=pyo.NonNegativeReals,
            bounds=lambda model, sc, g, t: (model.Pg_min_exst[g], model.Pg_max_exst[g])
        )
        if has_reactive_demand:
            model.Qg_exst = pyo.Var(
                model.Set_sgt_exst,
                within=pyo.Reals,
                bounds=lambda model, sc, g, t: (model.Qg_min_exst[g], model.Qg_max_exst[g])
            )
        else:
            model.Qg_exst = pyo.Param(model.Set_sgt_exst, default=0.0)

        # - Generation of new DGs:
        # Pg_new limited via constraints later
        model.Pg_new = pyo.Var(model.Set_sgt_new_nren, within=pyo.NonNegativeReals)

        if has_reactive_demand:
            # Qg_new limited via constraints later
            model.Qg_new = pyo.Var(model.Set_sgt_new_nren, within=pyo.Reals)
        else:
            model.Qg_new = pyo.Param(model.Set_sgt_new_nren, default=0.0)

        # - ESS operations of existing ESS
        model.Pess_charge_exst = pyo.Var(
            model.Set_set_exst,
            within=pyo.NonNegativeReals,
            bounds=lambda model, sc, e, t: (0, model.Pess_max_exst[e])
        )
        model.Pess_discharge_exst = pyo.Var(
            model.Set_set_exst,
            within=pyo.NonNegativeReals,
            bounds=lambda model, sc, e, t: (0, model.Pess_max_exst[e])
        )
        model.Eess_exst = pyo.Var(
            model.Set_set_exst,
            within=pyo.NonNegativeReals,
            bounds=lambda model, sc, e, t: (model.Eess_min_exst[e], model.Eess_max_exst[e])
        )
        model.SOC_exst = pyo.Var(
            model.Set_set_exst,
            within=pyo.NonNegativeReals,
            bounds=lambda model, sc, e, t: (model.SOC_min_exst[e], model.SOC_max_exst[e])
        )
        model.ess_charge_binary_exst = pyo.Var(model.Set_set_exst, within=pyo.Binary)
        model.ess_discharge_binary_exst = pyo.Var(model.Set_set_exst, within=pyo.Binary)

        # - ESS operations of new ESS
        model.Pess_charge_new = pyo.Var(model.Set_set_new, within=pyo.NonNegativeReals)
        model.Pess_discharge_new = pyo.Var(model.Set_set_new, within=pyo.NonNegativeReals)
        model.Eess_new = pyo.Var(model.Set_set_new, within=pyo.NonNegativeReals)
        model.SOC_new = pyo.Var(
            model.Set_set_new,
            within=pyo.NonNegativeReals,
            bounds=lambda model, sc, e, t: (model.SOC_min_new[e], model.SOC_max_new[e])
        )
        model.ess_charge_binary_new = pyo.Var(model.Set_set_new, within=pyo.Binary)
        model.ess_discharge_binary_new = pyo.Var(model.Set_set_new, within=pyo.Binary)

        # - Bus state
        model.theta = pyo.Var(model.Set_sbt, within=pyo.Reals)
        model.V2_dn = pyo.Var(model.Set_sbt_dn, within=pyo.NonNegativeReals)

        # - Power flows
        model.Pf_tn = pyo.Var(model.Set_slt_tn, within=pyo.Reals)
        model.Pf_dn = pyo.Var(model.Set_slt_dn, within=pyo.Reals)
        if has_reactive_demand:
            model.Qf_dn = pyo.Var(model.Set_slt_dn, within=pyo.Reals)
        else:
            model.Qf_dn = pyo.Param(model.Set_slt_dn, default=0.0)

        # - Load sheddings (curtailed load)
        model.Pc = pyo.Var(model.Set_sbt, within=pyo.NonNegativeReals)
        if has_reactive_demand:
            model.Qc = pyo.Var(model.Set_sbt_dn, within=pyo.NonNegativeReals)
        else:
            model.Qc = pyo.Param(model.Set_sbt_dn, default=0.0)

        # - Grid electricity import/export
        model.Pimp = pyo.Var(model.Set_st, within=pyo.NonNegativeReals)
        model.Pexp = pyo.Var(model.Set_st, within=pyo.NonNegativeReals)

        # Note: sice we use DC power flow at TN level, there's no need to define Qimp and Qexp

        # - branch status
        model.branch_status = pyo.Var(model.Set_slt_tn | model.Set_slt_dn, within=pyo.Binary)

        # - failure logic related (both tn and dn line failures are considered) (only 'lines' are failable)
        model.shifted_gust_speed = pyo.Var(model.Set_slt_lines,  # or Set_slt / Set_bch, depending on the block
                                           within=pyo.NonNegativeReals,  # keep ≥ 0
                                           bounds=(0, 120))  # upper bound ≥ largest breakpoint
        model.fail_prob = pyo.Var(model.Set_slt_lines,  #
                                  within=pyo.NonNegativeReals, bounds=(0, 1))
        model.fail_condition = pyo.Var(model.Set_slt_lines, within=pyo.Binary)
        model.fail_indicator = pyo.Var(model.Set_slt_lines, within=pyo.Binary)
        model.fail_applies = pyo.Var(model.Set_slt_lines, within=pyo.Binary)
        model.repair_applies = pyo.Var(model.Set_slt_lines, within=pyo.Binary)

        # ------------------------------------------------------------------
        # 5. Constraints
        # ------------------------------------------------------------------
        print("Building constraints...")

        print("  - Investment budget constraints...")

        # 5.1) Budget
        def investment_budget_rule(model):
            """Total investment cost must not exceed budget"""
            total_line_hrdn_cost = sum(model.line_hrdn[l] * model.line_hrdn_cost[l] for l in model.Set_bch_hrdn_lines)

            total_dg_install_cost = sum(model.dg_install_capacity[g] * model.dg_install_cost[g]
                                        for g in model.Set_gen_new_nren)

            total_ess_install_cost = sum(model.ess_install_capacity[e] * model.ess_install_cost[e]
                                         for e in model.Set_ess_new)

            return total_line_hrdn_cost + total_dg_install_cost + total_ess_install_cost <= model.investment_budget

        model.Constraint_InvestmentBudget = pyo.Constraint(rule=investment_budget_rule)

        # 5.2) Shifted gust speed (i.e. Line hardening) -- Note: only dn lines hardening are considered
        def shifted_gust_rule(model, sc, l, t):
            # line hardening amount can be non-zero for only lines
            hrdn = model.line_hrdn[l] if l in model.Set_bch_hrdn_lines else 0
            return model.shifted_gust_speed[sc, l, t] >= model.gust_speed[sc, t] - hrdn

        model.Constraint_ShiftedGust = pyo.Constraint(model.Set_slt_lines, rule=shifted_gust_rule)

        print("  - Piecewise fragility constraints...")
        # 5.3) Piece-wise fragility
        # compute linearized fragility data
        fragility_data = self.piecewise_linearize_fragility(net, line_idx=Set_bch_lines, num_pieces=6)

        gust_speeds = fragility_data["gust_speeds"]
        gust_index = {x: i for i, x in enumerate(gust_speeds)}

        def fragility_rule(model, sc, l, t, x):
            return fragility_data["fail_probs"][l][gust_index[x]]

        model.Piecewise_Fragility = pyo.Piecewise(
            model.Set_slt_lines,
            model.fail_prob,
            model.shifted_gust_speed,
            pw_pts=gust_speeds,
            f_rule=fragility_rule,
            pw_constr_type="EQ",
            pw_repn="DCC")

        print("  - Failure logic constraints...")
        # 5.4) Failure logic constraints (unchanged from the original version)

        # - If a branch is not failable, its status must be kept as 1
        def fix_nonline_status(model, sc, l, t):
            # net.data.net.bch_type is 1 for real lines, 0 for transformers/couplers
            if net.data.net.bch_type[l - 1] == 0:
                return model.branch_status[sc, l, t] == 1
            else:
                return pyo.Constraint.Skip

        model.Constraint_FixNonlineStatus = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn,
                                                           rule=fix_nonline_status)

        # - If the random number falls within the failure probability (rand_num <= fail_prob), the failure condition is
        #   met (fail_condition == 1), otherwise failure condition is not met (fail_condition == 0)
        def fail_cond_1(model, sc, l, t):
            return model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] <= model.BigM * model.fail_condition[sc, l, t]

        def fail_cond_2(model, sc, l, t):
            return (model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] >= (model.fail_condition[sc, l, t] - 1)
                                                                             * model.BigM)

        model.Constraint_FailCond1 = pyo.Constraint(model.Set_slt_lines, rule=fail_cond_1)
        model.Constraint_FailCond2 = pyo.Constraint(model.Set_slt_lines, rule=fail_cond_2)

        # - If failure condition is met (fail_condition == 1) and the branch is impacted (impacted_branch == 1),
        #   the failure indicator is on (fail_indicator == 1), otherwise it is off (fail_indicator == 0)
        def fail_ind_1(model, sc, l, t):
            return model.fail_indicator[sc, l, t] <= model.fail_condition[sc, l, t]

        def fail_ind_2(model, sc, l, t):
            return model.fail_indicator[sc, l, t] <= model.impacted_branches[sc, l, t]

        def fail_ind_3(model, sc, l, t):
            return model.fail_indicator[sc, l, t] >= model.fail_condition[sc, l, t] + \
                model.impacted_branches[sc, l, t] - 1

        model.Constraint_FailInd1 = pyo.Constraint(model.Set_slt_lines, rule=fail_ind_1)
        model.Constraint_FailInd2 = pyo.Constraint(model.Set_slt_lines, rule=fail_ind_2)
        model.Constraint_FailInd3 = pyo.Constraint(model.Set_slt_lines, rule=fail_ind_3)

        # - If failure indicator is on (fail_indicator == 1) and the branch is operational at the last timestep
        #   (branch_status[t-1] == 1), the branch fails (fail_applies == 1, and branch_status[t] == 0), otherwise
        #   nothing happens (fail_applies == 0)
        def fail_activation_rule_1(model, sc, l, t):
            """ fail_applies can be 1 only if both branch_status at timestep 't-1' is 1 """
            # Skip constraint for first timestep (assume branches start operational)
            if t == model.first_timestep[sc]:
                return pyo.Constraint.Skip
            return model.fail_applies[sc, l, t] <= model.branch_status[sc, l, t - 1]

        def fail_activation_rule_2(model, sc, l, t):
            """ fail_applies can be 1 only if fail_indicator is 1 """
            if t == model.first_timestep[sc]:
                return pyo.Constraint.Skip
            return model.fail_applies[sc, l, t] <= model.fail_indicator[sc, l, t] + (
                    1 - model.branch_status[sc, l, t - 1])

        def fail_activation_rule_3(model, sc, l, t):
            """ If both conditions are met, fail_applies must be 1 """
            if t == model.first_timestep[sc]:
                return pyo.Constraint.Skip
            return model.fail_applies[sc, l, t] >= model.fail_indicator[sc, l, t] - (
                    1 - model.branch_status[sc, l, t - 1])

        model.Constraint_FailActivation1 = pyo.Constraint(model.Set_slt_lines, rule=fail_activation_rule_1)
        model.Constraint_FailActivation2 = pyo.Constraint(model.Set_slt_lines, rule=fail_activation_rule_2)
        model.Constraint_FailActivation3 = pyo.Constraint(model.Set_slt_lines, rule=fail_activation_rule_3)

        # - If failure applies (fail_applies == 1), the branch status at this timestep must be failed
        #   (branch_status[t] == 0)
        def immediate_failure_rule(model, sc, l, t):
            return model.branch_status[sc, l, t] <= 1 - model.fail_applies[sc, l, t]

        model.Constraint_ImmediateFailure = pyo.Constraint(model.Set_slt_lines, rule=immediate_failure_rule)

        # - If the branch status is operational (branch_status == 1), there must not be no failure applied during the
        #   last 'ttr' hours (fail_applies[t'] == 0, for t' from (t - ttr) to t)
        def failure_persistence_rule(model, sc, l, t):
            window_start = max(model.first_timestep[sc], t - model.branch_ttr[sc, l] + 1)
            # Only sum over timesteps that exist in the scenario
            return (model.branch_status[sc, l, t]
                    + sum(model.fail_applies[sc, l, τ]
                          for τ in Set_ts_scn[sc]
                          if window_start <= τ <= t)
                    ) <= 1

        model.Constraint_FailurePersistence = pyo.Constraint(model.Set_slt_lines, rule=failure_persistence_rule)

        # - Repair must happen exactly ttr timesteps after failure
        def repair_timing_rule(model, sc, l, t):
            """
            If a failure occurred exactly ttr timesteps ago, repair must happen now
            (assuming we're still within the time horizon)
            """
            ttr = model.branch_ttr[sc, l]
            target_t = t - ttr

            # Check if the target timestep exists in the scenario
            if target_t in Set_ts_scn[sc]:
                # If there was a failure exactly ttr steps ago, we must repair now
                return model.repair_applies[sc, l, t] >= model.fail_applies[sc, l, target_t]
            else:
                # If the target timestep doesn't exist, skip this constraint
                return pyo.Constraint.Skip

        model.Constraint_RepairTiming = pyo.Constraint(model.Set_slt_lines, rule=repair_timing_rule)

        # - Repair cannot happen before ttr timesteps have passed since failure
        def no_early_repair_rule(model, sc, l, t):
            """
            Repair cannot happen if there was no failure exactly ttr timesteps ago
            This prevents repairs from happening too early or without a failure
            """
            ttr = model.branch_ttr[sc, l]
            target_t = t - ttr

            # Check if we're far enough into the scenario to look back ttr steps
            if target_t >= model.first_timestep[sc]:
                # Sum of all failures in the window (t-ttr+1, t-1)
                # Only consider timesteps that exist in the scenario
                recent_failures = sum(
                    model.fail_applies[sc, l, tau]
                    for tau in Set_ts_scn[sc]
                    if max(model.first_timestep[sc], t - ttr + 1) <= tau < t
                )
                return model.repair_applies[sc, l, t] <= 1 - recent_failures
            else:
                # Can't repair in the first ttr timesteps (no failure could have occurred ttr ago)
                return model.repair_applies[sc, l, t] == 0

        model.Constraint_NoEarlyRepair = pyo.Constraint(model.Set_slt_lines, rule=no_early_repair_rule)

        # - Repair cannot happen if the line is already operational
        def no_repair_if_operational_rule(model, sc, l, t):
            """
            If a line is operational at t-1, it cannot be repaired at t (only failed lines can be repaired)
            """
            if t == model.first_timestep[sc]:  # Use the parameter instead of hardcoded 1
                # No repairs at first timestep (all lines start operational)
                return model.repair_applies[sc, l, t] == 0
            else:
                return model.repair_applies[sc, l, t] <= 1 - model.branch_status[sc, l, t - 1]

        model.Constraint_NoRepairIfOperational = pyo.Constraint(model.Set_slt_lines, rule=no_repair_if_operational_rule)

        # - A Branch has to keep its status at last timestep unless failure or repair happens
        def bch_status_transition_rule(model, sc, l, t):
            # The scenario starts with all lines operational
            if t == model.first_timestep[sc]:
                return model.branch_status[sc, l, t] == 1
            return (model.branch_status[sc, l, t] == model.branch_status[sc, l, t - 1] - model.fail_applies[sc, l, t]
                    + model.repair_applies[sc, l, t])

        model.Constraint_BranchStatusTransition = pyo.Constraint(
            model.Set_slt_lines, rule=bch_status_transition_rule)

        # - Branch failure and repair cannot happen simultaneously
        def bch_fail_rep_exclusivity(model, sc, l, t):
            return model.fail_applies[sc, l, t] + model.repair_applies[sc, l, t] <= 1

        model.Constraint_FailureRepairExclusivity = pyo.Constraint(
            model.Set_slt_lines, rule=bch_fail_rep_exclusivity)

        print("  - Operational constraints...")

        # 5.5) Power flow definitions
        def dc_flow_rule(model, sc, l, t):
            i, j = net.data.net.bch[l - 1]
            return model.Pf_tn[sc, l, t] == model.base_MVA * model.B_tn[l] * (model.theta[sc, i, t]
                                                                              - model.theta[sc, j, t])

        model.Constraint_FlowDef_TN = pyo.Constraint(model.Set_slt_tn, rule=dc_flow_rule)

        # Maximum allowable angle difference at lines
        ang_diff_max = net.data.net.theta_limits[1] - net.data.net.theta_limits[0]

        def line_angle_difference_upper_rule(model, sc, l, t):
            # fetch the two end‐buses of branch l
            i, j = net.data.net.bch[l - 1]
            # only enforce on pure TN branches
            if net.data.net.branch_level[l] == 'T':
                return model.theta[sc, i, t] - model.theta[sc, j, t] <= ang_diff_max
            return pyo.Constraint.Skip

        def line_angle_difference_lower_rule(model, sc, l, t):
            i, j = net.data.net.bch[l - 1]
            if net.data.net.branch_level[l] == 'T':
                return model.theta[sc, i, t] - model.theta[sc, j, t] >= -ang_diff_max
            return pyo.Constraint.Skip

        # Now index over the (scenario, tn‐branch, time) tuple set
        model.Constraint_AngleDiffUpperLimit = pyo.Constraint(
            model.Set_slt_tn, rule=line_angle_difference_upper_rule
        )
        model.Constraint_AngleDiffLowerLimit = pyo.Constraint(
            model.Set_slt_tn, rule=line_angle_difference_lower_rule
        )

        def voltage_drop_rule(model, sc, l, t):
            i, j = net.data.net.bch[l - 1]
            return model.V2_dn[sc, i, t] - model.V2_dn[sc, j, t] == \
                2 * (model.R_dn[l] * model.Pf_dn[sc, l, t] + model.X_dn[l] * model.Qf_dn[sc, l, t])

        model.Constraint_VoltageDrop_DN = pyo.Constraint(model.Set_slt_dn, rule=voltage_drop_rule)

        # thermal limits TN (|P|<=Pmax*status)
        def flow_upper_limit_tn_rule(model, sc, l, t):
            return model.Pf_tn[sc, l, t] <= model.Pmax_tn[l] * model.branch_status[sc, l, t]

        def flow_lower_limit_tn_rule(model, sc, l, t):
            return model.Pf_tn[sc, l, t] >= -model.Pmax_tn[l] * model.branch_status[sc, l, t]

        model.Constraint_FlowUpperLimit_TN = pyo.Constraint(model.Set_slt_tn, rule=flow_upper_limit_tn_rule)
        model.Constraint_FlowLowerLimit_TN = pyo.Constraint(model.Set_slt_tn, rule=flow_lower_limit_tn_rule)

        # DN apparent-power box + 45° cuts (|P|,|Q|,|P±Q| <= √2 Smax * status)
        sqrt2 = math.sqrt(2)

        def S1_dn_rule(model, sc, l, t):
            return model.Pf_dn[sc, l, t] <= model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S2_dn_rule(model, sc, l, t):
            return model.Pf_dn[sc, l, t] >= -model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S3_dn_rule(model, sc, l, t):
            if not has_reactive_demand:
                return pyo.Constraint.Skip
            return model.Qf_dn[sc, l, t] <= model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S4_dn_rule(model, sc, l, t):
            if not has_reactive_demand:
                return pyo.Constraint.Skip
            return model.Qf_dn[sc, l, t] >= -model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S5_dn_rule(model, sc, l, t):
            if not has_reactive_demand:
                return pyo.Constraint.Skip
            return model.Pf_dn[sc, l, t] + model.Qf_dn[sc, l, t] \
                <= sqrt2 * model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S6_dn_rule(model, sc, l, t):
            if not has_reactive_demand:
                return pyo.Constraint.Skip
            return model.Pf_dn[sc, l, t] + model.Qf_dn[sc, l, t] \
                >= -sqrt2 * model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S7_dn_rule(model, sc, l, t):
            if not has_reactive_demand:
                return pyo.Constraint.Skip
            return model.Pf_dn[sc, l, t] - model.Qf_dn[sc, l, t] \
                <= sqrt2 * model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S8_dn_rule(model, sc, l, t):
            if not has_reactive_demand:
                return pyo.Constraint.Skip
            return model.Pf_dn[sc, l, t] - model.Qf_dn[sc, l, t] \
                >= -sqrt2 * model.Smax_dn[l] * model.branch_status[sc, l, t]

        model.Constraint_DN_S1 = pyo.Constraint(model.Set_slt_dn, rule=S1_dn_rule)
        model.Constraint_DN_S2 = pyo.Constraint(model.Set_slt_dn, rule=S2_dn_rule)
        model.Constraint_DN_S3 = pyo.Constraint(model.Set_slt_dn, rule=S3_dn_rule)
        model.Constraint_DN_S4 = pyo.Constraint(model.Set_slt_dn, rule=S4_dn_rule)
        model.Constraint_DN_S5 = pyo.Constraint(model.Set_slt_dn, rule=S5_dn_rule)
        model.Constraint_DN_S6 = pyo.Constraint(model.Set_slt_dn, rule=S6_dn_rule)
        model.Constraint_DN_S7 = pyo.Constraint(model.Set_slt_dn, rule=S7_dn_rule)
        model.Constraint_DN_S8 = pyo.Constraint(model.Set_slt_dn, rule=S8_dn_rule)

        # 5.6) Nodal balance constraints (TN: active; DN: active & reactive)
        slack_bus = net.data.net.slack_bus

        def P_balance_tn(model, sc, b, t):
            """Active power balance at TN buses including both existing and new DG/ESS"""
            # Existing generation
            Pg_exst = sum(model.Pg_exst[sc, g, t] for g in model.Set_gen_exst
                          if net.data.net.gen[g - 1] == b)

            # New DG generation (typically none at TN level, but included for completeness)
            Pg_new = sum(model.Pg_new[sc, g, t] for g in model.Set_gen_new_nren
                         if new_gen_to_bus_map[g] == b)

            # Existing ESS contribution (typically none at TN level)
            Pess_net_exst = 0
            if Set_ess_exst:
                Pess_net_exst = sum(model.Pess_discharge_exst[sc, e, t] - model.Pess_charge_exst[sc, e, t]
                                    for e in model.Set_ess_exst if net.data.net.ess[e - 1] == b)

            # New ESS contribution (typically none at TN level)
            Pess_net_new = sum(model.Pess_discharge_new[sc, e, t] - model.Pess_charge_new[sc, e, t]
                               for e in model.Set_ess_new if new_ess_to_bus_map[e] == b)

            # Power flows
            Pf_in = sum(model.Pf_tn[sc, l, t] for l in model.Set_bch_tn
                        if net.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf_tn[sc, l, t] for l in model.Set_bch_tn
                         if net.data.net.bch[l - 1][0] == b)

            # Grid import/export at slack bus
            Pgrid = (model.Pimp[sc, t] - model.Pexp[sc, t]) if b == slack_bus else 0

            return (Pg_exst + Pg_new + Pess_net_exst + Pess_net_new + Pgrid + Pf_in - Pf_out
                    == model.Pd[sc, b, t] - model.Pc[sc, b, t])

        model.Constraint_PBalance_TN = pyo.Constraint(model.Set_sbt_tn, rule=P_balance_tn)

        def P_balance_dn(model, sc, b, t):
            """Active power balance at DN buses including both existing and new DG/ESS"""
            # Existing generation
            Pg_exst = sum(model.Pg_exst[sc, g, t] for g in model.Set_gen_exst
                          if net.data.net.gen[g - 1] == b)

            # New DG generation
            Pg_new = sum(model.Pg_new[sc, g, t] for g in model.Set_gen_new_nren
                         if new_gen_to_bus_map[g] == b)

            # Existing ESS contribution
            Pess_net_exst = 0
            if Set_ess_exst:
                Pess_net_exst = sum(model.Pess_discharge_exst[sc, e, t] - model.Pess_charge_exst[sc, e, t]
                                    for e in model.Set_ess_exst if net.data.net.ess[e - 1] == b)

            # New ESS contribution
            Pess_net_new = sum(model.Pess_discharge_new[sc, e, t] - model.Pess_charge_new[sc, e, t]
                               for e in model.Set_ess_new if new_ess_to_bus_map[e] == b)

            # DN power flows
            Pf_in_dn = sum(model.Pf_dn[sc, l, t] for l in model.Set_bch_dn
                           if net.data.net.bch[l - 1][1] == b)
            Pf_out_dn = sum(model.Pf_dn[sc, l, t] for l in model.Set_bch_dn
                            if net.data.net.bch[l - 1][0] == b)

            # Coupling flows from TN
            Pf_cpl_in = sum(model.Pf_tn[sc, l, t] for l in model.Set_bch_tn
                            if net.data.net.branch_level[l] == 'T-D' and net.data.net.bch[l - 1][1] == b)
            Pf_cpl_out = sum(model.Pf_tn[sc, l, t] for l in model.Set_bch_tn
                             if net.data.net.branch_level[l] == 'T-D' and net.data.net.bch[l - 1][0] == b)

            return (Pg_exst + Pg_new + Pess_net_exst + Pess_net_new + Pf_in_dn - Pf_out_dn +
                    Pf_cpl_in - Pf_cpl_out == model.Pd[sc, b, t] - model.Pc[sc, b, t])

        model.Constraint_PBalance_DN = pyo.Constraint(model.Set_sbt_dn, rule=P_balance_dn)

        def Q_balance_dn(model, sc, b, t):
            """Reactive power balance at DN buses"""
            # Skip if no reactive demand
            if not has_reactive_demand:
                return pyo.Constraint.Skip

            # Existing reactive generation
            Qg_exst = sum(model.Qg_exst[sc, g, t] for g in model.Set_gen_exst
                          if net.data.net.gen[g - 1] == b)

            # New DG reactive generation
            Qg_new = sum(model.Qg_new[sc, g, t] for g in model.Set_gen_new_nren
                         if new_gen_to_bus_map[g] == b)

            # Reactive power flows
            Qf_in_dn = sum(model.Qf_dn[sc, l, t] for l in model.Set_bch_dn
                           if net.data.net.bch[l - 1][1] == b)
            Qf_out_dn = sum(model.Qf_dn[sc, l, t] for l in model.Set_bch_dn
                            if net.data.net.bch[l - 1][0] == b)

            # Note: ESS typically doesn't provide reactive power
            return Qg_exst + Qg_new + Qf_in_dn - Qf_out_dn == model.Qd[sc, b, t] - model.Qc[sc, b, t]

        model.Constraint_QBalance_DN = pyo.Constraint(model.Set_sbt_dn, rule=Q_balance_dn)

        # 5.7) Voltage limits DN
        def V2_dn_upper_limit_rule(model, sc, b, t):
            return model.V2_dn[sc, b, t] <= model.V2_max[b]

        def V2_dn_lower_limit_rule(model, sc, b, t):
            return model.V2_dn[sc, b, t] >= model.V2_min[b]

        model.Constraint_V2max = pyo.Constraint(model.Set_sbt_dn, rule=V2_dn_upper_limit_rule)
        model.Constraint_V2min = pyo.Constraint(model.Set_sbt_dn, rule=V2_dn_lower_limit_rule)

        # 5.8) Generation limits
        # - EXISTING generators
        def Pg_exst_upper_limit_rule(model, sc, g, t):
            """Upper bound for existing generation"""
            if has_renewables and g in model.Set_gen_exst_ren:
                return model.Pg_exst[sc, g, t] <= model.Pg_renewable_avail[g, t]
            else:
                return model.Pg_exst[sc, g, t] <= model.Pg_max_exst[g]

        def Pg_exst_lower_limit_rule(model, sc, g, t):
            return model.Pg_exst[sc, g, t] >= model.Pg_min_exst[g]

        model.Constraint_Pg_exst_upper = pyo.Constraint(model.Set_sgt_exst, rule=Pg_exst_upper_limit_rule)
        model.Constraint_Pg_exst_lower = pyo.Constraint(model.Set_sgt_exst, rule=Pg_exst_lower_limit_rule)

        # Reactive power for existing generators
        if has_reactive_demand:
            def Qg_exst_upper_limit_rule(model, sc, g, t):
                return model.Qg_exst[sc, g, t] <= model.Qg_max_exst[g]

            def Qg_exst_lower_limit_rule(model, sc, g, t):
                return model.Qg_exst[sc, g, t] >= model.Qg_min_exst[g]

            model.Constraint_Qg_exst_upper = pyo.Constraint(model.Set_sgt_exst, rule=Qg_exst_upper_limit_rule)
            model.Constraint_Qg_exst_lower = pyo.Constraint(model.Set_sgt_exst, rule=Qg_exst_lower_limit_rule)

        # - NEW generators
        def Pg_new_upper_limit_rule(model, sc, g, t):
            """Upper bound for new DG generation - limited by installed capacity"""
            return model.Pg_new[sc, g, t] <= model.dg_install_capacity[g]

        def Pg_new_lower_limit_rule(model, sc, g, t):
            """New DG can be turned off completely"""
            return model.Pg_new[sc, g, t] >= 0

        model.Constraint_Pg_new_upper = pyo.Constraint(model.Set_sgt_new_nren, rule=Pg_new_upper_limit_rule)
        model.Constraint_Pg_new_lower = pyo.Constraint(model.Set_sgt_new_nren, rule=Pg_new_lower_limit_rule)

        # Reactive power for new generators
        if has_reactive_demand:
            def Qg_new_upper_limit_rule(model, sc, g, t):
                bus_id = new_gen_to_bus_map[g]
                return model.Qg_new[sc, g, t] <= model.new_dg_q_p_ratio[bus_id] * model.dg_install_capacity[g]

            def Qg_new_lower_limit_rule(model, sc, g, t):
                bus_id = new_gen_to_bus_map[g]
                return model.Qg_new[sc, g, t] >= -model.new_dg_q_p_ratio[bus_id] * model.dg_install_capacity[g]

            model.Constraint_Qg_new_upper = pyo.Constraint(model.Set_sgt_new_nren, rule=Qg_new_upper_limit_rule)
            model.Constraint_Qg_new_lower = pyo.Constraint(model.Set_sgt_new_nren, rule=Qg_new_lower_limit_rule)

        # 5.9) ESS operation constraints
        # For EXISTING ESS
        # - ESS Charge/Discharge Power Limits
        def ess_charge_limit_exst_rule(model, sc, e, t):
            """Existing ESS charging cannot exceed rated power capacity"""
            return model.Pess_charge_exst[sc, e, t] <= model.Pess_max_exst[e] * model.ess_charge_binary_exst[sc, e, t]

        def ess_discharge_limit_exst_rule(model, sc, e, t):
            """Existing ESS discharging cannot exceed rated power capacity"""
            return model.Pess_discharge_exst[sc, e, t] <= model.Pess_max_exst[e] * model.ess_discharge_binary_exst[
                sc, e, t]

        model.Constraint_ESS_charge_limit_exst = pyo.Constraint(model.Set_set_exst,
                                                                rule=ess_charge_limit_exst_rule)
        model.Constraint_ESS_discharge_limit_exst = pyo.Constraint(model.Set_set_exst,
                                                                   rule=ess_discharge_limit_exst_rule)

        # - ESS Charge/Discharge Exclusivity
        def ess_exclusivity_exst_rule(model, sc, e, t):
            """Existing ESS cannot charge and discharge simultaneously"""
            return model.ess_charge_binary_exst[sc, e, t] + model.ess_discharge_binary_exst[sc, e, t] <= 1

        model.Constraint_ESS_exclusivity_exst = pyo.Constraint(model.Set_set_exst,
                                                               rule=ess_exclusivity_exst_rule)

        # - ESS SOC Dynamics
        def Eess_dynamics_exst_rule(model, sc, e, t):
            """Energy evolution for existing ESS considering charge/discharge efficiency"""
            ts_list = list(Set_ts_scn[sc])

            if t == ts_list[0]:  # First timestep in scenario
                # Initial energy stored (MWh)
                return model.Eess_exst[sc, e, t] == model.Eess_max_exst[e] * model.initial_SOC_exst[e]
            else:
                prev_t = ts_list[ts_list.index(t) - 1]
                # Energy change (MWh) = Power (MW) × efficiency × time (h)
                return model.Eess_exst[sc, e, t] == model.Eess_exst[sc, e, prev_t] + \
                    model.dt * (model.Pess_charge_exst[sc, e, t] * model.eff_ch_exst[e] -
                          model.Pess_discharge_exst[sc, e, t] / model.eff_dis_exst[e])

        model.Constraint_Eess_dynamics_exst = pyo.Constraint(
            model.Set_set_exst,
            rule=Eess_dynamics_exst_rule
        )

        # - ESS SOC Definition
        def soc_definition_exst_rule(model, sc, e, t):
            """SOC (p.u.) = Energy stored (MWh) / Energy capacity (MWh)"""
            return model.SOC_exst[sc, e, t] == model.Eess_exst[sc, e, t] / model.Eess_max_exst[e]

        model.Constraint_SOC_definition_exst = pyo.Constraint(
            model.Set_set_exst,
            rule=soc_definition_exst_rule
        )

        # - ESS SOC Limits
        def soc_min_exst_rule(model, sc, e, t):
            """SOC must stay above minimum for existing ESS"""
            return model.SOC_exst[sc, e, t] >= model.Eess_max_exst[e] * model.SOC_min_exst[e]

        def soc_max_exst_rule(model, sc, e, t):
            """SOC must stay below maximum for existing ESS"""
            return model.SOC_exst[sc, e, t] <= model.Eess_max_exst[e] * model.SOC_max_exst[e]

        model.Constraint_SOC_min_exst = pyo.Constraint(
            model.Set_set_exst,
            rule=soc_min_exst_rule
        )
        model.Constraint_SOC_max_exst = pyo.Constraint(
            model.Set_set_exst,
            rule=soc_max_exst_rule
        )

        # For NEW ESS
        # - ESS Charge/Discharge Power Limits
        def ess_charge_limit_new_rule(model, sc, e, t):
            """New ESS charging cannot exceed installed power capacity"""
            return model.Pess_charge_new[sc, e, t] <= model.ess_install_capacity[e] * model.ess_charge_binary_new[
                sc, e, t]

        def ess_discharge_limit_new_rule(model, sc, e, t):
            """New ESS discharging cannot exceed installed power capacity"""
            return model.Pess_discharge_new[sc, e, t] <= model.ess_install_capacity[e] * model.ess_discharge_binary_new[
                sc, e, t]

        model.Constraint_ESS_charge_limit_new = pyo.Constraint(model.Set_set_new,
                                                               rule=ess_charge_limit_new_rule)
        model.Constraint_ESS_discharge_limit_new = pyo.Constraint(model.Set_set_new,
                                                                  rule=ess_discharge_limit_new_rule)

        # - ESS Charge/Discharge Exclusivity
        def ess_exclusivity_new_rule(model, sc, e, t):
            """New ESS cannot charge and discharge simultaneously"""
            return model.ess_charge_binary_new[sc, e, t] + model.ess_discharge_binary_new[sc, e, t] <= 1

        model.Constraint_ESS_exclusivity_new = pyo.Constraint(model.Set_set_new,
                                                              rule=ess_exclusivity_new_rule)

        def Eess_dynamics_new_rule(model, sc, e, t):
            """Energy evolution for new ESS considering charge/discharge efficiency"""
            ts_list = list(Set_ts_scn[sc])

            if t == ts_list[0]:  # First timestep in scenario
                # Initial energy = installed power × power-to-energy ratio × initial SOC
                energy_capacity = model.ess_install_capacity[e] * model.ess_power_energy_ratio_new[e]
                return model.Eess_new[sc, e, t] == energy_capacity * model.initial_SOC_new[e]
            else:
                prev_t = ts_list[ts_list.index(t) - 1]
                # Energy change (MWh) = Power (MW) × efficiency × time (h)
                return model.Eess_new[sc, e, t] == model.Eess_new[sc, e, prev_t] + \
                    model.dt * (model.Pess_charge_new[sc, e, t] * model.eff_ch_new[e] -
                          model.Pess_discharge_new[sc, e, t] / model.eff_dis_new[e])

        model.Constraint_Eess_dynamics_new = pyo.Constraint(model.Set_set_new, rule=Eess_dynamics_new_rule)

        # ESS SOC Definition
        def soc_definition_new_rule(model, sc, e, t):
            """SOC (p.u.) = Energy stored (MWh) / Energy capacity (MWh)"""
            # energy_capacity = model.ess_install_capacity[e] * model.ess_power_energy_ratio_new[e]
            # if energy_capacity == 0:
            #     return model.SOC_new[sc, e, t] == 0
            # else:
            #     return model.SOC_new[sc, e, t] == model.Eess_new[sc, e, t] / energy_capacity
            return pyo.Constraint.Skip
  # it is skipped as the expression to compute SOC_new is non-linear

        model.Constraint_SOC_definition_new = pyo.Constraint(model.Set_set_new, rule=soc_definition_new_rule)

        # - ESS SOC Limits
        def soc_min_new_rule(model, sc, e, t):
            """SOC must stay above minimum for new ESS"""
            # energy_capacity = model.ess_install_capacity[e] * model.ess_power_energy_ratio_new[e]
            # return model.SOC_new[sc, e, t] >= energy_capacity * model.SOC_min_new[e]
            return pyo.Constraint.Skip


        def soc_max_new_rule(model, sc, e, t):
            """SOC must stay below maximum for new ESS"""
            # energy_capacity = model.ess_install_capacity[e] * model.ess_power_energy_ratio_new[e]
            # return model.SOC_new[sc, e, t] <= energy_capacity * model.SOC_max_new[e]
            return pyo.Constraint.Skip


        model.Constraint_SOC_min_new = pyo.Constraint(model.Set_set_new, rule=soc_min_new_rule)
        model.Constraint_SOC_max_new = pyo.Constraint(model.Set_set_new, rule=soc_max_new_rule)

        # ESS Energy Limits
        def Eess_min_new_rule(model, sc, e, t):
            """Energy stored must stay above minimum"""
            energy_capacity = model.ess_install_capacity[e] * model.ess_power_energy_ratio_new[e]
            return model.Eess_new[sc, e, t] >= energy_capacity * model.SOC_min_new[e]

        def Eess_max_new_rule(model, sc, e, t):
            """Energy stored must stay below maximum (capacity)"""
            energy_capacity = model.ess_install_capacity[e] * model.ess_power_energy_ratio_new[e]
            return model.Eess_new[sc, e, t] <= energy_capacity * model.SOC_max_new[e]

        model.Constraint_Eess_min_new = pyo.Constraint(model.Set_set_new, rule=Eess_min_new_rule)
        model.Constraint_Eess_max_new = pyo.Constraint(model.Set_set_new, rule=Eess_max_new_rule)

        # 5.10) Slack-bus reference theta = 0
        def slack_rule(model, sc, t):
            return model.theta[sc, slack_bus, t] == 0

        model.Constraint_Slack = pyo.Constraint([(sc, t) for sc in Set_scn for t in Set_ts_scn[sc]],
                                                rule=slack_rule)

        # Prepare parameters that will be used when defining the two expressions below
        num_scenarios = len(model.Set_scn)
        prob_factor = 1 / num_scenarios
        model.num_scenarios = pyo.Param(initialize=num_scenarios)
        model.prob_factor = pyo.Param(initialize=prob_factor)

        print("  - Resilience metric threshold constraints...")

        # 5.11) Resilience metric threshold constraint (resilience metric <= pre-defined threshold)
        # Code 'expected total operational cost at dn level across all ws scenarios' (the resilience metric)
        # as an expression
        # (currently as we assume all windstorm scenarios have equal probabilities, we simply find the average value)
        def expected_total_op_cost_dn_ws_rule(model):
            # 1) DN generation cost from EXISTING generators
            gen_cost_dn_exst = sum(
                (model.gen_cost_coef_exst[g, 0] + model.gen_cost_coef_exst[g, 1] * model.Pg_exst[sc, g, t])
                for (sc, g, t) in model.Set_sgt_dn_exst
            ) * model.prob_factor

            # 2) DN generation cost from NEW DGs
            gen_cost_dn_new = sum(
                (model.gen_cost_coef_new[g, 0] + model.gen_cost_coef_new[g, 1] * model.Pg_new[sc, g, t])
                for (sc, g, t) in model.Set_sgt_dn_new
            ) * model.prob_factor

            # 3) DN active load-shedding cost
            active_ls_cost_dn = sum(
                model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            ) * model.prob_factor

            # 4) DN reactive load-shedding cost
            reactive_ls_cost_dn = sum(
                model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            ) * model.prob_factor

            # 5) DN line repair cost
            rep_cost_dn = sum(
                model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_dn_lines
            ) * model.prob_factor

            return gen_cost_dn_exst + gen_cost_dn_new + active_ls_cost_dn + reactive_ls_cost_dn + rep_cost_dn

        model.exp_total_op_cost_dn_ws_expr = pyo.Expression(rule=expected_total_op_cost_dn_ws_rule)

        # Code "Expected Energy Not Supplied (EENS) at dn level across all ws scenarios" as an expression
        # Since we have hourly resolution, EENS = sum of all active power load shedding
        def expected_total_eens_dn_ws_rule(model):
            eens_dn = sum(
                model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            ) * model.prob_factor

            return eens_dn

        model.exp_total_eens_dn_ws_expr = pyo.Expression(rule=expected_total_eens_dn_ws_rule)

        if resilience_metric_threshold is not None:
            model.Constraint_ResilienceLevel = pyo.Constraint(
                # expr=model.exp_total_op_cost_dn_expr <= model.resilience_metric_threshold
                expr=model.exp_total_eens_dn_ws_expr <= model.resilience_metric_threshold
            )

        # ------------------------------------------------------------------
        # 6. Objective
        # ------------------------------------------------------------------
        print("Setting up objective function...")

        # Code total investment cost as an expression
        def total_inv_cost_rule(model):
            # Line hardening cost
            line_hrdn_cost = sum(model.line_hrdn_cost[l] * model.line_hrdn[l]
                                 for l in model.Set_bch_hrdn_lines)

            # New DG installation cost
            dg_install_cost = sum(model.dg_install_cost[g] * model.dg_install_capacity[g]
                                  for g in model.Set_gen_new_nren)

            # New ESS installation cost
            ess_install_cost = sum(model.ess_install_cost[e] * model.ess_install_capacity[e]
                                   for e in model.Set_ess_new)

            return line_hrdn_cost + dg_install_cost + ess_install_cost

        model.total_inv_cost_expr = pyo.Expression(rule=total_inv_cost_rule)

        def expected_total_op_cost_rule(model):
            # 1) Generation cost from EXISTING generators
            gen_cost_exst = sum(
                model.scn_prob[sc] * (
                            model.gen_cost_coef_exst[g, 0] + model.gen_cost_coef_exst[g, 1] * model.Pg_exst[sc, g, t])
                for (sc, g, t) in model.Set_sgt_exst
            )

            # 2) Generation cost from NEW DGs
            gen_cost_new = sum(
                model.scn_prob[sc] * (
                            model.gen_cost_coef_new[g, 0] + model.gen_cost_coef_new[g, 1] * model.Pg_new[sc, g, t])
                for (sc, g, t) in model.Set_sgt_new_nren
            )

            # 3) Grid import / export cost
            imp_exp_cost = sum(
                model.scn_prob[sc] * (model.Pimp_cost * model.Pimp[sc, t] +
                                      model.Pexp_cost * model.Pexp[sc, t])
                for (sc, t) in model.Set_st
            )

            # 4) Active load-shedding cost
            act_ls_cost = sum(
                model.scn_prob[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt
            )

            # 5) Reactive load-shedding cost (DN buses only)
            reac_ls_cost = sum(
                model.scn_prob[sc] * model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 6) Line repair cost
            rep_cost = sum(
                model.scn_prob[sc] * model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_lines
            )

            # Note: ESS operation typically doesn't have direct operational costs
            # (charging/discharging losses are already captured in efficiency parameters)

            # Windstorm window costs
            ws_window_cost = gen_cost_exst + gen_cost_new + imp_exp_cost + act_ls_cost + reac_ls_cost + rep_cost

            # If normal operation scenario is included, add annual normal operation costs
            if include_normal_scenario and normal_operation_opf_results:
                # Each windstorm scenario includes its window cost and a full year of normal operation
                annual_normal_cost = sum(
                    model.scn_prob[sc] * normal_operation_opf_results["total_cost"]
                    for sc in model.Set_scn
                )
                return ws_window_cost + annual_normal_cost
            else:
                return ws_window_cost

        model.exp_total_op_cost_expr = pyo.Expression(rule=expected_total_op_cost_rule)

        # def objective_rule(model):
        #     # First-stage: Investment cost
        #     inv_cost = model.total_inv_cost_expr
        #
        #     # Second-stage: Expected operational cost
        #     if include_normal_scenario and normal_operation_opf_results:
        #         # Normal scenario (full year) + Windstorm scenarios (window + full year)
        #         normal_contribution = normal_scenario_prob * normal_operation_opf_results["total_cost"]
        #
        #         ws_contribution = model.exp_total_op_cost_expr  # Already includes annual normal costs
        #         return inv_cost + normal_contribution + ws_contribution
        #     else:
        #         # Only windstorm scenarios (without annual normalization)
        #         return inv_cost + model.exp_total_op_cost_expr

        def objective_rule(model):
            # First-stage: Investment cost
            inv_cost = model.total_inv_cost_expr

            # Calculate present value factor for annuity
            r = model.discount_rate
            n = model.asset_lifetime
            if r == 0:
                pv_factor = n  # Handle zero discount rate case
            else:
                pv_factor = (1 - (1 + r) ** (-n)) / r

            # Second-stage: Expected operational cost
            if include_normal_scenario and normal_operation_opf_results:
                # Normal scenario (full year) + Windstorm scenarios (window + full year)
                normal_contribution = normal_scenario_prob * normal_operation_opf_results["total_cost"]
                ws_contribution = model.exp_total_op_cost_expr  # Already includes annual normal costs

                # Apply PV factor to total annual operational cost
                annual_op_cost = normal_contribution + ws_contribution
                return inv_cost + pv_factor * annual_op_cost
            else:
                # Only windstorm scenarios (without annual normalization)
                return inv_cost + pv_factor * model.exp_total_op_cost_expr

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Add parameter to track normal cost contribution
        if include_normal_scenario and normal_operation_opf_results:
            model.normal_cost_contribution = pyo.Param(
                initialize=normal_scenario_prob * normal_operation_opf_results["total_cost"],
                mutable=False
            )

            # Also create expressions for tracking DN-level normal costs
            model.normal_gen_cost_dn = pyo.Param(
                initialize=normal_scenario_prob * normal_operation_opf_results["gen_cost_dn"],
                mutable=False
            )

        print("\nModel building completed.")
        print("=" * 60)

        return model

    def solve_investment_model(
            self,
            model,
            solver_name: str = "gurobi",
            mip_gap: float = 5e-3,
            time_limit: int = 60,
            write_lp: bool = False,
            write_result: bool = False,
            result_path: str = None,
            **solver_options
    ):
        """
        Solve a Pyomo investment model and export essential results.

        Parameters
        ----------
        mip_gap       : Absolute/relative MIP gap (default 0.5 %)
        time_limit    : Wall-clock limit in seconds
        write_lp      : Dump the model to LP file *only when* write_lp is True **or** the solver fails
        result_path   : If not specified, a unique file name will be assigned
        solver_options: Extra options forwarded to the solver
        """

        print(f"\nSolving model with {solver_name} (time limit: {time_limit}s)...")

        opt = SolverFactory(solver_name)

        # --- solver-specific options -------------------------------------------------
        if solver_name.lower() == "gurobi":
            default_opts = {"MIPGap": mip_gap, "TimeLimit": time_limit}
        elif solver_name.lower() == "cbc":
            default_opts = {
                "ratioGap": mip_gap,
                "seconds": time_limit,
                "threads": 0,
                "presolve": "on",
                "cuts": "on",
            }
        elif solver_name.lower() == "glpk":
            default_opts = {
                "mipgap": mip_gap,
                "tmlim": time_limit,  # Time limit in seconds
            }
        else:
            default_opts = {}
            print(f"Warning: Using solver '{solver_name}' with default options")

        default_opts.update(solver_options)
        for k, v in default_opts.items():
            opt.options[k] = v

        # --- solve -----------------------------------------------------------
        results = opt.solve(model, tee=True)

        # --- check solution status --------------------------------------------
        status = results.solver.status
        term_cond = results.solver.termination_condition

        if status == pyo.SolverStatus.ok and term_cond == pyo.TerminationCondition.optimal:
            print("Optimal solution found.")
            best_obj = pyo.value(model.Objective)
            mip_gap_out = 0.0
        elif status == pyo.SolverStatus.ok:
            print("Feasible solution found.")
            best_obj = pyo.value(model.Objective) if hasattr(model, 'Objective') else None
            mip_gap_out = getattr(results.problem, 'lower_bound', None)
            if mip_gap_out and best_obj:
                mip_gap_out = abs(best_obj - mip_gap_out) / abs(best_obj)
        else:
            print("No feasible solution found.")
            best_obj = None
            mip_gap_out = None

        # --- write LP file if requested ---------------------------------------
        if write_lp:
            model.write("LP_Models/solved_model.lp",
                        io_options={"symbolic_solver_labels": True})

        # --- write results if requested ---------------------------------------
        if write_result:
            print("Writing results...")

            # If 'result_path' is not specified, create a unique file name with timestamp
            if result_path is None:
                # Example: results_20240613_151822_GB-Kearsley_seed101.csv
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                network = getattr(self, "network_name", "network")
                seed = getattr(self.meta, "ws_seed", None)
                # Shorten the original network name (e.g., "29_bus_GB_transmission_network_with_Kearsley_GSP_group")
                # and add it to the file name
                short_network_name = (network
                                      .replace("29_bus_", "29Bus")
                                      .replace("transmission_network_with_", "")
                                      .replace("GB_", "GB-")
                                      .replace("_GSP_group", ""))

                # Add the network information
                fname = f"results_network_{short_network_name}"

                # Add the windstorm information
                fname += f"_{self.meta.n_ws_scenarios}_ws"

                # Add the windstorm seed value
                if seed is not None:
                    fname += f"_seed_{seed}"

                # Add the resilience metric threshold value
                rthres = float(getattr(model, "resilience_metric_threshold", float('inf')))
                if rthres != float('inf'):
                    # Use scientific notation with 2 decimals (e.g., 2.5e8)
                    sci_str = f"{rthres:.2e}"
                    # Replace '+' with empty string for filesystem compatibility (e.g., 2.50e+08 → 2.50e08)
                    sci_str = sci_str.replace("+0", "").replace("+", "")
                    # Add to the file name
                    fname += f"_resilience_threshold_{sci_str}"
                else:
                    fname += f"_resilience_threshold_inf"

                # Add the timestamp
                fname += f"_{timestamp}"

                # File name finishes
                fname += ".xlsx"  # Changed from .csv to .xlsx since we're using Excel writer

                # Use absolute path for safety (ISSUE 2 FIX)
                from pathlib import Path
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent  # Go up from core/ to project root
                result_dir = project_root / "Optimization_Results" / "Investment_Model"
                result_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
                result_path = str(result_dir / fname)

            self._write_selected_variables_to_excel(model, result_path)
            print(f"Results written to: {result_path}")

        return {
            "objective": best_obj,
            "gap": mip_gap_out,
            "status": str(status),
            "termination": str(term_cond),
            "runtime_s": results.solver.time
        }

    def _write_selected_variables_to_excel(self, model,
                                           path: str = "Optimization_Results/Investment_Model/results.xlsx",
                                           meta: dict | None = None):
        """
        Export selected variables and key parameters to a multi-sheet .xlsx workbook.
        """
        important = (
            "line_hrdn",
            "Pg", "Qg", "Pc", "Qc",
            "Pf_tn", "Pf_dn",
            "rand_num", "shifted_gust_speed", "branch_status", "fail_prob", "fail_condition",
            "impacted_branches", "fail_indicator",
            "fail_applies", "repair_applies"
        )

        path = Path(path).with_suffix(".xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)

        if meta is None:
            meta = {
                "written_at": datetime.now().isoformat(timespec="seconds"),
                "network_name": self.network_name,
                "windstorm_name": self.windstorm_name,
                "windstorm_random_seed": getattr(self.meta, 'ws_seed', None),
                "number_of_ws_scenarios": getattr(self.meta, 'n_ws_scenarios', 0),
                "normal_scenario_included": getattr(self.meta, 'normal_scenario_included', False),
                "normal_scenario_probability": getattr(self.meta, 'normal_scenario_prob', 0),
                "resilience_metric_threshold": float(pyo.value(model.resilience_metric_threshold)),
                "objective_value": float(pyo.value(model.Objective)),
                "total_investment_cost": float(pyo.value(model.total_inv_cost_expr)),
                "expected_total_operational_cost": float(pyo.value(model.exp_total_op_cost_expr)),
                "ws_expected_total_operational_cost_dn": float(pyo.value(model.exp_total_op_cost_dn_ws_expr)),
                "ws_exp_total_eens_dn": float(pyo.value(model.exp_total_eens_dn_ws_expr)),
            }

            # Add normal scenario costs if included
            if hasattr(model, 'normal_cost_contribution'):
                meta["normal_operation_cost_contribution"] = float(pyo.value(model.normal_cost_contribution))

                # Also add the DN generation cost from normal scenario if available
                if hasattr(model, 'normal_gen_cost_dn'):
                    meta["normal_operation_gen_cost_dn"] = float(pyo.value(model.normal_gen_cost_dn))

                # Calculate total expected cost including normal
                # The windstorm operational cost is already weighted by (1 - normal_prob) in the objective
                # So the total is just the objective value
                meta["total_expected_operational_cost_with_normal"] = (
                        float(pyo.value(model.exp_total_op_cost_expr)) +
                        float(pyo.value(model.normal_cost_contribution))
                )

            # Add detailed normal operation info if available
            if hasattr(self.meta, 'normal_operation_opf_results') and self.meta.normal_operation_opf_results:
                meta["normal_hours_computed"] = self.meta.normal_operation_opf_results.get("hours_computed", "N/A")
                meta["normal_scale_factor"] = self.meta.normal_operation_opf_results.get("scale_factor", "N/A")
                meta["normal_solver_status"] = self.meta.normal_operation_opf_results.get("solver_status", "N/A")
                if hasattr(self.meta.normal_operation_opf_results, 'representative_days'):
                    meta["normal_representative_days"] = str(
                        self.meta.normal_operation_opf_results.get("representative_days", "N/A"))

        from factories.network_factory import make_network
        net = make_network(meta.get("network_name", self.network_name))

        with pd.ExcelWriter(path, engine="xlsxwriter") as xl:
            # 1) META sheet
            pd.Series(meta, name="value") \
                .to_frame() \
                .to_excel(xl, sheet_name="Meta", header=False)

            # 2) PARAMETER sheets
            branch_ids = list(range(1, len(net.data.net.bch) + 1))

            # Ensure branch_level is a list, not a dict
            bl = net.data.net.branch_level
            if isinstance(bl, dict):
                bl = pd.Series(bl).reindex(branch_ids).tolist()
            df_bl = pd.DataFrame({
                "branch": branch_ids,
                "branch_level": bl
            })
            df_bl.to_excel(xl, sheet_name="branch_level", index=False)

            # bch_type is expected as a list
            bt = net.data.net.bch_type
            df_bt = pd.DataFrame({
                "branch": branch_ids,
                "bch_type": bt
            })
            df_bt.to_excel(xl, sheet_name="bch_type", index=False)

            # 3) one sheet per selected variable
            for name in important:
                var = getattr(model, name, None)
                if var is None:
                    continue

                rows = []
                if not var.is_indexed():
                    rows.append({"index": "", "value": float(pyo.value(var))})
                else:
                    for idx in var:
                        rows.append({
                            "index": str(idx),
                            "value": float(pyo.value(var[idx]))
                        })

                df = pd.DataFrame(rows)
                sheet = name[:29]
                df.to_excel(xl, sheet_name=sheet, index=False)

    def piecewise_linearize_fragility(self, net, line_idx, num_pieces):

        """
        Return { "gust_speeds": [...],
                 "fail_probs" : { l: [p(t1),…,p(tn)] for l in line_idx } }
        """
        # restrict mins / maxs to *lines* only
        # gmin = min(net.data.frg.thrd_1[l - 1] for l in line_idx)
        # gmax = max(net.data.frg.thrd_2[l - 1] for l in line_idx)
        gmin = 0
        gmax = 120
        gust_speeds = np.linspace(gmin, gmax, num_pieces).tolist()

        fail_probs = {}
        for l in line_idx:
            mu, sg = net.data.frg.mu[l - 1], net.data.frg.sigma[l - 1]
            th1, th2 = net.data.frg.thrd_1[l - 1], net.data.frg.thrd_2[l - 1]
            sf = net.data.frg.shift_f[l - 1]

            fp = []
            for x in gust_speeds:
                z = x - sf
                if z < th1:
                    fp.append(0.0)
                elif z > th2:
                    fp.append(1.0)
                else:
                    fp.append(float(lognorm.cdf(z, s=sg, scale=np.exp(mu))))
            fail_probs[l] = fp

        return {"gust_speeds": gust_speeds, "fail_probs": fail_probs}

    def build_normal_scenario_opf_model(self,
                                        network_name: str,
                                        use_representative_days: bool = True,
                                        representative_days: list = None):
        """
        Build an OPF model for normal operation (baseline) using ONLY existing assets.
        This represents the system operation before any new investments.

        Args:
            network_name: Network configuration name
            use_representative_days: If True, use representative days instead of full year
            representative_days: List of day numbers (1-365)

        Returns:
            tuple: (model, scale_factor) where scale_factor annualizes the costs
        """

        # ------------------------------------------------------------------
        # 0. Setup
        # ------------------------------------------------------------------
        net = make_network(network_name)
        model = pyo.ConcreteModel()

        # Determine time periods
        if use_representative_days:
            days = representative_days or [15, 105, 195, 285]  # One day per season
            hours_per_day = 24
            total_hours = len(days) * hours_per_day
            # Create hour indices for representative days
            timesteps = []
            for day in days:
                start_hour = (day - 1) * 24
                timesteps.extend(range(start_hour, start_hour + 24))
            scale_factor = 365 / len(days)  # Scale to annual
        else:
            timesteps = list(range(8760))  # Full year
            total_hours = 8760
            scale_factor = 1.0

        model.scale_factor = pyo.Param(initialize=scale_factor, mutable=False)

        # ------------------------------------------------------------------
        # 1. Index Sets
        # ------------------------------------------------------------------
        # Basic sets - use dictionaries directly (same as build_investment_model)
        Set_ts = timesteps
        Set_bus = list(range(1, len(net.data.net.bus) + 1))
        Set_bus_tn = [b for b in net.data.net.bus if net.data.net.bus_level[b] == 'T']
        Set_bus_dn = [b for b in net.data.net.bus if net.data.net.bus_level[b] == 'D']
        Set_bch = list(range(1, len(net.data.net.bch) + 1))
        Set_bch_tn = [l for l in range(1, len(net.data.net.bch) + 1)
                      if net.data.net.branch_level[l] in ['T', 'T-D']]
        Set_bch_dn = [l for l in range(1, len(net.data.net.bch) + 1)
                      if net.data.net.branch_level[l] == 'D']
        # ONLY existing generators (no new investments in normal operation)
        Set_gen_exst = list(range(1, len(net.data.net.gen) + 1))

        # Classify existing generators by type
        Set_gen_exst_ren = []
        Set_gen_exst_nren = []
        for g in Set_gen_exst:
            gen_name = net.data.net.gen_name[g - 1] if hasattr(net.data.net, 'gen_name') else f"Gen_{g}"
            if any(keyword in gen_name.lower() for keyword in ['wind', 'solar', 'renewable']):
                Set_gen_exst_ren.append(g)
            else:
                Set_gen_exst_nren.append(g)

        # ONLY existing ESS (if any)
        Set_ess_exst = list(range(1, len(net.data.net.ess) + 1)) if hasattr(net.data.net, 'ess') else []

        # ------------------------------------------------------------------
        # 2. Initialize Pyomo Sets
        # ------------------------------------------------------------------
        model.Set_ts = pyo.Set(initialize=Set_ts)
        model.Set_bus = pyo.Set(initialize=Set_bus)
        model.Set_bus_tn = pyo.Set(initialize=Set_bus_tn)
        model.Set_bus_dn = pyo.Set(initialize=Set_bus_dn)
        model.Set_bch = pyo.Set(initialize=Set_bch)
        model.Set_bch_tn = pyo.Set(initialize=Set_bch_tn)
        model.Set_bch_dn = pyo.Set(initialize=Set_bch_dn)
        model.Set_gen_exst = pyo.Set(initialize=Set_gen_exst)
        model.Set_gen_exst_ren = pyo.Set(initialize=Set_gen_exst_ren)
        model.Set_gen_exst_nren = pyo.Set(initialize=Set_gen_exst_nren)
        model.Set_ess_exst = pyo.Set(initialize=Set_ess_exst)

        # ------------------------------------------------------------------
        # 3. Parameters
        # ------------------------------------------------------------------
        # 3.1) Base + slack
        model.base_MVA = pyo.Param(initialize=net.data.net.base_MVA)
        slack_bus = net.data.net.slack_bus

        # 3.2) Generator parameters (existing only)
        coef_len = len(net.data.net.gen_cost_coef[0])
        model.gen_cost_coef = pyo.Param(
            model.Set_gen_exst, range(coef_len),
            initialize={(g, c): net.data.net.gen_cost_coef[g - 1][c]
                        for g in model.Set_gen_exst for c in range(coef_len)}
        )
        model.Pg_max_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: net.data.net.Pg_max_exst[g - 1] for g in model.Set_gen_exst})
        model.Pg_min_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: net.data.net.Pg_min_exst[g - 1] for g in model.Set_gen_exst})
        model.Qg_max_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: net.data.net.Qg_max_exst[g - 1] for g in model.Set_gen_exst})
        model.Qg_min_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: net.data.net.Qg_min_exst[g - 1] for g in model.Set_gen_exst})

        # 3.3) Branch parameters — TN (DC) and DN (LinAC)
        model.B_tn = pyo.Param(model.Set_bch_tn,
                               initialize={l: 1.0 / net.data.net.bch_X[l - 1] for l in model.Set_bch_tn})
        model.Pmax_tn = pyo.Param(model.Set_bch_tn,
                                  initialize={l: net.data.net.bch_Smax[l - 1] for l in model.Set_bch_tn})
        model.R_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: net.data.net.bch_R[l - 1] for l in model.Set_bch_dn})
        model.X_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: net.data.net.bch_X[l - 1] for l in model.Set_bch_dn})
        model.Smax_dn = pyo.Param(model.Set_bch_dn,
                                  initialize={l: net.data.net.bch_Smax[l - 1] for l in model.Set_bch_dn})

        # 3.4) DN voltage limits (use bounds directly later)
        model.V2_max = pyo.Param(model.Set_bus_dn,
                                 initialize={b: net.data.net.V_max[b - 1] ** 2 for b in model.Set_bus_dn})
        model.V2_min = pyo.Param(model.Set_bus_dn,
                                 initialize={b: net.data.net.V_min[b - 1] ** 2 for b in model.Set_bus_dn})

        # 3.5) Load data (robust to missing profiles, like in network.py)
        model.Pd = pyo.Param(
            model.Set_bus, model.Set_ts,
            initialize={(b, t): (net.data.net.profile_Pd[b - 1][t]
                                 if net.data.net.profile_Pd[b - 1] is not None else 0.0)
                        for b in model.Set_bus for t in model.Set_ts}
        )
        model.Qd = pyo.Param(
            model.Set_bus_dn, model.Set_ts,
            initialize={(b, t): (net.data.net.profile_Qd[b - 1][t]
                                 if (net.data.net.profile_Qd[b - 1] is not None) else 0.0)
                        for b in model.Set_bus_dn for t in model.Set_ts},
            default=0.0
        )

        # has_reactive_demand = False

        # 3.6) Costs
        model.Pimp_cost = pyo.Param(initialize=net.data.net.Pimp_cost)
        model.Pexp_cost = pyo.Param(initialize=net.data.net.Pexp_cost)
        model.Pc_cost = pyo.Param(model.Set_bus,
                                  initialize={b: net.data.net.Pc_cost[b - 1] for b in model.Set_bus})
        model.Qc_cost = pyo.Param(model.Set_bus_dn,
                                  initialize={b: net.data.net.Qc_cost[b - 1] for b in model.Set_bus_dn})

        # 3.7) ESS parameters (if existing)
        if Set_ess_exst:
            model.Pess_max_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Pess_max_exst[e - 1] for e in Set_ess_exst})
            model.Pess_min_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Pess_min_exst[e - 1] for e in Set_ess_exst})
            model.Eess_max_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Eess_max_exst[e - 1] for e in Set_ess_exst})
            model.Eess_min_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: net.data.net.Eess_min_exst[e - 1] for e in Set_ess_exst})
            model.SOC_max_exst = pyo.Param(model.Set_ess_exst,
                                           initialize={e: net.data.net.SOC_max_exst[e - 1] for e in Set_ess_exst})
            model.SOC_min_exst = pyo.Param(model.Set_ess_exst,
                                           initialize={e: net.data.net.SOC_min_exst[e - 1] for e in Set_ess_exst})
            model.eff_ch_exst = pyo.Param(model.Set_ess_exst,
                                          initialize={e: net.data.net.eff_ch_exst[e - 1] for e in Set_ess_exst})
            model.eff_dis_exst = pyo.Param(model.Set_ess_exst,
                                           initialize={e: net.data.net.eff_dis_exst[e - 1] for e in Set_ess_exst})
            model.initial_SOC_exst = pyo.Param(model.Set_ess_exst,
                                               initialize={e: net.data.net.initial_SOC_exst[e - 1] for e in
                                                           Set_ess_exst})

        # ------------------------------------------------------------------
        # 4. Variables
        # ------------------------------------------------------------------
        # 4.1) Generation (existing generators only)
        model.Pg_exst = pyo.Var(
            model.Set_gen_exst, model.Set_ts,
            within=pyo.NonNegativeReals,
            bounds=lambda model, g, t: (model.Pg_min_exst[g], model.Pg_max_exst[g])
        )
        has_reactive_demand = any(model.Qd[b, t] > 0 for b in model.Set_bus_dn for t in model.Set_ts)
        if has_reactive_demand:
            model.Qg_exst = pyo.Var(
                model.Set_gen_exst, model.Set_ts,
                within=pyo.Reals,
                bounds=lambda model, g, t: (model.Qg_min_exst[g], model.Qg_max_exst[g])
            )
        else:
            model.Qg_exst = pyo.Param(model.Set_gen_exst, model.Set_ts, default=0.0)

        # 4.2) Grid import/export (active only here)
        model.Pimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)
        model.Pexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)

        # 4.3) States
        model.theta = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.Reals)
        # NOTE: add bounds directly (tighter & safer)
        model.V2_dn = pyo.Var(
            model.Set_bus_dn, model.Set_ts,
            within=pyo.NonNegativeReals,
            bounds=lambda model, b, t: (model.V2_min[b], model.V2_max[b])
        )

        # 4.4) Flows
        model.Pf_tn = pyo.Var(model.Set_bch_tn, model.Set_ts, within=pyo.Reals)
        model.Pf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)
        if has_reactive_demand:
            model.Qf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)
        else:
            model.Qf_dn = pyo.Param(model.Set_bch_dn, model.Set_ts, default=0.0)

        # 4.5) Load shedding
        model.Pc = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals,
                           bounds=lambda model, b, t: (0, max(0.0, pyo.value(model.Pd[b, t])))
                           )
        if has_reactive_demand:
            model.Qc = pyo.Var(model.Set_bus_dn, model.Set_ts, within=pyo.NonNegativeReals)
        else:
            model.Qc = pyo.Param(model.Set_bus_dn, model.Set_ts, default=0.0)

        # 4.6) ESS variables (if existing)
        if Set_ess_exst:
            model.Pess_charge_exst = pyo.Var(model.Set_ess_exst, model.Set_ts,
                                             within=pyo.NonNegativeReals)
            model.Pess_discharge_exst = pyo.Var(model.Set_ess_exst, model.Set_ts,
                                                within=pyo.NonNegativeReals)
            model.Eess_exst = pyo.Var(model.Set_ess_exst, model.Set_ts,
                                      within=pyo.NonNegativeReals)
            model.SOC_exst = pyo.Var(model.Set_ess_exst, model.Set_ts,
                                     within=pyo.NonNegativeReals,
                                     bounds=lambda model, e, t: (model.SOC_min_exst[e], model.SOC_max_exst[e]))
            model.ess_charge_binary_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, within=pyo.Binary)
            model.ess_discharge_binary_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, within=pyo.Binary)

        # ------------------------------------------------------------------
        # 5. Constraints
        # ------------------------------------------------------------------
        # 5.1) Slack angle
        def slack_angle_rule(model, t):
            return model.theta[slack_bus, t] == 0.0

        model.Constraint_SlackAngle = pyo.Constraint(model.Set_ts, rule=slack_angle_rule)

        # 5.2) DC Power flow (TN)
        def dc_power_flow_rule(model, l, t):
            fr_bus = net.data.net.bch[l - 1][0]
            to_bus = net.data.net.bch[l - 1][1]
            return model.Pf_tn[l, t] == model.B_tn[l] * (model.theta[fr_bus, t] - model.theta[to_bus, t])

        model.Constraint_DC_PowerFlow = pyo.Constraint(model.Set_bch_tn, model.Set_ts, rule=dc_power_flow_rule)

        # 5.3) Linearized DistFlow (DN)
        def lin_distflow_P_rule(model, l, t):
            i, j = net.data.net.bch[l - 1]
            return model.Pf_dn[l, t] == (model.V2_dn[i, t] - model.V2_dn[j, t]) / model.R_dn[l]

        model.Constraint_LinDistFlow_P = pyo.Constraint(model.Set_bch_dn, model.Set_ts,
                                                        rule=lin_distflow_P_rule)

        if has_reactive_demand:
            def lin_distflow_Q_rule(model, l, t):
                i, j = net.data.net.bch[l - 1]
                return model.Qf_dn[l, t] == (model.V2_dn[i, t] - model.V2_dn[j, t]) / model.X_dn[l]

            model.Constraint_LinDistFlow_Q = pyo.Constraint(model.Set_bch_dn, model.Set_ts,
                                                            rule=lin_distflow_Q_rule)

        # 5.4) Branch limits
        # - Branch limit at TN-level:
        def pmax_tn_upper_rule(model, l, t):
            return model.Pf_tn[l, t] <= model.Pmax_tn[l]

        def pmax_tn_lower_rule(model, l, t):
            return -model.Pmax_tn[l] <= model.Pf_tn[l, t]

        model.Constraint_Pf_tn_upper = pyo.Constraint(model.Set_bch_tn, model.Set_ts, rule=pmax_tn_upper_rule)
        model.Constraint_Pf_tn_lower = pyo.Constraint(model.Set_bch_tn, model.Set_ts, rule=pmax_tn_lower_rule)

        # - Branch limit at DN-level:
        def smax_dn_rule(model, l, t):
            return (model.Pf_dn[l, t]) ** 2 + (model.Qf_dn[l, t]) ** 2 <= (model.Smax_dn[l]) ** 2

        model.Constraint_Smax_dn = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=smax_dn_rule)

        # 5.5) Nodal balance
        # - Nodal balance at TN-level
        def power_balance_tn_rule(model, b, t):
            if b not in Set_bus_tn:
                return pyo.Constraint.Skip
            inflow_tn = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if net.data.net.bch[l - 1][1] == b)
            outflow_tn = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if net.data.net.bch[l - 1][0] == b)
            Pg_b = sum(model.Pg_exst[g, t] for g in model.Set_gen_exst if net.data.net.gen[g - 1] == b)
            Pgrid = (model.Pimp[t] - model.Pexp[t]) if b == slack_bus else 0.0

            Pess_net = 0.0
            if Set_ess_exst:
                Pess_net = sum(
                    model.Pess_discharge_exst[e, t] - model.Pess_charge_exst[e, t]
                    for e in model.Set_ess_exst
                    if hasattr(net.data.net, 'ess') and net.data.net.ess[e - 1] == b
                )

            return Pg_b + Pgrid + Pess_net + inflow_tn - outflow_tn == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_PowerBalance_TN = pyo.Constraint(model.Set_bus_tn, model.Set_ts,
                                                          rule=power_balance_tn_rule)

        # # - Nodal balance at DN-level
        def power_balance_dn_rule(model, b, t):
            # DN internal flows (D-level branches only)
            inflow_dn = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][1] == b)
            outflow_dn = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][0] == b)

            # Coupling flow (only branch_level == 'T-D')
            inflow_cpl = sum(model.Pf_tn[l, t]
                             for l in model.Set_bch_tn
                             if net.data.net.branch_level[l] == 'T-D'
                             and net.data.net.bch[l - 1][1] == b)
            outflow_cpl = sum(model.Pf_tn[l, t]
                              for l in model.Set_bch_tn
                              if net.data.net.branch_level[l] == 'T-D'
                              and net.data.net.bch[l - 1][0] == b)

            # Existing DG at DN buses
            Pg_exst = sum(model.Pg_exst[g, t] for g in model.Set_gen_exst if net.data.net.gen[g - 1] == b)

            # ESS (if any) at this DN bus: discharge adds, charge subtracts
            ess_dis = sum(
                model.Pess_discharge_exst[e, t] for e in getattr(model, 'Set_ess_exst', []) if net.data.net.ess[e - 1] == b)
            ess_ch = sum(
                model.Pess_charge_exst[e, t] for e in getattr(model, 'Set_ess_exst', []) if net.data.net.ess[e - 1] == b)

            return (inflow_dn + inflow_cpl + Pg_exst + ess_dis - 
                    (outflow_dn + outflow_cpl + model.Pd[b, t] - model.Pc[b, t] + ess_ch) ) == 0

        model.PowerBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts, rule=power_balance_dn_rule)

        if has_reactive_demand:
            def reactive_balance_dn_rule(model, b, t):
                inflow_q = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][1] == b)
                outflow_q = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][0] == b)
                Qg_b = sum(model.Qg_exst[g, t] for g in model.Set_gen_exst if net.data.net.gen[g - 1] == b)
                return Qg_b + inflow_q - outflow_q == model.Qd[b, t] - model.Qc[b, t]

            model.Constraint_ReactiveBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                                 rule=reactive_balance_dn_rule)

        # 5.6) Reactive power balance (DN)
        if has_reactive_demand:
            def reactive_balance_dn_rule(model, b, t):
                Qg = sum(model.Qg_exst[g, t] for g in model.Set_gen_exst if net.data.net.gen[g - 1] == b)
                Qf_in = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][1] == b)
                Qf_out = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][0] == b)
                return Qg + Qf_in - Qf_out == model.Qd[b, t] - model.Qc[b, t]

            model.Constraint_ReactiveBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                                 rule=reactive_balance_dn_rule)

        # # 5.7) Curtailment upper bound
        # def pc_limit_rule(model, b, t):
        #     return model.Pc[b, t] <= model.Pd[b, t]
        #
        # model.Constraint_Pc_limit = pyo.Constraint(model.Set_bus, model.Set_ts, rule=pc_limit_rule)

        # 5.8) Voltage limits
        model.Constraint_V2_max = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                 rule=lambda model, b, t: model.V2_dn[b, t] <= model.V2_max[b])
        model.Constraint_V2_min = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                 rule=lambda model, b, t: model.V2_dn[b, t] >= model.V2_min[b])

        # 5.9) ESS constraints
        if Set_ess_exst:
            # - Charge/discharge power limits
            def ess_charge_ub_rule(model, e, t):
                return model.Pess_charge_exst[e, t] <= model.Pess_max_exst[e] * model.ess_charge_binary_exst[e, t]

            model.ESS_ChargeUB = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_charge_ub_rule)

            def ess_discharge_ub_rule(model, e, t):
                return model.Pess_discharge_exst[e, t] <= model.Pess_max_exst[e] * model.ess_discharge_binary_exst[e, t]

            model.ESS_DischargeUB = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_discharge_ub_rule)

            # - Charge/discharge exclusivity
            def ess_exclusivity_rule(model, e, t):
                return model.ess_charge_binary_exst[e, t] + model.ess_discharge_binary_exst[e, t] <= 1

            model.ESS_Exclusivity = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_exclusivity_rule)

            # - SOC limits
            def ess_e_bounds_rule(model, e, t):
                return pyo.inequality(model.Eess_min_exst[e], model.Eess_exst[e, t], model.Eess_max_exst[e])

            model.ESS_Energy_Bounds = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_e_bounds_rule)

            def ess_soc_bounds_rule(model, e, t):
                return pyo.inequality(model.SOC_min_exst[e], model.SOC_exst[e, t], model.SOC_max_exst[e])

            model.ESS_SOC_Bounds = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_soc_bounds_rule)

            # - Link battery energy with SOC
            def ess_e_soc_link_rule(model, e, t):
                return model.Eess_exst[e, t] == model.SOC_exst[e, t] * model.Eess_max_exst[e]

            model.ESS_E_SOC_Link = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_e_soc_link_rule)

            # - Inter-temporal energy balance (i.e., ESS dynamics)
            ts_sorted = sorted(list(model.Set_ts.data()))

            def ess_dyn_rule(model, e, t):
                idx = ts_sorted.index(t)
                if idx == 0:
                    # initial E from initial SOC
                    E_prev = model.initial_SOC_exst[e] * model.Eess_max_exst[e]
                else:
                    E_prev = model.Eess_exst[e, ts_sorted[idx - 1]]
                return model.Eess_exst[e, t] == E_prev + model.eff_ch_exst[e] * model.Pess_charge_exst[e, t] - (
                            1 / model.eff_dis_exst[e]) * model.Pess_discharge_exst[e, t]

            model.ESS_Dynamics = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_dyn_rule)

        # ------------------------------------------------------------------
        # 6. Objective
        # ------------------------------------------------------------------
        def objective_rule(model):
            # Generation cost
            gen_cost = sum(
                model.gen_cost_coef[g, 0] + model.gen_cost_coef[g, 1] * model.Pg_exst[g, t]
                for g in model.Set_gen_exst for t in model.Set_ts
            )

            # Grid import/export cost
            grid_cost = sum(
                model.Pimp_cost * model.Pimp[t] + model.Pexp_cost * model.Pexp[t]
                for t in model.Set_ts
            )

            # Load shedding cost (should be minimal/zero in normal operation)
            ls_cost = sum(
                model.Pc_cost[b] * model.Pc[b, t]
                for b in model.Set_bus
                for t in model.Set_ts
            )

            if has_reactive_demand:
                ls_cost += sum(
                    model.Qc_cost[b] * model.Qc[b, t]
                    for b in model.Set_bus_dn
                    for t in model.Set_ts
                )

            # Apply scale factor to annualize costs
            return (gen_cost + grid_cost + ls_cost) * scale_factor

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return model, scale_factor

    def solve_normal_scenario_opf_model(self, model, solver='gurobi', print_summary=True):
        """
        Solve the normal scenario OPF model and extract cost components.

        Args:
            model: The built normal scenario OPF model
            solver: Solver to use (default: 'gurobi')
            print_summary: Whether to print a summary of results

        Returns:
            dict: Cost breakdown and key metrics
        """
        from pyomo.opt import SolverFactory
        from factories.network_factory import make_network

        opt = SolverFactory(solver)
        results = opt.solve(model, tee=False)

        if results.solver.status != 'ok':
            raise RuntimeError("Failed to solve normal operation OPF model")

        # Extract total cost (already annualized by scale factor)
        total_cost = pyo.value(model.Objective)

        # ------------------------------------------------------------------
        # Extract component costs (before scaling, then apply scale factor)
        # ------------------------------------------------------------------

        # 1. Generation costs - separate by level
        gen_cost_total = 0
        gen_cost_tn = 0
        gen_cost_dn = 0

        net = make_network(self.network_name)

        for g in model.Set_gen_exst:
            for t in model.Set_ts:
                # Calculate generation cost for this generator at this time
                pg_val = pyo.value(model.Pg_exst[g, t])
                cost = model.gen_cost_coef[g, 0] + model.gen_cost_coef[g, 1] * pg_val
                gen_cost_total += cost

                # Determine if generator is at TN or DN level
                gen_bus = net.data.net.gen[g - 1]

                if net.data.net.bus_level[gen_bus] == 'T':  # Direct dict access
                    gen_cost_tn += cost
                else:
                    gen_cost_dn += cost

        # Apply scale factor to annualize
        gen_cost_total *= model.scale_factor
        gen_cost_tn *= model.scale_factor
        gen_cost_dn *= model.scale_factor

        # 2. Grid import/export costs
        grid_cost = 0
        total_import = 0
        total_export = 0

        for t in model.Set_ts:
            import_val = pyo.value(model.Pimp[t])
            export_val = pyo.value(model.Pexp[t])
            total_import += import_val
            total_export += export_val
            grid_cost += model.Pimp_cost * import_val + model.Pexp_cost * export_val

        grid_cost *= model.scale_factor
        total_import *= model.scale_factor
        total_export *= model.scale_factor

        # 3. Load shedding costs (should be zero or minimal in normal operation)
        ls_cost_active = 0
        ls_cost_reactive = 0
        total_pc = 0
        total_qc = 0

        # Active load shedding
        for b in model.Set_bus:
            for t in model.Set_ts:
                pc_val = pyo.value(model.Pc[b, t])
                total_pc += pc_val
                ls_cost_active += model.Pc_cost[b] * pc_val

        # Reactive load shedding (only if reactive demand exists)
        has_reactive = hasattr(model, 'Qc') and not isinstance(model.Qc, pyo.Param)
        if has_reactive:
            for b in model.Set_bus_dn:
                for t in model.Set_ts:
                    qc_val = pyo.value(model.Qc[b, t])
                    total_qc += qc_val
                    ls_cost_reactive += model.Qc_cost[b] * qc_val

        ls_cost_total = (ls_cost_active + ls_cost_reactive) * model.scale_factor
        total_pc *= model.scale_factor
        total_qc *= model.scale_factor

        # 4. ESS operation statistics (if existing ESS)
        ess_stats = {}
        if hasattr(model, 'Set_ess_exst') and len(model.Set_ess_exst) > 0:
            total_charge = 0
            total_discharge = 0

            for e in model.Set_ess_exst:
                for t in model.Set_ts:
                    total_charge += pyo.value(model.Pess_charge_exst[e, t])
                    total_discharge += pyo.value(model.Pess_discharge_exst[e, t])

            ess_stats = {
                'total_charge_MWh': total_charge * model.scale_factor,
                'total_discharge_MWh': total_discharge * model.scale_factor,
                'cycling_efficiency_loss_MWh': (total_charge - total_discharge) * model.scale_factor
            }

        # ------------------------------------------------------------------
        # Prepare results dictionary
        # ------------------------------------------------------------------
        results_dict = {
            # Main costs (annualized)
            'total_cost': total_cost,
            'gen_cost_total': gen_cost_total,
            'gen_cost_tn': gen_cost_tn,
            'gen_cost_dn': gen_cost_dn,
            'grid_cost': grid_cost,
            'ls_cost_total': ls_cost_total,
            'ls_cost_active': ls_cost_active * model.scale_factor,
            'ls_cost_reactive': ls_cost_reactive * model.scale_factor,

            # Quantities (annualized)
            'total_import_MWh': total_import,
            'total_export_MWh': total_export,
            'total_pc_MWh': total_pc,
            'total_qc_MVArh': total_qc,

            # Model info
            'scale_factor': pyo.value(model.scale_factor),
            'hours_computed': len(model.Set_ts),
            'solver_status': str(results.solver.status),
            'termination_condition': str(results.solver.termination_condition),

            # ESS stats (if applicable)
            **ess_stats
        }

        # ------------------------------------------------------------------
        # Print summary if requested
        # ------------------------------------------------------------------
        if print_summary:
            print("\n" + "=" * 60)
            print("NORMAL OPERATION OPF RESULTS")
            print("=" * 60)

            print(f"Model solved: {results_dict['solver_status']}")
            print(f"Hours simulated: {results_dict['hours_computed']}")
            print(f"Scale factor: {results_dict['scale_factor']:.2f}")

            print("\nANNUALIZED COSTS:")
            print(f"  Total Cost: £{results_dict['total_cost']:,.2f}")
            print(f"  - Generation: £{results_dict['gen_cost_total']:,.2f}")
            print(f"    - TN level: £{results_dict['gen_cost_tn']:,.2f}")
            print(f"    - DN level: £{results_dict['gen_cost_dn']:,.2f}")
            print(f"  - Grid Import/Export: £{results_dict['grid_cost']:,.2f}")
            print(f"  - Load Shedding: £{results_dict['ls_cost_total']:,.2f}")

            if total_pc > 1e-6 or total_qc > 1e-6:
                print("\nWARNING: Load shedding detected in normal operation!")
                print(f"  - Active: {results_dict['total_pc_MWh']:.2f} MWh")
                print(f"  - Reactive: {results_dict['total_qc_MVArh']:.2f} MVArh")

            print("\nANNUALIZED ENERGY FLOWS:")
            print(f"  Grid Import: {results_dict['total_import_MWh']:,.2f} MWh")
            print(f"  Grid Export: {results_dict['total_export_MWh']:,.2f} MWh")

            if ess_stats:
                print("\nESS OPERATION:")
                print(f"  Total Charge: {ess_stats['total_charge_MWh']:,.2f} MWh")
                print(f"  Total Discharge: {ess_stats['total_discharge_MWh']:,.2f} MWh")
                print(f"  Efficiency Loss: {ess_stats['cycling_efficiency_loss_MWh']:,.2f} MWh")

            print("=" * 60 + "\n")

        return results_dict
