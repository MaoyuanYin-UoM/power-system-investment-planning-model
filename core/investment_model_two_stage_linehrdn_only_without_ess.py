"""
OLD VERSION of Two-Stage Stochastic Investment Planning Model

This is a fallback version that uses CONTINUOUS hardening variables instead of binary.

Key differences from the new version (investment_model_two_stage.py):
1. Line hardening: Continuous shift amount (hrdn_shift) with bounds [0, max_shift]
   - New version uses: Binary decision (harden or not) with fixed shift amount
   - Old version uses: Continuous optimization of shift amount
2. Investment options: Only line hardening (no DG installation)
3. Cost model: Proportional to hardening amount (cost_rate * hrdn_shift)
   - New version uses: Fixed cost per hardened line

Usage:
    from core.investment_model_two_stage_old import InvestmentClassOld
    inv = InvestmentClassOld()
    model = inv.build_investment_model(...)
"""

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
from pyomo.contrib.iis import write_iis  # Pyomo IIS tool (uses solver APIs under the hood)
from datetime import datetime

from factories.windstorm_factory import make_windstorm
from factories.network_factory import make_network

class Object(object):
    pass


class InvestmentClassOld():

    def __init__(self, obj=None):
        # Get default values from InvestmentConfig
        if obj is None:
            obj = InvestmentConfig()

        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def build_investment_model(self,
                               network_name: str = "29_bus_GB_transmission_network_with_Kearsley_GSP_group",
                               windstorm_name: str = "windstorm_29_bus_GB_transmission_network",
                               path_ws_scenario_library: str = None,
                               include_normal_scenario: bool = True,
                               normal_scenario_prob: float = 0.99,
                               use_representative_days: bool = True,
                               representative_days: list = None,
                               resilience_metric_threshold: float = None,
                               investment_budget: float = None,
                               solver_for_normal_opf: str = 'gurobi'):
        """
        Build a Pyomo MILP model for resilience enhancement investment planning (line hardening only)
        against windstorms, using the form of stochastic programming over multiple scenarios.

        OLD VERSION: This model uses continuous hardening variables (not binary).
        - Only line hardening investment (no DG)
        - Continuous hardening shift amount (instead of fixed binary decision)
        - Cost proportional to hardening amount

        * 1st-stage:   continuous line-hardening shift  x_hrdn_l  (in m/s, identical for TN & DN)
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

        # Store windstorm library info
        self.ws_library_path = path_ws_scenario_library  # Store the library path
        # self.ws_library_total_num_scenarios =   # todo: to be added

        # Read windstorm scenarios data (able to handle both old and new scenario format)
        if "scenarios" in data:
            # New library format: convert dictionary to list
            ws_scenarios = []
            scenario_id_mapping = {}  # not necessary but kept for potential debugging use

            for idx, (scenario_id, scenario_data) in enumerate(sorted(data["scenarios"].items()), start=1):
                # Use sequential 1-indexed IDs instead of extracting from scenario_id
                scenario_data["simulation_id"] = idx  # 1, 2, 3, ... (1-indexed scenario ID)
                scenario_id_mapping[idx] = scenario_id  # store original ID (in the form of e.g., 'ws_0038')
                ws_scenarios.append(scenario_data)

                # Optional: store the mapping for reference
            self.scenario_id_mapping = scenario_id_mapping
        else:
            raise ValueError("JSON does not contain 'scenarios'")

        # Store windstorm info into metadata
        self.meta = Object()
        self.meta.ws_seed = metadata.get("seed", metadata.get("base_seed", None))
        self.meta.total_num_ws_scenarios = metadata.get("number_of_ws_simulations",
                                                metadata.get("num_scenarios", len(ws_scenarios)))
        self.meta.num_ws_scenarios_used = len(data["scenarios"])
        self.meta.library_type = metadata.get("library_type", "windstorm_scenarios")

        # Extract the relative ws scenario probabilities
        self.meta.ws_scenario_probabilities_relative = None
        if "scenario_probabilities" in data:
            prob_dict = data["scenario_probabilities"]

            # Since we have the scenario IDs from data["scenarios"], extract in the same order
            if "scenarios" in data:
                # Extract probabilities in the same order as we created ws_scenarios (sorted by original ID)
                sorted_scenario_ids = sorted(data["scenarios"].keys())
                # Store as a list where index (i-1) corresponds to simulation_id i
                self.meta.ws_scenario_probabilities_relative = [
                    prob_dict.get(sc_id, 0) for sc_id in sorted_scenario_ids
                ]
            else:
                print("Warning: Could not extract scenario probabilities - 'scenarios' not found in data.")
        else:
            print("Warning: 'scenario_probabilities' not found in JSON file.")

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

        # Compute absolute ws scenario probabilities
        self.meta.ws_scenario_probabilities_abs = {}
        if include_normal_scenario:
            # Scale relative probabilities by the windstorm portion (1 - normal_prob)
            ws_total_prob = 1 - normal_scenario_prob
            for sc in Set_scn:
                # Access list with 0-based index (sc-1)
                if self.meta.ws_scenario_probabilities_relative:
                    rel_prob = self.meta.ws_scenario_probabilities_relative[sc - 1]
                else:
                    rel_prob = 1.0 / len(Set_scn)
                self.meta.ws_scenario_probabilities_abs[sc] = ws_total_prob * rel_prob
        else:
            # No normal scenario - ws scenarios get full probability
            for sc in Set_scn:
                if self.meta.ws_scenario_probabilities_relative:
                    self.meta.ws_scenario_probabilities_abs[sc] = self.meta.ws_scenario_probabilities_relative[
                        sc - 1]
                else:
                    self.meta.ws_scenario_probabilities_abs[sc] = 1.0 / len(Set_scn)

        # Store normal scenario probability for reference
        self.meta.normal_scenario_prob = normal_scenario_prob if include_normal_scenario else 0

        # ------------------------------------------------------------------
        # 1. Index sets -- tn for transmission level network, dn for distribution level network
        # ------------------------------------------------------------------
        print("\nCreating index sets...")

        # 1.1) Single sets
        # - Basic sets
        Set_bus = net.data.net.bus[:]
        Set_bch = list(range(1, len(net.data.net.bch) + 1))
        Set_gen = list(range(1, len(net.data.net.gen) + 1))

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

        # OLD VERSION: DG installation removed
        # No new DG installations in old version
        Set_gen_new_nren = []
        Set_gen_new_ren = []
        Set_gen_new = []
        Set_bus_dg_available = []
        new_gen_to_bus_map = {}

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
                        print(f"max_ttr = {max_ttr}")
                    else:
                        max_ttr = ws.data.WS.event.ttr[1]

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
        model.Set_bus_dg_available = pyo.Set(initialize=Set_bus_dg_available)

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
        # - Relative probabilities
        if self.meta.ws_scenario_probabilities_relative:
            # Use probabilities from clustered library
            scn_prob_relative = {sc: self.meta.ws_scenario_probabilities_relative[sc - 1] for sc in Set_scn}
        else:
            # Fallback to equal distribution among windstorm scenarios
            scn_prob_relative = {sc: 1.0 / len(Set_scn) for sc in Set_scn}

        model.scn_prob_relative = pyo.Param(model.Set_scn, initialize=scn_prob_relative)

        # - Absolute probabilities
        if self.meta.ws_scenario_probabilities_abs:
            # Use probabilities from clustered library (already scaled for normal scenario if needed)
            scn_prob_abs = {sc: self.meta.ws_scenario_probabilities_abs[sc] for sc in Set_scn}
        else:
            # Fallback to equal probabilities
            if include_normal_scenario:
                ws_total_prob = 1 - normal_scenario_prob
                scn_prob_abs = {sc: ws_total_prob / len(Set_scn) for sc in Set_scn}
            else:
                scn_prob_abs = {sc: 1 / len(Set_scn) for sc in Set_scn}

        model.scn_prob_abs = pyo.Param(model.Set_scn, initialize=scn_prob_abs)

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

        # 3.9) Cost-related parameters
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
                                   initialize={l: net.data.net.cost_bch_repair[l - 1]
                                               for l in range(1, len(net.data.net.bch) + 1)},
                                   within=pyo.NonNegativeReals)

        # OLD VERSION: Proportional hardening cost (per m/s) instead of fixed cost
        # Cost model: Total cost = hrdn_cost_rate_old (£/km/(m/s)) × line_length (km) × shift_amount (m/s)
        # Check for hrdn_cost_rate_old (cost per km per m/s), then multiply by line length
        if hasattr(net.data.net, 'hrdn_cost_rate_old'):
            # Multiply rate by line length to get cost per m/s for each line
            hrdn_cost_rate_dict = {
                l: net.data.net.hrdn_cost_rate_old[l - 1] * net.data.net.bch_length_km[l - 1]
                for l in model.Set_bch_hrdn_lines
            }
        else:
            reference_shift = net.data.net.bch_hrdn_limits[1]  # Use max shift as reference
            print(f"  WARNING: 'hrdn_cost_rate_old' not found. Deriving cost rate from 'cost_bch_hrdn_fixed' / {reference_shift} m/s")
            hrdn_cost_rate_dict = {
                l: net.data.net.cost_bch_hrdn_fixed[l - 1] / reference_shift
                for l in model.Set_bch_hrdn_lines
            }

        model.line_hrdn_cost_rate = pyo.Param(
            model.Set_bch_hrdn_lines,
            initialize=hrdn_cost_rate_dict,
            within=pyo.NonNegativeReals,
            doc="Cost per m/s of hardening shift (= rate × line_length)"
        )
        model.dg_install_cost = pyo.Param(model.Set_gen_new_nren,
                                          initialize={g: net.data.net.dg_install_cost[new_gen_to_bus_map[g] - 1]
                                                      for g in Set_gen_new_nren})
        model.investment_budget = pyo.Param(initialize=net.data.investment_budget
                                                       if net.data.investment_budget else 1e10)

        # 3.12) Line hardening related parameters (OLD VERSION: continuous bounds instead of fixed shift)
        model.hrdn_shift_min = pyo.Param(
            model.Set_bch_hrdn_lines,
            initialize={l: net.data.bch_hrdn_limits[0] for l in model.Set_bch_hrdn_lines},
            within=pyo.NonNegativeReals,
            doc="Minimum hardening shift (m/s)"
        )
        model.hrdn_shift_max = pyo.Param(
            model.Set_bch_hrdn_lines,
            initialize={l: net.data.bch_hrdn_limits[1] for l in model.Set_bch_hrdn_lines},
            within=pyo.NonNegativeReals,
            doc="Maximum hardening shift (m/s)"
        )

        # 3.13) Other parameters
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
        # - Line hardening (OLD VERSION: continuous shift amount in m/s, not binary)
        model.hrdn_shift = pyo.Var(
            model.Set_bch_hrdn_lines,
            within=pyo.NonNegativeReals,
            bounds=lambda model, l: (model.hrdn_shift_min[l], model.hrdn_shift_max[l]),
            doc="Continuous hardening shift amount (m/s)"
        )

        # - (Gas) DG installation capacity (continuous, MW)
        model.dg_install_capacity = pyo.Var(
            model.Set_gen_new_nren,
            within=pyo.NonNegativeReals,
            bounds=lambda model, g: (model.dg_install_capacity_min[g],
                                     model.dg_install_capacity_max[g])
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
        model.Pc = pyo.Var(model.Set_sbt, within=pyo.NonNegativeReals,
                           bounds=lambda model, s, b, t: (0, max(0.0, pyo.value(model.Pd[s, b, t]))))
        if has_reactive_demand:
            model.Qc = pyo.Var(model.Set_sbt_dn, within=pyo.NonNegativeReals,
                               bounds=lambda model, s, b, t: (0, max(0.0, pyo.value(model.Pd[s, b, t]))))
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
            """Total investment cost must not exceed budget (OLD VERSION: only hardening, continuous cost)"""
            # OLD VERSION: Continuous hardening - cost proportional to shift amount
            total_line_hrdn_cost = sum(model.hrdn_shift[l] * model.line_hrdn_cost_rate[l]
                                       for l in model.Set_bch_hrdn_lines)

            # DG removed in old version (set is empty, so this sum = 0)
            total_dg_install_cost = sum(model.dg_install_capacity[g] * model.dg_install_cost[g]
                                        for g in model.Set_gen_new_nren)

            return total_line_hrdn_cost + total_dg_install_cost <= model.investment_budget

        model.Constraint_InvestmentBudget = pyo.Constraint(rule=investment_budget_rule)

        # 5.2) Shifted gust speed (i.e. Line hardening) -- OLD VERSION: continuous hardening shift
        def shifted_gust_rule(model, sc, l, t):
            # OLD VERSION: Continuous hardening shift (not binary * fixed)
            if l in model.Set_bch_hrdn_lines:
                hrdn = model.hrdn_shift[l]  # Continuous shift amount
            else:
                hrdn = 0
            return model.shifted_gust_speed[sc, l, t] >= model.gust_speed[sc, t] - hrdn

        model.Constraint_ShiftedGust = pyo.Constraint(model.Set_slt_lines, rule=shifted_gust_rule)

        print("  - Piecewise fragility constraints...")
        # 5.3) Piece-wise fragility with line-specific breakpoints
        fragility_data = self.piecewise_linearize_fragility(net, line_idx=Set_bch_lines, num_pieces=4)

        # Group lines by their breakpoints
        breakpoint_groups = {}
        for l, data in fragility_data.items():
            # Use tuple of breakpoints as key
            bpts_tuple = tuple(data["breakpoints"])
            if bpts_tuple not in breakpoint_groups:
                breakpoint_groups[bpts_tuple] = {
                    "lines": [],
                    "breakpoints": data["breakpoints"],
                    "probs_by_line": {}
                }
            breakpoint_groups[bpts_tuple]["lines"].append(l)
            breakpoint_groups[bpts_tuple]["probs_by_line"][l] = data["probabilities"]

        # Create a separate Piecewise component for each group
        for group_idx, (bpts_tuple, group_data) in enumerate(breakpoint_groups.items()):
            lines_in_group = group_data["lines"]
            breakpoints = group_data["breakpoints"]
            probs_by_line = group_data["probs_by_line"]

            # Create index set for this group
            group_index_set = [(sc, l, t)
                               for sc in model.Set_scn
                               for l in lines_in_group
                               for t in Set_ts_scn[sc]]

            # Create fragility rule for this group
            def make_fragility_rule(probs_dict, bpts):
                def fragility_rule(model, sc, l, t, x):
                    # Get the probabilities for this specific line
                    probs = probs_dict[l]
                    # Find the index for interpolation
                    for i in range(len(bpts) - 1):
                        if bpts[i] <= x <= bpts[i + 1]:
                            # Linear interpolation
                            weight = (x - bpts[i]) / (bpts[i + 1] - bpts[i])
                            return probs[i] * (1 - weight) + probs[i + 1] * weight
                    # Edge cases
                    if x <= bpts[0]:
                        return probs[0]
                    else:
                        return probs[-1]

                return fragility_rule

            # Create the Piecewise component for this group
            piecewise_name = f'Piecewise_Fragility_Group_{group_idx}'
            setattr(model, piecewise_name,
                    pyo.Piecewise(
                        group_index_set,
                        model.fail_prob,
                        model.shifted_gust_speed,
                        pw_pts=breakpoints,
                        f_rule=make_fragility_rule(probs_by_line, breakpoints),
                        pw_constr_type="EQ",
                        pw_repn="DCC"
                    ))

            print(f"    - Created {piecewise_name}: {len(lines_in_group)} lines, "
                  f"{len(breakpoints)} breakpoints [{breakpoints[0]:.1f}, {breakpoints[-1]:.1f}]")

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
            """Active power balance at TN buses including both existing and new DG"""
            # Existing generation
            Pg_exst = sum(model.Pg_exst[sc, g, t] for g in model.Set_gen_exst
                          if net.data.net.gen[g - 1] == b)

            # New DG generation (typically none at TN level, but included for completeness)
            Pg_new = sum(model.Pg_new[sc, g, t] for g in model.Set_gen_new_nren
                         if new_gen_to_bus_map[g] == b)

            # Power flows
            Pf_in = sum(model.Pf_tn[sc, l, t] for l in model.Set_bch_tn
                        if net.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf_tn[sc, l, t] for l in model.Set_bch_tn
                         if net.data.net.bch[l - 1][0] == b)

            # Grid import/export at slack bus
            Pgrid = (model.Pimp[sc, t] - model.Pexp[sc, t]) if b == slack_bus else 0

            return (Pg_exst + Pg_new + Pgrid + Pf_in - Pf_out
                    == model.Pd[sc, b, t] - model.Pc[sc, b, t])

        model.Constraint_PBalance_TN = pyo.Constraint(model.Set_sbt_tn, rule=P_balance_tn)

        def P_balance_dn(model, sc, b, t):
            """Active power balance at DN buses including both existing and new DG"""
            # Existing generation
            Pg_exst = sum(model.Pg_exst[sc, g, t] for g in model.Set_gen_exst
                          if net.data.net.gen[g - 1] == b)

            # New DG generation
            Pg_new = sum(model.Pg_new[sc, g, t] for g in model.Set_gen_new_nren
                         if new_gen_to_bus_map[g] == b)

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

            return (Pg_exst + Pg_new + Pf_in_dn - Pf_out_dn +
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

        # 5.9) Slack-bus reference theta = 0
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
        # Code 'expected total operational cost at dn level across all ws scenarios' as an expression
        def ws_expected_total_op_cost_relprob_dn_rule(model):
            # 1) DN generation cost from EXISTING generators
            gen_cost_dn_exst = sum(
                model.scn_prob_relative[sc] * (
                            model.gen_cost_coef_exst[g, 0] + model.gen_cost_coef_exst[g, 1] * model.Pg_exst[sc, g, t])
                for (sc, g, t) in model.Set_sgt_dn_exst
            )

            # 2) DN generation cost from NEW generators
            gen_cost_dn_new = sum(
                model.scn_prob_relative[sc] * (
                            model.gen_cost_coef_new[g, 0] + model.gen_cost_coef_new[g, 1] * model.Pg_new[sc, g, t])
                for (sc, g, t) in model.Set_sgt_dn_new
            )

            # 3) DN active load-shedding
            active_ls_cost_dn = sum(
                model.scn_prob_relative[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 4) DN reactive load-shedding
            reactive_ls_cost_dn = sum(
                model.scn_prob_relative[sc] * model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 5) DN line repair
            rep_cost_dn = sum(
                model.scn_prob_relative[sc] * model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_dn_lines
            )

            return gen_cost_dn_exst + gen_cost_dn_new + active_ls_cost_dn + reactive_ls_cost_dn + rep_cost_dn

        model.ws_exp_total_op_cost_relprob_dn_expr = pyo.Expression(rule=ws_expected_total_op_cost_relprob_dn_rule)

        # Code "Expected Energy Not Supplied (EENS) at dn level across all ws scenarios" as an expression
        # Since we have hourly resolution, EENS = sum of all active power load shedding
        def ws_expected_total_eens_relprob_dn_rule(model):
            # EENS = sum of all active power load shedding, weighted by RELATIVE ws probability
            # This gives the expected value across windstorm scenarios only
            eens_dn = sum(
                model.scn_prob_relative[sc] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )
            return eens_dn

        model.ws_exp_total_eens_relprob_dn_expr = pyo.Expression(rule=ws_expected_total_eens_relprob_dn_rule)

        if resilience_metric_threshold is not None:
            model.Constraint_ResilienceLevel = pyo.Constraint(
                # expr=model.exp_total_op_cost_dn_expr <= model.resilience_metric_threshold
                expr=model.ws_exp_total_eens_relprob_dn_expr <= model.resilience_metric_threshold
            )

        # ------------------------------------------------------------------
        # 6. Objective
        # ------------------------------------------------------------------
        print("Setting up objective function...")

        # Code total investment cost as an expression
        def total_inv_cost_rule(model):
            # OLD VERSION: Continuous hardening cost (proportional to shift amount)
            line_hrdn_cost = sum(model.line_hrdn_cost_rate[l] * model.hrdn_shift[l]
                                 for l in model.Set_bch_hrdn_lines)

            # DG removed in old version (set is empty, so this sum = 0)
            dg_install_cost = sum(model.dg_install_cost[g] * model.dg_install_capacity[g]
                                  for g in model.Set_gen_new_nren)

            return line_hrdn_cost + dg_install_cost

        model.total_inv_cost_expr = pyo.Expression(rule=total_inv_cost_rule)

        # Expression: total operational cost during the windstorm hours calculated using absolute probabilities
        def ws_exp_total_op_cost_absprob_expr(model):
            # All components already use model.scn_prob_abs[sc] correctly
            # 1) Generation cost from existing generators
            gen_cost_exst = sum(
                model.scn_prob_abs[sc] * (
                        model.gen_cost_coef_exst[g, 0] + model.gen_cost_coef_exst[g, 1] * model.Pg_exst[sc, g, t])
                for (sc, g, t) in model.Set_sgt_exst
            )

            # 2) Generation cost from new generators
            gen_cost_new = sum(
                model.scn_prob_abs[sc] * (
                        model.gen_cost_coef_new[g, 0] + model.gen_cost_coef_new[g, 1] * model.Pg_new[sc, g, t])
                for (sc, g, t) in model.Set_sgt_new
            )

            # 3) Grid import/export
            imp_exp_cost = sum(
                model.scn_prob_abs[sc] * (model.Pimp_cost * model.Pimp[sc, t] + model.Pexp_cost * model.Pexp[sc, t])
                for (sc, t) in model.Set_st
            )

            # 4) Active load-shedding
            act_ls_cost = sum(
                model.scn_prob_abs[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt
            )

            # 5) Reactive load-shedding (DN buses only)
            reac_ls_cost = sum(
                model.scn_prob_abs[sc] * model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 6) Line repair
            rep_cost = sum(
                model.scn_prob_abs[sc] * model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_lines
            )

            # Total windstorm window costs ONLY (no normal annual cost added)
            ws_window_cost = gen_cost_exst + gen_cost_new + imp_exp_cost + act_ls_cost + reac_ls_cost + rep_cost

            # REMOVED: Don't add annual_normal_cost here
            return ws_window_cost

        model.ws_exp_total_op_cost_absprob_expr = pyo.Expression(rule=ws_exp_total_op_cost_absprob_expr)

        print(model.scn_prob_abs[sc] for sc in model.Set_scn)

        def objective_rule(model):
            # First-stage: Investment cost
            inv_cost = model.total_inv_cost_expr

            # Calculate present value factor for annuity
            r = model.discount_rate
            n = model.asset_lifetime
            if r == 0:
                pv_factor = n
            else:
                pv_factor = (1 - (1 + r) ** (-n)) / r

            # Second-stage: Expected operational cost
            if include_normal_scenario and normal_operation_opf_results:
                # Normal scenario contributes its probability * annual cost
                normal_contribution = normal_scenario_prob * normal_operation_opf_results["total_cost"]
                # Windstorm scenarios contribute their expected cost (already probability-weighted)
                ws_contribution = model.ws_exp_total_op_cost_absprob_expr

                # Total annual operational cost
                annual_op_cost = normal_contribution + ws_contribution
                return inv_cost + pv_factor * annual_op_cost
            else:
                # Only windstorm scenarios
                return inv_cost + pv_factor * model.ws_exp_total_op_cost_absprob_expr

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


        # ------------------------------------------------------------------
        # 7. Additional expressions (Recorded for future result writing or data post-processing)
        # ------------------------------------------------------------------
        print("Defining additional expressions...")

        def total_inv_cost_line_hrdn_rule(model):
            # OLD VERSION: Continuous hardening cost
            return sum(model.line_hrdn_cost_rate[l] * model.hrdn_shift[l]
                       for l in model.Set_bch_hrdn_lines)

        model.total_inv_cost_line_hrdn_expr = pyo.Expression(rule=total_inv_cost_line_hrdn_rule)

        def total_inv_cost_dg_rule(model):
            return sum(model.dg_install_cost[g] * model.dg_install_capacity[g]
                       for g in model.Set_gen_new_nren)

        model.total_inv_cost_dg_expr = pyo.Expression(rule=total_inv_cost_dg_rule)

        def ws_exp_total_op_cost_relprob_rule(model):
            # 1) Generation cost from existing generators
            gen_cost_exst = sum(
                model.scn_prob_relative[sc] * (
                        model.gen_cost_coef_exst[g, 0] + model.gen_cost_coef_exst[g, 1] * model.Pg_exst[sc, g, t])
                for (sc, g, t) in model.Set_sgt_exst
            )

            # 2) Generation cost from new generators
            gen_cost_new = sum(
                model.scn_prob_relative[sc] * (
                        model.gen_cost_coef_new[g, 0] + model.gen_cost_coef_new[g, 1] * model.Pg_new[sc, g, t])
                for (sc, g, t) in model.Set_sgt_new
            )

            # 3) Grid import/export
            imp_exp_cost = sum(
                model.scn_prob_relative[sc] * (
                            model.Pimp_cost * model.Pimp[sc, t] + model.Pexp_cost * model.Pexp[sc, t])
                for (sc, t) in model.Set_st
            )

            # 4) Active load-shedding
            act_ls_cost = sum(
                model.scn_prob_relative[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt
            )

            # 5) Reactive load-shedding (DN buses only)
            reac_ls_cost = sum(
                model.scn_prob_relative[sc] * model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 6) Line repair
            rep_cost = sum(
                model.scn_prob_relative[sc] * model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_lines
            )

            return gen_cost_exst + gen_cost_new + imp_exp_cost + act_ls_cost + reac_ls_cost + rep_cost

        model.ws_exp_total_op_cost_relprob_expr = pyo.Expression(rule=ws_exp_total_op_cost_relprob_rule)

        def ws_exp_total_op_cost_absprob_dn_rule(model):
            # 1) DN generation cost from existing generators
            gen_cost_dn_exst = sum(
                model.scn_prob_abs[sc] * (
                        model.gen_cost_coef_exst[g, 0] + model.gen_cost_coef_exst[g, 1] * model.Pg_exst[sc, g, t])
                for (sc, g, t) in model.Set_sgt_dn_exst
            )

            # 2) DN generation cost from new generators
            gen_cost_dn_new = sum(
                model.scn_prob_abs[sc] * (
                        model.gen_cost_coef_new[g, 0] + model.gen_cost_coef_new[g, 1] * model.Pg_new[sc, g, t])
                for (sc, g, t) in model.Set_sgt_dn_new
            )

            # 3) DN active load-shedding
            active_ls_cost_dn = sum(
                model.scn_prob_abs[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 4) DN reactive load-shedding
            reactive_ls_cost_dn = sum(
                model.scn_prob_abs[sc] * model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 5) DN line repair
            rep_cost_dn = sum(
                model.scn_prob_abs[sc] * model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_dn_lines
            )

            return gen_cost_dn_exst + gen_cost_dn_new + active_ls_cost_dn + reactive_ls_cost_dn + rep_cost_dn

        model.ws_exp_total_op_cost_absprob_dn_expr = pyo.Expression(rule=ws_exp_total_op_cost_absprob_dn_rule)

        def ws_exp_total_eens_absprob_dn_rule(model):
            # EENS = sum of all active power load shedding, weighted by ABSOLUTE ws probability
            eens_dn = sum(
                model.scn_prob_abs[sc] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )
            return eens_dn

        model.ws_exp_total_eens_absprob_dn_expr = pyo.Expression(rule=ws_exp_total_eens_absprob_dn_rule)

        def ws_exp_total_ls_cost_absprob_dn_rule(model):
            # Active load-shedding cost at DN level with ABSOLUTE probability
            active_ls_cost_dn = sum(
                model.scn_prob_abs[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )
            return active_ls_cost_dn

        model.ws_exp_total_ls_cost_absprob_dn_expr = pyo.Expression(rule=ws_exp_total_ls_cost_absprob_dn_rule)

        def ws_exp_total_ls_cost_relprob_dn_rule(model):
            # Active load-shedding cost at DN level with RELATIVE probability
            active_ls_cost_dn = sum(
                model.scn_prob_relative[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )
            return active_ls_cost_dn

        model.ws_exp_total_ls_cost_relprob_dn_expr = pyo.Expression(rule=ws_exp_total_ls_cost_relprob_dn_rule)

        if include_normal_scenario and normal_operation_opf_results:
            model.normal_annual_op_cost_absprob = pyo.Param(
                initialize=normal_scenario_prob * normal_operation_opf_results["total_cost"],
                mutable=False
            )

            model.normal_annual_op_cost_relprob = pyo.Param(
                initialize=1.0 * normal_operation_opf_results["total_cost"],
                mutable=False
            )

            model.normal_gen_cost_absprob_dn = pyo.Param(
                initialize=normal_scenario_prob * normal_operation_opf_results["gen_cost_dn"],
                mutable=False
            )

            model.normal_gen_cost_relprob_dn = pyo.Param(
                initialize=1.0 * normal_operation_opf_results["gen_cost_dn"],
                mutable=False
            )


        print("\nModel building completed.")
        print("=" * 60)

        return model

    def solve_investment_model(
            self,
            model,
            solver_name: str = "gurobi",
            mip_gap: float = 1e-3,
            mip_gap_abs: float = 1e5,
            time_limit: int = 3600,
            numeric_focus: int = 0,
            mip_focus: int = 0,
            method: int = -1,
            heuristics: float = 0.2,
            cuts: int = 1,
            presolve: int = -1,
            write_lp: bool = False,
            write_result: bool = False,
            result_path: str = None,
            log_file_path: str = None,
            additional_notes: str = None,
            print_gap_callback: bool = False,
            gap_print_interval: float = 10.0,  # seconds between prints
            **solver_options
    ):
        """
        Solve a Pyomo investment model and export essential results.

        Parameters
        ----------
        mip_gap       : Absolute/relative MIP gap (default 0.5 %)
        time_limit    : Wall-clock limit in seconds
        write_lp      : Dump the model to LP file (suggest doing this only for debugging)
        result_path   : If not specified, a unique file name will be assigned
        solver_options: Extra options forwarded to the solver
        """

        # --- write LP file if requested ---------------------------------------
        if write_lp:
            write_lp_path = "Output_Results/LP_Models/debug_model.lp"
            print(f"\nWriting model as a .lp file into: \"{write_lp_path}\" ...")
            model.write(write_lp_path, io_options={"symbolic_solver_labels": True})

        print(f"\nSolving model with {solver_name} (time limit: {time_limit}s)...")

        opt = SolverFactory(solver_name)

        # --- solver-specific options -------------------------------------------------
        # If we want a precise live gap, use the Gurobi persistent interface so we can attach a callback.
        use_persistent = (solver_name.lower() == "gurobi") and print_gap_callback

        if use_persistent:
            # --- Persistent solver so we can register a Gurobi callback
            opt = pyo.SolverFactory("gurobi_persistent")
            opt.set_instance(model)

            # Map your options into real Gurobi params on the persistent solver
            def set_param(k, v):
                try:
                    opt.set_gurobi_param(k, v)
                except Exception:
                    # Fallback: some names differ slightly; ignore unknowns
                    pass

            # Default Gurobi options (keep your existing ones)
            # We mirror the options you already set in the non-persistent branch.
            set_param("MIPGap", mip_gap)
            set_param("MIPGapAbs", mip_gap_abs)
            set_param("TimeLimit", time_limit)
            set_param("NumericFocus", numeric_focus)
            set_param("MIPFocus", mip_focus)
            set_param("Method", method)
            set_param("Heuristics", heuristics)
            set_param("Cuts", cuts)
            set_param("Presolve", presolve)
            set_param("DisplayInterval", 1)  # show frequent updates

            # Create default log file path if not provided
            if log_file_path is None:
                from pathlib import Path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = Path(__file__).parent.parent / "Optimization_Results" / "Solver_Logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file_path = str(log_dir / f"gurobi_log_{timestamp}.log")
                print(f"Writing detailed Gurobi log to: {log_file_path}")

            # Set the log file parameter
            set_param("LogFile", log_file_path)
            set_param("LogToConsole", 1)

            # -------------------------
            # Register the progress callback
            # -------------------------
            import math
            import gurobipy as gp

            last_print_time = {"t": -1e9}  # mutable closure to throttle prints

            def _progress_cb(cb_m, cb_opt, cb_where):
                # Pyomo persistent callback signature: (pyomo_model, solver_interface, where)
                # Use cb_opt.cbGet(...) to query runtime / best / bound.
                if cb_where == gp.GRB.Callback.MIP:
                    runtime = cb_opt.cbGet(gp.GRB.Callback.RUNTIME)

                    # throttle printing
                    if runtime - last_print_time["t"] < max(0.1, float(gap_print_interval)):
                        return

                    best = cb_opt.cbGet(gp.GRB.Callback.MIP_OBJBST)
                    bnd = cb_opt.cbGet(gp.GRB.Callback.MIP_OBJBND)

                    gap = None
                    if best not in (None, gp.GRB.INFINITY) and not (math.isinf(best) or math.isnan(best)):
                        if bnd not in (None,) and not (math.isinf(bnd) or math.isnan(bnd)):
                            denom = max(1e-16, abs(best))
                            gap = abs(best - bnd) / denom

                    if gap is not None:
                        # e.g., Gap=0.00000087%
                        print(f"[{runtime:8.1f}s] Best={best:.10g}  Bound={bnd:.10g}  Gap={gap:.8%}")

                    last_print_time["t"] = runtime

            opt.set_callback(_progress_cb)

        else:
            # --- Your existing non-persistent path (unchanged) ---
            opt = SolverFactory(solver_name)

            if solver_name.lower() == "gurobi":
                # Create log file path if not specified (kept from your code)
                if log_file_path is None:
                    from pathlib import Path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_dir = Path(__file__).parent.parent / "Optimization_Results" / "Solver_Logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_file_path = str(log_dir / f"gurobi_log_{timestamp}.log")
                    print("debug message")
                    print(f"log_file_path: {log_file_path}")

                default_opts = {
                    "MIPGap": mip_gap,
                    "MIPGapAbs": mip_gap_abs,
                    "TimeLimit": time_limit,
                    "NumericFocus": numeric_focus,
                    "MIPFocus": mip_focus,
                    "Method": method,
                    "Heuristics": heuristics,
                    "LogFile": log_file_path,
                    "LogToConsole": 1,
                    "DisplayInterval": 5,
                }
                print(f"Writing detailed Gurobi log to: {log_file_path}")
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
                    "tmlim": time_limit,
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

        if term_cond == pyo.TerminationCondition.infeasible:
            print("\n" + "=" * 60)
            print("MODEL IS INFEASIBLE - Computing IIS...")
            print("=" * 60)

            try:
                # --- paths (absolute, under project root) ---------------------------
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent
                lp_dir = project_root / "Optimization_Results" / "LP_Models"
                iis_dir = project_root / "Optimization_Results" / "IIS"
                lp_dir.mkdir(parents=True, exist_ok=True)
                iis_dir.mkdir(parents=True, exist_ok=True)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                lp_path = lp_dir / f"infeasible_{ts}.lp"
                iis_path = iis_dir / f"iis_{ts}.ilp"

                # --- write the LP snapshot with readable names ----------------------
                model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
                print(f"Wrote LP snapshot to: {lp_path}")

                # --- preferred: use Pyomo’s IIS writer (Gurobi/CPLEX/Xpress supported) ---
                out_file = write_iis(model, str(iis_path), solver="gurobi")
                print(f"✓ IIS written to: {out_file}")
                print("Open the .ilp in a text editor to see conflicting constraints/bounds.")

            except Exception as e:
                # Fallback: call gurobi_cl on the LP to compute IIS
                print(f"Pyomo IIS utility failed ({e}). Falling back to gurobi_cl...")
                try:
                    import subprocess
                    cmd = [
                        "gurobi_cl",
                        str(lp_path),  # model file
                        "IISMethod=-1",  # choose IIS algorithm (optional)
                        f"ResultFile={iis_path}"  # .ilp output
                    ]
                    print("Running:", " ".join(cmd))
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.stdout:
                        print(res.stdout[-1000:])  # tail of gurobi_cl output for context

                    if iis_path.exists():
                        print(f"✓ IIS written to: {iis_path}")
                    else:
                        print("✗ IIS file not created. Try running the above command manually in a terminal.")
                except Exception as ee:
                    print(f"Fallback IIS attempt failed: {ee}")



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
                fname += f"_{self.meta.total_num_ws_scenarios}_ws"

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

            self._write_selected_variables_to_excel(model, result_path, additional_notes=additional_notes)
            print(f"Results written to: {result_path}")

        return {
            "objective": best_obj,
            "gap": mip_gap_out,
            "status": str(status),
            "termination": str(term_cond),
            # "runtime_s": results.solver.time
        }

    def _write_selected_variables_to_excel(self, model,
                                           path: str = None,
                                           meta: dict | None = None,
                                           additional_notes: str = None,
                                           ):
        """
        Export values of selected variables and expressions to a multi-sheet .xlsx workbook.
        """
        important = (
            "hrdn_shift",  # OLD VERSION: continuous hardening variable (not binary)
            "dg_install_capacity",

            "Pg", "Qg", "Pc", "Qc",
            "Pf_tn", "Pf_dn",
            "rand_num", "shifted_gust_speed", "branch_status", "fail_prob", "fail_condition",
            "impacted_branches", "fail_indicator",
            "fail_applies", "repair_applies"
        )

        path = Path(path).with_suffix(".xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Construct metadata
        if meta is None:
            meta = {
                "written_at": datetime.now().isoformat(timespec="seconds"),
                "network_name": self.network_name,
                "windstorm_name": self.windstorm_name,

                "ws_library_path": getattr(self, 'ws_library_path', 'Not available'),
                "ws_library_total_num_scenarios": getattr(self.meta, 'total_num_ws_scenarios'),
                "windstorm_library_random_seed": getattr(self.meta, 'ws_seed', None),
                "num_ws_scenarios_used": getattr(self.meta, 'num_ws_scenarios_used', 0),
                "ws_scenario_probs_abs": str(getattr(self.meta, 'ws_scenario_probabilities_abs', [])),

                "normal_scenario_included": getattr(self.meta, 'normal_scenario_included', False),
                "normal_scenario_probability": getattr(self.meta, 'normal_scenario_prob', 0),

                "resilience_metric_threshold": float(pyo.value(model.resilience_metric_threshold)),
                "objective_value": float(pyo.value(model.Objective)),

                "total_investment_cost": float(pyo.value(model.total_inv_cost_expr)),
                "investment_cost_line_hardening": float(pyo.value(model.total_inv_cost_line_hrdn_expr)),
                "investment_cost_dg": float(pyo.value(model.total_inv_cost_dg_expr)),

                "ws_exp_total_eens_absprob_dn": float(pyo.value(model.ws_exp_total_eens_absprob_dn_expr)),
                "ws_exp_total_eens_relprob_dn": float(pyo.value(model.ws_exp_total_eens_relprob_dn_expr)),

                "ws_expected_total_operational_cost_absprob": float(pyo.value(model.ws_exp_total_op_cost_absprob_expr)),
                "ws_expected_total_operational_cost_relprob": float(pyo.value(model.ws_exp_total_op_cost_relprob_expr)),

                "ws_expected_total_operational_cost_absprob_dn": float(pyo.value(model.ws_exp_total_op_cost_absprob_dn_expr)),
                "ws_expected_total_operational_cost_relprob_dn": float(pyo.value(model.ws_exp_total_op_cost_relprob_dn_expr)),

                "ws_expected_total_ls_cost_absprob_dn_rule": float(pyo.value(model.ws_exp_total_ls_cost_absprob_dn_expr)),
                "ws_expected_total_ls_cost_relprob_dn_rule": float(pyo.value(model.ws_exp_total_ls_cost_relprob_dn_expr)),
            }

        # Add normal scenario costs if included
        if hasattr(model, 'normal_annual_op_cost_absprob'):
            meta["normal_annual_op_cost_absprob"] = float(pyo.value(model.normal_annual_op_cost_absprob))

            meta["expected_total_operational_cost_(objective_value)"] = (
                    float(pyo.value(model.ws_exp_total_op_cost_absprob_expr)) +
                    float(pyo.value(model.normal_annual_op_cost_absprob))
            )

        else:
            meta["expected_total_operational_cost_(objective_value)"] = (
                    float(pyo.value(model.ws_exp_total_op_cost_relprob_expr))
            )

        if hasattr(model, 'normal_annual_op_cost_relprob'):
            meta["normal_annual_op_cost_relprob"] = float(pyo.value(model.normal_annual_op_cost_relprob))

        # Also add the DN generation cost from normal scenario if available
        if hasattr(model, 'normal_gen_cost_absprob_dn'):
            meta["normal_gen_cost_absprob_dn"] = float(pyo.value(model.normal_gen_cost_absprob_dn))

        if hasattr(model, 'normal_gen_cost_relprob_dn'):
            meta["normal_gen_cost_relprob_dn"] = float(pyo.value(model.normal_gen_cost_relprob_dn))

        # Add detailed normal operation info if available
        if hasattr(self.meta, 'normal_operation_opf_results') and self.meta.normal_operation_opf_results:
            meta["normal_hours_computed"] = self.meta.normal_operation_opf_results.get("hours_computed", "N/A")
            meta["normal_scale_factor"] = self.meta.normal_operation_opf_results.get("scale_factor", "N/A")
            meta["normal_solver_status"] = self.meta.normal_operation_opf_results.get("solver_status", "N/A")
            if hasattr(self.meta.normal_operation_opf_results, 'representative_days'):
                meta["normal_representative_days"] = str(
                    self.meta.normal_operation_opf_results.get("representative_days", "N/A"))

        # Add additional notes
        meta["additional_notes"] = additional_notes if additional_notes else "None"

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
        Create line-specific piecewise linearization with adaptive breakpoints.
        Each line gets breakpoints focused on its transition region [thrd_1, thrd_2].
        """

        # Group lines by their fragility parameters to avoid redundancy
        line_groups = {}

        for l in line_idx:
            mu = net.data.frg.mu[l - 1]
            sg = net.data.frg.sigma[l - 1]
            th1 = net.data.frg.thrd_1[l - 1]
            th2 = net.data.frg.thrd_2[l - 1]
            sf = net.data.frg.shift_f[l - 1]

            # Create a key for grouping (round to avoid floating point issues)
            key = (round(mu, 4), round(sg, 4), round(th1, 2), round(th2, 2), round(sf, 2))

            if key not in line_groups:
                line_groups[key] = []
            line_groups[key].append(l)

        print(f"  - Found {len(line_groups)} unique fragility parameter sets among {len(line_idx)} lines")

        # Create breakpoints and probabilities for each line
        line_specific_data = {}

        for group_idx, (params, lines) in enumerate(line_groups.items()):
            mu, sg, th1, th2, sf = params

            # Define bounds
            global_min = 0
            global_max = 120

            # Effective thresholds after shift
            effective_th1 = th1 + sf
            effective_th2 = th2 + sf

            # Create adaptive breakpoints
            breakpoints = []

            # Add global minimum (start of first flat region)
            breakpoints.append(global_min)

            # Add the first threshold (end of first flat region, start of transition)
            breakpoints.append(effective_th1)

            # Add (num_pieces - 1) points within transition region
            if num_pieces > 2:
                transition_points = np.linspace(effective_th1, effective_th2, num_pieces + 1)[1:-1]
                breakpoints.extend(transition_points.tolist())

            # Add the second threshold (end of transition, start of second flat region)
            breakpoints.append(effective_th2)

            # Add global maximum (end of second flat region)
            breakpoints.append(global_max)

            # Remove duplicates and sort (shouldn't have any, but just in case)
            breakpoints = sorted(list(set(breakpoints)))

            # Calculate failure probabilities at each breakpoint
            fail_probs = []
            for x in breakpoints:
                z = x - sf  # Apply shift
                if z <= th1:
                    fail_probs.append(0.0)
                elif z >= th2:
                    fail_probs.append(1.0)
                else:
                    fail_probs.append(float(lognorm.cdf(z, s=sg, scale=np.exp(mu))))

            # Store data for each line in this group
            for l in lines:
                line_specific_data[l] = {
                    "breakpoints": breakpoints,
                    "probabilities": fail_probs
                }

            # Debug output
            print(f"\n    Group {group_idx}: Lines {lines[:3]}{'...' if len(lines) > 3 else ''}")
            print(f"      Parameters: mu={mu:.3f}, sigma={sg:.3f}, thrd=[{th1:.1f}, {th2:.1f}], shift={sf:.1f}")
            print(f"      Breakpoints: {[f'{x:.1f}' for x in breakpoints]}")
            print(
                f"      Structure: [0-{effective_th1:.0f}] flat | [{effective_th1:.0f}-{effective_th2:.0f}] transition ({num_pieces} pieces) | [{effective_th2:.0f}-120] flat")

        return line_specific_data

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

            return Pg_b + Pgrid + inflow_tn - outflow_tn == model.Pd[b, t] - model.Pc[b, t]

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

            return (inflow_dn + inflow_cpl + Pg_exst -
                    (outflow_dn + outflow_cpl + model.Pd[b, t] - model.Pc[b, t]) ) == 0

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
