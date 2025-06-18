import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.stats import lognorm
from network import NetworkClass
from windstorm import WindClass
from config import InvestmentConfig
from network_factory import make_network

from pathlib import Path
from datetime import datetime
import math
import json
import os
import csv

from windstorm_factory import make_windstorm


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
                               path_all_ws_scenarios: str = "Scenario_Results/Extracted_Windstorm_Scenarios/6_selected_ws_scenarios_network_29BusGB-KearsleyGSPGroup_windstorm_GB_year.json",
                               include_normal_scenario: bool = True,
                               normal_scenario_prob: float = 0.99,
                               use_representative_days: bool = True,  # New default
                               representative_days: list = None,  # New parameter
                               resilience_metric_threshold: float = None,
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

            # Clean up the normal model to free memory
            del normal_model

            print("Normal operation cost computation completed.\n")

        # 0.2 Read Windstorm Scenarios
        with open(path_all_ws_scenarios) as f:
            data = json.load(f)

        if "metadata" in data:
            metadata = data["metadata"]
        else:
            raise ValueError("JSON does not contain 'metadata'")

        if "ws_scenarios" in data:
            ws_scenarios = data["ws_scenarios"]
        else:
            raise ValueError("JSON does not contain 'ws_scenarios'")

        # Store windstorm infor into metadata
        self.meta = Object()
        self.meta.ws_seed = metadata.get("seed", None)
        self.meta.n_ws_scenarios = metadata.get("number_of_ws_simulations", len(ws_scenarios))
        self.meta.period_type = metadata.get("period_type", "year")

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

        _abs_start_ts = {sim["simulation_id"]: sim["events"][0]["bgn_hr"]
                      for sim in ws_scenarios}


        # ------------------------------------------------------------------
        # 1. Index sets -- tn for transmission level network, dn for distribution level network
        # ------------------------------------------------------------------
        Set_bus_tn = [b for b in net.data.net.bus
                      if net.data.net.bus_level[b] == "T"]
        Set_bus_dn = [b for b in net.data.net.bus
                      if net.data.net.bus_level[b] == "D"]

        Set_bch_tn = [l for l in range(1, len(net.data.net.bch) + 1)
                      if net.data.net.branch_level[l] in ("T", "T-D")]

        Set_bch_dn = [l for l in range(1, len(net.data.net.bch) + 1)
                      if net.data.net.branch_level[l] == "D"]

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

        Set_gen = list(range(1, len(net.data.net.gen) + 1))

        # scenario-specific timestep sets ----------------------------------
        Set_ts_scn = {
            sim["simulation_id"]: list(range(1, len(sim["bch_rand_nums"][0]) + 1))
            for sim in ws_scenarios
        }

        # Tuple sets:
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

        Set_sgt = [(sc, g, t) for sc in Set_scn
                   for g in Set_gen
                   for t in Set_ts_scn[sc]]

        Set_sgt_dn = [
            (sc, g, t)
            for (sc, g, t) in Set_sgt
            if net.data.net.bus_level[net.data.net.gen[g - 1]] == 'D'
        ]

        # ------------------------------------------------------------------
        # 2. Initialize Pyomo sets
        # ------------------------------------------------------------------
        model = pyo.ConcreteModel()

        model.Set_scn = pyo.Set(initialize=Set_scn)
        model.Set_bus_tn = pyo.Set(initialize=Set_bus_tn)
        model.Set_bus_dn = pyo.Set(initialize=Set_bus_dn)
        model.Set_bch_tn = pyo.Set(initialize=Set_bch_tn)
        model.Set_bch_dn = pyo.Set(initialize=Set_bch_dn)
        model.Set_bch_tn_lines = pyo.Set(initialize=Set_bch_tn_lines)
        model.Set_bch_dn_lines = pyo.Set(initialize=Set_bch_dn_lines)
        model.Set_bch_lines = pyo.Set(initialize=Set_bch_lines)
        model.Set_bch_hrdn_lines = pyo.Set(initialize=Set_bch_hrdn_lines)
        model.Set_gen = pyo.Set(initialize=Set_gen)

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
        model.Set_sgt = pyo.Set(initialize=Set_sgt, dimen=3)
        model.Set_sgt_dn = pyo.Set(initialize=Set_sgt_dn, dimen=3)

        # ------------------------------------------------------------------
        # 3. Variables
        # ------------------------------------------------------------------
        # 3.1) First-stage decision variables:
        model.line_hrdn = pyo.Var(model.Set_bch_hrdn_lines,
                                  bounds=(net.data.bch_hrdn_limits[0], net.data.bch_hrdn_limits[1]),
                                  within=pyo.NonNegativeReals)

        # 3.2) Second-stage recourse variables  (indexed by scenario)

        # Before defining variables, first check if the model include reactive power demand
        temp_qd_dict = {}  # You need to build Qd_dict early
        for sim in ws_scenarios:
            sc = sim["simulation_id"]
            bgn = _abs_start_ts[sc]
            for t in Set_ts_scn[sc]:
                abs_hr = bgn + t - 1
                for b in Set_bus_dn:
                    temp_qd_dict[(sc, b, t)] = net.data.net.profile_Qd[b - 1][abs_hr - 1]

        has_reactive_demand = any(v > 0 for v in temp_qd_dict.values())
        print(f"Reactive demand detected: {has_reactive_demand}")


        # - generation
        model.Pg = pyo.Var(model.Set_sgt, within=pyo.NonNegativeReals)
        if has_reactive_demand:
            model.Qg = pyo.Var(model.Set_sgt, within=pyo.Reals)
        else:
            model.Qg = pyo.Param(model.Set_sgt, default=0.0)

        # - bus state
        model.theta = pyo.Var(model.Set_sbt, within=pyo.Reals)
        model.V2_dn = pyo.Var(model.Set_sbt_dn, within=pyo.NonNegativeReals)

        # - flows
        model.Pf_tn = pyo.Var(model.Set_slt_tn, within=pyo.Reals)
        model.Pf_dn = pyo.Var(model.Set_slt_dn, within=pyo.Reals)
        if has_reactive_demand:
            model.Qf_dn = pyo.Var(model.Set_slt_dn, within=pyo.Reals)
        else:
            model.Qf_dn = pyo.Param(model.Set_slt_dn, default=0.0)

        # - load shedding (curtailed load)
        model.Pc = pyo.Var(model.Set_sbt, within=pyo.NonNegativeReals)
        if has_reactive_demand:
            model.Qc = pyo.Var(model.Set_sbt_dn, within=pyo.NonNegativeReals)
        else:
            model.Qc = pyo.Param(model.Set_sbt_dn, default=0.0)

        # - grid import/export
        model.Pimp = pyo.Var(model.Set_st, within=pyo.NonNegativeReals)
        model.Pexp = pyo.Var(model.Set_st, within=pyo.NonNegativeReals)

        # branch status
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
        # 4. Parameters
        # ------------------------------------------------------------------
        # 4.0) Resilience level threshold
        model.resilience_metric_threshold = pyo.Param(
            initialize=(resilience_metric_threshold
                        if resilience_metric_threshold is not None
                        else float('inf')),
            mutable=False
        )

        # 4.1)  Scenario probability
        model.scn_prob = pyo.Param(model.Set_scn, initialize=scn_prob)

        # 4.2)  Network static data  (costs, limits …)  – match names exactly
        # - base value
        model.base_MVA = pyo.Param(initialize=net.data.net.base_MVA)

        # - generator cost (c0 + c1·P)
        coef_len = len(net.data.net.gen_cost_coef[0])
        model.gen_cost_coef = pyo.Param(
            model.Set_gen, range(coef_len),
            initialize={(g, c): net.data.net.gen_cost_coef[g - 1][c]
                        for g in model.Set_gen for c in range(coef_len)}
        )
        # - generation limits
        model.Pg_max = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Pg_max[g - 1] for g in model.Set_gen})
        model.Pg_min = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Pg_min[g - 1] for g in model.Set_gen})
        model.Qg_max = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Qg_max[g - 1] for g in model.Set_gen})
        model.Qg_min = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Qg_min[g - 1] for g in model.Set_gen})

        # - branch constants
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

        # - voltage limits (squared)
        model.V2_min = pyo.Param(model.Set_bus_dn,
                                 initialize={b: net.data.net.V_min[net.data.net.bus.index(b)] ** 2
                                             for b in Set_bus_dn})
        model.V2_max = pyo.Param(model.Set_bus_dn,
                                 initialize={b: net.data.net.V_max[net.data.net.bus.index(b)] ** 2
                                             for b in Set_bus_dn})

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
                                   initialize={l: net.data.cost_bch_rep[l - 1]
                                               for l in range(1, len(net.data.net.bch) + 1)},
                                   within=pyo.NonNegativeReals)
        model.line_hrdn_cost = pyo.Param(model.Set_bch_tn | model.Set_bch_dn,
                                         initialize={l: net.data.cost_bch_hrdn[l - 1]
                                                     for l in range(1, len(net.data.net.bch) + 1)},
                                         within=pyo.NonNegativeReals)
        model.budget = pyo.Param(initialize=net.data.budget_bch_hrdn)

        # - demand profiles (Pd, Qd) by absolute hour
        Pd_param = {}
        Qd_param = {}
        for sim in ws_scenarios:
            sc = sim["simulation_id"]
            bgn = _abs_start_ts[sc]
            for t in Set_ts_scn[sc]:
                abs_hr = bgn + t - 1
                for b in net.data.net.bus:
                    Pd_param[(sc, b, t)] = net.data.net.profile_Pd[b - 1][abs_hr - 1]
                    if b in Set_bus_dn:
                        Qd_param[(sc, b, t)] = net.data.net.profile_Qd[b - 1][abs_hr - 1]

        model.Pd = pyo.Param(model.Set_sbt, initialize=Pd_param)
        model.Qd = pyo.Param(model.Set_sbt_dn, initialize=Qd_param)

        # - windstorm stochastic data  (gust, rand, impact, ttr)
        gust_dict = {}
        rand_dict = {}
        impact_dict = {}
        ttr_dict = {}

        for sim in ws_scenarios:
            sc = sim["simulation_id"]
            gust_series = sim["events"][0]["gust_speed"]
            for t in Set_ts_scn[sc]:
                gs = gust_series[t - 1] if t <= len(gust_series) else 0
                gust_dict[(sc, t)] = gs
            # branch-level arrays
            for l in range(1, len(net.data.net.bch) + 1):
                ttr_dict[(sc, l)] = sim["bch_ttr"][l - 1]
                for t, val in enumerate(sim["bch_rand_nums"][l - 1], start=1):
                    rand_dict[(sc, l, t)] = val
                for t, val in enumerate(sim["flgs_impacted_bch"][l - 1], start=1):
                    impact_dict[(sc, l, t)] = val

        model.gust_speed = pyo.Param(model.Set_st,
                                     initialize=gust_dict)
        model.rand_num = pyo.Param(model.Set_slt_tn | model.Set_slt_dn,
                                   initialize=rand_dict)
        model.impacted_branches = pyo.Param(model.Set_slt_tn | model.Set_slt_dn,
                                            initialize=impact_dict, within=pyo.Binary)
        model.branch_ttr = pyo.Param(model.Set_scn, model.Set_bch_tn | model.Set_bch_dn,
                                     initialize=ttr_dict)


        # ------------------------------------------------------------------
        # 5. Constraints
        # ------------------------------------------------------------------
        # 5.1) Budget
        def investment_budget_rule(model):
            return sum(
                model.line_hrdn_cost[l] * model.line_hrdn[l]
                for l in model.Set_bch_hrdn_lines
            ) <= model.budget

        model.Constraint_InvestmentBudget = pyo.Constraint(rule=investment_budget_rule)

        # 5.2) Shifted gust speed (i.e. Line hardening) -- Note: only dn lines hardening are considered
        def shifted_gust_rule(model, sc, l, t):
            # line hardening amount can be non-zero for only lines
            hrdn = model.line_hrdn[l] if l in model.Set_bch_hrdn_lines else 0
            return model.shifted_gust_speed[sc, l, t] >= model.gust_speed[sc, t] - hrdn

        model.Constraint_ShiftedGust = pyo.Constraint(model.Set_slt_lines, rule=shifted_gust_rule)

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

        # 5.4) Failure logic constraints (unchanged from the original version)
        BigM = 1e4

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
            return model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] <= BigM * model.fail_condition[sc, l, t]

        def fail_cond_2(model, sc, l, t):
            return model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] >= (model.fail_condition[sc, l, t] - 1) * BigM

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
            if t > 1:
                return model.fail_applies[sc, l, t] <= model.branch_status[sc, l, t - 1]
            else:
                return pyo.Constraint.Skip

        def fail_activation_rule_2(model, sc, l, t):
            """ fail_applies can be 1 only if fail_indicator is 1 """
            return model.fail_applies[sc, l, t] <= model.fail_indicator[sc, l, t]

        def fail_activation_rule_3(model, sc, l, t):
            """ If both conditions are met, fail_applies must be 1 """
            if t > 1:
                return (model.fail_applies[sc, l, t] >= model.branch_status[sc, l, t - 1]
                        + model.fail_indicator[sc, l, t] - 1)
            else:
                return pyo.Constraint.Skip

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
            window_start = max(1, t - model.branch_ttr[sc, l] + 1)
            return (model.branch_status[sc, l, t]
                    + sum(model.fail_applies[sc, l, τ] for τ in range(window_start, t + 1))
                    ) <= 1

        model.Constraint_FailurePersistence = pyo.Constraint(model.Set_slt_lines, rule=failure_persistence_rule)

        # Repair must happen exactly ttr timesteps after failure
        def repair_timing_rule(model, sc, l, t):
            """
            If a failure occurred exactly ttr timesteps ago, repair must happen now
            (assuming we're still within the time horizon)
            """
            ttr = model.branch_ttr[sc, l]

            # Check if we're far enough into the scenario to look back ttr steps
            if t > ttr:
                # If there was a failure exactly ttr steps ago, we must repair now
                return model.repair_applies[sc, l, t] >= model.fail_applies[sc, l, t - ttr]
            else:
                return pyo.Constraint.Skip

        model.Constraint_RepairTiming = pyo.Constraint(model.Set_slt_lines, rule=repair_timing_rule)

        # Repair cannot happen before ttr timesteps have passed since failure
        def no_early_repair_rule(model, sc, l, t):
            """
            Repair cannot happen if there was no failure exactly ttr timesteps ago
            This prevents repairs from happening too early or without a failure
            """
            ttr = model.branch_ttr[sc, l]

            if t > ttr:
                # Sum of all failures in the window (t-ttr+1, t-1)
                # If any failure happened in this window (except exactly ttr ago), no repair allowed
                recent_failures = sum(
                    model.fail_applies[sc, l, tau]
                    for tau in range(max(1, t - ttr + 1), t)
                )
                return model.repair_applies[sc, l, t] <= 1 - recent_failures
            else:
                # Can't repair in the first ttr timesteps (no failure could have occurred ttr ago)
                return model.repair_applies[sc, l, t] == 0

        model.Constraint_NoEarlyRepair = pyo.Constraint(model.Set_slt_lines, rule=no_early_repair_rule)

        # Repair cannot happen if the line is already operational
        def no_repair_if_operational_rule(model, sc, l, t):
            """
            If a line is operational at t-1, it cannot be repaired at t
            (you can only repair failed lines)
            """
            if t > 1:
                return model.repair_applies[sc, l, t] <= 1 - model.branch_status[sc, l, t - 1]
            else:
                # No repairs at t=1 (all lines start operational)
                return model.repair_applies[sc, l, t] == 0

        model.Constraint_NoRepairIfOperational = pyo.Constraint(model.Set_slt_lines, rule=no_repair_if_operational_rule)

        # - A Branch has to keep its status at last timestep unless failure or repair happens
        def bch_status_transition_rule(model, sc, l, t):
            # The scenario starts with all lines operational
            if t == 1:
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

        # 5.5) Power flow definitions
        def dc_flow_rule(model, sc, l, t):
            i, j = net.data.net.bch[l - 1]
            return model.Pf_tn[sc, l, t] == model.base_MVA * model.B_tn[l] * (model.theta[sc, i, t]
                                                                              - model.theta[sc, j, t])

        model.Constraint_FlowDef_TN = pyo.Constraint(model.Set_slt_tn, rule=dc_flow_rule)

        # - Maximum allowable angle difference at lines
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

        # 5.6) Power balance constraints (TN: active; DN: active & reactive)
        slack_bus = net.data.net.slack_bus

        def P_balance_tn(model, sc, b, t):
            Pg = sum(model.Pg[sc, g, t]
                     for g in model.Set_gen
                     if net.data.net.gen[g - 1] == b)
            Pf_in = sum(model.Pf_tn[sc, l, t]
                        for l in model.Set_bch_tn
                        if net.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf_tn[sc, l, t]
                         for l in model.Set_bch_tn
                         if net.data.net.bch[l - 1][0] == b)
            Pgrid = (model.Pimp[sc, t] - model.Pexp[sc, t]) if b == slack_bus else 0

            return Pg + Pgrid + Pf_in - Pf_out == model.Pd[sc, b, t] - model.Pc[sc, b, t]

        model.Constraint_PBalance_TN = pyo.Constraint(model.Set_sbt_tn, rule=P_balance_tn)

        def P_balance_dn(model, sc, b, t):
            Pg = sum(model.Pg[sc, g, t]
                     for g in model.Set_gen
                     if net.data.net.gen[g - 1] == b)
            Pf_in_dn = sum(model.Pf_dn[sc, l, t]
                           for l in model.Set_bch_dn
                           if net.data.net.bch[l - 1][1] == b)
            Pf_out_dn = sum(model.Pf_dn[sc, l, t]
                            for l in model.Set_bch_dn
                            if net.data.net.bch[l - 1][0] == b)
            Pf_cpl_in = sum(model.Pf_tn[sc, l, t]
                            for l in model.Set_bch_tn
                            if net.data.net.branch_level[l] == 'T-D'
                            and net.data.net.bch[l - 1][1] == b)
            Pf_cpl_out = sum(model.Pf_tn[sc, l, t]
                             for l in model.Set_bch_tn
                             if net.data.net.branch_level[l] == 'T-D'
                             and net.data.net.bch[l - 1][0] == b)

            return Pg + Pf_in_dn - Pf_out_dn + Pf_cpl_in - Pf_cpl_out == model.Pd[sc, b, t] - model.Pc[sc, b, t]

        model.Constraint_PBalance_DN = pyo.Constraint(model.Set_sbt_dn, rule=P_balance_dn)

        def Q_balance_dn(model, sc, b, t):
            # Skip this constraint if no reactive demand exists
            if not has_reactive_demand:
                return pyo.Constraint.Skip

            Qg = sum(model.Qg[sc, g, t] for g in model.Set_gen if net.data.net.gen[g - 1] == b)
            Qf_in_dn = sum(model.Qf_dn[sc, l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][1] == b)
            Qf_out_dn = sum(model.Qf_dn[sc, l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][0] == b)
            return Qg + Qf_in_dn - Qf_out_dn == model.Qd[sc, b, t] - model.Qc[sc, b, t]

        model.Constraint_QBalance_DN = pyo.Constraint(model.Set_sbt_dn, rule=Q_balance_dn)

        # 5.7) Voltage limits DN
        def V2_dn_upper_limit_rule(model, sc, b, t):
            return model.V2_dn[sc, b, t] <= model.V2_max[b]

        def V2_dn_lower_limit_rule(model, sc, b, t):
            return model.V2_dn[sc, b, t] >= model.V2_min[b]

        model.Constraint_V2max = pyo.Constraint(model.Set_sbt_dn, rule=V2_dn_upper_limit_rule)
        model.Constraint_V2min = pyo.Constraint(model.Set_sbt_dn, rule=V2_dn_lower_limit_rule)

        # 5.8) Generator limits
        def Pg_upper_limit_rule(model, sc, g, t):
            return model.Pg[sc, g, t] <= model.Pg_max[g]

        def Pg_lower_limit_rule(model, sc, g, t):
            return model.Pg[sc, g, t] >= model.Pg_min[g]

        def Qg_upper_limit_rule(model, sc, g, t):
            # Skip this constraint if no reactive demand exists
            if not has_reactive_demand:
                return pyo.Constraint.Skip

            return model.Qg[sc, g, t] <= model.Qg_max[g]

        def Qg_lower_limit_rule(model, sc, g, t):
            # Skip this constraint if no reactive demand exists
            if not has_reactive_demand:
                return pyo.Constraint.Skip

            return model.Qg[sc, g, t] >= model.Qg_min[g]

        model.Constraint_PgUpperLimit = pyo.Constraint(model.Set_sgt, rule=Pg_upper_limit_rule)
        model.Constraint_PgLowerLimit = pyo.Constraint(model.Set_sgt, rule=Pg_lower_limit_rule)
        model.Constraint_QgUpperLimit = pyo.Constraint(model.Set_sgt, rule=Qg_upper_limit_rule)
        model.Constraint_QgLowerLimit = pyo.Constraint(model.Set_sgt, rule=Qg_lower_limit_rule)

        # 5.9) Slack-bus reference theta = 0
        def slack_rule(model, sc, t):
            return model.theta[sc, slack_bus, t] == 0

        model.Constraint_Slack = pyo.Constraint([(sc, t) for sc in Set_scn
                                                 for t in Set_ts_scn[sc]],
                                                rule=slack_rule)

        # Code 'expected total operational cost at dn level across all ws scenarios' (the resilience metric)
        # as an expression
        # (currently as we assume all windstorm scenarios have equal probabilities, we simply find the average value)
        def expected_total_op_cost_dn_ws_rule(model):
            num_scenarios = len(model.Set_scn)

            # 1) DN generation cost
            gen_cost_dn = sum(
                (model.gen_cost_coef[g, 0] + model.gen_cost_coef[g, 1] * model.Pg[sc, g, t])
                for (sc, g, t) in model.Set_sgt_dn
            ) / num_scenarios

            # 2) DN active load-shedding
            active_ls_cost_dn = sum(
                model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            ) / num_scenarios

            # 3) DN reactive load-shedding
            reactive_ls_cost_dn = sum(
                model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            ) / num_scenarios

            # 4) DN line repair
            rep_cost_dn = sum(
                model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_dn_lines
            ) / num_scenarios

            return gen_cost_dn + active_ls_cost_dn + reactive_ls_cost_dn + rep_cost_dn

        model.exp_total_op_cost_dn_ws_expr = pyo.Expression(rule=expected_total_op_cost_dn_ws_rule)

        # Code "Expected Energy Not Supplied (EENS) at dn level across all ws scenarios" as an expression
        # Since we have hourly resolution, EENS = sum of all active power load shedding
        def expected_total_eens_dn_ws_rule(model):
            num_scenarios = len(model.Set_scn)

            eens_dn = sum(
                model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            ) / num_scenarios

            return eens_dn

        model.exp_total_eens_dn_ws_expr = pyo.Expression(rule=expected_total_eens_dn_ws_rule)

        # 5.10) resilience-level constraint (resilience metric <= pre-defined threshold)
        if resilience_metric_threshold is not None:
            model.Constraint_ResilienceLevel = pyo.Constraint(
                # expr=model.exp_total_op_cost_dn_expr <= model.resilience_metric_threshold
                expr=model.exp_total_eens_dn_expr <= model.resilience_metric_threshold
            )

        # ------------------------------------------------------------------
        # 6. Objective
        # ------------------------------------------------------------------
        # Code total investment cost as an expression
        def total_inv_cost_rule(model):
            return sum(model.line_hrdn_cost[l] * model.line_hrdn[l]
                       for l in model.Set_bch_hrdn_lines)

        model.total_inv_cost_expr = pyo.Expression(rule=total_inv_cost_rule)

        def expected_total_op_cost_rule(model):
            # 1) Generation cost
            gen_cost = sum(
                model.scn_prob[sc] * (model.gen_cost_coef[g, 0] + model.gen_cost_coef[g, 1] * model.Pg[sc, g, t])
                for (sc, g, t) in model.Set_sgt
            )

            # 2) Grid import / export
            imp_exp_cost = sum(
                model.scn_prob[sc] * (model.Pimp_cost * model.Pimp[sc, t] +
                                      model.Pexp_cost * model.Pexp[sc, t])
                for (sc, t) in model.Set_st
            )

            # 3) Active load-shedding
            act_ls_cost = sum(
                model.scn_prob[sc] * model.Pc_cost[b] * model.Pc[sc, b, t]
                for (sc, b, t) in model.Set_sbt
            )

            # 4) Reactive load-shedding (DN buses only)
            reac_ls_cost = sum(
                model.scn_prob[sc] * model.Qc_cost[b] * model.Qc[sc, b, t]
                for (sc, b, t) in model.Set_sbt_dn
            )

            # 5) Line repair
            rep_cost = sum(
                model.scn_prob[sc] * model.rep_cost[l] * model.repair_applies[sc, l, t]
                for (sc, l, t) in model.Set_slt_lines
            )

            # Windstorm window costs
            ws_window_cost = gen_cost + imp_exp_cost + act_ls_cost + reac_ls_cost + rep_cost

            # Add normal annual operation cost to each windstorm scenario
            # This accounts for the rest of the year outside the windstorm window
            if include_normal_scenario and normal_operation_opf_results:
                # Each windstorm scenario includes its window cost and a full year of normal operation
                # (theoretically we should exclude the hours from the windstorm window in the normal operation hours,
                # but for simplicity, we simply add the operational cost from a full year)
                annual_normal_cost = sum(
                    model.scn_prob[sc] * normal_operation_opf_results["total_cost"]
                    for sc in model.Set_scn
                )

                return ws_window_cost + annual_normal_cost
            else:
                return ws_window_cost

        model.exp_total_op_cost_expr = pyo.Expression(rule=expected_total_op_cost_rule)

        def objective_rule(model):
            # First-stage: Investment cost
            inv_cost = model.total_inv_cost_expr

            # Second-stage: Expected operational cost
            if include_normal_scenario and normal_operation_opf_results:
                # Normal scenario (full year) + Windstorm scenarios (window + full year)
                normal_contribution = normal_scenario_prob * normal_operation_opf_results["total_cost"]

                ws_contribution = model.exp_total_op_cost_expr  # Already includes annual normal costs
                return inv_cost + normal_contribution + ws_contribution
            else:
                # Only windstorm scenarios (without annual normalization)
                return inv_cost + model.exp_total_op_cost_expr

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

        opt = SolverFactory(solver_name)

        # --- default options -------------------------------------------------
        default_opts = {"MIPGap": mip_gap, "TimeLimit": time_limit}
        default_opts.update(solver_options)
        for k, v in default_opts.items():
            opt.options[k] = v

        # --- solve -----------------------------------------------------------
        results = opt.solve(model, tee=True)

        status = results.solver.status
        term_cond = results.solver.termination_condition
        best_obj = pyo.value(model.Objective, exception=False)
        mip_gap_out = getattr(results.solver, "gap", None)

        # --- basic checking --------------------------------------------------
        ok = (status == pyo.SolverStatus.ok
              and term_cond in (pyo.TerminationCondition.optimal,
                                pyo.TerminationCondition.feasible))
        if not ok:
            msg = f"Solver finished with status={status}, " \
                  f"termination={term_cond}, gap={mip_gap_out}"
            if write_lp:  # dump model only when really needed
                model.write("LP_Models/model_at_failure.lp",
                            io_options={"symbolic_solver_labels": True})
            raise RuntimeError(msg)

        # --- optional exports ------------------------------------------------
        if write_lp:
            model.write("LP_Models/solved_model.lp",
                        io_options={"symbolic_solver_labels": True})

        if write_result:
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

                # Add the resilience level threshold value
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
                fname += ".csv"
                result_path = os.path.join("Optimization_Results", "Investment_Model", fname)

            self._write_selected_variables_to_excel(model, result_path)

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

        from network_factory import make_network
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
        Build an OPF model for normal operation using representative days.

        Args:
            network_name: Network configuration name
            use_representative_days: If True, use representative days instead of full year
            representative_days: List of day numbers (1-365). If None, uses default seasonal days

        Returns:
            tuple: (model, scale_factor) where scale_factor is for annualizing costs
        """
        from network_factory import make_network

        net = make_network(network_name)

        # Determine hours to compute
        if use_representative_days:
            if representative_days is None:
                # Default: one representative day per season
                representative_days = [15, 105, 195, 285]  # Mid-Jan, Apr, Jul, Oct

            hours_to_compute = []
            for day in representative_days:
                day_start = (day - 1) * 24 + 1
                hours_to_compute.extend(range(day_start, day_start + 24))

            # Scale factor to annualize costs
            scale_factor = 365.0 / len(representative_days)
        else:
            # Full year (fallback option)
            hours_to_compute = list(range(1, 8761))
            scale_factor = 1.0

        # Build model
        model = pyo.ConcreteModel()

        # Store metadata
        model.hours_computed = len(hours_to_compute)
        model.scale_factor = scale_factor
        model.network_name = network_name
        model.representative_days = representative_days if use_representative_days else None

        # ------------------------------------------------------------------
        # 1. Sets
        # ------------------------------------------------------------------
        Set_bus_tn = [b for b in net.data.net.bus if net.data.net.bus_level[b] == 'T']
        Set_bus_dn = [b for b in net.data.net.bus if net.data.net.bus_level[b] == 'D']
        Set_bch_tn = [l for l in range(1, len(net.data.net.bch) + 1)
                      if net.data.net.branch_level[l] in ('T', 'T-D')]
        Set_bch_dn = [l for l in range(1, len(net.data.net.bch) + 1)
                      if net.data.net.branch_level[l] == 'D']
        Set_gen = list(range(1, len(net.data.net.gen) + 1))
        Set_ts = list(range(1, len(hours_to_compute) + 1))  # Indexed from 1

        # Pyomo sets
        model.Set_bus_tn = pyo.Set(initialize=Set_bus_tn)
        model.Set_bus_dn = pyo.Set(initialize=Set_bus_dn)
        model.Set_bch_tn = pyo.Set(initialize=Set_bch_tn)
        model.Set_bch_dn = pyo.Set(initialize=Set_bch_dn)
        model.Set_gen = pyo.Set(initialize=Set_gen)
        model.Set_ts = pyo.Set(initialize=Set_ts)

        # Get slack bus
        slack_bus = net.data.net.slack_bus

        # ------------------------------------------------------------------
        # 2. Parameters (COMPLETE)
        # ------------------------------------------------------------------
        model.base_MVA = pyo.Param(initialize=net.data.net.base_MVA)

        # Demand profiles for selected hours
        Pd_dict = {}
        Qd_dict = {}
        for t_idx, abs_hr in enumerate(hours_to_compute, 1):
            for b in net.data.net.bus:
                Pd_dict[(b, t_idx)] = net.data.net.profile_Pd[b - 1][abs_hr - 1]
                if b in Set_bus_dn:
                    Qd_dict[(b, t_idx)] = net.data.net.profile_Qd[b - 1][abs_hr - 1]

        model.Pd = pyo.Param(model.Set_bus_tn | model.Set_bus_dn, model.Set_ts,
                             initialize=Pd_dict, mutable=False)
        model.Qd = pyo.Param(model.Set_bus_dn, model.Set_ts,
                             initialize=Qd_dict, mutable=False)

        # After loading Qd_dict, check if all reactive demands are zero:
        has_reactive_demand = any(v > 0 for v in Qd_dict.values())

        # Generator parameters (THIS WAS MISSING!)
        model.Pg_max = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Pg_max[g - 1] for g in Set_gen})
        model.Pg_min = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Pg_min[g - 1] for g in Set_gen})
        model.Qg_max = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Qg_max[g - 1] for g in Set_gen})
        model.Qg_min = pyo.Param(model.Set_gen,
                                 initialize={g: net.data.net.Qg_min[g - 1] for g in Set_gen})

        # Generator cost coefficients (THIS WAS MISSING!)
        coef_len = len(net.data.net.gen_cost_coef[0])
        model.gen_cost_coef = pyo.Param(
            model.Set_gen, range(coef_len),
            initialize={(g, c): net.data.net.gen_cost_coef[g - 1][c]
                        for g in model.Set_gen for c in range(coef_len)}
        )

        # Branch parameters TN
        model.B_tn = pyo.Param(model.Set_bch_tn,
                               initialize={l: 1.0 / net.data.net.bch_X[l - 1] for l in Set_bch_tn})
        model.Pmax_tn = pyo.Param(model.Set_bch_tn,
                                  initialize={l: net.data.net.bch_Pmax[l - 1] for l in Set_bch_tn})

        # Branch parameters DN
        model.R_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: net.data.net.bch_R[l - 1] for l in Set_bch_dn})
        model.X_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: net.data.net.bch_X[l - 1] for l in Set_bch_dn})
        model.Smax_dn = pyo.Param(model.Set_bch_dn,
                                  initialize={l: net.data.net.bch_Smax[l - 1] for l in Set_bch_dn})

        # Voltage limits
        model.V2_min = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                 initialize={bus: net.data.net.V_min[i] ** 2
                                             for i, bus in enumerate(net.data.net.bus)})
        model.V2_max = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                 initialize={bus: net.data.net.V_max[i] ** 2
                                             for i, bus in enumerate(net.data.net.bus)})

        # Load shedding costs
        model.Pc_cost = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                  initialize={b: net.data.net.Pc_cost[b - 1] for b in net.data.net.bus})
        model.Qc_cost = pyo.Param(model.Set_bus_dn,
                                  initialize={b: net.data.net.Qc_cost[net.data.net.bus.index(b)]
                                              for b in Set_bus_dn})

        # Electricity import/export costs
        model.Pimp_cost = pyo.Param(initialize=net.data.net.Pimp_cost)
        model.Pexp_cost = pyo.Param(initialize=net.data.net.Pexp_cost)

        # ------------------------------------------------------------------
        # 3. Variables
        # ------------------------------------------------------------------
        # Generation
        model.Pg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.NonNegativeReals)
        model.Qg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.Reals)

        # Voltage angles (TN) and squared magnitudes (DN)
        model.theta = pyo.Var(model.Set_bus_tn, model.Set_ts, within=pyo.Reals)
        model.V2_dn = pyo.Var(model.Set_bus_dn, model.Set_ts, within=pyo.NonNegativeReals)

        # Power flows
        model.Pf_tn = pyo.Var(model.Set_bch_tn, model.Set_ts, within=pyo.Reals)
        model.Pf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)
        model.Qf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)

        # Load shedding (should be minimal/zero in normal operation)
        model.Pc = pyo.Var(model.Set_bus_tn | model.Set_bus_dn, model.Set_ts,
                           within=pyo.NonNegativeReals)
        model.Qc = pyo.Var(model.Set_bus_dn, model.Set_ts,
                           within=pyo.NonNegativeReals)

        # Electricity import/export at the slack bus
        model.Pimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)
        model.Pexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)

        # ------------------------------------------------------------------
        # 4. Constraints
        # ------------------------------------------------------------------

        # 4.1) DC Power Flow (TN) - Handle T-D branches properly
        def dc_power_flow_rule(model, l, t):
            fr_bus = net.data.net.bch[l - 1][0]
            to_bus = net.data.net.bch[l - 1][1]

            # Check if both buses are in transmission set
            if fr_bus in Set_bus_tn and to_bus in Set_bus_tn:
                return model.Pf_tn[l, t] == model.B_tn[l] * (model.theta[fr_bus, t] - model.theta[to_bus, t])
            else:
                # Skip if either end is a distribution bus (T-D branch)
                return pyo.Constraint.Skip

        model.Constraint_DCPowerFlow = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                      rule=dc_power_flow_rule)

        # 4.2) Power Balance TN
        def power_balance_tn_rule(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if net.data.net.gen[g - 1] == b)
            Pf_in = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if net.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if net.data.net.bch[l - 1][0] == b)

            Pgrid = (model.Pimp[t] - model.Pexp[t]) if b == slack_bus else 0

            return Pg + Pgrid + Pf_in - Pf_out == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_PowerBalance_TN = pyo.Constraint(model.Set_bus_tn, model.Set_ts,
                                                          rule=power_balance_tn_rule)

        # 4.3) LinDistFlow DN
        def lindistflow_P_rule(model, l, t):
            fr_bus = net.data.net.bch[l - 1][0]
            to_bus = net.data.net.bch[l - 1][1]
            return (model.V2_dn[to_bus, t] == model.V2_dn[fr_bus, t]
                    - 2 * (model.R_dn[l] * model.Pf_dn[l, t] + model.X_dn[l] * model.Qf_dn[l, t]))

        model.Constraint_LinDistFlow = pyo.Constraint(model.Set_bch_dn, model.Set_ts,
                                                      rule=lindistflow_P_rule)

        # 4.4) Power Balance DN
        def power_balance_dn_rule(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if net.data.net.gen[g - 1] == b)
            Pf_in_dn = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][1] == b)
            Pf_out_dn = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][0] == b)

            # Coupling from TN
            Pf_cpl_in = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn
                            if net.data.net.branch_level[l] == 'T-D' and net.data.net.bch[l - 1][1] == b)
            Pf_cpl_out = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn
                             if net.data.net.branch_level[l] == 'T-D' and net.data.net.bch[l - 1][0] == b)

            return Pg + Pf_in_dn - Pf_out_dn + Pf_cpl_in - Pf_cpl_out == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_PowerBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                          rule=power_balance_dn_rule)

        def reactive_balance_dn_rule(model, b, t):
            # If no reactive demand in the system, make this constraint trivial
            if not has_reactive_demand:
                return pyo.Constraint.Skip  # Skip this constraint

            Qg = sum(model.Qg[g, t] for g in model.Set_gen if net.data.net.gen[g - 1] == b)
            Qf_in = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][1] == b)
            Qf_out = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][0] == b)
            return Qg + Qf_in - Qf_out == model.Qd[b, t] - model.Qc[b, t]

        model.Constraint_ReactiveBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                             rule=reactive_balance_dn_rule)

        # 4.5) Branch Limits
        def pf_tn_upper_rule(model, l, t):
            return model.Pf_tn[l, t] <= model.Pmax_tn[l]

        def pf_tn_lower_rule(model, l, t):
            return model.Pf_tn[l, t] >= -model.Pmax_tn[l]

        model.Constraint_Pf_tn_upper = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                      rule=pf_tn_upper_rule)
        model.Constraint_Pf_tn_lower = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                      rule=pf_tn_lower_rule)

        def apparent_power_dn_rule(model, l, t):
            # If no reactive power, just use active power limit
            if not has_reactive_demand:
                return model.Pf_dn[l, t] ** 2 <= model.Smax_dn[l] ** 2
            else:
                return model.Pf_dn[l, t] ** 2 + model.Qf_dn[l, t] ** 2 <= model.Smax_dn[l] ** 2

        model.Constraint_Smax_dn = pyo.Constraint(model.Set_bch_dn, model.Set_ts,
                                                  rule=apparent_power_dn_rule)

        # 4.6) Voltage Limits
        def v2_upper_rule(model, b, t):
            return model.V2_dn[b, t] <= model.V2_max[b]

        def v2_lower_rule(model, b, t):
            return model.V2_dn[b, t] >= model.V2_min[b]

        model.Constraint_V2_upper = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                   rule=v2_upper_rule)
        model.Constraint_V2_lower = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                   rule=v2_lower_rule)

        # 4.7) Generator Limits
        def pg_upper_rule(model, g, t):
            return model.Pg[g, t] <= model.Pg_max[g]

        def pg_lower_rule(model, g, t):
            return model.Pg[g, t] >= model.Pg_min[g]

        def qg_upper_rule(model, g, t):
            return model.Qg[g, t] <= model.Qg_max[g]

        def qg_lower_rule(model, g, t):
            return model.Qg[g, t] >= model.Qg_min[g]

        model.Constraint_Pg_upper = pyo.Constraint(model.Set_gen, model.Set_ts, rule=pg_upper_rule)
        model.Constraint_Pg_lower = pyo.Constraint(model.Set_gen, model.Set_ts, rule=pg_lower_rule)
        model.Constraint_Qg_upper = pyo.Constraint(model.Set_gen, model.Set_ts, rule=qg_upper_rule)
        model.Constraint_Qg_lower = pyo.Constraint(model.Set_gen, model.Set_ts, rule=qg_lower_rule)

        # 4.8) Slack Bus
        if slack_bus:
            def slack_rule(model, t):
                return model.theta[slack_bus, t] == 0

            model.Constraint_Slack = pyo.Constraint(model.Set_ts, rule=slack_rule)

        # ------------------------------------------------------------------
        # 5. Objective (with scale factor)
        # ------------------------------------------------------------------
        def objective_rule(model):
            # Generation cost
            gen_cost = sum(
                model.gen_cost_coef[g, 0] + model.gen_cost_coef[g, 1] * model.Pg[g, t]
                for g in model.Set_gen for t in model.Set_ts
            )

            # Grid import/export cost
            grid_cost = sum(
                model.Pimp_cost * model.Pimp[t] + model.Pexp_cost * model.Pexp[t]
                for t in model.Set_ts
            )

            # Load shedding cost (should be minimal/zero)
            ls_cost = sum(
                model.Pc_cost[b] * model.Pc[b, t]
                for b in model.Set_bus_tn | model.Set_bus_dn
                for t in model.Set_ts
            )

            ls_cost += sum(
                model.Qc_cost[b] * model.Qc[b, t]
                for b in model.Set_bus_dn
                for t in model.Set_ts
            )

            # Apply scale factor to get annual cost
            return (gen_cost + grid_cost + ls_cost) * scale_factor

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return model, scale_factor


    def solve_normal_scenario_opf_model(self, model, solver='gurobi', print_summary=True):
        """
        Solve the normal scenario OPF model and extract cost components.

        Returns:
            dict: Cost breakdown and key metrics
        """
        opt = SolverFactory(solver)
        results = opt.solve(model, tee=False)

        if results.solver.status != 'ok':
            raise RuntimeError("Failed to solve normal operation OPF model")

        # Extract total cost (already annualized)
        total_cost = pyo.value(model.Objective)

        # Extract component costs (before scaling)
        gen_cost_total = 0
        gen_cost_tn = 0
        gen_cost_dn = 0

        # Extract grid import/export costs
        grid_cost = 0
        for t in model.Set_ts:
            grid_cost += (model.Pimp_cost * pyo.value(model.Pimp[t]) +
                          model.Pexp_cost * pyo.value(model.Pexp[t]))
        grid_cost *= model.scale_factor

        # Extract load shedding costs (not just quantities)
        ls_cost_active = 0
        ls_cost_reactive = 0
        total_pc = 0
        total_qc = 0

        for b in model.Set_bus_tn | model.Set_bus_dn:
            for t in model.Set_ts:
                pc_val = pyo.value(model.Pc[b, t])
                total_pc += pc_val
                ls_cost_active += model.Pc_cost[b] * pc_val

        for b in model.Set_bus_dn:
            for t in model.Set_ts:
                qc_val = pyo.value(model.Qc[b, t])
                total_qc += qc_val
                ls_cost_reactive += model.Qc_cost[b] * qc_val

        ls_cost_total = (ls_cost_active + ls_cost_reactive) * model.scale_factor
        total_pc *= model.scale_factor
        total_qc *= model.scale_factor

        from network_factory import make_network
        net = make_network(model.network_name)

        for g in model.Set_gen:
            for t in model.Set_ts:
                cost = (model.gen_cost_coef[g, 0] +
                        model.gen_cost_coef[g, 1] * pyo.value(model.Pg[g, t]))
                gen_cost_total += cost

                # Check if generator is at TN or DN level
                gen_bus = net.data.net.gen[g - 1]
                if net.data.net.bus_level[gen_bus] == 'T':
                    gen_cost_tn += cost
                else:
                    gen_cost_dn += cost

        # Apply scale factor to component costs
        gen_cost_total *= model.scale_factor
        gen_cost_tn *= model.scale_factor
        gen_cost_dn *= model.scale_factor

        # Verify total cost breakdown
        calculated_total = gen_cost_total + grid_cost + ls_cost_total

        # Check for load shedding
        total_eens = sum(pyo.value(model.Pc[b, t])
                              for b in model.Set_bus_tn | model.Set_bus_dn
                              for t in model.Set_ts)

        if print_summary:
            print("\n" + "=" * 60)
            print("Normal Operation Scenario Results")
            print("=" * 60)
            # ... existing print statements ...
            print(f"Total annual cost: ${total_cost:,.2f}")
            print(f"  - Generation cost: ${gen_cost_total:,.2f}")
            print(f"    - TN generation: ${gen_cost_tn:,.2f}")
            print(f"    - DN generation: ${gen_cost_dn:,.2f}")
            print(f"  - Grid import/export cost: ${grid_cost:,.2f}")
            print(f"  - Load shedding cost: ${ls_cost_total:,.2f}")
            print(f"    - Active (Pc): ${ls_cost_active * model.scale_factor:,.2f}")
            print(f"    - Reactive (Qc): ${ls_cost_reactive * model.scale_factor:,.2f}")
            print(f"Total load shed: {total_pc:.4f} MW active, {total_qc:.4f} MVAr reactive")
            print(f"Total EENS: {total_eens:.4f} MWh")
            print(f"Cost verification: ${calculated_total:,.2f} (should equal total)")
            print("=" * 60 + "\n")

        return {
            "total_cost": total_cost,
            "gen_cost_total": gen_cost_total,
            "gen_cost_tn": gen_cost_tn,
            "gen_cost_dn": gen_cost_dn,
            "grid_cost": grid_cost,
            "ls_cost_total": ls_cost_total,
            "load_shed_mw": total_pc,
            "load_shed_mvar": total_qc,
            "hours_computed": model.hours_computed,
            "scale_factor": model.scale_factor,
            "representative_days": model.representative_days,
            "solver_status": str(results.solver.status)
        }