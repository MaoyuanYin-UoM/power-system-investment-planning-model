

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.stats import lognorm
from network import NetworkClass
from windstorm import WindClass
from config import InvestmentConfig
from network_factory import make_network
from pathlib import Path
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
                               path_all_ws_scenarios: str = "Scenario_Results/Extracted_Scenarios/all_ws_scenarios_UK-Kearsley_network_seed_100.json"):
        """
        Build a Pyomo MILP model for resilience enhancement investment planning (line hardening)
        against windstorms, using the form of stochastic programming over multiple scenarios.


        * 1st-stage:   line-hardening shift  Δv_l  ( identical for TN & DN )
        * 2nd-stage:   scenario-wise DC OPF on TN  +  LinDistFlow on DN
                       with wind-storm-driven stochastic failures / repairs.
        """

        # ------------------------------------------------------------------
        # 0. Preliminaries
        # ------------------------------------------------------------------
        ws = make_windstorm("windstorm_UK_transmission_network")
        net = make_network("UK_transmission_network_with_kearsley_GSP_group")

        with open(path_all_ws_scenarios) as f:
            ws_scenarios = json.load(f)

        Set_scn = [sim["simulation_id"] for sim in ws_scenarios]
        scn_prob = {sc: 1.0 / len(Set_scn) for sc in Set_scn}

        # helper: absolute-hour mapping (t → hour in full-profile)
        _abs_start = {sim["simulation_id"]: sim["events"][0]["bgn_hr"]
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

        Set_bch_dn_lines = [l for l in Set_bch_dn
                            if net.data.net.bch_type[l - 1] == 1]  # hardenable branches (i.e., distribution lines)

        Set_gen = list(range(1, len(net.data.net.gen) + 1))

        # scenario-specific timestep sets ----------------------------------
        Set_ts_scn = {
            sim["simulation_id"]: list(range(1, len(sim["bch_rand_nums"][0]) + 1))
            for sim in ws_scenarios
        }

        # build the usual tuple sets because Pyomo cannot index a Set by Set
        #  - sbt: scenario, bus, timestep
        #  - slt: scenario, branch, timestep
        #  - sgt: scenario, gen, timestep
        Set_sbt = [(sc, b, t) for sc in Set_scn
                   for b in net.data.net.bus
                   for t in Set_ts_scn[sc]]
        Set_sbt_dn = [(sc, b, t) for (sc, b, t) in Set_sbt if b in Set_bus_dn]

        Set_slt_tn = [(sc, l, t) for sc in Set_scn
                      for l in Set_bch_tn
                      for t in Set_ts_scn[sc]]
        Set_slt_dn = [(sc, l, t) for sc in Set_scn
                      for l in Set_bch_dn
                      for t in Set_ts_scn[sc]]

        Set_sgt = [(sc, g, t) for sc in Set_scn
                   for g in Set_gen
                   for t in Set_ts_scn[sc]]

        # ------------------------------------------------------------------
        # 2. Sets
        # ------------------------------------------------------------------
        model = pyo.ConcreteModel()

        model.Set_scn = pyo.Set(initialize=Set_scn)
        model.Set_bus_tn = pyo.Set(initialize=Set_bus_tn)
        model.Set_bus_dn = pyo.Set(initialize=Set_bus_dn)
        model.Set_bch_tn = pyo.Set(initialize=Set_bch_tn)
        model.Set_bch_dn = pyo.Set(initialize=Set_bch_dn)
        model.Set_bch_dn_lines = pyo.Set(initialize=Set_bch_dn_lines)
        model.Set_gen = pyo.Set(initialize=Set_gen)

        model.Set_ts_scn = {sc: pyo.Set(initialize=Set_ts_scn[sc])
                            for sc in Set_scn}  # for iteration ease

        model.Set_sbt = pyo.Set(initialize=Set_sbt, dimen=3)
        model.Set_sbt_dn = pyo.Set(initialize=Set_sbt_dn, dimen=3)
        model.Set_slt_tn = pyo.Set(initialize=Set_slt_tn, dimen=3)
        model.Set_slt_dn = pyo.Set(initialize=Set_slt_dn, dimen=3)
        model.Set_sgt = pyo.Set(initialize=Set_sgt, dimen=3)

        # ------------------------------------------------------------------
        # 3. Variables
        # ------------------------------------------------------------------
        # 3.1) First-stage decision variables:
        model.line_hrdn = pyo.Var(model.Set_bch_dn_lines,
                                 bounds=self.data.bch_hrdn_limits,
                                 within=pyo.NonNegativeReals)

        # 3.2) Second-stage recourse variables  (indexed by scenario)
        # - generation
        model.Pg = pyo.Var(model.Set_sgt, within=pyo.NonNegativeReals)
        model.Qg = pyo.Var(model.Set_sgt, within=pyo.Reals)

        # - bus state
        model.theta = pyo.Var(model.Set_sbt, within=pyo.Reals)
        model.V2_dn = pyo.Var(model.Set_sbt_dn, within=pyo.NonNegativeReals)

        # - flows
        model.Pf_tn = pyo.Var(model.Set_slt_tn, within=pyo.Reals)
        model.Pf_dn = pyo.Var(model.Set_slt_dn, within=pyo.Reals)
        model.Qf_dn = pyo.Var(model.Set_slt_dn, within=pyo.Reals)

        # - load shedding (curtailed load)
        model.Pc = pyo.Var(model.Set_sbt, within=pyo.NonNegativeReals)
        model.Qc = pyo.Var(model.Set_sbt_dn, within=pyo.NonNegativeReals)

        # - failure logic (same naming as old file)
        model.shifted_gust_speed = pyo.Var(model.Set_slt_tn | model.Set_slt_dn,
                                           within=pyo.NonNegativeReals, bounds=(0, 100))
        model.fail_prob = pyo.Var(model.Set_slt_tn | model.Set_slt_dn,
                                  within=pyo.NonNegativeReals, bounds=(0, 1))
        model.branch_status = pyo.Var(model.Set_slt_tn | model.Set_slt_dn, within=pyo.Binary)
        model.fail_condition = pyo.Var(model.Set_slt_tn | model.Set_slt_dn, within=pyo.Binary)
        model.fail_indicator = pyo.Var(model.Set_slt_tn | model.Set_slt_dn, within=pyo.Binary)
        model.fail_applies = pyo.Var(model.Set_slt_tn | model.Set_slt_dn, within=pyo.Binary)
        model.repair_applies = pyo.Var(model.Set_slt_tn | model.Set_slt_dn, within=pyo.Binary)

        # ------------------------------------------------------------------
        # 4. Parameters
        # ------------------------------------------------------------------
        # 4.1)  Scenario probability
        model.scn_prob = pyo.Param(model.Set_scn, initialize=scn_prob)

        # 4.2)  Network static data  (costs, limits …)  – match names exactly
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

        # - load shedding (curtailment), repair, hardening costs
        model.Pc_cost = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                     initialize={b: self.data.cost_bus_ls[b - 1]
                                                 for b in net.data.net.bus})
        model.Qc_cost = pyo.Param(model.Set_bus_dn,
                                     initialize={b: self.data.cost_bus_ls[b - 1]
                                                 for b in Set_bus_dn})
        model.rep_cost = pyo.Param(model.Set_bch_tn | model.Set_bch_dn,
                                      initialize={l: self.data.cost_bch_rep[l - 1]
                                                  for l in range(1, len(net.data.net.bch) + 1)})
        model.line_hrdn_cost = pyo.Param(model.Set_bch_tn | model.Set_bch_dn,
                                      initialize={l: self.data.cost_bch_hrdn[l - 1]
                                                  for l in range(1, len(net.data.net.bch) + 1)})
        model.budget = pyo.Param(initialize=self.data.budget_bch_hrdn)

        # - demand profiles (Pd, Qd) by absolute hour
        Pd_param = {}
        Qd_param = {}
        for sim in ws_scenarios:
            sc = sim["simulation_id"]
            bgn = _abs_start[sc]
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

        model.gust_speed = pyo.Param(model.Set_scn,
                                     initialize={sc: 0})  # dummy, used only in expression
        model.gust_speed_ts = pyo.Param(model.Set_scn,
                                        default=None, mutable=True)
        # easier: store as dict inside rule – see constraints below
        model.rand_num = pyo.Param(model.Set_slt_tn | model.Set_slt_dn,
                                   initialize=rand_dict)
        model.impacted_branches = pyo.Param(model.Set_slt_tn | model.Set_slt_dn,
                                            initialize=impact_dict, within=pyo.Binary)
        model.branch_ttr = pyo.Param(model.Set_scn, model.Set_bch_tn | model.Set_bch_dn,
                                     initialize=ttr_dict)

        # ------------------------------------------------------------------
        # 4. Constraints
        # ------------------------------------------------------------------
        # 4.1) Budget
        def investment_budget_rule(model):
            return sum(
                model.line_hrdn_cost[l] * model.line_hrdn[l]
                for l in model.Set_bch_dn_lines
            ) <= model.budget

        model.Constraint_InvestmentBudget = pyo.Constraint(rule=investment_budget_rule)

        # 4.2) Shifted gust speed
        def shifted_gust_rule(model, sc, l, t):
            gs = gust_dict[(sc, t)]
            return model.shifted_gust_speed[sc, l, t] == gs - model.line_hrdn[l]

        model.Constraint_ShiftedGust = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn,
                                                      rule=shifted_gust_rule)

        # 4.3) Piece-wise fragility
        fragility_data = self.piecewise_linearize_fragility(net, num_pieces=6)

        # Precompute a lookup so fragility_rule can turn an x-value into its piecewise index
        gust_speeds = fragility_data["gust_speeds"]
        gust_index_map = {gs: i for i, gs in enumerate(gust_speeds)}

        def fragility_rule(model, sc, l, t, x):
            idx = gust_index_map[x]
            return fragility_data["fail_probs"][l][idx]

        model.Piecewise_Fragility = pyo.Piecewise(
            model.Set_slt_tn | model.Set_slt_dn,
            model.fail_prob,
            model.shifted_gust_speed,
            pw_pts=gust_speeds,
            f_rule=fragility_rule,
            pw_constr_type="EQ",
            pw_repn="DCC"
        )

        # 4.4) Failure logic constraints (unchanged from the original version)
        BigM = 1e3

        def fail_cond_1(model, sc, l, t):
            return model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] <= BigM * model.fail_condition[sc, l, t]

        def fail_cond_2(model, sc, l, t):
            return model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] >= (model.fail_condition[sc, l, t] - 1) * BigM

        model.Constraint_FailCond1 = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn, rule=fail_cond_1)
        model.Constraint_FailCond2 = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn, rule=fail_cond_2)

        def fail_ind_1(model, sc, l, t):
            return model.fail_indicator[sc, l, t] <= model.fail_condition[sc, l, t]

        def fail_ind_2(model, sc, l, t):
            return model.fail_indicator[sc, l, t] <= model.impacted_branches[sc, l, t]

        def fail_ind_3(model, sc, l, t):
            return model.fail_indicator[sc, l, t] >= model.fail_condition[sc, l, t] + \
                model.impacted_branches[sc, l, t] - 1

        model.Constraint_FailInd1 = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn, rule=fail_ind_1)
        model.Constraint_FailInd2 = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn, rule=fail_ind_2)
        model.Constraint_FailInd3 = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn, rule=fail_ind_3)

        # branch_status = 0 if failed and before repair duration ends
        def failure_duration_rule(model, sc, l, t):
            if t > model.branch_ttr[sc, l]:
                return model.branch_status[sc, l, t] == \
                    1 - sum(model.fail_applies[sc, l, tp]
                            for tp in range(t - model.branch_ttr[sc, l], t))
            return pyo.Constraint.Skip

        model.Constraint_FailureDuration = pyo.Constraint(model.Set_slt_tn | model.Set_slt_dn,
                                                          rule=failure_duration_rule)

        # 4.5) Power flow definitions
        def dc_flow_rule(model, sc, l, t):
            i, j = net.data.net.bch[l - 1]
            return model.Pf_tn[sc, l, t] == model.B_tn[l] * (model.theta[sc, i, t] - model.theta[sc, j, t])

        model.Constraint_FlowDef_TN = pyo.Constraint(model.Set_slt_tn, rule=dc_flow_rule)

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
            return model.Qf_dn[sc, l, t] <= model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S4_dn_rule(model, sc, l, t):
            return model.Qf_dn[sc, l, t] >= -model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S5_dn_rule(model, sc, l, t):
            return model.Pf_dn[sc, l, t] + model.Qf_dn[sc, l, t] \
                <= sqrt2 * model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S6_dn_rule(model, sc, l, t):
            return model.Pf_dn[sc, l, t] + model.Qf_dn[sc, l, t] \
                >= -sqrt2 * model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S7_dn_rule(model, sc, l, t):
            return model.Pf_dn[sc, l, t] - model.Qf_dn[sc, l, t] \
                <= sqrt2 * model.Smax_dn[l] * model.branch_status[sc, l, t]

        def S8_dn_rule(model, sc, l, t):
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

        # 4.6) Power balance constraints (TN: active; DN: active & reactive)
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
            Pgrid = (model.Pimp[t] - model.Pexp[t]) if b == slack_bus else 0

            return Pg + Pgrid + Pf_in - Pf_out == model.Pd[sc, b, t] - model.Pc[sc, b, t]

        model.Constraint_PBalance_TN = pyo.Constraint(model.Set_sbt, filter=lambda idx: idx[1] in Set_bus_tn,
                                                  rule=P_balance_tn)

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
            Qg = sum(model.Qg[sc, g, t] for g in model.Set_gen if net.data.net.gen[g - 1] == b)
            Qf_in_dn = sum(model.Qf_dn[sc, l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][1] == b)
            Qf_out_dn = sum(model.Qf_dn[sc, l, t] for l in model.Set_bch_dn if net.data.net.bch[l - 1][0] == b)
            return Qg + Qf_in_dn - Qf_out_dn == model.Qd[sc, b, t] - model.Qc[sc, b, t]

        model.Constraint_QBalance_DN = pyo.Constraint(model.Set_sbt_dn, rule=Q_balance_dn)

        # 4.7) Voltage limits DN
        def V2_dn_upper_limit_rule(model, sc, b, t):
            return model.V2_dn[sc, b, t] <= model.V2_max[b]

        def V2_dn_lower_limit_rule(model, sc, b, t):
            return model.V2_dn[sc, b, t] >= model.V2_min[b]

        model.Constraint_V2max = pyo.Constraint(model.Set_sbt_dn, rule=V2_dn_upper_limit_rule)
        model.Constraint_V2min = pyo.Constraint(model.Set_sbt_dn, rule=V2_dn_lower_limit_rule)

        # 4.8) Generator limits
        def Pg_upper_limit_rule(model, sc, g, t):
            return model.Pg[sc, g, t] <= model.Pg_max[g]

        def Pg_lower_limit_rule(model, sc, g, t):
            return model.Pg[sc, g, t] >= model.Pg_min[g]

        def Qg_upper_limit_rule(model, sc, g, t):
            return model.Qg[sc, g, t] <= model.Qg_max[g]

        def Qg_lower_limit_rule(model, sc, g, t):
            return model.Qg[sc, g, t] >= model.Qg_min[g]

        model.Constraint_PgUpperLimit = pyo.Constraint(model.Set_sgt, rule=Pg_upper_limit_rule)
        model.Constraint_PgLowerLimit = pyo.Constraint(model.Set_sgt, rule=Pg_lower_limit_rule)
        model.Constraint_QgUpperLimit = pyo.Constraint(model.Set_sgt, rule=Qg_upper_limit_rule)
        model.Constraint_QgLowerLimit = pyo.Constraint(model.Set_sgt, rule=Qg_lower_limit_rule)

        # 4.9) Slack-bus reference theta = 0
        def slack_rule(model, sc, t):
            return model.theta[sc, slack_bus, t] == 0

        model.Constraint_Slack = pyo.Constraint([(sc, t) for sc in Set_scn
                                                 for t in Set_ts_scn[sc]],
                                                rule=slack_rule)

        # ------------------------------------------------------------------
        # 5. Objective
        # ------------------------------------------------------------------
        def objective_rule(model):
            inv_cost = sum(
                # line hardening cost
                model.line_hrdn_cost[l] * model.line_hrdn[l]
                for l in model.Set_bch_tn | model.Set_bch_dn
            )
            exp_op_cost = sum(
                model.scn_prob[sc] * (
                    # generation cost
                    sum(model.gen_cost_coef[g, 0] + model.gen_cost_coef[g, 1] * model.Pg[sc, g, t]
                        for g in model.Set_gen for t in Set_ts_scn[sc])
                    # load shedding cost
                    + sum(model.Pc_cost[b] * model.Pc[sc, b, t]
                          for b in net.data.net.bus for t in Set_ts_scn[sc])
                    + sum(model.Qc_cost[b] * model.Qc[sc, b, t]
                          for b in Set_bus_dn for t in Set_ts_scn[sc])
                    # repair cost
                    + sum(model.rep_cost[l] * model.repair_applies[sc, l, t]
                          for l in model.Set_bch_tn | model.Set_bch_dn
                          for t in Set_ts_scn[sc])
                )
                for sc in Set_scn
            )
            return inv_cost + exp_op_cost

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # done
        return model

    def solve_investment_model(
            self,
            model,
            solver_name: str = "gurobi",
            mip_gap: float = 5e-3,
            time_limit: int = 60,
            write_lp: bool = False,
            csv_path: str | None = None,
            **solver_options
    ):
        """
        Solve a Pyomo investment model and export essential results.

        Parameters
        ----------
        mip_gap       : Absolute/relative MIP gap (default 0.5 %)
        time_limit    : Wall-clock limit in seconds
        write_lp      : Dump the model to LP file *only when*
                        write_lp is True **or** the solver fails
        csv_path      : If given, write a tidy CSV of selected variables
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
            if write_lp or not csv_path:  # dump model only when really needed
                model.write("LP_Models/model_at_failure.lp",
                            io_options={"symbolic_solver_labels": True})
            raise RuntimeError(msg)

        # --- optional exports ------------------------------------------------
        if write_lp:
            model.write("LP_Models/solved_model.lp",
                        io_options={"symbolic_solver_labels": True})

        if csv_path:
            self._write_selected_variables_to_csv(model, csv_path)

        return {
            "objective": best_obj,
            "gap": mip_gap_out,
            "status": str(status),
            "termination": str(term_cond),
            "runtime_s": results.solver.time
        }


    def _write_selected_variables_to_csv(model, path: str = "Optimization_Results/Investment_Model/results"):
        """
        Write a subset of model variables to <path> as tidy CSV.

        Each row has:  variable , index_tuple , value
          • For scalar variables index_tuple is left blank (“”).
          • For indexed variables the full Pyomo index tuple is written as a
            literal Python tuple, e.g. "(scenario-7,  14,  3)".

        Parameters
        ----------
        model : pyomo.core.base.PyomoModel.ConcreteModel
        path  : str or pathlib.Path
            Output filename (parent folders are created if necessary).
        """
        # -- choose which variables to export --------------------------------
        important = (
            # 1st-stage
            "line_hrdn",
            # generation & curtailment
            "Pg", "Qg", "Pc", "Qc",
            # power flows
            "Pf_tn", "Pf_dn",
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["variable", "index_tuple", "value"])

            for name in important:
                var = getattr(model, name, None)
                if var is None:  # model might lack a given var
                    continue

                if not var.is_indexed():  # scalar variable
                    wr.writerow([name, "", pyo.value(var)])
                else:  # indexed variable
                    for idx in var:
                        wr.writerow([name, str(idx), pyo.value(var[idx])])


    def piecewise_linearize_fragility(self, net, num_pieces):
        """
        Piecewise linearize the fragility curve function

        Parameters:
        - net: Instance of NetworkClass (contains fragility curve data).
        - num_pieces: Number of linear segments for approximation.

        Returns:
        - A dictionary containing piecewise linear points (gust speed vs failure probability).
        """

        gust_speeds = np.linspace(min(net.data.fragility.thrd_1),
                                  max(net.data.fragility.thrd_2),
                                  num_pieces).tolist()
        # for each branch, build a little array of fail‐probs
        fail_probs = {
            l: [
                0.0 if (x - net.data.fragility.shift_f[l - 1]) < net.data.fragility.thrd_1[l - 1]
                else 1.0 if (x - net.data.fragility.shift_f[l - 1]) > net.data.fragility.thrd_2[l - 1]
                else float(lognorm.cdf(
                    x - net.data.fragility.shift_f[l - 1],
                    s=net.data.fragility.sigma[l - 1],
                    scale=math.exp(net.data.fragility.mu[l - 1])
                ))
                for x in gust_speeds
            ]
            for l in range(1, len(net.data.net.bch) + 1)
        }
        return {"gust_speeds": gust_speeds, "fail_probs": fail_probs}

    def _get_bch_hrdn_limits(self):
        return self.data.bch_hrdn_limits

    def _get_cost_bch_hrdn(self):
        return self.data.cost_bch_hrdn

    def _get_cost_bch_rep(self):
        return self.data.cost_bch_rep

    def _get_cost_bus_ls(self):
        return self.data.cost_bus_ls

    def _get_budget_bch_hrdn(self):
        return self.data.budget_bch_hrdn
