import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.stats import lognorm
from network_linear import NetworkClass
from windstorm import WindClass
from config import InvestmentConfig
from network_factory import make_network
import json
import os
import csv


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
                               path_all_ws_scenarios: str = "Scenario_Results/all_ws_scenarios_year.json"):
        """
        Build a Pyomo MILP model for resilience enhancement investment planning (line hardening)
        against windstorms, using the form of stochastic programming over multiple scenarios.
        """
        # Create an instance for the windstorm and network class
        ws = WindClass()
        net = make_network('matpower_case22')


        # Load windstorm scenarios from JSON file
        with open(path_all_ws_scenarios) as f:
            all_ws_scenarios = json.load(f)
        scn_ids = [sim["simulation_id"] for sim in all_ws_scenarios]
        scn_prob = {scn: 1.0 / len(scn_ids) for scn in scn_ids} # Assumes each scenario has equal probability

        # Define Pyomo model
        model = pyo.ConcreteModel()

        # 1. Sets
        # 1.1) Single sets:
        model.Set_scn = pyo.Set(initialize=scn_ids)
        model.Set_bus = pyo.Set(initialize=net.data.net.bus)  # Buses
        model.Set_bch = pyo.Set(initialize=range(1, len(net.data.net.bch) + 1))  # Branches
        model.Set_gen = pyo.Set(initialize=range(1, len(net.data.net.gen) + 1))  # Generators

        # Note the number of timesteps varies between scenarios
        scn_ts_dict = {sim["simulation_id"]: list(range(1, len(sim["bch_rand_nums"][0]) + 1))
                       for sim in all_ws_scenarios}
        model.Set_ts_scn = pyo.Set(model.Set_scn, initialize=scn_ts_dict) # Timesteps are sets indexed over Set_scn

        # 1.2) Tuple sets:
        # -----------------------------------------------------------
        # Since Pyomo does not support indexed set (i.e., the Set of timesteps can not be indexed over the Set of scenarios),
        # tuple sets should be used when parameter is based on both Set_scn and Set_ts simultaneously:
        # 1.2.1) 2-tuple set for scenario and timestep (sc, t)
        st_index = [
            (sc, t)
            for sc in scn_ids
            for t in scn_ts_dict[sc]
        ]
        model.Set_st = pyo.Set(initialize=st_index, dimen=2)

        # 1.2.2) 3-tuple set for scenario, bus, and timestep (sc, b, t)
        sbt_index = [
            (sc, b, t)
            for sc in scn_ids
            for b in net.data.net.bus
            for t in scn_ts_dict[sc]
        ]
        model.Set_sbt = pyo.Set(initialize=sbt_index, dimen=3)

        # 1.2.3) 3-tuple set for scenario, branch, and timestep (sc, l, t)
        slt_index = [
            (sc, l, t)
            for sc in scn_ids
            for l in range(1, len(net.data.net.bch) + 1)
            for t in scn_ts_dict[sc]
        ]
        model.Set_slt = pyo.Set(initialize=slt_index, dimen=3)

        # 1.2.4) 3-tuple set for scenario, gen, and timestep (sc, g, t)
        sgt_index = [
            (sc, g, t)
            for sc in scn_ids
            for g in range(1, len(net.data.net.gen)+1)
            for t in scn_ts_dict[sc]
        ]
        model.Set_sgt = pyo.Set(initialize=sgt_index, dimen=3)

        # 2. Parameters
        # 2.1) Initialize scenario parameters:
        model.scn_prob = pyo.Param(model.Set_scn, initialize=scn_prob, mutable=False)

        # 2.2) Initialize network parameters:
        # 2.2.1) Set demand value for each (scn, bus, ts):
        demand_dict = {}
        for sim in all_ws_scenarios:
            sc = sim["simulation_id"]
            # absolute start hour in full-year
            abs_bgn = sim["events"][0]["bgn_hr"]
            for b in model.Set_bus:
                for t in model.Set_ts_scn[sc]:
                    abs_t = abs_bgn + t - 1
                    demand = net.data.net.demand_profile_active[b - 1][abs_t - 1]
                    demand_dict[(sc, b, t)] = demand
        model.demand = pyo.Param(model.Set_sbt, initialize=demand_dict)

        # 2.2.2) Generation cost coefficients and limits
        coef_len = len(net.data.net.gen_cost_coef[0])
        model.gen_cost_coef = pyo.Param(model.Set_gen, range(coef_len),
                                        initialize={(i + 1, j): net.data.net.gen_cost_coef[i][j]
                                                    for i in range(len(net.data.net.gen_cost_coef))
                                                    for j in range(coef_len)})

        model.gen_active_max = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                                    enumerate(net.data.net.gen_active_max)})

        model.gen_active_min = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                                    enumerate(net.data.net.gen_active_min)})

        model.bch_cap = pyo.Param(model.Set_bch, initialize={i + 1: bc for i, bc in
                                                             enumerate(net.data.net.bch_cap)})

        # 2.2.3) Calculate susceptance B for each line (under the assumption of DC power flow)
        model.bch_B = pyo.Param(model.Set_bch, initialize={i + 1: 1 / X for i, X in
                                                           enumerate(net.data.net.bch_X)})

        # 2.2.4) Initialize investment parameters (budget and costs):
        # initialize dictionaries
        cost_hrdn = {i + 1: c for i, c in enumerate(self._get_cost_bch_hrdn())} # hardening cost
        cost_rep = {i + 1: c for i, c in enumerate(self._get_cost_bch_rep())} # repair cost
        cost_ls = {i + 1: c for i, c in enumerate(self._get_cost_bus_ls())} # load shedding cost
        # set parameters
        model.budget = pyo.Param(initialize=self._get_budget_bch_hrdn()) # total investment budget
        model.cost_bch_hrdn = pyo.Param(model.Set_bch, initialize=cost_hrdn)
        model.cost_repair = pyo.Param(model.Set_bch, initialize=cost_rep)
        model.cost_load_shed = pyo.Param(model.Set_bus, initialize=cost_ls)

        # 2.3) Initialize windstorm parameters:

        # 2.3.1) initialize gust speed for each (scn, ts)
        # (it is assumed the gust speed is the same for all branches at each timestep)
        gust_dict = {}
        for sim in all_ws_scenarios:
            sc = sim["simulation_id"]
            dur = len(sim["events"][0]["gust_speed"])
            for t in model.Set_ts_scn[sc]:
                gust = sim["events"][0]["gust_speed"][t-1] if t<=dur else 0 # gust speed is 0 beyond windstorm duration
                gust_dict[(sc, t)] = gust
        model.gust_speed = pyo.Param(model.Set_st, initialize=gust_dict)

        # 2.3.2) initialize random numbers for line failure sampling for each (scn, bch, ts)
        rand_dict = {(sim["simulation_id"], l, t): sim["bch_rand_nums"][l - 1][t - 1]
                     for sim in all_ws_scenarios
                     for l in model.Set_bch
                     for t in model.Set_ts_scn[sim["simulation_id"]]}
        model.rand_num = pyo.Param(model.Set_slt, initialize=rand_dict)

        # 2.3.3) Initialize impacted branches flags for each (scn, bch, ts)
        impact_dict = {(sim["simulation_id"], l, t): sim["flgs_impacted_bch"][l - 1][t - 1]
                    for sim in all_ws_scenarios
                    for l in model.Set_bch
                    for t in model.Set_ts_scn[sim["simulation_id"]]}
        model.impacted_branches = pyo.Param(model.Set_slt, initialize=impact_dict, within=pyo.Binary)

        # 2.3.4) Initialize time to repair values for each (scn, bch)
        ttr_dict = {(sim["simulation_id"], l): sim["bch_ttr"][l - 1]
                    for sim in all_ws_scenarios
                    for l in model.Set_bch}
        model.branch_ttr = pyo.Param(model.Set_scn, model.Set_bch, initialize=ttr_dict)


        # 3. Variables
        # 3.1) First-stage hardening decisions
        model.bch_hrdn = pyo.Var(model.Set_bch, within=pyo.NonNegativeReals,
                                 bounds=(self.data.bch_hrdn_limits[0],
                                         self.data.bch_hrdn_limits[1]))  # Fragility curve shift made by hardening
        # 3.2) Second-stage recourse variables indexed by scenarios
        model.P_gen = pyo.Var(model.Set_sgt, within=pyo.NonNegativeReals)
        model.load_shed = pyo.Var(model.Set_sbt, within=pyo.NonNegativeReals)
        model.theta = pyo.Var(model.Set_sbt, within=pyo.Reals)
        model.P_flow = pyo.Var(model.Set_slt, within=pyo.Reals)
        model.gen_cost = pyo.Var(model.Set_sgt, within=pyo.NonNegativeReals)
        model.shifted_gust_speed = pyo.Var(model.Set_slt, within=pyo.NonNegativeReals, bounds=(0, 100))
        model.fail_prob = pyo.Var(model.Set_slt, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.branch_status = pyo.Var(model.Set_slt, within=pyo.Binary)
        model.fail_condition = pyo.Var(model.Set_slt, within=pyo.Binary)
        model.fail_indicator = pyo.Var(model.Set_slt, within=pyo.Binary)
        model.fail_applies = pyo.Var(model.Set_slt, within=pyo.Binary)
        model.repair_applies = pyo.Var(model.Set_slt, within=pyo.Binary)


        # 4. Constraints
        # 4.1) Budget constraint:
        def budget_rule(model):
            return sum(model.cost_bch_hrdn[l] * model.bch_hrdn[l] for l in model.Set_bch) <= model.budget

        model.Constraint_Budget = pyo.Constraint(rule=budget_rule)

        # 4.2) Power balance constraint:
        def power_balance_rule(model, sc, b, t):
            gen_out = sum(model.P_gen[sc,g,t] for g in model.Set_gen if net.data.net.gen[g-1]==b)
            inflow  = sum(model.P_flow[sc,l,t] for l in model.Set_bch if net.data.net.bch[l-1][1]==b)
            outflow = sum(model.P_flow[sc,l,t] for l in model.Set_bch if net.data.net.bch[l-1][0]==b)
            return gen_out + inflow - outflow + model.load_shed[sc,b,t] == model.demand[sc,b,t]

        model.Constraint_PowerBalance = pyo.Constraint(model.Set_sbt, rule=power_balance_rule)

        # 4.3) Branch power flow constraint:
        def power_flow_def_rule(model, sc, l, t):
            i,j = net.data.net.bch[l-1]
            return model.P_flow[sc,l,t] == model.bch_B[l] * (model.theta[sc,i,t] - model.theta[sc,j,t])

        model.Constraint_FlowDefinition = pyo.Constraint(
            model.Set_slt, rule=power_flow_def_rule)

        # 4.4) Line thermal limit constraints:
        # (If branch_status is 0, below two constraints enforces the branch's power flow to be 0)
        def flow_upper_limit_rule(model, sc, l, t):
            return model.P_flow[sc, l, t] <= model.bch_cap[l] * model.branch_status[sc, l, t]

        def flow_lower_limit_rule(model, sc, l, t):
            return model.P_flow[sc, l, t] >= -model.bch_cap[l] * model.branch_status[sc, l, t]

        model.Constraint_FlowLimit_Upper = pyo.Constraint(model.Set_slt, rule=flow_upper_limit_rule)
        model.Constraint_FlowLimit_Lower = pyo.Constraint(model.Set_slt, rule=flow_lower_limit_rule)

        # 4.5) Generation limit constraints:
        def gen_upper_limit_rule(model, sc, g, t):
            return model.P_gen[sc, g, t] <= model.gen_active_max[g]

        def gen_lower_limit_rule(model, sc, g, t):
            return model.P_gen[sc, g, t] >= model.gen_active_min[g]

        model.Constraint_GenUpperLimit = pyo.Constraint(model.Set_sgt, rule=gen_upper_limit_rule)
        model.Constraint_GenLowerLimit = pyo.Constraint(model.Set_sgt, rule=gen_lower_limit_rule)

        # 4.6) Generation cost constraint:
        def gen_cost_rule(model, sc, g, t):
            return model.gen_cost[sc,g,t] == model.gen_cost_coef[g,0] + model.gen_cost_coef[g,1]*model.P_gen[sc,g,t]

        model.Constraint_GenCost = pyo.Constraint(model.Set_sgt, rule=gen_cost_rule)

        # 4.7) Slack bus angle constraint:
        def slack_rule(model, sc, t):
            return model.theta[sc, net.data.net.slack_bus, t] == 0

        model.Constraint_SlackBus = pyo.Constraint(model.Set_st, rule=slack_rule)

        # 4.8) Shifted gust speed constraint
        def shifted_gust_speed_rule(model, sc, l, t):
            return model.shifted_gust_speed[sc,l,t] == model.gust_speed[sc,t] - model.bch_hrdn[l]

        model.Constraint_ShiftedGustSpeed = pyo.Constraint(model.Set_slt, rule=shifted_gust_speed_rule)

        # 4.9) Piecewise Linear Fragility Approximation
        # generate breakpoints and fragility function values
        fragility_data = self.piecewise_linearize_fragility(ws, num_pieces=6)

        # Precompute index map for gust speeds
        gust_speeds = fragility_data["gust_speeds"]
        gust_index_map = {x: i for i, x in enumerate(gust_speeds)}

        # Define a function to return the failure probability for each (bch, ts, x)
        def fragility_rule(model, scn, l, t, x):
            idx = gust_index_map[x]
            return fragility_data["fail_probs"][idx]

        model.Piecewise_Fragility = pyo.Piecewise(
            model.Set_slt,
            model.fail_prob,
            model.shifted_gust_speed,
            pw_pts=gust_speeds,
            f_rule=fragility_rule,
            pw_constr_type='EQ',
            pw_repn='DCC'
        )


        # 4.10) Line failure and repair constraints
        BigM = 1e3

        def fail_condition_rule_1(model, sc, l, t):
            """ Enforce fail_condition[l, t] = 1 when fail_prob[l, t] > rand_num[l, t] """
            return model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] <= model.fail_condition[sc, l, t] * BigM

        def fail_condition_rule_2(model, sc, l, t):
            """ Enforce fail_condition[l, t] = 0 when fail_prob[l, t] <= rand_num[l, t] """
            return model.fail_prob[sc, l, t] - model.rand_num[sc, l, t] >= (model.fail_condition[sc, l, t] - 1) * BigM

        model.Constraint_FailCondition1 = pyo.Constraint(model.Set_slt, rule=fail_condition_rule_1)
        model.Constraint_FailCondition2 = pyo.Constraint(model.Set_slt, rule=fail_condition_rule_2)

        # 2) Ensure fail_indicator is 1 only when fail_condition = 1 and impacted_branches = 1
        def fail_indicator_rule_1(model, sc, l, t):
            """ fail_indicator[l, t] = 1 only if fail_condition[l, t] = 1 and impacted_branches[l, t] = 1 """
            return model.fail_indicator[sc, l, t] <= model.fail_condition[sc, l, t]

        def fail_indicator_rule_2(model, sc, l, t):
            """ fail_indicator[l, t] = 1 only if impacted_branches[l, t] = 1 """
            return model.fail_indicator[sc, l, t] <= model.impacted_branches[sc, l, t]

        def fail_indicator_rule_3(model, sc, l, t):
            """ fail_indicator[l, t] = 1 if both fail_condition and impacted_branches are 1 """
            return (model.fail_indicator[sc, l, t] >= model.fail_condition[sc, l, t]
                                                      + model.impacted_branches[sc, l, t] - 1)

        model.Constraint_FailIndicator1 = pyo.Constraint(model.Set_slt, rule=fail_indicator_rule_1)
        model.Constraint_FailIndicator2 = pyo.Constraint(model.Set_slt, rule=fail_indicator_rule_2)
        model.Constraint_FailIndicator3 = pyo.Constraint(model.Set_slt, rule=fail_indicator_rule_3)

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

        model.Constraint_FailActivation1 = pyo.Constraint(model.Set_slt, rule=fail_activation_rule_1)
        model.Constraint_FailActivation2 = pyo.Constraint(model.Set_slt, rule=fail_activation_rule_2)
        model.Constraint_FailActivation3 = pyo.Constraint(model.Set_slt, rule=fail_activation_rule_3)

        def enforce_failure_duration_rule(model, sc, l, t):
            """
            Ensures a failed line stays down for branch_ttr[l] timesteps,
            and restores after branch_ttr[l] timesteps
            """
            if t > model.branch_ttr[sc, l]:
                return model.branch_status[sc, l, t] == 1 - sum(
                    model.fail_applies[sc, l, t_prime] for t_prime in range(t - model.branch_ttr[sc, l], t))
            else:
                return pyo.Constraint.Skip

        model.Constraint_FailureDuration = pyo.Constraint(model.Set_slt, rule=enforce_failure_duration_rule)

        def repair_applies_rule_1(model, sc, l, t):
            """ Enforce repair_applies to be 1 when branch_status switches from 0 to 1. """
            if t > 1:
                return (model.repair_applies[sc, l, t] >= model.branch_status[sc, l, t]
                                                          - model.branch_status[sc, l, t - 1])
            else:
                return pyo.Constraint.Skip

        def repair_applies_rule_2(model, sc, l, t):
            """ Ensures repair_applies is 0 when branch_status[l, t] is 0. """
            if t > 1:
                return model.repair_applies[sc, l, t] <= model.branch_status[sc, l, t]
            else:
                return pyo.Constraint.Skip

        def repair_applies_rule_3(model, sc, l, t):
            """ Ensures repair_applies is 0 when branch_status[l, t-1] is 1. """
            if t > 1:
                return model.repair_applies[sc, l, t] <= 1 - model.branch_status[sc, l, t - 1]
            else:
                return pyo.Constraint.Skip

        model.Constraint_RepairAppliesRule1 = pyo.Constraint(model.Set_slt, rule=repair_applies_rule_1)
        model.Constraint_RepairAppliesRule2 = pyo.Constraint(model.Set_slt, rule=repair_applies_rule_2)
        model.Constraint_RepairAppliesRule3 = pyo.Constraint(model.Set_slt, rule=repair_applies_rule_3)


        # 5. Objective function: Minimize total cost = investment cost + expected recourse cost
        def objective_function(model):
            inv_cost = sum(model.cost_bch_hrdn[l] * model.bch_hrdn[l] for l in model.Set_bch)
            rec_cost = sum(
                model.scn_prob[sc] * (
                    sum(model.cost_load_shed[b]*model.load_shed[sc,b,t]
                        for b in model.Set_bus for t in model.Set_ts_scn[sc]) # load shedding cost
                  + sum(model.cost_repair[l]*model.repair_applies[sc,l,t]
                        for l in model.Set_bch for t in model.Set_ts_scn[sc]) # branch repair cost
                  + sum(model.gen_cost_coef[g,0] + model.gen_cost_coef[g,1]*model.P_gen[sc,g,t]
                        for g in model.Set_gen for t in model.Set_ts_scn[sc]) # generation cost
                )
                for sc in model.Set_scn
            )
            return inv_cost + rec_cost

        model.Objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

        return model


    def solve_investment_model(self, model, solver='gurobi'):
        # Solve the model
        solver = SolverFactory(solver)
        # solver.options['feasibility relaxation'] = True
        # results = solver.solve(model, tee=True, options={"ResultFile": "feas_relaxed.sol",
        #                                                  "MIPGap": 0.01,
        #                                                  "TimeLimit": 60,
        #                                                  })

        solver.options["MIPGap"] = 0.005
        solver.options["TimeLimit"] = 60

        results = solver.solve(model, tee=True, logfile="solver_log.txt")

        # solver.solve(model, tee=True, keepfiles=True, logfile="gurobi_log.txt", warmstart=True,
        #              symbolic_solver_labels=True)

        model.write("LP_Models/infeasible_model.lp", io_options={"symbolic_solver_labels": True})

        # Ensure output directory exists
        output_dir = "Optimization_Results"
        os.makedirs(output_dir, exist_ok=True)

        # Define output file path
        output_file = os.path.join(output_dir, "optimization_results.csv")

        # Write results to a CSV file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Variable", "Index", "Value"])  # Header row

            for v in model.component_objects(pyo.Var, active=True):
                var_object = getattr(model, v.name)
                for index in var_object:
                    writer.writerow([v.name, index, var_object[index].value])

        print(f"\nOptimization results saved to: {output_file}")


        return results


    def piecewise_linearize_fragility(self, ws, num_pieces):
        """
        Piecewise linearize the fragility curve function

        Parameters:
        - ws: Instance of WindClass (contains fragility curve data).
        - num_pieces: Number of linear segments for approximation.

        Returns:
        - A dictionary containing piecewise linear points (gust speed vs failure probability).
        """

        # Extract fragility parameters
        mu = ws._get_frg_mu()
        sigma = ws._get_frg_sigma()
        thrd_1 = ws._get_frg_thrd_1()
        thrd_2 = ws._get_frg_thrd_2()
        shift_f = ws._get_frg_shift_f()

        # Generate breakpoints for linearization
        gust_speeds = np.linspace(thrd_1, thrd_2, num_pieces).tolist()  # Convert to list
        fail_probs = []

        # Compute failure probabilities at each breakpoint
        shape = sigma
        scale = np.exp(mu)

        for speed in gust_speeds:
            adjusted_speed = float(speed - shift_f)
            if adjusted_speed < thrd_1:
                fail_probs.append(0.0)
            elif adjusted_speed > thrd_2:
                fail_probs.append(1.0)
            else:
                pof = float(lognorm.cdf(adjusted_speed, s=shape, scale=scale))  # Convert to float
                fail_probs.append(pof)

        # Add breakpoint at gust_speed = 0 with fail_prob = 0
        gust_speeds.insert(0, 0)  # Insert 0 at the beginning
        fail_probs.insert(0, 0)  # Ensure failure probability is 0 at gust_speed = 0

        # Return piecewise linear data
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
