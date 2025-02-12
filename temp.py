import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.stats import lognorm
from network_linear import NetworkClass
from windstorm import WindClass
from config import InvestmentConfig
import json


class Object(object):
    pass


class InvestmentClass():

    def __init__(self, obj=None):
        # Get default values from InvestmentConfig
        if obj is None:
            obj = InvestmentConfig()

        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def build_investment_model(self):
        """
        Build a Pyomo MILP model for investment planning to enhance power system resilience against windstorms.
        """
        # Create an instance for the windstorm and network class
        ws = WindClass()
        net = NetworkClass()

        # Load windstorm scenarios from JSON file
        with open("Results/all_scenarios.json", "r") as f:
            all_results = json.load(f)

        # Define Pyomo model
        model = pyo.ConcreteModel()

        # 1. Sets
        model.Set_bus = pyo.Set(initialize=net.data.net.bus)  # Buses
        model.Set_bch = pyo.Set(initialize=range(1, len(net.data.net.bch) + 1))  # Branches
        model.Set_gen = pyo.Set(initialize=range(1, len(net.data.net.gen) + 1))  # Generators
        model.Set_ts = pyo.Set(initialize=range(1, ws._get_num_hrs_prd() + 1))  # Timesteps in a period


        # 2. Parameters
        # 1) Initialize network parameters:

        # Convert the demand profiles into a dictionary and initialize Pyomo demand profile parameters
        demand_dict = {
            (bus, ts): net.data.net.demand_profile_active[bus - 1][ts - 1]
            for bus in range(1, len(net.data.net.demand_profile_active) + 1)  # Iterate over buses (1-indexed)
            for ts in range(1, len(net.data.net.demand_profile_active[0]) + 1)  # Iterate over timesteps (1-indexed)
        }

        model.demand = pyo.Param(model.Set_bus, model.Set_ts, initialize=demand_dict)

        model.gen_cost_coef = pyo.Param(model.Set_gen, range(len(net.data.net.gen_cost_coef[0])),
                                        initialize={
                                            (i + 1, j): self.data.net.gen_cost_coef[i][j]
                                            for i in range(len(net.data.net.gen_cost_coef))
                                            for j in range(len(net.data.net.gen_cost_coef[i]))})

        model.gen_active_max = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                                    enumerate(net.data.net.gen_active_max)})

        model.gen_active_min = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                                    enumerate(net.data.net.gen_active_min)})

        model.bch_cap = pyo.Param(model.Set_bch, initialize={i + 1: bc for i, bc in
                                                             enumerate(net.data.net.bch_cap)})

        # calculate susceptance B for each line (under the assumption of DC power flow)
        model.bch_B = pyo.Param(model.Set_bch, initialize={i + 1: 1 / X for i, X in
                                                           enumerate(net.data.net.bch_X)})

        # 2) Initialize windstorm parameters:

        # 2.1) initialize gust speed (it is assumed the gust speed is the same for all branches at each timestep)
        # create a dictionary for initialization
        gust_speed_data = {ts: 0 for ts in model.Set_ts}  # Initialize all timesteps with zero wind speed (no storm)
        for event in all_results[0]["events"]:  # Loop through all windstorm events
            event_start = event["bgn_hr"]
            event_end = event_start + len(event["gust_speed"]) - 1
            # Assign wind speeds for timesteps within the event duration
            for ts in range(event_start, event_end + 1):
                gust_speed_data[ts] = max(gust_speed_data[ts], event["gust_speed"][ts - event_start])

        # assign the dictionary to the pyomo parameter
        model.gust_speed = pyo.Param(model.Set_ts, initialize=gust_speed_data)

        # 2.2) initialize random numbers for line failure sampling
        # create dictionary
        rand_num_data = {(l, t): all_results[0]["bch_rand_nums"][l - 1][t - 1]
                         for l in model.Set_bch
                         for t in model.Set_ts}
        # assign dictionary
        model.rand_num = pyo.Param(model.Set_bch, model.Set_ts, initialize=rand_num_data)

        # 2.3) initialize time to repair values
        # create dictionary
        ttr_data = {l: all_results[0]["bch_ttr"][l - 1] for l in model.Set_bch}
        # assign dictionary
        model.branch_ttr = pyo.Param(model.Set_bch, initialize=ttr_data)


        # 3) Initialize investment parameters (budget and costs):
        # convert lists to dictionaries
        cost_bch_hrdn_dict = {i + 1: cost for i, cost in enumerate(self._get_cost_bch_hrdn())}
        cost_bch_rep_dict = {i + 1: cost for i, cost in enumerate(self._get_cost_bch_rep())}
        cost_bus_ls_dict = {i + 1: cost for i, cost in enumerate(self._get_cost_bus_ls())}
        # set parameters
        model.budget = pyo.Param(initialize=self._get_budget_bch_hrdn())
        model.cost_bch_hrdn = pyo.Param(model.Set_bch, initialize=cost_bch_hrdn_dict)
        model.cost_repair = pyo.Param(model.Set_bch, initialize=cost_bch_rep_dict)
        model.cost_load_shed = pyo.Param(model.Set_bus, initialize=cost_bus_ls_dict)


        # 3. Variables
        model.bch_hrdn = pyo.Var(model.Set_bch, within=pyo.NonNegativeReals)  # Shift in fragility curve
        model.P_gen = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.NonNegativeReals)  # Generator output
        model.load_shed = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # Load shedding
        model.theta = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.Reals)  # Bus voltage angle
        model.P_flow = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Reals)  # Power flow on branches

        model.shifted_gust_speed = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.NonNegativeReals)
        model.fail_prob = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.NonNegativeReals)

        model.branch_status = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Binary)
        model.fail_indicator = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Binary)
        model.fail_applies = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Binary)
        model.repair_applies = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Binary)
        model.repair_complete = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Binary)


        # 4. Constraints
        # 1) Budget Constraint
        def budget_rule(model):
            return sum(model.cost_bch_hrdn[l] * model.bch_hrdn[l] for l in model.Set_bch) <= model.budget

        model.Constraint_Budget = pyo.Constraint(rule=budget_rule)

        # 2) Power Balance Constraint
        def power_balance_rule(model, b, t):
            total_gen = sum(model.P_gen[g, t] for g in model.Set_gen if net.data.net.gen[g - 1] == b)
            inflow = sum(model.P_flow[l, t] for l in model.Set_bch if net.data.net.bch[l - 1][1] == b)
            outflow = sum(model.P_flow[l, t] for l in model.Set_bch if net.data.net.bch[l - 1][0] == b)
            return total_gen + inflow - outflow == net.data.net.demand_active[b - 1]

        model.Constraint_PowerBalance = pyo.Constraint(model.Set_bus, model.Set_ts, rule=power_balance_rule)

        # 3) Branch power Flow Constraint
        BigM = 1e6

        def power_flow_rule_upper(model, l, t):
            """ Upper bound on power flow considering line failures """
            i, j = net.data.net.bch[l - 1]
            # if branch_status is 0, this constraint is relaxed and let the "flow_limit_rule" below to take control
            return model.P_flow[l, t] <= model.bch_B[l] * (model.theta[i, t] - model.theta[j, t]) + BigM * (
                        1 - model.branch_status[l, t])

        def power_flow_rule_lower(model, l, t):
            """ Lower bound on power flow considering line failures """
            i, j = net.data.net.bch[l - 1]
            # similar to above
            return model.P_flow[l, t] >= model.bch_B[l] * (model.theta[i, t] - model.theta[j, t]) - BigM * (
                        1 - model.branch_status[l, t])

        model.Constraint_PowerFlow_Upper = pyo.Constraint(model.Set_bch, model.Set_ts, rule=power_flow_rule_upper)
        model.Constraint_PowerFlow_Lower = pyo.Constraint(model.Set_bch, model.Set_ts, rule=power_flow_rule_lower)

        # If branch_status is 0, below two constraints enforces the branch's power flow to be 0
        def flow_limit_rule_upper(model, l, t):
            """ Upper bound on power flow considering line failures """
            return model.P_flow[l, t] <= model.bch_cap[l] * model.branch_status[l, t]

        def flow_limit_rule_lower(model, l, t):
            """ Lower bound on power flow considering line failures """
            return model.P_flow[l, t] >= -model.bch_cap[l] * model.branch_status[l, t]

        model.Constraint_FlowLimit_Upper = pyo.Constraint(model.Set_bch, model.Set_ts, rule=flow_limit_rule_upper)
        model.Constraint_FlowLimit_Lower = pyo.Constraint(model.Set_bch, model.Set_ts, rule=flow_limit_rule_lower)

        # 4) Generator limit constraints:
        def gen_upper_limit_rule(model, gen_idx):
            return model.P_gen[gen_idx] <= model.gen_active_max[gen_idx]

        def gen_lower_limit_rule(model, gen_idx):
            return model.P_gen[gen_idx] >= model.gen_active_min[gen_idx]

        model.Constraint_GenUpperLimit = pyo.Constraint(model.Set_gen, rule=gen_upper_limit_rule)
        model.Constraint_GenLowerLimit = pyo.Constraint(model.Set_gen, rule=gen_lower_limit_rule)

        # 5) Slack bus constraint
        def slack_bus_rule(model, t):
            return model.theta[net.data.net.slack_bus, t] == 0

        model.Constraint_SlackBus = pyo.Constraint(model.Set_ts, rule=slack_bus_rule)


        # 6) Shifted gust speed constraint
        def shifted_gust_speed_rule(model, l, t):
            return model.shifted_gust_speed[l, t] == model.gust_speed[t] - model.bch_hrdn[l]

        model.Constraint_ShiftedGustSpeed = pyo.Constraint(model.Set_bch, model.Set_ts, rule=shifted_gust_speed_rule)

        # 7) Piecewise Linear Fragility Approximation
        # generate breakpoints and fragility function values
        fragility_data = self.piecewise_linearize_fragility(ws, num_pieces=10)
        # use the break points and corresponding values to link "fail_prob" with "shifted_gust_speed"
        model.Piecewise_Fragility = pyo.Piecewise(
            model.Set_bch, model.Set_ts,
            model.fail_prob,
            model.shifted_gust_speed,
            pw_pts=fragility_data["gust_speeds"],
            f_rule=fragility_data["fail_probs"],
            pw_constr_type='CC'
        )


        # 8) Line failure and repair constraints
        BigM = 1e6

        def fail_indicator_rule_1(model, l, t):
            """
            Ensures fail_indicator[l, t] is 1 if rand_num[l, t] <= fail_prob[l, t].
            Uses Big-M to enforce the binary condition.
            """
            return model.fail_indicator[l, t] * BigM >= model.fail_prob[l, t] - model.rand_num[l, t]

        def fail_indicator_rule_2(model, l, t):
            """ Ensures fail_indicator[l, t] is 0 if rand_num[l, t] > fail_prob[l, t] """
            return model.fail_indicator[l, t] <= (model.fail_prob[l, t] - model.rand_num[l, t]) + 1

        model.Constraint_FailIndicator1 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=fail_indicator_rule_1)
        model.Constraint_FailIndicator2 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=fail_indicator_rule_2)

        def fail_activation_rule_1(model, l, t):
            """ fail_applies can be 1 only if both branch_status at timestep 't-1' is 1 """
            if t > 1:
                return model.fail_applies[l, t] <= model.branch_status[l, t - 1]
            else:
                return pyo.Constraint.Skip

        def fail_activation_rule_2(model, l, t):
            """ fail_applies can be 1 only if fail_indicator is 1 """
            return model.fail_applies[l, t] <= model.fail_indicator[l, t]

        def fail_activation_rule_3(model, l, t):
            """ If both conditions are met, fail_applies must be 1 """
            return model.fail_applies[l, t] >= model.branch_status[l, t - 1] + model.fail_indicator[l, t] - 1

        model.Constraint_FailActivation1 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=fail_activation_rule_1)
        model.Constraint_FailActivation2 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=fail_activation_rule_2)
        model.Constraint_FailActivation3 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=fail_activation_rule_3)

        def line_failure_rule(model, l, t):
            """ Forces a line to fail if fail_applies[l, t] = 1 """
            if t > 1:
                return model.branch_status[l, t] <= 1 - model.fail_applies[l, t]
            else:
                return model.branch_status[l, t] <= 1 - model.fail_indicator[l, t]  # Initial timestep case

        model.Constraint_LineFailure = pyo.Constraint(model.Set_bch, model.Set_ts, rule=line_failure_rule)

        def repair_activation_rule_1(model, l, t):
            """ repair_applies can be 1 only if the branch was failed at the last timestep """
            if t > 1:
                return model.repair_applies[l, t] <= 1 - model.branch_status[l, t - 1]
            else:
                return pyo.Constraint.Skip

        def repair_activation_rule_2(model, l, t):
            """ repair_applies can be 1 only if repair_complete[l, t] is 1 """
            return model.repair_applies[l, t] <= model.repair_complete[l, t]

        def repair_activation_rule_3(model, l, t):
            """ If both conditions are met, repair_applies must be 1 """
            return model.repair_applies[l, t] >= (1 - model.branch_status[l, t - 1]) + model.repair_complete[l, t] - 1

        model.Constraint_RepairActivation1 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=repair_activation_rule_1)
        model.Constraint_RepairActivation2 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=repair_activation_rule_2)
        model.Constraint_RepairActivation3 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=repair_activation_rule_3)

        def repair_rule(model, l, t):
            """ Forces a branch to be restored when repair_applies is 1 """
            if t > 1:
                return model.branch_status[l, t] >= model.repair_applies[l, t]
            else:
                return pyo.Constraint.Skip  # No repairs at t=1

        model.Constraint_Repair = pyo.Constraint(model.Set_bch, model.Set_ts, rule=repair_rule)

        # Below two constraints ensures branch repair takes place at the correct timestep
        def repair_complete_rule_1(model, l, t):
            """
            Ensures repair_complete[l, t] can be 1 only if the line status was failed at 'branch_ttr' timesteps ago.
            """
            if t > model.branch_ttr[l]:  # Ensure valid indexing
                return model.repair_complete[l, t] <= (1 - model.branch_status[l, t - model.branch_ttr[l]])
            else:
                return pyo.Constraint.Skip

        def repair_complete_rule_2(model, l, t):
            """
            Ensures repair_complete[l, t] can be 1 only if the line status is still failed at the current timestep 't'.
            This, combined with 'repair_complete_rule_1', ensures the line has been failed for 'branch_ttr' timesteps.
            """
            return model.repair_complete[l, t] <= (1 - model.branch_status[l, t])

        model.Constraint_RepairComplete1 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=repair_complete_rule_1)
        model.Constraint_RepairComplete2 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=repair_complete_rule_2)

        def failure_repair_exclusivity_rule(model, l, t):
            """
            A line cannot both fail and be repaired at the same timestep.
            """
            return model.fail_applies[l, t] + model.repair_applies[l, t] <= 1

        model.Constraint_ExclusiveFailRepair = pyo.Constraint(model.Set_bch, model.Set_ts,
                                                              rule=failure_repair_exclusivity_rule)



        # 5. Objective Function: Minimize Total Cost
        def objective_function(model):
            total_cost_investment = sum(model.cost_bch_hrdn[l] * model.bch_hrdn[l] for l in model.Set_bch)
            total_cost_load_shed = sum(
                model.cost_load_shed[b] * model.load_shed[b, t] for b in model.Set_bus for t in model.Set_ts)
            total_cost_repair = sum(model.cost_repair[l] * (1 - model.bch_hrdn[l]) for l in model.Set_bch)
            return total_cost_investment + total_cost_load_shed + total_cost_repair

        model.Objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

        return model


    def solve_investment_model(self, model, solver='glpk'):
        # Solve the model
        solver = SolverFactory(solver)
        results = solver.solve(model, tee=True)

        # Display Results
        for v in model.component_objects(pyo.Var, active=True):
            print(f"Variable {v.name}:")
            var_object = getattr(model, v.name)
            for index in var_object:
                print(f"  Index {index}: Value = {var_object[index].value}")

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
        gust_speeds = np.linspace(thrd_1, thrd_2, num_pieces)
        fail_probs = []

        # Compute failure probabilities at each breakpoint
        shape = sigma
        scale = np.exp(mu)

        for speed in gust_speeds:
            adjusted_speed = speed - shift_f
            if adjusted_speed < thrd_1:
                fail_probs.append(0.0)
            elif adjusted_speed > thrd_2:
                fail_probs.append(1.0)
            else:
                pof = lognorm.cdf(adjusted_speed, s=shape, scale=scale)
                fail_probs.append(pof)

        # Return piecewise linear data
        return {"gust_speeds": gust_speeds.tolist(), "fail_probs": fail_probs}


    def _get_cost_bch_hrdn(self):
        return self.data.cost_bch_hrdn

    def _get_cost_bch_rep(self):
        return self.data.cost_bch_rep

    def _get_cost_bus_ls(self):
        return self.data.cost_bus_ls

    def _get_budget_bch_hrdn(self):
        return self.data.budget_bch_hrdn
