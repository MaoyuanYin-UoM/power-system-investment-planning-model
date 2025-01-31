import pyomo.environ as pyo
from pyomo.opt import SolverFactory
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
        model.Set_bch = pyo.Set(initialize=range(1, len(net.data.net.bch) + 1))  # Branches
        model.Set_gen = pyo.Set(initialize=range(1, len(net.data.net.gen) + 1))  # Generators
        model.Set_bus = pyo.Set(initialize=net.data.net.bus)  # Buses
        model.Set_ts = pyo.Set(initialize=range(ws._get_num_hrs_prd()))  # Timesteps in a period


        # 2. Parameters
        # 1) Initialize network parameters:
        model.demand = pyo.Param(model.Set_bus, model.Set_ts, initialize={})

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

        # 2) Initilize investment parameters (budget and costs):
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

        # 3) Line Flow Constraint
        def line_flow_rule(model, l, t):
            i, j = net.data.net.bch[l - 1]
            return model.P_flow[l, t] == net.data.net.bch_X[l - 1] ** -1 * (model.theta[i, t] - model.theta[j, t])

        model.Constraint_LineFlow = pyo.Constraint(model.Set_bch, model.Set_ts, rule=line_flow_rule)

        # 4) Line limit constraints
        def line_upper_limit_rule(model, line_idx):
            return model.P_flow[line_idx] <= model.bch_cap[line_idx]

        def line_lower_limit_rule(model, line_idx):
            return model.P_flow[line_idx] >= -model.bch_cap[line_idx]

        model.Constraint_LineUpperLimit = pyo.Constraint(model.Set_bch, rule=line_upper_limit_rule)
        model.Constraint_LineLowerLimit = pyo.Constraint(model.Set_bch, rule=line_lower_limit_rule)

        # 5) Generator limit constraints:
        def gen_lower_limit_rule(model, gen_idx):
            return model.P_gen[gen_idx] >= model.gen_active_min[gen_idx]

        def gen_upper_limit_rule(model, gen_idx):
            return model.P_gen[gen_idx] <= model.gen_active_max[gen_idx]

        model.Constraint_GenLowerLimit = pyo.Constraint(model.Set_gen, rule=gen_lower_limit_rule)
        model.Constraint_GenUpperLimit = pyo.Constraint(model.Set_gen, rule=gen_upper_limit_rule)

        # 6) Slack Bus Constraint
        def slack_bus_rule(model, t):
            return model.theta[net.data.net.slack_bus, t] == 0

        model.Constraint_SlackBus = pyo.Constraint(model.Set_ts, rule=slack_bus_rule)


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


    def _get_cost_bch_hrdn(self):
        return self.data.cost_bch_hrdn

    def _get_cost_bch_rep(self):
        return self.data.cost_bch_rep

    def _get_cost_bus_ls(self):
        return self.data.cost_bus_ls

    def _get_budget_bch_hrdn(self):
        return self.data.budget_bch_hrdn
