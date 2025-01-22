# This script contains the network engine using linear power flow models


import numpy as np
import cmath, math
import scipy.sparse
import pyomo.environ as pyo
from pyomo.contrib.mpc.examples.cstr.model import initialize_model
from pyomo.opt import SolverFactory
from traitlets import Float

from config import NetConfig

class Object(object):
    pass


class NetworkClass:
    def __init__(self, obj=None):

        # Get default values from config
        if obj == None:
            obj = NetConfig()

        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))


    def build_dc_opf_model(self):
        """The model that calculates DC optimal power flow based on the pyomo optimisation package"""
        # Define a concrete model
        model = pyo.ConcreteModel()

        # 1. Sets (indices for buses, branches, generators):
        model.Set_bus = pyo.Set(initialize=self.data.net.bus)
        model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch)+1))
        model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen)+1))

        # 2. Parameters:
        model.demand = pyo.Param(model.Set_bus, initialize={i+1: d for i, d in
                                                                    enumerate(self.data.net.demand_active)})

        # model.gen_cost_model = pyo.Param(model.Set_gen, initialize=self.data.net.gen_cost_model)

        model.gen_cost_coef = pyo.Param(model.Set_gen, range(len(self.data.net.gen_cost_coef[0])),
                                        initialize={
                                            (i + 1, j): self.data.net.gen_cost_coef[i][j]
                                            for i in range(len(self.data.net.gen_cost_coef))
                                            for j in range(len(self.data.net.gen_cost_coef[i]))})


        model.gen_active_max = pyo.Param(model.Set_gen, initialize={i+1: g for i, g in
                                                                            enumerate(self.data.net.gen_active_max)})

        model.gen_active_min = pyo.Param(model.Set_gen, initialize={i+1: g for i, g in
                                                                            enumerate(self.data.net.gen_active_min)})


        model.bch_cap = pyo.Param(model.Set_bch, initialize={i+1: bc for i, bc in
                                                                     enumerate(self.data.net.bch_cap)})

        # calculate susceptance B for each line (under the assumption of DC power flow)
        model.bch_B = pyo.Param(model.Set_bch, initialize={i+1: 1/X for i, X in
                                                                   enumerate(self.data.net.bch_X)})
        
        # 3. Variables:
        model.theta = pyo.Var(model.Set_bus, within=pyo.Reals)
        model.P_gen = pyo.Var(model.Set_gen, within=pyo.Reals)
        model.P_flow = pyo.Var(model.Set_bch, within=pyo.Reals)

        
        # 4. Constraints:
        # 1) Power balance at each bus
        def power_balance_rule(model, bus_idx):
            total_gen_at_bus = sum(model.P_gen[i] for i in model.Set_gen if self.data.net.gen[i-1] == bus_idx)
            inflow = sum(model.P_flow[i] for i in model.Set_bch if self.data.net.bch[i-1][1] == bus_idx)
            outflow = sum(model.P_flow[i] for i in model.Set_bch if self.data.net.bch[i-1][0] == bus_idx)
            return total_gen_at_bus + inflow - outflow == model.demand[bus_idx]

        model.Constraint_PowerBalance = pyo.Constraint(model.Set_bus, rule=power_balance_rule)

        # 2) Line flow constraints
        def line_flow_rule(model, line_idx):
            i, j = self.data.net.bch[line_idx-1]
            return model.P_flow[line_idx] == model.bch_B[line_idx] * (model.theta[i] - model.theta[j])

        model.Constraint_LineFlow = pyo.Constraint(model.Set_bch, rule=line_flow_rule)

        # 3) Line limit constraints
        def line_upper_limit_rule(model, line_idx):
            return model.P_flow[line_idx] <= model.bch_cap[line_idx]

        def line_lower_limit_rule(model, line_idx):
            return model.P_flow[line_idx] >= -model.bch_cap[line_idx]

        model.Constraint_LineUpperLimit = pyo.Constraint(model.Set_bch, rule=line_upper_limit_rule)
        model.Constraint_LineLowerLimit = pyo.Constraint(model.Set_bch, rule=line_lower_limit_rule)

        # 4) Generator limit constraints:
        def gen_lower_limit_rule(model, gen_idx):
            return model.P_gen[gen_idx] >= model.gen_active_min[gen_idx]

        def gen_upper_limit_rule(model, gen_idx):
            return model.P_gen[gen_idx] <= model.gen_active_max[gen_idx]

        model.Constraint_GenLowerLimit = pyo.Constraint(model.Set_gen, rule=gen_lower_limit_rule)
        model.Constraint_GenUpperLimit = pyo.Constraint(model.Set_gen, rule=gen_upper_limit_rule)

        # 5) Slack bus constraint (zero phase angle at slack bus):
        def slack_bus_rule(model):
            slack_bus = self.data.net.slack_bus
            return model.theta[slack_bus] == 0

        model.Constraint_SlackBus = pyo.Constraint(rule=slack_bus_rule)


        # 5. Objective function (minimise total generation cost)
        def Objective_rule(model):
            total_cost = 0
            for g in model.Set_gen:
                generator_cost = 0
                for c in range(len(self.data.net.gen_cost_coef[0])):
                    term = model.gen_cost_coef[g, c] * (model.P_gen[g] ** c)
                    print(f"Gen {g}, Coeff {c}, Term: {term}")
                    generator_cost += term
                total_cost += generator_cost
            return total_cost

        model.Objective_MinimiseTotalGenCost = pyo.Objective(rule=Objective_rule, sense=1)

        return model


    def solve_dc_opf(self, solver='glpk'):
        """Solve the DC OPF model"""
        model = self.build_dc_opf_model()
        solver = SolverFactory(solver)
        results = solver.solve(model)

        # Extract results and print some of them
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            # Display optimization results
            print(results)

            # Display variable values
            for v in model.component_objects(pyo.Var, active=True):
                print(f"Variable {v.name}:")
                var_object = getattr(model, v.name)
                for index in var_object:
                    print(f"  Index {index}: Value = {var_object[index].value}")

            # Display objective value
            for obj in model.component_objects(pyo.Objective, active=True):
                print(f"Objective {obj.name}: Value = {pyo.value(obj)}")

        else:
            print("Solver failed to find an optimal solution.")

        return results



    # Gets:
    def _get_resistance(self):
        return self.data.net.bch_R

    def _get_reactance(self):
        return self.data.net.bch_X