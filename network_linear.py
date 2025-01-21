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


    def _build_dc_opf_model(self):
        """The model that calculates DC optimal power flow based on the pyomo optimisation package"""
        # Define a concrete model
        model = pyo.ConcreteModel()

        # 1. Sets (indices for buses, branches, generators):
        model.Set_bus = pyo.Set(initialize=self.data.net.bus)
        model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch)+1))
        model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen)+1))

        # 2. Parameters:
        model.demand = pyo.Param(model.Set_Bus, initialize={i+1: d for i, d in
                                                                    enumerate(self.data.net.demand_active)})
        model.gen_cost_model = pyo.Param(model.Set_gen, initialize={i+1: gcm for i, gcm in
                                                                    enumerate(self.data.net.gen_cost_model)})

        model.gen_cost_coef = pyo.Param(model.Set_Gen, initialize={
                                                                    (i+1, j+1): self.data.net.gen_cost_model[i][j]
                                                                    for i in range(len(self.data.net.gen_cost_model))
                                                                    for j in range(len(self.data.net.gen_cost_model[i]))
                                                                    })

        model.gen_active_max = pyo.Param(model.Set_Gen, initialize={i+1: g for i, g in
                                                                            enumerate(self.data.net.gen_active_max)})

        model.gen_active_min = pyo.Param(model.Set_Gen, initialize={i+1: g for i, g in
                                                                            enumerate(self.data.net.gen_active_min)})


        model.bch_cap = pyo.Param(model.Set_Bch, initialize={i+1: bc for i, bc in
                                                                     enumerate(self.data.net.bch_cap)})

        # calculate susceptance B for each line (under the assumption of DC power flow)
        model.bch_B = pyo.Param(model.Set_Bch, initialize={i+1: 1/X for i, X in
                                                                   enumerate(self.data.net.bch_X)})
        
        # 3. Variables:
        model.theta = pyo.Var(model.Set_bus, within=pyo.NonNegativeReals)
        model.P_gen = pyo.Var(model.Set_bus, within=pyo.Reals)
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
            return model.P_flow[line_idx] == model.B[line_idx] * (model.theta[i] - model.theta[j])

        model.Constraint_LineFlow = pyo.Constraint(model.Set_bch, rule=line_flow_rule)

        # 3) Line limit constraints
        def line_limit_rule(model, line_idx):
            return abs(model.P_flow[line_idx]) <= model.bch_cap[line_idx]

        model.Constraint_LineLimit = pyo.Constraint(model.Set_bch, rule=line_limit_rule)

        # 4) Generator limit constraints:
        def gen_limit_rule(model, gen_idx):
            return model.gen_active_min[gen_idx] <= model.P_gen[gen_idx] <= model.gen_active_max[gen_idx]

        model.Constraint_GenCapacity = pyo.Constraint(model.Set_gen, rule=gen_limit_rule)

        # 5) Slack bus constraint (zero phase angle at slack bus):
        def slack_bus_rule(model):
            slack_bus = self.data.net.slack_bus
            return model.theta[slack_bus] == 0

        model.Constraint_SlackBus = pyo.Constraint(rule=slack_bus_rule)


        # 5. Objective function (minimise total generation cost)
        def Objective_rule(model):
            total_gen_cost = 0
            gen_idx = 0
            for i in self.data.net.gen_cost_coef:
                bus_idx = self.data.net.gen[gen_idx]
                gen_cost = sum(self.data.net.gen_cost_coef[i-1][j] * (model.P_gen[bus_idx]**j)
                           for j in range(self.data.net.gen_cost_coef[0]))  # sum: coef[j] * x^j over all coefficients j
                total_gen_cost += gen_cost
                gen_idx += 1
            return total_gen_cost

        model.Objective_MinimiseTotalGenCost = pyo.Objective(rule=Objective_rule, sense=1)

        return model


    def _solve_dc_opf(self, solver='glpk'):
        """Solve the DC OPF model"""
        model = self._build_dc_opf_model()
        solver = SolverFactory(solver)
        results = solver.solve(model)

        # Extract results and print some of them
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            print("Optimal solution found!")
            print("Generation values:")
            print({g: model.P_gen[g].value for g in model.Set_gen})

            minimised_total_cost = model.Objective_MinimiseTotalGenCost()
            print("Total generation cost:")
            print(minimised_total_cost)

        else:
            print("Solver failed to find an optimal solution.")

        return results






    # Gets:
    def _get_resistance(self):
        return self.data.net.bch_R

    def _get_reactance(self):
        return self.data.net.bch_X