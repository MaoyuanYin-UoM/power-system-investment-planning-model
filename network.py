# This script contains the network engine using linear power flow models


import numpy as np
import pandas as pd
import cmath, math
import scipy.sparse
import pyomo.environ as pyo
from pyomo.contrib.mpc.examples.cstr.model import initialize_model
from pyomo.opt import SolverFactory

from config import NetConfig
from windstorm import WindClass


class Object(object):
    pass


class NetworkClass:
    def __init__(self, obj=None):

        # Get default values from config
        if obj == None:
            obj = NetConfig()

        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # Import windclass
        ws = WindClass()

        # Set scaled demand profiles for all buses
        DP_file_path = "Input_Data/Demand_Profile/normalized_hourly_demand_profile_year.xlsx"
        df = pd.read_excel(DP_file_path, header=0)  # ignore any header row
        if ws.data.MC.lng_prd == 'year':
            normalized_profile = df.iloc[:, 0].tolist()  # Extract the whole column (1 year) and convert to list
        elif ws.data.MC.lng_prd == 'month':
            normalized_profile = df.iloc[0:720, 0].tolist()  # Extract the first month
        elif ws.data.MC.lng_prd == 'week':
            normalized_profile = df.iloc[0:168, 0].tolist()  # Extract the first week
        self.set_scaled_profile_for_buses(normalized_profile)

    def build_dc_opf_model(self):
        """
        The model that calculates DC optimal power flow based on the pyomo optimisation package
        Note: Only static power flow, i.e., set of timesteps is not considered
        """

        # Define a concrete model
        model = pyo.ConcreteModel()

        # 1. Sets (indices for buses, branches, generators):
        model.Set_bus = pyo.Set(initialize=self.data.net.bus)
        model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch) + 1))
        model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen) + 1))

        # 2. Parameters:
        model.Pd = pyo.Param(model.Set_bus, initialize={b + 1: self.data.net.Pd_max[b]
                                                        for b in range(len(self.data.net.Pd_max))})

        model.Pg_max = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                            enumerate(self.data.net.Pg_max)})

        model.Pg_min = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                            enumerate(self.data.net.Pg_min)})

        model.bch_Pmax = pyo.Param(model.Set_bch, initialize={i + 1: bc for i, bc in
                                                              enumerate(self.data.net.bch_Pmax)})

        # calculate susceptance B for each line (under the assumption of DC power flow)
        model.bch_B = pyo.Param(model.Set_bch, initialize={i + 1: 1 / X for i, X in
                                                           enumerate(self.data.net.bch_X)})

        # model.gen_cost_model = pyo.Param(model.Set_gen, initialize=self.data.net.gen_cost_model)

        model.gen_cost_coef = pyo.Param(model.Set_gen, range(len(self.data.net.gen_cost_coef[0])),
                                        initialize={
                                            (i + 1, j): self.data.net.gen_cost_coef[i][j]
                                            for i in range(len(self.data.net.gen_cost_coef))
                                            for j in range(len(self.data.net.gen_cost_coef[i]))})

        model.Pimp_cost = pyo.Param(initialize=self.data.net.Pimp_cost)
        model.Pexp_cost = pyo.Param(initialize=self.data.net.Pexp_cost)
        model.Pc_cost = pyo.Param(
            model.Set_bus,
            initialize={b: self.data.net.Pc_cost[b - 1] for b in self.data.net.bus},
            within=pyo.NonNegativeReals,
        )

        # 3. Variables:
        model.theta = pyo.Var(model.Set_bus, within=pyo.Reals)
        model.Pg = pyo.Var(model.Set_gen, within=pyo.Reals)  # Active generation
        model.Pf = pyo.Var(model.Set_bch, within=pyo.Reals)  # Active power flow in branch
        model.Pc = pyo.Var(model.Set_bus, within=pyo.NonNegativeReals)  # curtailed load (load shedding) at bus

        model.Pimp = pyo.Var(within=pyo.NonNegativeReals)  # cost of grid import/export
        model.Pexp = pyo.Var(within=pyo.NonNegativeReals)  # cost of grid import/export

        # 4. Constraints:
        # 1) Power balance at each bus

        # (old version, deprecated as it leads to error with islanded bus with zero load and gen)
        # def power_balance_rule(model, bus_idx):
        #     total_gen_at_bus = sum(model.Pg[i] for i in model.Set_gen if self.data.net.gen[i-1] == bus_idx)
        #     inflow = sum(model.Pf[i] for i in model.Set_bch if self.data.net.bch[i-1][1] == bus_idx)
        #     outflow = sum(model.Pf[i] for i in model.Set_bch if self.data.net.bch[i-1][0] == bus_idx)
        #     return total_gen_at_bus + inflow - outflow == model.Pd[bus_idx]
        #
        # model.Constraint_PowerBalance = pyo.Constraint(model.Set_bus, rule=power_balance_rule)

        def power_balance_rule(model, bus_idx):
            # “zero” expressed as a Pyomo expression:
            zero_expr = 0 * model.theta[bus_idx]

            # sum of generation at this bus (might be empty)
            gen_terms = [
                model.Pg[g]
                for g in model.Set_gen
                if self.data.net.gen[g - 1] == bus_idx
            ]
            Pg_sum = sum(gen_terms) if gen_terms else zero_expr

            # if this is the slack bus, add the grid import/export 'Pgrid'
            Pgrid = zero_expr
            if bus_idx == self.data.net.slack_bus:
                Pgrid = model.Pimp - model.Pexp

            # sum of power flowing into the bus (might be empty)
            inflow_terms = [
                model.Pf[l]
                for l in model.Set_bch
                if self.data.net.bch[l - 1][1] == bus_idx
            ]
            Pf_in_sum = sum(inflow_terms) if inflow_terms else zero_expr

            # sum of power flowing out of the bus (might be empty)
            outflow_terms = [
                model.Pf[l]
                for l in model.Set_bch
                if self.data.net.bch[l - 1][0] == bus_idx
            ]
            Pf_out_sum = sum(outflow_terms) if outflow_terms else zero_expr

            # now this is always a Pyomo expression, never a bare Python bool
            return Pg_sum + Pf_in_sum - Pf_out_sum + Pgrid == model.Pd[bus_idx] - model.Pc[bus_idx]

        model.Constraint_PowerBalance = pyo.Constraint(model.Set_bus, rule=power_balance_rule)

        # 2) Line flow constraints
        def line_flow_rule(model, line_idx):
            i, j = self.data.net.bch[line_idx - 1]
            return model.Pf[line_idx] == model.bch_B[line_idx] * (model.theta[i] - model.theta[j])

        model.Constraint_LineFlow = pyo.Constraint(model.Set_bch, rule=line_flow_rule)

        # 3) Line limit constraints
        def line_upper_limit_rule(model, line_idx):
            return model.Pf[line_idx] <= model.bch_Pmax[line_idx]

        def line_lower_limit_rule(model, line_idx):
            return model.Pf[line_idx] >= -model.bch_Pmax[line_idx]

        model.Constraint_LineUpperLimit = pyo.Constraint(model.Set_bch, rule=line_upper_limit_rule)
        model.Constraint_LineLowerLimit = pyo.Constraint(model.Set_bch, rule=line_lower_limit_rule)

        # 4) Generator limit constraints:
        def gen_lower_limit_rule(model, gen_idx):
            return model.Pg[gen_idx] >= model.Pg_min[gen_idx]

        def gen_upper_limit_rule(model, gen_idx):
            return model.Pg[gen_idx] <= model.Pg_max[gen_idx]

        model.Constraint_GenLowerLimit = pyo.Constraint(model.Set_gen, rule=gen_lower_limit_rule)
        model.Constraint_GenUpperLimit = pyo.Constraint(model.Set_gen, rule=gen_upper_limit_rule)

        # 5) Slack bus constraint (zero phase angle at slack bus):
        def slack_bus_rule(model):
            slack_bus = self.data.net.slack_bus
            return model.theta[slack_bus] == 0

        model.Constraint_SlackBus = pyo.Constraint(rule=slack_bus_rule)

        # 5. Objective function (minimise total generation cost)
        def objective_rule(model):
            total_gen_cost = sum(
                sum(model.gen_cost_coef[g, c] * (model.Pg[g] ** c)
                    for c in range(len(self.data.net.gen_cost_coef[0])))
                for g in model.Set_gen
            )

            grid_cost = model.Pimp_cost * model.Pimp + model.Pexp_cost * model.Pexp

            total_ls_cost = sum(model.Pc_cost[b] * model.Pc[b] for b in model.Set_bus)

            return total_gen_cost + grid_cost + total_ls_cost

        model.Objective_MinimiseTotalCost = pyo.Objective(rule=objective_rule, sense=1)

        return model

    def solve_dc_opf(self, model, solver='gurobi'):
        """
        Solve the DC OPF model
        Note: use 'solve_linearized_ac_opf' instead to solve the linearized ac model
        """
        solver = SolverFactory(solver)
        results = solver.solve(model)

        # Extract results and print some of them
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            # Display optimization results
            print(results)

            # Display variable values
            for V2 in model.component_objects(pyo.Var, active=True):
                print(f"Variable {V2.name}:")
                var_object = getattr(model, V2.name)
                for index in var_object:
                    print(f"  Index {index}: Value = {var_object[index].value}")

            # Display objective value
            for obj in model.component_objects(pyo.Objective, active=True):
                print(f"Objective {obj.name}: Value = {pyo.value(obj)}")

        else:
            print("Solver failed to find an optimal solution.")

        return results

    def build_linearized_ac_opf_model(self):
        """
        The model that calculates AC optimal power flow (LinDistFlow Model) based on the pyomo optimisation package
        Note: A set of timesteps - hourly resolution over a 1-year period - is considered
        """
        model = pyo.ConcreteModel()

        # 1) Sets
        model.Set_bus = pyo.Set(initialize=self.data.net.bus)
        model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch) + 1))
        model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen) + 1))
        model.Set_ts = pyo.Set(initialize=range(1, 1 + 1))

        # 2) Parameters
        # – maximum active/reactive demand
        model.Pd_max = pyo.Param(model.Set_bus,
                                 initialize={b + 1: self.data.net.Pd_max[b] for b in range(len(self.data.net.Pd_max))})
        model.Qd_max = pyo.Param(model.Set_bus,
                                 initialize={b + 1: self.data.net.Qd_max[b] for b in range(len(self.data.net.Qd_max))})

        # - active/reactive demand profiles
        model.Pd = pyo.Param(model.Set_bus, model.Set_ts,
                             initialize={
                                 (b, t): self.data.net.profile_Pd[b - 1][t - 1]
                                 for b in model.Set_bus
                                 for t in model.Set_ts
                             })
        model.Qd = pyo.Param(model.Set_bus, model.Set_ts,
                             initialize={
                                 (b, t): self.data.net.profile_Qd[b - 1][t - 1]
                                 for b in model.Set_bus
                                 for t in model.Set_ts
                             })

        # – branch parameters
        model.R = pyo.Param(model.Set_bch,
                            initialize={i + 1: self.data.net.bch_R[i] for i in range(len(self.data.net.bch_R))})
        model.X = pyo.Param(model.Set_bch,
                            initialize={i + 1: self.data.net.bch_X[i] for i in range(len(self.data.net.bch_X))})
        model.S_max = pyo.Param(model.Set_bch, initialize={i + 1: self.data.net.bch_Smax[i] for i in
                                                           range(len(self.data.net.bch_Smax))})
        # – voltage limits
        model.V_min = pyo.Param(model.Set_bus,
                                initialize={b + 1: self.data.net.V_min[b] for b in range(len(self.data.net.V_min))})
        model.V_max = pyo.Param(model.Set_bus,
                                initialize={b + 1: self.data.net.V_max[b] for b in range(len(self.data.net.V_max))})
        # – generator limits
        model.Pg_max = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in enumerate(self.data.net.Pg_max)})
        model.Pg_min = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in enumerate(self.data.net.Pg_min)})
        model.Qg_max = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in enumerate(self.data.net.Qg_max)})
        model.Qg_min = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in enumerate(self.data.net.Qg_min)})

        # - generation costs
        model.gen_cost_coef = pyo.Param(model.Set_gen, range(len(self.data.net.gen_cost_coef[0])),
                                        initialize={(i + 1, j): self.data.net.gen_cost_coef[i][j]
                                                    for i in range(len(self.data.net.gen_cost_coef))
                                                    for j in range(len(self.data.net.gen_cost_coef[0]))})

        # grid import/export cost
        model.Pimp_cost = pyo.Param(initialize=self.data.net.Pimp_cost)  # grid active power import cost
        model.Pexp_cost = pyo.Param(initialize=self.data.net.Pexp_cost)  # grid active power import cost
        model.Qimp_cost = pyo.Param(initialize=self.data.net.Qimp_cost)  # grid reactive power import cost
        model.Qexp_cost = pyo.Param(initialize=self.data.net.Qexp_cost)  # grid reactive power import cost

        # - cost for active/reactive load shedding
        model.Pc_cost = pyo.Param(model.Set_bus,
                                  initialize={b + 1: self.data.net.Pc_cost[b]
                                              for b in range(len(self.data.net.Pc_cost))})
        model.Qc_cost = pyo.Param(model.Set_bus,
                                  initialize={b + 1: self.data.net.Qc_cost[b]
                                              for b in range(len(self.data.net.Qc_cost))})

        # 3) Variables
        model.Pg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.NonNegativeReals)  # active generation
        model.Qg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.Reals)  # reactive generation
        model.Pimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)  # grid active power import
        model.Pexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)  # grid active power import
        model.Qimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)  # grid reactive power import
        model.Qexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)  # grid reactive power import
        model.Pf = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Reals)  # active power flow
        model.Qf = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Reals)  # reactive power flow
        model.Pc = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # curtailed active demand
        model.Qc = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # curtailed reactive demand
        model.V2 = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # squared voltage

        # model.I2 = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.NonNegativeReals)  # squared current

        # 4) Constraints
        # 4.1) Active power balance at each bus
        def P_balance_at_bus_rule(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            Pf_in = sum(model.Pf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][0] == b)
            slack = self.data.net.slack_bus
            Pgrid =  model.Pimp[t] - model.Pexp[t] if b == slack else 0

            return Pf_in - Pf_out + Pg + Pgrid == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_ActivePowerBalance = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                             rule=P_balance_at_bus_rule)

        # 4.2) Reactive‐power balance at each bus
        def Q_balance_at_bus_rule(model, b, t):
            Qg = sum(model.Qg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            Qf_in = sum(model.Qf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][1] == b)
            Qf_out = sum(model.Qf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][0] == b)
            slack = self.data.net.slack_bus
            Qgrid =  model.Qimp[t] - model.Qexp[t] if b == slack else 0

            return Qf_in - Qf_out + Qg + Qgrid == model.Qd[b, t] - model.Qc[b, t]

        model.Constraint_ReactivePowerBalance = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                               rule=Q_balance_at_bus_rule)

        # # 4.3) Voltage drop constraints
        # def volt_drop_rule(model, l, t):
        #     i, j = self.data.net.bch[l - 1]
        #     return (model.V2[i, t] - model.V2[j, t]
        #             == 2 * (model.R[l] * model.Pf[l, t] + model.X[l] * model.Qf[l, t])
        #             - (model.R[l] ** 2 + model.X[l] ** 2) * model.I2[l, t])
        #
        # model.Constraint_VoltageDrop = pyo.Constraint(model.Set_bch, model.Set_ts, rule=volt_drop_rule)
        #
        # # 4.4) Current‐flow relation
        # def current_flow_rule(model, l, t):
        #     i, j = self.data.net.bch[l - 1]
        #     # Pf[l,t]^2 + Qf[l,t]^2 == I2[l,t] * V2[i,t]
        #     return model.Pf[l, t] ** 2 + model.Qf[l, t] ** 2 == model.I2[l, t] * model.V2[i, t]
        #
        # model.Constraint_CurrentFlow = pyo.Constraint(
        #     model.Set_bch, model.Set_ts,
        #     rule=current_flow_rule
        # )

        # 4.3) Voltage drop constraints (quadratic loss term ignored)

        def volt_drop_lin_rule(model, l, t):
            # “l” is the branch index; self.data.net.bch[l-1] = (i, j)
            i, j = self.data.net.bch[l - 1]
            return model.V2[i, t] - model.V2[j, t] == 2 * (model.R[l] * model.Pf[l, t] + model.X[l] * model.Qf[l, t])

        model.Constraint_VoltageDrop = pyo.Constraint(model.Set_bch, model.Set_ts, rule=volt_drop_lin_rule)

        # 4.5) Apparent power flow limits of each branch (i.e., line thermal capacity):
        def S1_rule(model, l, t): return model.Pf[l, t] <= model.S_max[l]

        def S2_rule(model, l, t): return model.Pf[l, t] >= -1 * model.S_max[l]

        def S3_rule(model, l, t): return model.Qf[l, t] <= model.S_max[l]

        def S4_rule(model, l, t): return model.Qf[l, t] >= -1 * model.S_max[l]

        def S5_rule(model, l, t): return model.Pf[l, t] + model.Qf[l, t] <= 2 * model.S_max[l]

        def S6_rule(model, l, t): return model.Pf[l, t] + model.Qf[l, t] >= -2 * model.S_max[l]

        def S7_rule(model, l, t): return model.Pf[l, t] - model.Qf[l, t] <= 2 * model.S_max[l]

        def S8_rule(model, l, t): return model.Pf[l, t] - model.Qf[l, t] >= -2 * model.S_max[l]

        model.Constraint_PowerFlowLimit_1 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S1_rule)
        model.Constraint_PowerFlowLimit_2 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S2_rule)
        model.Constraint_PowerFlowLimit_3 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S3_rule)
        model.Constraint_PowerFlowLimit_4 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S4_rule)
        model.Constraint_PowerFlowLimit_5 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S5_rule)
        model.Constraint_PowerFlowLimit_6 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S6_rule)
        model.Constraint_PowerFlowLimit_7 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S7_rule)
        model.Constraint_PowerFlowLimit_8 = pyo.Constraint(model.Set_bch, model.Set_ts, rule=S8_rule)

        # 4.6) Voltage‐magnitude limits at each bus
        def v_min_rule(model, b, t): return model.V2[b, t] >= model.V_min[b]

        def v_max_rule(model, b, t): return model.V2[b, t] <= model.V_max[b]

        model.Constraint_Vmin = pyo.Constraint(model.Set_bus, model.Set_ts, rule=v_min_rule)
        model.Constraint_Vmax = pyo.Constraint(model.Set_bus, model.Set_ts, rule=v_max_rule)

        # 4.7) Generator limits
        def Pg_upper_limit_rule(model, g, t): return model.Pg[g, t] <= model.Pg_max[g]

        def Pg_lower_limit_rule(model, g, t): return model.Pg[g, t] >= model.Pg_min[g]

        def Qg_upper_limit_rule(model, g, t): return model.Qg[g, t] <= model.Qg_max[g]

        def Qg_lower_limit_rule(model, g, t): return model.Qg[g, t] >= model.Qg_min[g]

        model.Constraint_Pg_upper = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Pg_upper_limit_rule)
        model.Constraint_Pg_lower = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Pg_lower_limit_rule)
        model.Constraint_Qg_upper = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Qg_upper_limit_rule)
        model.Constraint_Qg_lower = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Qg_lower_limit_rule)

        # 5) Objective – Minimizing total generation cost
        def objective_rule(model):
            # 5.1) generator production costs (quadratic, linear …)
            gen_cost = sum(
                sum(model.gen_cost_coef[g, c] * model.Pg[g, t] ** c
                    for c in range(len(self.data.net.gen_cost_coef[0])))
                for g in model.Set_gen
                for t in model.Set_ts
            )

            # 5.2) active load-shedding penalties
            active_ls_cost = sum(
                model.Pc_cost[b] * model.Pc[b, t]
                for b in model.Set_bus
                for t in model.Set_ts
            )

            # 5.3) reactive load-shedding penalties
            reactive_ls_cost = sum(
                model.Qc_cost[b] * model.Qc[b, t]
                for b in model.Set_bus
                for t in model.Set_ts
            )

            # 5.4) cost to external import / export
            grid_cost = sum(
                model.Pimp_cost * model.Pimp[t] + model.Pexp_cost * model.Pexp[t] +
                model.Qimp_cost * model.Qimp[t] + model.Qexp_cost * model.Qexp[t]
                for t in model.Set_ts
            )

            return gen_cost + active_ls_cost + reactive_ls_cost + grid_cost

        model.Objective_MinimizingTotalCost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return model

    def solve_linearized_ac_opf(self, model, solver='gurobi', print_all_variables=True):
        """Solve the linearized AC OPF model"""
        solver = SolverFactory(solver)
        results = solver.solve(model, tee=True)

        # Extract results and print some of them
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            # Display optimization results
            print(results)

            # Display variable values
            if print_all_variables:
                for V2 in model.component_objects(pyo.Var, active=True):
                    print(f"Variable {V2.name}:")
                    var_object = getattr(model, V2.name)
                    for index in var_object:
                        print(f"  Index {index}: Value = {var_object[index].value}")

                # Display objective value
                for obj in model.component_objects(pyo.Objective, active=True):
                    print(f"Objective {obj.name}: Value = {pyo.value(obj)}")

        else:
            print("Solver failed to find an optimal solution.")

        return results

    def build_combined_dc_linearized_ac_opf_model(self):
        """
        This method builds the model compatible with a composite network (including both transmission and distribution
        level) and can perform both DC power flow (for transmission level network) and linearized AC power flow (for
        distribution level network)

        Note:
        """
        # ------------------------------------------------------------------
        # 1. Sets
        # ------------------------------------------------------------------
        Set_bus_tn = [b for b in self.data.net.bus if self.data.net.bus_level[b] == 'T']
        Set_bus_dn = [b for b in self.data.net.bus if self.data.net.bus_level[b] == 'D']
        Set_bch_tn = [i for i in range(1, len(self.data.net.bch) + 1) if self.data.net.branch_level[i] == 'T']
        Set_bch_dn = [i for i in range(1, len(self.data.net.bch) + 1) if self.data.net.branch_level[i] == 'D']
        Set_gen = list(range(1, len(self.data.net.gen) + 1))
        Set_ts = list(range(1, 1 + 1))

        model = pyo.ConcreteModel()

        model.Set_bus_tn = pyo.Set(initialize=Set_bus_tn)
        model.Set_bus_dn = pyo.Set(initialize=Set_bus_dn)
        model.Set_bch_tn = pyo.Set(initialize=Set_bch_tn)
        model.Set_bch_dn = pyo.Set(initialize=Set_bch_dn)
        model.Set_gen = pyo.Set(initialize=Set_gen)
        model.Set_ts = pyo.Set(initialize=Set_ts)

        # ------------------------------------------------------------------
        # 2.  Parameters
        # ------------------------------------------------------------------
        # 2.1  Demand profiles (for both Pd and Qd) for every (bus, t)
        Pd = {(b, t): self.data.net.demand_profile_active[b - 1][t - 1]
              for b in self.data.net.bus for t in Set_ts}
        model.Pd = pyo.Param(model.Set_bus_tn | model.Set_bus_dn, model.Set_ts,
                             initialize=Pd, mutable=False)

        Qd = {(b, t): self.data.net.profile_Qd[b - 1][t - 1]
              for b in self.data.net.bus for t in Set_ts}
        model.Qd = pyo.Param(model.Set_bus_dn, model.Set_ts, initialize=Qd, mutable=False)

        # 2.2  Generator limits & costs
        model.Pg_max = pyo.Param(model.Set_gen,
                                 initialize={i + 1: v for i, v in enumerate(self.data.net.Pg_max)})
        model.Pg_min = pyo.Param(model.Set_gen,
                                 initialize={i + 1: v for i, v in enumerate(self.data.net.Pg_min)})
        model.Qg_max = pyo.Param(model.Set_gen,
                                 initialize={i + 1: v for i, v in enumerate(self.data.net.Qg_max)})
        model.Qg_min = pyo.Param(model.Set_gen,
                                 initialize={i + 1: v for i, v in enumerate(self.data.net.Qg_min)})

        model.gen_cost_coef = pyo.Param(
            model.Set_gen, range(len(self.data.net.gen_cost_coef[0])),
            initialize={(g, c): self.data.net.gen_cost_coef[g - 1][c]
                        for g in model.Set_gen
                        for c in range(len(self.data.net.gen_cost_coef[0]))}
        )

        # 2.3  Branch data
        # ––– TN  (DC → susceptance in p.u.)
        model.B_tn = pyo.Param(model.Set_bch_tn,
                               initialize={l: 1.0 / self.data.net.bch_X[l - 1] for l in model.Set_bch_tn})

        model.Pmax_tn = pyo.Param(model.Set_bch_tn,
                                  initialize={l: self.data.net.bch_Pmax[l - 1] for l in model.Set_bch_tn})

        # ––– DN  (LinDistFlow → R/X + Smax)
        model.R_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: self.data.net.bch_R[l - 1] for l in model.Set_bch_dn})
        model.X_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: self.data.net.bch_X[l - 1] for l in model.Set_bch_dn})
        model.Smax_dn = pyo.Param(model.Set_bch_dn,
                                  initialize={l: self.data.net.bch_Smax[l - 1] for l in model.Set_bch_dn})

        # 2.4  Cost of load shedding / external import
        model.Pc_cost = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                  initialize={b: self.data.net.Pc_cost[b - 1] for b in self.data.net.bus})
        model.Qc_cost = pyo.Param(model.Set_bus_dn,
                                  initialize={b: self.data.net.Qc_cost[b - 1] for b in self.data.net.bus})

        model.Pext_cost = pyo.Param(initialize=self.data.net.Pext_cost)
        model.Qext_cost = pyo.Param(initialize=getattr(self.data.net, "Qext_cost", 0.0))

        # ------------------------------------------------------------------
        # 3.  Decision variables
        # ------------------------------------------------------------------
        # — generation
        model.Pg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.NonNegativeReals)
        model.Qg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.Reals)

        # — bus quantities
        model.theta = pyo.Var(model.Set_bus_tn | model.Set_bus_dn, model.Set_ts, within=pyo.Reals)
        model.V2_dn = pyo.Var(model.Set_bus_dn, model.Set_ts, within=pyo.NonNegativeReals)

        # — line flows
        model.Pf_tn = pyo.Var(model.Set_bch_tn, model.Set_ts, within=pyo.Reals)
        model.Pf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)
        model.Qf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)

        # — import / shedding
        model.Pext = pyo.Var(model.Set_ts, within=pyo.Reals)
        model.Qext = pyo.Var(model.Set_ts, within=pyo.Reals)
        model.Pc = pyo.Var(model.Set_bus_tn | model.Set_bus_dn, model.Set_ts,
                           within=pyo.NonNegativeReals)
        model.Qc = pyo.Var(model.Set_bus_dn, model.Set_ts, within=pyo.NonNegativeReals)

        # ------------------------------------------------------------------
        # 4.  Constraints
        # ------------------------------------------------------------------
        slack = self.data.net.slack_bus

        # 4.1 Slack-bus reference (angle)
        def _slack_rule(model, t):
            return model.theta[slack, t] == 0

        model.Constraint_SlackBus = pyo.Constraint(model.Set_ts, rule=_slack_rule)

        # 4.2 DC flow on TN branches
        def _flow_tn_rule(model, l, t):
            i, j = self.data.net.bch[l - 1]
            return model.Pf_tn[l, t] == model.B_tn[l] * (model.theta[i, t] - model.theta[j, t])

        model.Constraint_FlowDef_TN = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                     rule=_flow_tn_rule)

        # 4.3 TN thermal limits
        def _flow_tn_u(model, l, t):
            return model.Pf_tn[l, t] <= model.Pmax_tn[l]

        def _flow_tn_l(model, l, t):
            return model.Pf_tn[l, t] >= -model.Pmax_tn[l]

        model.Constraint_TN_LimitU = pyo.Constraint(model.Set_bch_tn, model.Set_ts, rule=_flow_tn_u)
        model.Constraint_TN_LimitL = pyo.Constraint(model.Set_bch_tn, model.Set_ts, rule=_flow_tn_l)

        # 4.4 LinDistFlow on DN branches  (quadratic losses ignored)
        def _volt_drop_dn(model, l, t):
            i, j = self.data.net.bch[l - 1]
            return model.V2_dn[i, t] - model.V2_dn[j, t] == \
                2 * (model.R_dn[l] * model.Pf_dn[l, t] + model.X_dn[l] * model.Qf_dn[l, t])

        model.Constraint_VDrop_DN = pyo.Constraint(model.Set_bch_dn, model.Set_ts,
                                                   rule=_volt_drop_dn)

        # 4.5 DN apparent-power limits (|P|,|Q| ≤ Smax)
        def _P_dn_u(model, l, t):
            return model.Pf_dn[l, t] <= model.Smax_dn[l]

        def _P_dn_l(model, l, t):
            return model.Pf_dn[l, t] >= -model.Smax_dn[l]

        def _Q_dn_u(model, l, t):
            return model.Qf_dn[l, t] <= model.Smax_dn[l]

        def _Q_dn_l(model, l, t):
            return model.Qf_dn[l, t] >= -model.Smax_dn[l]

        model.Constraint_DN_PU = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=_P_dn_u)
        model.Constraint_DN_PL = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=_P_dn_l)
        model.Constraint_DN_QU = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=_Q_dn_u)
        model.Constraint_DN_QL = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=_Q_dn_l)

        # 4.6  Power-balance at TN buses (active only)
        def _balance_tn(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            inflow = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if self.data.net.bch[l - 1][1] == b)
            outflow = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if self.data.net.bch[l - 1][0] == b)
            # branch flows from DN that end at this TN bus
            dn_in = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][1] == b)
            dn_out = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][0] == b)
            extra = model.Pext[t] if b == slack else 0
            return Pg + inflow + dn_in + extra - (outflow + dn_out) + model.Pc[b, t] == model.Pd[b, t]

        model.Constraint_Balance_TN = pyo.Constraint(model.Set_bus_tn, model.Set_ts,
                                                     rule=_balance_tn)

        # 4.7  Power-balance at DN buses (active + reactive)
        def _balance_dn_P(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            inflow = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][1] == b)
            outflow = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][0] == b)
            return Pg + inflow - outflow + model.Pc[b, t] == model.Pd[b, t]

        def _balance_dn_Q(model, b, t):
            Qg = sum(model.Qg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            inflow = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][1] == b)
            outflow = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][0] == b)
            extra = model.Qext[t] if b == slack else 0
            return Qg + inflow - outflow + extra + model.Qc[b, t] == model.Qd[b, t]

        model.Constraint_Balance_DN_P = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                       rule=_balance_dn_P)
        model.Constraint_Balance_DN_Q = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                       rule=_balance_dn_Q)

        # 4.8  Generator & voltage limits (same style as other build_* methods)
        def _Pg_max(model, g, t):
            return model.Pg[g, t] <= model.Pg_max[g]

        def _Pg_min(model, g, t):
            return model.Pg[g, t] >= model.Pg_min[g]

        def _Qg_max(model, g, t):
            return model.Qg[g, t] <= model.Qg_max[g]

        def _Qg_min(model, g, t):
            return model.Qg[g, t] >= model.Qg_min[g]

        model.Constraint_PgU = pyo.Constraint(model.Set_gen, model.Set_ts, rule=_Pg_max)
        model.Constraint_PgL = pyo.Constraint(model.Set_gen, model.Set_ts, rule=_Pg_min)
        model.Constraint_QgU = pyo.Constraint(model.Set_gen, model.Set_ts, rule=_Qg_max)
        model.Constraint_QgL = pyo.Constraint(model.Set_gen, model.Set_ts, rule=_Qg_min)

        # voltage magnitude limits (DN only, values are V²)
        if hasattr(self.data.net, "V_min") and self.data.net.V_min is not None:
            def _vmin(model, b, t): return model.V2_dn[b, t] >= self.data.net.V_min[self.data.net.bus.index(b)]

            def _vmax(model, b, t): return model.V2_dn[b, t] <= self.data.net.V_max[self.data.net.bus.index(b)]

            model.Constraint_Vmin = pyo.Constraint(model.Set_bus_dn, model.Set_ts, rule=_vmin)
            model.Constraint_Vmax = pyo.Constraint(model.Set_bus_dn, model.Set_ts, rule=_vmax)

        # ------------------------------------------------------------------
        # 5.  Objective – total cost (generation + import + load shed)
        # ------------------------------------------------------------------
        def _total_cost(model):
            gen_cost = sum(model.gen_cost_coef[g, 0] +
                           model.gen_cost_coef[g, 1] * model.Pg[g, t]
                           for g in model.Set_gen for t in model.Set_ts)
            import_cost = sum(model.Pext_cost * model.Pext[t] + model.Qext_cost * model.Qext[t]
                              for t in model.Set_ts)
            shed_cost = sum(model.Pc_cost[b] * model.Pc[b, t] for b in self.data.net.bus for t in model.Set_ts) + \
                        sum(model.Qc_cost[b] * model.Qc[b, t] for b in Set_bus_dn for t in model.Set_ts)
            return gen_cost + import_cost + shed_cost

        model.Objective = pyo.Objective(rule=_total_cost, sense=pyo.minimize)

        return model

    def read_normalized_profile(self, file_path):
        """Read normalized demand profile from given file path into a Python list"""
        # read into a pandas dataframe
        demand_profile_df = pd.read_excel(file_path)
        # convert it into a python list
        demand_profile = demand_profile_df.iloc[:, 0].tolist()

        return demand_profile

    def set_scaled_profile_for_buses(self, normalized_profile):
        """
        Set demand profiles (both Pd and Qd) scaled from a normalized profile:

        Note: If Pd_min and Qd_min are provided, do a linear stretch between min and max values;
              otherwise scale by max values only.
        """
        Pd_max = self.data.net.Pd_max
        Pd_min = self.data.net.Pd_min
        Qd_max = self.data.net.Qd_max
        Qd_min = self.data.net.Qd_min

        profile_Pd = []
        profile_Qd = []

        # Case 1: we have both min and max
        if Pd_min is not None and Qd_min is not None:
            for pd_max, pd_min, qd_max, qd_min in zip(Pd_max, Pd_min, Qd_max, Qd_min):
                # linear stretch: scaled = min + norm * (max - min)
                pd_profile = [pd_min + v * (pd_max - pd_min) for v in normalized_profile]
                qd_profile = [qd_min + v * (qd_max - qd_min) for v in normalized_profile]
                profile_Pd.append(pd_profile)
                profile_Qd.append(qd_profile)

        # Case 2: no min values provided → scale by max only
        else:
            for pd_max, qd_max in zip(Pd_max, Qd_max):
                pd_profile = [v * pd_max for v in normalized_profile]
                qd_profile = [v * qd_max for v in normalized_profile]
                profile_Pd.append(pd_profile)
                profile_Qd.append(qd_profile)

        self.data.net.profile_Pd = profile_Pd
        self.data.net.profile_Qd = profile_Qd

    def find_islanded_buses(self):
        """
        Return a sorted list of all buses that are not connected
        to any branch (i.e. never appear in self.data.net.bch).
        """
        # 1) Build a set of all bus IDs
        all_buses = set(self.data.net.bus)

        # 2) Gather every bus that appears in any branch
        connected = set()
        for (i, j) in self.data.net.bch:
            connected.add(i)
            connected.add(j)

        # 3) Islanded = buses in all_buses but not in connected
        islanded = sorted(all_buses - connected)
        return islanded

    # Gets and Sets:
    def _get_resistance(self):
        """Get bch_R"""
        return self.data.net.bch_R

    def _get_reactance(self):
        """Get bch_X"""
        return self.data.net.bch_X

    def _get_bch(self):
        """Get bch"""
        return self.data.net.bch

    def _get_bus_lon(self):
        """Get bus_lon"""
        return self.data.net.bus_lon

    def _get_bus_lat(self):
        """Get bus_lat"""
        return self.data.net.bus_lat

    def set_all_bus_coords_in_tuple(self):
        """Get all bus coordinates in the form of [(lon1, lat1), (lon2, lat2), ...]"""
        self.data.net.all_bus_coords_in_tuple = \
            [(self.data.net.bus_lon[i], self.data.net.bus_lat[i]) for i in range(len(self.data.net.bus_lon))]

    def set_gis_data(self):
        """Set gis_bgn and gis_end for functions such as "compare_circle" """
        bch = self._get_bch()
        self.set_all_bus_coords_in_tuple()
        gis_bgn = []
        gis_end = []
        for b in range(len(bch)):
            gis_bgn.append(self.data.net.all_bus_coords_in_tuple[bch[b][0] - 1])
            gis_end.append(self.data.net.all_bus_coords_in_tuple[bch[b][1] - 1])

        self.data.net.bch_gis_bgn = gis_bgn
        self.data.net.bch_gis_end = gis_end

    def _get_bch_gis_bgn(self):
        """Get bch_gis_bgn"""
        return self.data.net.bch_gis_bgn

    def _get_bch_gis_end(self):
        """Get bch_gis_end"""
        return self.data.net.bch_gis_end
