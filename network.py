# This script contains the network engine using linear power flow models


import numpy as np
import pandas as pd
import cmath, math
import scipy.sparse
import pyomo.environ as pyo
from pyomo.contrib.mpc.examples.cstr.model import initialize_model
from pyomo.opt import SolverFactory
import os, pathlib
from datetime import datetime

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

        model.Pc = pyo.Var(
            model.Set_bus,
            bounds=lambda m, b: (0, m.Pd[b])  # ensure load shedding value does not exceed the load value
        )  # curtailed load (load shedding) at bus

        model.Pimp = pyo.Var(within=pyo.NonNegativeReals)  # cost of grid import/export
        model.Pexp = pyo.Var(within=pyo.NonNegativeReals)  # cost of grid import/export

        # 4. Constraints:
        # 1) Power balance at each bus
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

        # 3) Line angle difference constraints
        ANGLE_MAX = math.radians(30)
        def angle_diff_upper_rule(model, l):
            i, j = self.data.net.bch[l - 1]
            return model.theta[i] - model.theta[j] <= ANGLE_MAX

        def angle_diff_lower_rule(model, l):
            i, j = self.data.net.bch[l - 1]
            return model.theta[i] - model.theta[j] >= -ANGLE_MAX

        model.Constraint_AngleDiffUpper = pyo.Constraint(model.Set_bch,
                                                         rule=angle_diff_upper_rule)
        model.Constraint_AngleDiffLower = pyo.Constraint(model.Set_bch,
                                                         rule=angle_diff_lower_rule)

        # 4) Line limit constraints
        def line_upper_limit_rule(model, line_idx):
            return model.Pf[line_idx] <= model.bch_Pmax[line_idx]

        def line_lower_limit_rule(model, line_idx):
            return model.Pf[line_idx] >= -model.bch_Pmax[line_idx]

        model.Constraint_LineUpperLimit = pyo.Constraint(model.Set_bch, rule=line_upper_limit_rule)
        model.Constraint_LineLowerLimit = pyo.Constraint(model.Set_bch, rule=line_lower_limit_rule)

        # 5) Generator limit constraints:
        def gen_lower_limit_rule(model, gen_idx):
            return model.Pg[gen_idx] >= model.Pg_min[gen_idx]

        def gen_upper_limit_rule(model, gen_idx):
            return model.Pg[gen_idx] <= model.Pg_max[gen_idx]

        model.Constraint_GenLowerLimit = pyo.Constraint(model.Set_gen, rule=gen_lower_limit_rule)
        model.Constraint_GenUpperLimit = pyo.Constraint(model.Set_gen, rule=gen_upper_limit_rule)

        # 6) Slack bus constraint (zero phase angle at slack bus):
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

    def solve_dc_opf(self, model, solver='gurobi', write_xlsx: bool = False, out_dir: str = "Optimization_Results/DC"):
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

            # Write results
            if write_xlsx:
                prefix = "dc_"
                net_name = getattr(self, "name", "network")
                book = pathlib.Path(out_dir) / f"results_{prefix}{net_name}.xlsx"
                self._write_results_to_excel(model, results, book)

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
        model.V2_min = pyo.Param(model.Set_bus,
                                 initialize={b + 1: self.data.net.V_min[b] ** 2 for b in
                                             range(len(self.data.net.V_min))})
        model.V2_max = pyo.Param(model.Set_bus,
                                 initialize={b + 1: self.data.net.V_max[b] ** 2 for b in
                                             range(len(self.data.net.V_max))})
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
        model.Pc = pyo.Var(
            model.Set_bus, model.Set_ts,
            bounds=lambda m, b, t: (0, m.Pd[b, t])  # ensure load shedding values do not exceed load values
        )  # curtailed active demand
        model.Qc = pyo.Var(
            model.Set_bus, model.Set_ts,
            bounds=lambda m, b, t: (0, m.Qd[b, t])
        )  # curtailed reactive demand
        model.V2 = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # squared voltage

        # model.I2 = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.NonNegativeReals)  # squared current

        # 4) Constraints
        # 4.1) Active power balance at each bus
        def P_balance_at_bus_rule(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            Pf_in = sum(model.Pf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][0] == b)
            slack = self.data.net.slack_bus
            Pgrid = model.Pimp[t] - model.Pexp[t] if b == slack else 0

            return Pf_in - Pf_out + Pg + Pgrid == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_ActivePowerBalance = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                             rule=P_balance_at_bus_rule)

        # 4.2) Reactive‐power balance at each bus
        def Q_balance_at_bus_rule(model, b, t):
            Qg = sum(model.Qg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            Qf_in = sum(model.Qf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][1] == b)
            Qf_out = sum(model.Qf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][0] == b)
            slack = self.data.net.slack_bus
            Qgrid = model.Qimp[t] - model.Qexp[t] if b == slack else 0

            return Qf_in - Qf_out + Qg + Qgrid == model.Qd[b, t] - model.Qc[b, t]

        model.Constraint_ReactivePowerBalance = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                               rule=Q_balance_at_bus_rule)

        # 4.3) Voltage drop constraints (quadratic loss term ignored)
        def voltage_drop_lin_rule(model, l, t):
            # “l” is the branch index; self.data.net.bch[l-1] = (i, j)
            i, j = self.data.net.bch[l - 1]
            return model.V2[i, t] - model.V2[j, t] == 2 * (model.R[l] * model.Pf[l, t] + model.X[l] * model.Qf[l, t])

        model.Constraint_VoltageDrop = pyo.Constraint(model.Set_bch, model.Set_ts, rule=voltage_drop_lin_rule)

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
        def voltage_squared_lower_limit_rule(model, b, t): return model.V2[b, t] >= model.V2_min[b]

        def voltage_squared_upper_limit_rule(model, b, t): return model.V2[b, t] <= model.V2_max[b]

        model.Constraint_V2min = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                rule=voltage_squared_lower_limit_rule)
        model.Constraint_V2max = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                rule=voltage_squared_upper_limit_rule)

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


    def solve_linearized_ac_opf(self, model, solver='gurobi', print_all_variables=True,
                                write_xlsx: bool = False,
                                out_dir: str = "Optimization_Results/Linearized_AC"):
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

            # Write results
            if write_xlsx:
                net_name = getattr(self, "name", "network")
                book = pathlib.Path(out_dir) / f"results_{net_name}.xlsx"
                self._write_results_to_excel(model, results, book)

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
        Set_bus_tn = [b for b in self.data.net.bus if self.data.net.bus_level[b] == 'T']  # tn: transmission network
        Set_bus_dn = [b for b in self.data.net.bus if self.data.net.bus_level[b] == 'D']  # dn: distribution network
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
        Pd = {(b, t): self.data.net.profile_Pd[b - 1][t - 1]
              for b in self.data.net.bus for t in Set_ts}
        model.Pd = pyo.Param(model.Set_bus_tn | model.Set_bus_dn, model.Set_ts,
                             initialize=Pd, mutable=False)

        Qd_all = {(b, t): self.data.net.profile_Qd[b - 1][t - 1]
                  for b in self.data.net.bus for t in Set_ts}
        model.Qd = pyo.Param(model.Set_bus_dn, model.Set_ts,
                             initialize={k: v for k, v in Qd_all.items() if k[0] in Set_bus_dn})  # Qd only for dn

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

        # 2.4 Voltage limits
        model.V2_min = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                 initialize={bus: self.data.net.V_min[i] ** 2
                                             for i, bus in enumerate(self.data.net.bus)})
        model.V2_max = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                 initialize={bus: self.data.net.V_max[i] ** 2
                                             for i, bus in enumerate(self.data.net.bus)})

        # 2.5  Cost of load shedding / external import
        model.Pc_cost = pyo.Param(model.Set_bus_tn | model.Set_bus_dn,
                                  initialize={b: self.data.net.Pc_cost[b - 1] for b in self.data.net.bus})
        model.Qc_cost = pyo.Param(model.Set_bus_dn,
                                  initialize={b: self.data.net.Qc_cost[self.data.net.bus.index(b)] for b in Set_bus_dn})

        model.Pimp_cost = pyo.Param(initialize=self.data.net.Pimp_cost)
        model.Pexp_cost = pyo.Param(initialize=self.data.net.Pexp_cost)
        model.Qimp_cost = pyo.Param(initialize=self.data.net.Qimp_cost)
        model.Qexp_cost = pyo.Param(initialize=self.data.net.Qexp_cost)

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

        # — grid import / export
        model.Pimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)
        model.Pexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)
        model.Qimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)
        model.Qexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)

        # — load shedding
        model.Pc = pyo.Var(model.Set_bus_tn | model.Set_bus_dn, model.Set_ts,
                           bounds=lambda m, b, t: (0, m.Pd[b, t]))  # ensure Pc does not exceed Pd
        model.Qc = pyo.Var(model.Set_bus_dn, model.Set_ts,
                           bounds=lambda m, b, t: (0, m.Qd[b, t]))

        # ------------------------------------------------------------------
        # 4.  Constraints
        # ------------------------------------------------------------------
        slack = self.data.net.slack_bus

        # 4.1 Slack-bus reference (angle)

        def slack_bus_rule(model, t):
            return model.theta[slack, t] == 0

        model.Constraint_SlackBus = pyo.Constraint(model.Set_ts, rule=slack_bus_rule)

        # 4.2 DC power flow on transmission branches
        def DC_line_flow_tn_rule(model, l, t):
            i, j = self.data.net.bch[l - 1]
            return model.Pf_tn[l, t] == model.B_tn[l] * (model.theta[i, t] - model.theta[j, t])

        model.Constraint_FlowDef_TN = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                     rule=DC_line_flow_tn_rule)

        # 4.3 Angle-difference at lines
        ANGLE_MAX = math.radians(30)  # maximum allowable angle difference: 30° (approx. 0.524 rad)
        def line_angle_difference_upper_rule(m, l, t):
            i, j = self.data.net.bch[l - 1]
            if self.data.net.branch_level[l] == 'T':
                return m.theta[i, t] - m.theta[j, t] <= ANGLE_MAX
            return pyo.Constraint.Skip

        def line_angle_difference_lower_rule(m, l, t):
            i, j = self.data.net.bch[l - 1]
            if self.data.net.branch_level[l] == 'T':
                return m.theta[i, t] - m.theta[j, t] >= -ANGLE_MAX
            return pyo.Constraint.Skip

        model.Constraint_AngleDiffUpperLimit = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                      rule=line_angle_difference_upper_rule)
        model.Constraint_AngleDiffLowerLimit = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                      rule=line_angle_difference_lower_rule)

        # 4.4 transmission branches thermal limits
        def line_flow_upper_limit_tn_rule(model, l, t):
            return model.Pf_tn[l, t] <= model.Pmax_tn[l]

        def line_flow_lower_limit_tn_rule(model, l, t):
            return model.Pf_tn[l, t] >= -model.Pmax_tn[l]

        model.Constraint_TN_LimitU = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                    rule=line_flow_upper_limit_tn_rule)
        model.Constraint_TN_LimitL = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                    rule=line_flow_lower_limit_tn_rule)

        # 4.5 Linearized AC power flow on distributed branches (quadratic losses ignored)
        def voltage_drop_dn_rule(model, l, t):
            i, j = self.data.net.bch[l - 1]
            return model.V2_dn[i, t] - model.V2_dn[j, t] == \
                2 * (model.R_dn[l] * model.Pf_dn[l, t] + model.X_dn[l] * model.Qf_dn[l, t])

        model.Constraint_VDrop_DN = pyo.Constraint(model.Set_bch_dn, model.Set_ts,
                                                   rule=voltage_drop_dn_rule)

        # 4.6 distribution branches apparent-power (thermal) limits (|P|,|Q| ≤ Smax)
        # 1) |P| ≤ Smax
        def S1_dn_rule(model, l, t):
            return model.Pf_dn[l, t] <= model.Smax_dn[l]

        def S2_dn_rule(model, l, t):
            return model.Pf_dn[l, t] >= -model.Smax_dn[l]

        # 2) |Q| ≤ Smax
        def S3_dn_rule(model, l, t):
            return model.Qf_dn[l, t] <= model.Smax_dn[l]

        def S4_dn_rule(model, l, t):
            return model.Qf_dn[l, t] >= -model.Smax_dn[l]

        # 3) ±45°‐cuts: |P ± Q| ≤ √2 Smax
        _diag = math.sqrt(2)

        def S5_dn_rule(model, l, t):
            return model.Pf_dn[l, t] + model.Qf_dn[l, t] <= _diag * model.Smax_dn[l]

        def S6_dn_rule(model, l, t):
            return model.Pf_dn[l, t] + model.Qf_dn[l, t] >= -_diag * model.Smax_dn[l]

        def S7_dn_rule(model, l, t):
            return model.Pf_dn[l, t] - model.Qf_dn[l, t] <= _diag * model.Smax_dn[l]

        def S8_dn_rule(model, l, t):
            return model.Pf_dn[l, t] - model.Qf_dn[l, t] >= -_diag * model.Smax_dn[l]

        model.Constraint_PowerFlowLimit_1 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S1_dn_rule)
        model.Constraint_PowerFlowLimit_2 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S2_dn_rule)
        model.Constraint_PowerFlowLimit_3 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S3_dn_rule)
        model.Constraint_PowerFlowLimit_4 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S4_dn_rule)
        model.Constraint_PowerFlowLimit_5 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S5_dn_rule)
        model.Constraint_PowerFlowLimit_6 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S6_dn_rule)
        model.Constraint_PowerFlowLimit_7 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S7_dn_rule)
        model.Constraint_PowerFlowLimit_8 = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=S8_dn_rule)

        # 4.7  Power-balance at TN buses (active only)
        def P_balance_at_bus_tn(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            inflow = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if self.data.net.bch[l - 1][1] == b)
            outflow = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if self.data.net.bch[l - 1][0] == b)
            # branch flows from DN that end at this TN bus
            dn_in = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][1] == b)
            dn_out = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][0] == b)
            slack = self.data.net.slack_bus
            Pgrid = model.Pimp[t] - model.Pexp[t] if b == slack else 0
            return Pg + inflow + dn_in + Pgrid - (outflow + dn_out) == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_ActivePowerBalance_TN = pyo.Constraint(model.Set_bus_tn, model.Set_ts,
                                                                rule=P_balance_at_bus_tn)

        # 4.8  Power-balance at DN buses (active + reactive)
        def P_balance_at_bus_dn(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            inflow = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][1] == b)
            outflow = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][0] == b)
            return Pg + inflow - outflow == model.Pd[b, t] - model.Pc[b, t]

        def Q_balance_at_bus_dn(model, b, t):
            Qg = sum(model.Qg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            inflow = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][1] == b)
            outflow = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn if self.data.net.bch[l - 1][0] == b)
            return Qg + inflow - outflow == model.Qd[b, t] - model.Qc[b, t]

        model.Constraint_ActivePowerBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                                rule=P_balance_at_bus_dn)
        model.Constraint_ReactivePowerBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                                  rule=Q_balance_at_bus_dn)

        # 4.9  Generator & voltage limits (same style as other build_* methods)
        def Pg_upper_limit_rule(model, g, t):
            return model.Pg[g, t] <= model.Pg_max[g]

        def Pg_lower_limit_rule(model, g, t):
            return model.Pg[g, t] >= model.Pg_min[g]

        def Qg_upper_limit_rule(model, g, t):
            return model.Qg[g, t] <= model.Qg_max[g]

        def Qg_lower_limit_rule(model, g, t):
            return model.Qg[g, t] >= model.Qg_min[g]

        model.Constraint_PgUpperLimit = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Pg_upper_limit_rule)
        model.Constraint_PgLowerLimit = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Pg_lower_limit_rule)
        model.Constraint_QgUpperLimit = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Qg_upper_limit_rule)
        model.Constraint_QgLowerLimit = pyo.Constraint(model.Set_gen, model.Set_ts, rule=Qg_lower_limit_rule)

        # 4.10 voltage magnitude limits (DN only, values are V²)
        def voltage_squared_lower_limit_rule(model, b, t):
            return model.V2_dn[b, t] >= model.V2_min[b]

        def voltage_squared_upper_limit_rule(model, b, t):
            return model.V2_dn[b, t] <= model.V2_max[b]

        model.Constraint_V2min = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                rule=voltage_squared_lower_limit_rule)
        model.Constraint_V2max = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                rule=voltage_squared_upper_limit_rule)

        # ------------------------------------------------------------------
        # 5.  Objective – minimizing total cost (generation + import + load shed)
        # ------------------------------------------------------------------
        def objective_rule(model):
            gen_cost = sum(model.gen_cost_coef[g, 0] +
                           model.gen_cost_coef[g, 1] * model.Pg[g, t]
                           for g in model.Set_gen for t in model.Set_ts)
            grid_cost = sum(model.Pimp_cost * model.Pimp[t] + model.Pexp_cost * model.Pexp[t]
                            for t in model.Set_ts)
            shed_cost = sum(model.Pc_cost[b] * model.Pc[b, t] for b in model.Set_bus_dn for t in model.Set_ts) + \
                        sum(model.Qc_cost[b] * model.Qc[b, t] for b in model.Set_bus_dn for t in model.Set_ts)
            return gen_cost + grid_cost + shed_cost

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return model


    def solve_combined_dc_linearized_ac_opf(self, model, solver='gurobi',
                                            write_xlsx: bool = False,
                                            out_dir: str = "Optimization_Results/Combined_DC_and_Linearized_AC"):
        """
        Solve the combined DC (transmission) + linearized AC (distribution) OPF model.
        """
        # 1) Create solver and solve with output streamed to console
        opt = SolverFactory(solver)
        results = opt.solve(model, tee=True)

        # 2) Check solver status and print results
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            print(results)

            # Print all decision‐variables
            for var in model.component_objects(pyo.Var, active=True):
                print(f"\nVariable {var.name}:")
                var_object = getattr(model, var.name)
                for idx in var_object:
                    print(f"  Index {idx}: Value = {var_object[idx].value}")

            # Print objective value
            for obj in model.component_objects(pyo.Objective, active=True):
                print(f"\nObjective {obj.name}: Value = {pyo.value(obj)}")

            # Write results
            if write_xlsx:
                prefix = "combined_dc_and_linearized_ac_"
                net_name = getattr(self, "name", "network")
                book = pathlib.Path(out_dir) / f"results_{prefix}{net_name}.xlsx"
                self._write_results_to_excel(model, results, book)
        else:
            print("Solver failed to find an optimal solution.")
        return results


    def _write_results_to_excel(self, model, results, book_path: str):
        """
        Write every active Var, Param (and Objective value) of *model* to a
        multi-sheet .xlsx workbook.

        Parameters
        ----------
        model : ConcreteModel   (must be solved already)
        book_path : str | Path  full path incl. filename.xlsx
        """
        book_path = pathlib.Path(book_path)
        book_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(book_path, engine="xlsxwriter") as xl:

            # --- variables ---------------------------------------------------
            for var in model.component_objects(pyo.Var, active=True):
                # build a dict { index_tuple: numeric or None }
                data = {}
                for idx in var:
                    val = var[idx].value
                    key = idx if idx is not None else ("",)
                    # if no value, write None (will appear blank in Excel)
                    data[key] = float(val) if val is not None else None

                df = (
                    pd.Series(data)
                    .rename("value")
                    .to_frame()
                    .reset_index()
                    .rename(columns={"level_0": "index"})
                )
                df.to_excel(xl, sheet_name=f"Var_{var.name[:28]}", index=False)

            # --- parameters --------------------------------------------------
            for par in model.component_objects(pyo.Param, active=True):
                data = {}
                for idx in par:
                    val = par[idx]
                    key = idx if idx is not None else ("",)
                    data[key] = float(val) if val is not None else None

                df = (
                    pd.Series(data)
                    .rename("value")
                    .to_frame()
                    .reset_index()
                    .rename(columns={"level_0": "index"})
                )
                df.to_excel(xl, sheet_name=f"Par_{par.name[:28]}", index=False)

            # --- objective value --------------------------------------------
            for obj in model.component_objects(pyo.Objective, active=True):
                pd.DataFrame({"Objective": [obj.name],
                              "Value": [pyo.value(obj)]}
                             ).to_excel(xl, sheet_name="Objective", index=False)

            # --- meta-information -------------------------------------------
            status = results.solver.status
            termc = results.solver.termination_condition
            gap = getattr(results.solver, "gap", None)
            meta = pd.DataFrame({
                "solved_at": [datetime.now().isoformat(timespec="seconds")],
                "pyomo_status": [str(status)],
                "termination_cond": [str(termc)],
                "pyomo_gap": [gap]
            })
            meta.to_excel(xl, sheet_name="meta", index=False)


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
