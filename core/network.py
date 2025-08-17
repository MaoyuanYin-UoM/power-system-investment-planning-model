# This script contains the network engine using linear power flow models


import numpy as np
import pandas as pd
import cmath, math
import scipy.sparse
from scipy.stats import lognorm
import pyomo.environ as pyo
from pyomo.contrib.mpc.examples.cstr.model import initialize_model
from pyomo.opt import SolverFactory
import os, pathlib
from datetime import datetime
from pathlib import Path

from core.config import NetConfig
from core.windstorm import WindClass


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
        # - Get the correct absolute directory
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        DP_file_path = project_root / "Input_Data" / "Demand_Profile" / "normalized_hourly_demand_profile_year.xlsx"
        # - Read and set the demand profile
        df = pd.read_excel(DP_file_path, header=0)  # ignore any header row
        if ws.data.MC.lng_prd == 'year':
            normalized_profile = df.iloc[:, 0].tolist()  # Extract the whole column (1 year) and convert to list
        elif ws.data.MC.lng_prd == 'month':
            normalized_profile = df.iloc[0:720, 0].tolist()  # Extract the first month
        elif ws.data.MC.lng_prd == 'week':
            normalized_profile = df.iloc[0:168, 0].tolist()  # Extract the first week
        
        self.set_scaled_demand_profile_for_buses(normalized_profile)

        # Set renewable generation profiles if applicable
        # This will check if generator types are defined and load appropriate profiles
        if hasattr(self.data.net, 'gen_type'):
            self.set_renewable_generation_profiles()

    def build_dc_opf_model(self):
        """
        The model that calculates DC optimal power flow based on the pyomo optimisation package
        Note: Only static power flow, i.e., set of timesteps is not considered
        """
        # todo: outdated version, to be updated
        # Define a concrete model
        model = pyo.ConcreteModel()

        # 1. Sets (indices for buses, branches, generators):
        model.Set_bus = pyo.Set(initialize=self.data.net.bus)
        model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch) + 1))
        model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen) + 1))

        # 2. Parameters:
        model.base_MVA = pyo.Param(initialize=self.data.net.base_MVA)

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
            bounds=lambda model, b: (0, model.Pd[b])  # ensure load shedding value does not exceed the load value
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
            return model.Pf[line_idx] == model.base_MVA * model.bch_B[line_idx] * (model.theta[i] - model.theta[j])

        model.Constraint_LineFlow = pyo.Constraint(model.Set_bch, rule=line_flow_rule)

        # 3) Line angle difference constraints
        ang_diff_max = (self.data.net.theta_limits[1] -
                        self.data.net.theta_limits[0])  # maximum allowable angle difference (rad)
        def angle_diff_upper_rule(model, l):
            i, j = self.data.net.bch[l - 1]
            return model.theta[i] - model.theta[j] <= ang_diff_max

        def angle_diff_lower_rule(model, l):
            i, j = self.data.net.bch[l - 1]
            return model.theta[i] - model.theta[j] >= -ang_diff_max

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
        solver_obj = SolverFactory(solver)

        # Add solver-specific options
        if solver.lower() == 'cbc':
            solver_obj.options['threads'] = 0  # Use all available threads
            solver_obj.options['presolve'] = 'on'
            solver_obj.options['cuts'] = 'on'
        elif solver.lower() == 'glpk':
            # GLPK options for LP problems
            solver_obj.options['tmlim'] = 3600  # Time limit in seconds (1 hour default)
            solver_obj.options['msg_lev'] = 'GLP_MSG_ON'  # Enable output messages
        elif solver.lower() == 'gurobi':
            # Gurobi typically doesn't need special options for LP
            pass

        results = solver_obj.solve(model)

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
        # todo: outdated version, to be updated
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
            bounds=lambda model, b, t: (0, model.Pd[b, t])  # ensure load shedding values do not exceed load values
        )  # curtailed active demand
        model.Qc = pyo.Var(
            model.Set_bus, model.Set_ts,
            bounds=lambda model, b, t: (0, model.Qd[b, t])
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
        solver_obj = SolverFactory(solver)

        # Add solver-specific options
        if solver.lower() == 'cbc':
            solver_obj.options['threads'] = 0
            solver_obj.options['presolve'] = 'on'
            solver_obj.options['cuts'] = 'on'
        elif solver.lower() == 'glpk':
            solver_obj.options['tmlim'] = 3600
            solver_obj.options['msg_lev'] = 'GLP_MSG_ON'

        results = solver_obj.solve(model, tee=True)

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

    def build_combined_dc_linearized_ac_opf_model(self, timesteps=None):
        """
        Build a combined DC (transmission) + linearized AC (distribution) OPF model.
        Updated to match current network factory structure.

        Args:
            timesteps: List of timestep indices. If None, uses single timestep [0]

        Returns:
            Pyomo ConcreteModel
        """

        # ------------------------------------------------------------------
        # 0. Setup
        # ------------------------------------------------------------------
        model = pyo.ConcreteModel()

        # Handle timesteps
        if timesteps is None:
            timesteps = [0]  # Single timestep for testing

        # ------------------------------------------------------------------
        # 1. Index Sets
        # ------------------------------------------------------------------
        # Use dictionaries directly as in build_investment_model
        Set_ts = timesteps
        Set_bus = list(range(1, len(self.data.net.bus) + 1))
        Set_bus_tn = [b for b in self.data.net.bus if self.data.net.bus_level[b] == 'T']
        Set_bus_dn = [b for b in self.data.net.bus if self.data.net.bus_level[b] == 'D']
        Set_bch = list(range(1, len(self.data.net.bch) + 1))
        Set_bch_tn = [l for l in range(1, len(self.data.net.bch) + 1)
                      if self.data.net.branch_level[l] in ['T', 'T-D']]
        Set_bch_dn = [l for l in range(1, len(self.data.net.bch) + 1)
                      if self.data.net.branch_level[l] == 'D']

        # Generators - only existing
        Set_gen_exst = list(range(1, len(self.data.net.gen) + 1))

        # ESS - only existing (if any)
        Set_ess_exst = list(range(1, len(self.data.net.ess) + 1)) if hasattr(self.data.net, 'ess') else []

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
        model.Set_ess_exst = pyo.Set(initialize=Set_ess_exst)

        # Get slack bus
        slack_bus = self.data.net.slack_bus

        # ------------------------------------------------------------------
        # 3. Parameters
        # ------------------------------------------------------------------
        # Network parameters
        model.base_MVA = pyo.Param(initialize=self.data.net.base_MVA)

        # Generator parameters (existing only)
        coef_len = len(self.data.net.gen_cost_coef[0])
        model.gen_cost_coef = pyo.Param(
            model.Set_gen_exst, range(coef_len),
            initialize={(g, c): self.data.net.gen_cost_coef[g - 1][c]
                        for g in model.Set_gen_exst for c in range(coef_len)}
        )

        model.Pg_max_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: self.data.net.Pg_max_exst[g - 1] for g in model.Set_gen_exst})
        model.Pg_min_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: self.data.net.Pg_min_exst[g - 1] for g in model.Set_gen_exst})
        model.Qg_max_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: self.data.net.Qg_max_exst[g - 1] for g in model.Set_gen_exst})
        model.Qg_min_exst = pyo.Param(model.Set_gen_exst,
                                      initialize={g: self.data.net.Qg_min_exst[g - 1] for g in model.Set_gen_exst})

        # Branch parameters - TN (DC power flow)
        model.B_tn = pyo.Param(model.Set_bch_tn,
                               initialize={l: 1.0 / self.data.net.bch_X[l - 1]
                                           for l in model.Set_bch_tn})
        model.Pmax_tn = pyo.Param(model.Set_bch_tn,
                                  initialize={l: self.data.net.bch_Pmax[l - 1]
                                              for l in model.Set_bch_tn})

        # Branch parameters - DN (Linearized AC)
        model.R_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: self.data.net.bch_R[l - 1] for l in model.Set_bch_dn})
        model.X_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: self.data.net.bch_X[l - 1] for l in model.Set_bch_dn})
        model.Smax_dn = pyo.Param(model.Set_bch_dn,
                                  initialize={l: self.data.net.bch_Smax[l - 1] for l in model.Set_bch_dn})

        # Voltage limits (squared for linearized AC)
        if hasattr(self.data.net, 'V2_max'):
            model.V2_max = pyo.Param(model.Set_bus_dn,
                                     initialize={b: self.data.net.V2_max[b - 1] for b in model.Set_bus_dn})
            model.V2_min = pyo.Param(model.Set_bus_dn,
                                     initialize={b: self.data.net.V2_min[b - 1] for b in model.Set_bus_dn})
        else:
            # Square V_max/V_min if only those exist
            model.V2_max = pyo.Param(model.Set_bus_dn,
                                     initialize={b: self.data.net.V_max[b - 1] ** 2 for b in model.Set_bus_dn})
            model.V2_min = pyo.Param(model.Set_bus_dn,
                                     initialize={b: self.data.net.V_min[b - 1] ** 2 for b in model.Set_bus_dn})

        # Load data
        model.Pd = pyo.Param(model.Set_bus, model.Set_ts,
                             initialize={(b, t): self.data.net.profile_Pd[b - 1][t]
                             if self.data.net.profile_Pd[b - 1] is not None else 0
                                         for b in model.Set_bus for t in model.Set_ts})

        model.Qd = pyo.Param(model.Set_bus_dn, model.Set_ts,
                             initialize={(b, t): self.data.net.profile_Qd[b - 1][t]
                             if b in model.Set_bus_dn and self.data.net.profile_Qd[b - 1] is not None else 0
                                         for b in model.Set_bus_dn for t in model.Set_ts},
                             default=0)

        # Cost parameters
        model.Pimp_cost = pyo.Param(initialize=self.data.net.Pimp_cost)
        model.Pexp_cost = pyo.Param(initialize=self.data.net.Pexp_cost)
        model.Pc_cost = pyo.Param(model.Set_bus,
                                  initialize={b: self.data.net.Pc_cost[b - 1] for b in model.Set_bus})
        model.Qc_cost = pyo.Param(model.Set_bus_dn,
                                  initialize={b: self.data.net.Qc_cost[b - 1] for b in model.Set_bus_dn})

        # ESS parameters
        if Set_ess_exst:
            model.Pess_max_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: self.data.net.Pess_max_exst[e - 1] for e in Set_ess_exst})
            model.Pess_min_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: self.data.net.Pess_min_exst[e - 1] for e in Set_ess_exst})
            model.Eess_max_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: self.data.net.Eess_max_exst[e - 1] for e in Set_ess_exst})
            model.Eess_min_exst = pyo.Param(model.Set_ess_exst,
                                            initialize={e: self.data.net.Eess_min_exst[e - 1] for e in Set_ess_exst})
            model.SOC_max_exst = pyo.Param(model.Set_ess_exst,
                                           initialize={e: self.data.net.SOC_max_exst[e - 1] for e in Set_ess_exst})
            model.SOC_min_exst = pyo.Param(model.Set_ess_exst,
                                           initialize={e: self.data.net.SOC_min_exst[e - 1] for e in Set_ess_exst})
            model.eff_ch_exst = pyo.Param(model.Set_ess_exst,
                                          initialize={e: self.data.net.eff_ch_exst[e - 1] for e in Set_ess_exst})
            model.eff_dis_exst = pyo.Param(model.Set_ess_exst,
                                           initialize={e: self.data.net.eff_dis_exst[e - 1] for e in Set_ess_exst})
            model.initial_SOC_exst = pyo.Param(model.Set_ess_exst,
                                               initialize={e: self.data.net.initial_SOC_exst[e - 1] for e in
                                                           Set_ess_exst})

        # ------------------------------------------------------------------
        # 4. Variables
        # ------------------------------------------------------------------
        # Generation (existing only)
        model.Pg_exst = pyo.Var(model.Set_gen_exst, model.Set_ts,
                                within=pyo.NonNegativeReals,
                                bounds=lambda model, g, t: (model.Pg_min_exst[g], model.Pg_max_exst[g]))

        # Check if reactive demand exists
        has_reactive_demand = any(model.Qd[b, t] > 0 for b in model.Set_bus_dn for t in model.Set_ts)

        if has_reactive_demand:
            model.Qg_exst = pyo.Var(model.Set_gen_exst, model.Set_ts,
                                    within=pyo.Reals,
                                    bounds=lambda model, g, t: (model.Qg_min_exst[g], model.Qg_max_exst[g]))
        else:
            model.Qg_exst = pyo.Param(model.Set_gen_exst, model.Set_ts, default=0.0)

        # Grid import/export
        model.Pimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)
        model.Pexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)

        # Bus angles and voltages
        model.theta = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.Reals)
        model.V2_dn = pyo.Var(model.Set_bus_dn, model.Set_ts,
                              within=pyo.NonNegativeReals,
                              bounds=lambda model, b, t: (model.V2_min[b], model.V2_max[b]))

        # Power flows
        model.Pf_tn = pyo.Var(model.Set_bch_tn, model.Set_ts, within=pyo.Reals)
        model.Pf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)

        if has_reactive_demand:
            model.Qf_dn = pyo.Var(model.Set_bch_dn, model.Set_ts, within=pyo.Reals)
        else:
            model.Qf_dn = pyo.Param(model.Set_bch_dn, model.Set_ts, default=0.0)

        # Load shedding
        model.Pc = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals,
                           bounds=lambda model, b, t: (0, max(0.0, pyo.value(model.Pd[b, t])))
                           )

        if has_reactive_demand:
            model.Qc = pyo.Var(model.Set_bus_dn, model.Set_ts,
                               within=pyo.NonNegativeReals,
                               bounds=lambda model, b, t: (0, model.Qd[b, t] if b in model.Set_bus_dn else 0))
        else:
            model.Qc = pyo.Param(model.Set_bus_dn, model.Set_ts, default=0.0)

        # ESS variables
        if Set_ess_exst:
            model.Pess_charge_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, within=pyo.NonNegativeReals)
            model.Pess_discharge_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, within=pyo.NonNegativeReals)
            model.Eess_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, within=pyo.NonNegativeReals)
            model.SOC_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, bounds=(0, 1))
            model.ess_charge_binary_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, within=pyo.Binary)
            model.ess_discharge_binary_exst = pyo.Var(model.Set_ess_exst, model.Set_ts, within=pyo.Binary)

        # ------------------------------------------------------------------
        # 5. Constraints
        # ------------------------------------------------------------------
        # DC Power flow (TN)
        def dc_power_flow_rule(model, l, t):
            fr_bus = self.data.net.bch[l - 1][0]
            to_bus = self.data.net.bch[l - 1][1]
            return model.Pf_tn[l, t] == model.B_tn[l] * (model.theta[fr_bus, t] - model.theta[to_bus, t])

        model.Constraint_DC_PowerFlow = pyo.Constraint(model.Set_bch_tn, model.Set_ts, rule=dc_power_flow_rule)

        # Linearized DistFlow (DN)
        def lindistflow_rule(model, l, t):
            fr_bus = self.data.net.bch[l - 1][0]
            to_bus = self.data.net.bch[l - 1][1]
            return (model.V2_dn[to_bus, t] == model.V2_dn[fr_bus, t] -
                    2 * (model.R_dn[l] * model.Pf_dn[l, t] + model.X_dn[l] * model.Qf_dn[l, t]))

        model.Constraint_LinDistFlow = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=lindistflow_rule)

        # Power balance (TN)
        def power_balance_tn_rule(model, b, t):
            Pg = sum(model.Pg_exst[g, t] for g in model.Set_gen_exst
                     if self.data.net.gen[g - 1] == b)
            Pf_in = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn
                        if self.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn
                         if self.data.net.bch[l - 1][0] == b)
            Pgrid = (model.Pimp[t] - model.Pexp[t]) if b == slack_bus else 0

            Pess_net = 0.0
            if len(model.Set_ess_exst):
                Pess_net = sum(
                    model.Pess_discharge_exst[e, t] - model.Pess_charge_exst[e, t]
                    for e in model.Set_ess_exst if self.data.net.ess[e - 1] == b
                )

            return Pg + Pgrid + Pess_net + Pf_in - Pf_out == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_PowerBalance_TN = pyo.Constraint(model.Set_bus_tn, model.Set_ts, rule=power_balance_tn_rule)

        # Power balance (DN)
        def power_balance_dn_rule(model, b, t):
            Pg = sum(model.Pg_exst[g, t] for g in model.Set_gen_exst
                     if self.data.net.gen[g - 1] == b)
            Pf_in_dn = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn
                           if self.data.net.bch[l - 1][1] == b)
            Pf_out_dn = sum(model.Pf_dn[l, t] for l in model.Set_bch_dn
                            if self.data.net.bch[l - 1][0] == b)

            Pf_cpl_in = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn
                            if self.data.net.branch_level[l] == 'T-D' and self.data.net.bch[l - 1][1] == b)
            Pf_cpl_out = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn
                             if self.data.net.branch_level[l] == 'T-D' and self.data.net.bch[l - 1][0] == b)

            # NEW: ESS net injection at this DN bus
            Pess_net = 0.0
            if len(model.Set_ess_exst):
                Pess_net = sum(
                    model.Pess_discharge_exst[e, t] - model.Pess_charge_exst[e, t]
                    for e in model.Set_ess_exst if self.data.net.ess[e - 1] == b
                )

            return Pg + Pess_net + Pf_in_dn - Pf_out_dn + Pf_cpl_in - Pf_cpl_out == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_PowerBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts, rule=power_balance_dn_rule)

        # Reactive power balance (DN) if needed
        if has_reactive_demand:
            def reactive_balance_dn_rule(model, b, t):
                Qg = sum(model.Qg_exst[g, t] for g in model.Set_gen_exst
                         if self.data.net.gen[g - 1] == b)
                Qf_in = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn
                            if self.data.net.bch[l - 1][1] == b)
                Qf_out = sum(model.Qf_dn[l, t] for l in model.Set_bch_dn
                             if self.data.net.bch[l - 1][0] == b)
                return Qg + Qf_in - Qf_out == model.Qd[b, t] - model.Qc[b, t]

            model.Constraint_ReactiveBalance_DN = pyo.Constraint(model.Set_bus_dn, model.Set_ts,
                                                                 rule=reactive_balance_dn_rule)

        # Branch flow limits
        model.Constraint_Pf_tn_upper = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                      rule=lambda model, l, t: model.Pf_tn[l, t] <= model.Pmax_tn[l])
        model.Constraint_Pf_tn_lower = pyo.Constraint(model.Set_bch_tn, model.Set_ts,
                                                      rule=lambda model, l, t: model.Pf_tn[l, t] >= -model.Pmax_tn[l])

        # Apparent power limit for DN
        if has_reactive_demand:
            def apparent_power_dn_rule(model, l, t):
                return model.Pf_dn[l, t] ** 2 + model.Qf_dn[l, t] ** 2 <= model.Smax_dn[l] ** 2
        else:
            def apparent_power_dn_rule(model, l, t):
                return model.Pf_dn[l, t] ** 2 <= model.Smax_dn[l] ** 2

        model.Constraint_Smax_dn = pyo.Constraint(model.Set_bch_dn, model.Set_ts, rule=apparent_power_dn_rule)

        # Slack bus constraint
        model.Constraint_Slack = pyo.Constraint(model.Set_ts,
                                                rule=lambda model, t: model.theta[slack_bus, t] == 0)

        if Set_ess_exst:
            # Charge/discharge power limits
            def ess_charge_ub_rule(model, e, t):
                return model.Pess_charge_exst[e, t] <= model.Pess_max_exst[e] * model.ess_charge_binary_exst[e, t]

            model.ESS_ChargeUB = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_charge_ub_rule)

            def ess_discharge_ub_rule(model, e, t):
                return model.Pess_discharge_exst[e, t] <= model.Pess_max_exst[e] * model.ess_discharge_binary_exst[e, t]

            model.ESS_DischargeUB = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_discharge_ub_rule)

            # Charge/discharge exclusivity
            def ess_exclusivity_rule(model, e, t):
                return model.ess_charge_binary_exst[e, t] + model.ess_discharge_binary_exst[e, t] <= 1

            model.ESS_Exclusivity = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_exclusivity_rule)

            def ess_e_bounds_rule(model, e, t):
                return pyo.inequality(model.Eess_min_exst[e], model.Eess_exst[e, t], model.Eess_max_exst[e])

            model.ESS_Energy_Bounds = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_e_bounds_rule)
            
            # SOC limits
            def ess_soc_bounds_rule(model, e, t):
                return pyo.inequality(model.SOC_min_exst[e], model.SOC_exst[e, t], model.SOC_max_exst[e])

            model.ESS_SOC_Bounds = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_soc_bounds_rule)

            # Link battery energy with SOC
            def ess_e_soc_link_rule(model, e, t):
                return model.Eess_exst[e, t] == model.SOC_exst[e, t] * model.Eess_max_exst[e]

            model.ESS_E_SOC_Link = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_e_soc_link_rule)

            ts_sorted = sorted(list(model.Set_ts.data()))

            # Inter-temporal energy balance (i.e., ESS dynamics)
            def ess_dyn_rule(model, e, t):
                idx = ts_sorted.index(t)
                E_prev = model.initial_SOC_exst[e] * model.Eess_max_exst[e] if idx == 0 else model.Eess_exst[e, ts_sorted[idx - 1]]
                return model.Eess_exst[e, t] == E_prev + model.eff_ch_exst[e] * model.Pess_charge_exst[e, t] - (
                        1 / model.eff_dis_exst[e]) * model.Pess_discharge_exst[e, t]

            model.ESS_Dynamics = pyo.Constraint(model.Set_ess_exst, model.Set_ts, rule=ess_dyn_rule)

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

            # Load shedding cost
            ls_cost = sum(
                model.Pc_cost[b] * model.Pc[b, t]
                for b in model.Set_bus for t in model.Set_ts
            )

            if has_reactive_demand:
                ls_cost += sum(
                    model.Qc_cost[b] * model.Qc[b, t]
                    for b in model.Set_bus_dn for t in model.Set_ts
                )

            return gen_cost + grid_cost + ls_cost

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return model


    def solve_combined_dc_linearized_ac_opf(self,
                                            model,
                                            solver='gurobi',
                                            mip_gap: float = 1e-3,
                                            mip_gap_abs: float = 1e5,
                                            time_limit: float = 3600,
                                            write_xlsx: bool = False,
                                            out_dir: str = "Optimization_Results/Combined_DC_and_Linearized_AC"):
        """
        Solve the combined DC (transmission) + linearized AC (distribution) OPF model.
        """
        # 1) Create solver and solve with output streamed to console
        opt = SolverFactory(solver)

        # Add solver-specific options
        if solver.lower() == 'gurobi':
            opt.options['MIPGap'] = mip_gap
            opt.options['MIPGapAbs'] = mip_gap_abs
            opt.options['TimeLimit'] = time_limit

        elif solver.lower() == 'cbc':
            opt.options['threads'] = 0
            opt.options['presolve'] = 'on'
            opt.options['cuts'] = 'on'

        elif solver.lower() == 'glpk':
            opt.options['tmlim'] = 3600
            opt.options['msg_lev'] = 'GLP_MSG_ON'

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

    def build_combined_opf_model_under_ws_scenarios(self,
                                                    single_ws_scenario: dict,
                                                    scenario_probability: float = 1.0,
                                                    ):
        """
        Build a simplified OPF model for a single windstorm scenario without investment decisions.

        Args:
            single_ws_scenario: Single windstorm scenario dictionary with events data
            scenario_probability: Probability of this scenario (default 1.0 for single scenario)

        Returns:
            Pyomo ConcreteModel for the windstorm scenario OPF
        """

        # todo: the model built by this method seems be infeasible under the scenarios 'ws_0278', 'ws_0644', 'ws_0403'
        #  in 'ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15.json' --> to be debugged

        # ------------------------------------------------------------------
        # 0. Preliminaries
        # ------------------------------------------------------------------

        # Store metadata for single scenario
        self.meta = Object()
        self.meta.n_ws_scenarios = 1
        self.meta.period_type = "year"  # Assuming year period

        # Process single scenario
        ws_scenarios = {"scenario_1": single_ws_scenario}

        # Extract timesteps from events
        all_timesteps = set()
        for event in single_ws_scenario.get("events", []):
            bgn_hr = event["bgn_hr"]
            duration = event["duration"]
            for t in range(bgn_hr, bgn_hr + duration):
                all_timesteps.add(t + 1)  # 1-indexed

        # Add repair hours if needed
        ttr_max = max(single_ws_scenario.get("bch_ttr", [0]))
        if ttr_max > 0:
            max_timestep = max(all_timesteps) if all_timesteps else 0
            for t in range(max_timestep + 1, min(max_timestep + ttr_max + 1, 8761)):
                all_timesteps.add(t)

        # ------------------------------------------------------------------
        # 1. Indexing sets
        # ------------------------------------------------------------------
        num_bus = len(self.data.net.bus)
        num_bch = len(self.data.net.bch)
        num_gen = len(self.data.net.gen)

        Set_bus = list(range(1, num_bus + 1))
        Set_bch = list(range(1, num_bch + 1))
        Set_gen = list(range(1, num_gen + 1))

        # Level-specific sets
        Set_bus_tn = [b for b in Set_bus if self.data.net.bus_level[b] == 'T']
        Set_bus_dn = [b for b in Set_bus if self.data.net.bus_level[b] == 'D']

        Set_bch_tn = [l for l in Set_bch if self.data.net.branch_level[l] == 'T']
        Set_bch_dn = [l for l in Set_bch if self.data.net.branch_level[l] == 'D']
        Set_bch_td = [l for l in Set_bch if self.data.net.branch_level[l] == 'T-D']

        # Line types
        Set_bch_tn_lines = [l for l in Set_bch_tn if self.data.net.bch_type[l - 1] == 'Line']
        Set_bch_dn_lines = [l for l in Set_bch_dn if self.data.net.bch_type[l - 1] == 'Line']
        Set_bch_lines = Set_bch_tn_lines + Set_bch_dn_lines

        # Check for renewable generators
        has_renewables = hasattr(self.data.net, 'profile_Pg_renewable') and \
                         any(profile is not None for profile in self.data.net.profile_Pg_renewable)

        Set_gen_ren = [g for g in Set_gen
                       if has_renewables and self.data.net.profile_Pg_renewable[g - 1] is not None]
        Set_gen_nren = [g for g in Set_gen if g not in Set_gen_ren]

        Set_ts = sorted(list(all_timesteps))

        # Combined index sets
        Set_bt = [(b, t) for b in Set_bus for t in Set_ts]
        Set_bt_dn = [(b, t) for b in Set_bus_dn for t in Set_ts]
        Set_bt_tn = [(b, t) for b in Set_bus_tn for t in Set_ts]

        Set_lt = [(l, t) for l in Set_bch_lines for t in Set_ts]
        Set_lt_tn_lines = [(l, t) for l in Set_bch_tn_lines for t in Set_ts]
        Set_lt_dn_lines = [(l, t) for l in Set_bch_dn_lines for t in Set_ts]
        Set_lt_lines = Set_lt_tn_lines + Set_lt_dn_lines  # Union of both

        Set_gt = [(g, t) for g in Set_gen for t in Set_ts]
        Set_gt_ren = [(g, t) for g in Set_gen_ren for t in Set_ts]
        Set_gt_nren = [(g, t) for g in Set_gen_nren for t in Set_ts]
        Set_gt_dn = [(g, t) for g in Set_gen for t in Set_ts
                     if self.data.net.bus_level[self.data.net.gen[g - 1]] == 'D']

        Set_ess = [e for e in range(1, len(self.data.net.ess) + 1)]

        # Check if we have non-zero reactive demand values in this model
        def _has_Qd():
            TOL = 1e-9
            if not hasattr(self.data.net, 'profile_Qd') or self.data.net.profile_Qd is None:
                return False
            for b in Set_bus_dn:
                prof = self.data.net.profile_Qd[b - 1]
                if prof is None:
                    continue
                # Set_ts are absolute indices; adjust if your profiles are 0-based
                if any(abs(prof[t - 1]) > TOL for t in Set_ts):
                    return True
            return False

        has_reactive_demand = _has_Qd()

        print("has_reactive_demand:")
        print(has_reactive_demand)

        # ------------------------------------------------------------------
        # 2. Initialize Pyomo model and sets
        # ------------------------------------------------------------------
        print("Building OPF model for single windstorm scenario...")

        model = pyo.ConcreteModel()

        model.Set_bus = pyo.Set(initialize=Set_bus)
        model.Set_bus_tn = pyo.Set(initialize=Set_bus_tn)
        model.Set_bus_dn = pyo.Set(initialize=Set_bus_dn)
        model.Set_bch = pyo.Set(initialize=Set_bch)
        model.Set_bch_tn = pyo.Set(initialize=Set_bch_tn)
        model.Set_bch_dn = pyo.Set(initialize=Set_bch_dn)
        model.Set_bch_tn_lines = pyo.Set(initialize=Set_bch_tn_lines)
        model.Set_bch_dn_lines = pyo.Set(initialize=Set_bch_dn_lines)
        model.Set_bch_lines = pyo.Set(initialize=Set_bch_lines)
        model.Set_gen = pyo.Set(initialize=Set_gen)
        model.Set_gen_ren = pyo.Set(initialize=Set_gen_ren)
        model.Set_gen_nren = pyo.Set(initialize=Set_gen_nren)
        model.Set_ess = pyo.Set(initialize=Set_ess)

        model.Set_ts = pyo.Set(initialize=Set_ts)

        model.Set_bt = pyo.Set(initialize=Set_bt, dimen=2)
        model.Set_bt_dn = pyo.Set(initialize=Set_bt_dn, dimen=2)
        model.Set_bt_tn = pyo.Set(initialize=Set_bt_tn, dimen=2)
        model.Set_lt_tn_lines = pyo.Set(initialize=Set_lt_tn_lines, dimen=2)
        model.Set_lt_dn_lines = pyo.Set(initialize=Set_lt_dn_lines, dimen=2)
        model.Set_lt_lines = pyo.Set(initialize=Set_lt_lines, dimen=2)
        model.Set_gt = pyo.Set(initialize=Set_gt, dimen=2)
        model.Set_gt_ren = pyo.Set(initialize=Set_gt_ren, dimen=2)
        model.Set_gt_nren = pyo.Set(initialize=Set_gt_nren, dimen=2)
        model.Set_gt_dn = pyo.Set(initialize=Set_gt_dn, dimen=2)
        model.Set_et = pyo.Set(initialize=[(e, t) for e in Set_ess for t in model.Set_ts])

        # ------------------------------------------------------------------
        # 3. Parameters
        # ------------------------------------------------------------------
        # 3.1) Base & slack
        model.base_MVA = pyo.Param(initialize=self.data.net.base_MVA)
        slack_bus = self.data.net.slack_bus

        # 3.2) Branch parameters - TN (DC) and DN (LinAC)
        model.B_tn = pyo.Param(model.Set_bch_tn,
                               initialize={l: 1.0 / self.data.net.bch_X[l - 1] for l in model.Set_bch_tn})
        model.Pmax_tn = pyo.Param(model.Set_bch_tn,
                                  initialize={l: self.data.net.bch_Smax[l - 1] for l in model.Set_bch_tn})
        model.R_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: self.data.net.bch_R[l - 1] for l in model.Set_bch_dn})
        model.X_dn = pyo.Param(model.Set_bch_dn,
                               initialize={l: self.data.net.bch_X[l - 1] for l in model.Set_bch_dn})
        model.Smax_dn = pyo.Param(model.Set_bch_dn,
                                  initialize={l: self.data.net.bch_Smax[l - 1] for l in model.Set_bch_dn})

        # 3.3) Demand profiles
        demand_data = {}
        reactive_demand_data = {} if has_reactive_demand else None

        for b in Set_bus:
            bus_idx = b - 1
            profile = self.data.net.profile_Pd[bus_idx]
            max_demand = self.data.net.Pd_max[bus_idx]

            for t in Set_ts:
                demand_data[(b, t)] = profile[t - 1] * max_demand if profile else max_demand

                if has_reactive_demand:
                    q_profile = self.data.net.profile_Qd[bus_idx]
                    max_q_demand = self.data.net.Qd_max[bus_idx]
                    reactive_demand_data[(b, t)] = q_profile[t - 1] * max_q_demand if q_profile else max_q_demand

        model.Pd = pyo.Param(model.Set_bt, initialize=demand_data)
        if has_reactive_demand:
            model.Qd = pyo.Param(model.Set_bt, initialize=reactive_demand_data)

        # 3.4) Generator parameters (existing only)
        coef_len = len(self.data.net.gen_cost_coef[0])
        model.gen_cost_coef = pyo.Param(
            model.Set_gen, range(coef_len),
            initialize={(g, c): self.data.net.gen_cost_coef[g - 1][c]
                        for g in model.Set_gen for c in range(coef_len)}
        )
        model.Pg_max = pyo.Param(model.Set_gen,
                                 initialize={g: self.data.net.Pg_max_exst[g - 1] for g in model.Set_gen})
        model.Pg_min = pyo.Param(model.Set_gen,
                                 initialize={g: self.data.net.Pg_min_exst[g - 1] for g in model.Set_gen})
        model.Qg_max = pyo.Param(model.Set_gen,
                                 initialize={g: self.data.net.Qg_max_exst[g - 1] for g in model.Set_gen})
        model.Qg_min = pyo.Param(model.Set_gen,
                                 initialize={g: self.data.net.Qg_min_exst[g - 1] for g in model.Set_gen})

        # Renewable generation availability
        if has_renewables and model.Set_gen_ren:
            renewable_availability = {}
            for g in model.Set_gen_ren:
                gen_idx = g - 1
                profile = self.data.net.profile_Pg_renewable[gen_idx]
                for hr in Set_ts:
                    renewable_availability[(g, hr)] = \
                        profile[hr - 1] if profile else self.data.net.Pg_max_exst[gen_idx]

            model.Pg_renewable_avail = pyo.Param(
                model.Set_gen_ren,
                model.Set_ts,
                initialize=renewable_availability
            )

        # 3.5) ESS parameters
        if Set_ess:
            model.Pess_max = pyo.Param(model.Set_ess,
                                       initialize={e: self.data.net.Pess_max_exst[e - 1] for e in Set_ess})
            model.Eess_max = pyo.Param(model.Set_ess,
                                       initialize={e: self.data.net.Eess_max_exst[e - 1] for e in Set_ess})
            model.Eess_min = pyo.Param(model.Set_ess,
                                       initialize={e: self.data.net.Eess_min_exst[e - 1] for e in Set_ess})
            model.SOC_max = pyo.Param(model.Set_ess,
                                      initialize={e: self.data.net.SOC_max_exst[e - 1] for e in Set_ess})
            model.SOC_min = pyo.Param(model.Set_ess,
                                      initialize={e: self.data.net.SOC_min_exst[e - 1] for e in Set_ess})
            model.eff_ch = pyo.Param(model.Set_ess,
                                     initialize={e: self.data.net.eff_ch_exst[e - 1] for e in Set_ess})
            model.eff_dis = pyo.Param(model.Set_ess,
                                      initialize={e: self.data.net.eff_dis_exst[e - 1] for e in Set_ess})
            model.initial_SOC = pyo.Param(model.Set_ess,
                                          initialize={e: self.data.net.initial_SOC_exst[e - 1] for e in Set_ess})

        # 3.6) Load shedding costs
        model.Pc_cost = pyo.Param(model.Set_bus,
                                  initialize={b: self.data.net.Pc_cost[b - 1] for b in model.Set_bus})
        model.Qc_cost = pyo.Param(model.Set_bus,
                                  initialize={b: self.data.net.Qc_cost[b - 1] for b in model.Set_bus})

        # 3.7) Repair costs
        model.rep_cost = pyo.Param(model.Set_bch_lines,
                                   initialize={l: self.data.net.cost_bch_rep[l - 1] for l in model.Set_bch_lines})

        # 3.8) Scenario probability
        model.prob_factor = pyo.Param(initialize=scenario_probability)

        # 3.9) Windstorm parameters
        # Process windstorm spatial-temporal data
        windstorm_params = {}
        impact_flags = {}

        # Initialize all timesteps with no wind
        for t in Set_ts:
            windstorm_params[t] = 0.0
            for l in Set_bch_lines:
                impact_flags[(l, t)] = 0

        # Process each event (directly from single_ws_scenario)
        for event in single_ws_scenario.get("events", []):
            bgn_hr = event["bgn_hr"]
            gust_speeds = event["gust_speed"]

            # Get impact flags from event
            event_impact_flags = np.array(event.get("flgs_impacted_bch", []))

            for idx, speed in enumerate(gust_speeds):
                t = bgn_hr + idx + 1  # 1-indexed
                if t in Set_ts:
                    windstorm_params[t] = speed

                    # Set impact flags for this timestep
                    if len(event_impact_flags) > 0:
                        for l in Set_bch_lines:
                            l_idx = l - 1
                            if l_idx < len(event_impact_flags) and len(event_impact_flags[l_idx]) > idx:
                                impact_flags[(l, t)] = int(event_impact_flags[l_idx][idx])

        model.wind_gust_speed = pyo.Param(model.Set_ts, initialize=windstorm_params)
        model.impact_flag = pyo.Param(model.Set_lt_lines, initialize=impact_flags, mutable=False)

        # Random numbers for failure probability
        rand_nums_data = {}
        bch_rand_nums = single_ws_scenario.get("bch_rand_nums", [])

        for l in Set_bch_lines:
            l_idx = l - 1
            for t in Set_ts:
                if l_idx < len(bch_rand_nums) and (t - 1) < len(bch_rand_nums[l_idx]):
                    rand_nums_data[(l, t)] = bch_rand_nums[l_idx][t - 1]
                else:
                    rand_nums_data[(l, t)] = 1.0  # No failure if no random number

        model.rand_num = pyo.Param(model.Set_lt_lines, initialize=rand_nums_data)

        # Time to repair
        ttr_data = {}
        bch_ttr = single_ws_scenario.get("bch_ttr", [])

        for l in Set_bch_lines:
            l_idx = l - 1
            if l_idx < len(bch_ttr):
                ttr_data[l] = bch_ttr[l_idx]
            else:
                ttr_data[l] = 24  # Default TTR

        model.ttr = pyo.Param(model.Set_bch_lines, initialize=ttr_data)

        # Fragility curve parameters
        model.frg_mu = pyo.Param(model.Set_bch_lines,
                                 initialize={l: self.data.frg.mu[l - 1] for l in model.Set_bch_lines})
        model.frg_sigma = pyo.Param(model.Set_bch_lines,
                                    initialize={l: self.data.frg.sigma[l - 1] for l in model.Set_bch_lines})
        model.frg_thrd_1 = pyo.Param(model.Set_bch_lines,
                                     initialize={l: self.data.frg.thrd_1[l - 1] for l in model.Set_bch_lines})
        model.frg_thrd_2 = pyo.Param(model.Set_bch_lines,
                                     initialize={l: self.data.frg.thrd_2[l - 1] for l in model.Set_bch_lines})

        # Number of timesteps in each scenario (for easy access)
        model.num_timesteps = len(Set_ts)

        # ------------------------------------------------------------------
        # 4. Variables
        # ------------------------------------------------------------------
        # 4.1) Voltage angles (TN)
        model.theta = pyo.Var(model.Set_bt_tn, within=pyo.Reals)

        # 4.2) Voltage magnitude squared (DN)
        model.V2 = pyo.Var(model.Set_bt_dn,
                           bounds=lambda model, b, t: (0.9 ** 2, 1.1 ** 2))

        # 4.3) Power flows
        model.Pf_tn = pyo.Var([(l, t) for l in Set_bch_tn + Set_bch_td
                               for t in Set_ts], within=pyo.Reals)
        model.Pf_dn = pyo.Var([(l, t) for l in Set_bch_dn
                               for t in Set_ts], within=pyo.Reals)
        if has_reactive_demand:
            model.Qf_dn = pyo.Var([(l, t) for l in Set_bch_dn
                                   for t in Set_ts], within=pyo.Reals)

        # 4.4) Grid electricity import/export
        model.Pimp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)
        model.Pexp = pyo.Var(model.Set_ts, within=pyo.NonNegativeReals)

        # 4.5) Generation variables
        model.Pg = pyo.Var(model.Set_gt, within=pyo.Reals)
        if has_reactive_demand:
            model.Qg = pyo.Var(model.Set_gt, within=pyo.Reals)

        # 4.6) ESS variables
        if Set_ess:
            model.Pess_charge = pyo.Var(
                model.Set_et,
                within=pyo.NonNegativeReals,
                bounds=lambda model, e, t: (0, model.Pess_max[e])
            )
            model.Pess_discharge = pyo.Var(
                model.Set_et,
                within=pyo.NonNegativeReals,
                bounds=lambda model, e, t: (0, model.Pess_max[e])
            )
            model.Eess = pyo.Var(
                model.Set_et,
                within=pyo.NonNegativeReals,
                bounds=lambda model, e, t: (model.Eess_min[e], model.Eess_max[e])
            )
            model.SOC = pyo.Var(
                model.Set_et,
                within=pyo.NonNegativeReals,
                bounds=lambda model, e, t: (model.SOC_min[e], model.SOC_max[e])
            )
            model.ess_charge_binary = pyo.Var(model.Set_et, within=pyo.Binary)
            model.ess_discharge_binary = pyo.Var(model.Set_et, within=pyo.Binary)

        # 4.7) Load shedding
        model.Pc = pyo.Var(model.Set_bt, within=pyo.NonNegativeReals,
                           bounds=lambda model, b, t: (0, max(0.0, pyo.value(model.Pd[b, t]))))
        if has_reactive_demand:
            model.Qc = pyo.Var(model.Set_bt_dn, within=pyo.NonNegativeReals,
                               bounds=lambda model, b, t: (0, max(0.0, pyo.value(model.Qd[b, t]))))

        # 4.8) Line status and failure variables
        model.bch_status = pyo.Var(model.Set_lt_lines, within=pyo.Binary)
        model.fail_prob = pyo.Var(model.Set_lt_lines, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.alpha = pyo.Var(model.Set_lt_lines, within=pyo.Binary)
        model.beta = pyo.Var(model.Set_lt_lines, within=pyo.Binary)
        model.fail_occurs = pyo.Var(model.Set_lt_lines, within=pyo.Binary)
        model.repair_applies = pyo.Var(model.Set_lt_lines, within=pyo.Binary)

        # ------------------------------------------------------------------
        # 5. Constraints
        # ------------------------------------------------------------------
        # 5.1) Power flow constraints - TN (DC)
        def dc_power_flow_rule(model, l, t):  
            fr_bus = self.data.net.bch[l - 1][0]
            to_bus = self.data.net.bch[l - 1][1]

            if fr_bus in Set_bus_tn and to_bus in Set_bus_tn:
                return model.Pf_tn[l, t] == model.B_tn[l] * (
                        model.theta[fr_bus, t] - model.theta[to_bus, t])  
            else:
                return pyo.Constraint.Skip

        model.Constraint_DCPowerFlow = pyo.Constraint(
            [(l, t) for l in Set_bch_tn for t in Set_ts],
            rule=dc_power_flow_rule
        )

        # 5.2) Power balance - TN
        def power_balance_tn_rule(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)

            Pgrid = (model.Pimp[t] - model.Pexp[t]) if b == slack_bus else 0

            Pf_in = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if self.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf_tn[l, t] for l in model.Set_bch_tn if self.data.net.bch[l - 1][0] == b)

            Pess = 0
            if Set_ess:
                Pess = sum(model.Pess_discharge[e, t] - model.Pess_charge[e, t]
                           for e in model.Set_ess if self.data.net.ess[e - 1] == b)

            return Pg + Pgrid + Pess + Pf_in - Pf_out == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_PowerBalance_TN = pyo.Constraint(model.Set_bt_tn, rule=power_balance_tn_rule)

        # 5.3) LinDistFlow - DN
        def lindistflow_P_rule(model, l, t):  
            fr_bus = self.data.net.bch[l - 1][0]
            to_bus = self.data.net.bch[l - 1][1]

            return model.V2[fr_bus, t] - model.V2[to_bus, t] == \
                2 * model.R_dn[l] * model.Pf_dn[l, t] + \
                (2 * model.X_dn[l] * model.Qf_dn[l, t] if has_reactive_demand else 0)  

        model.Constraint_LinDistFlow = pyo.Constraint(
            [(l, t) for l in Set_bch_dn for t in Set_ts],  # SIMPLIFIED
            rule=lindistflow_P_rule
        )

        # 5.4) Power balance - DN (active)
        def power_balance_dn_P_rule(model, b, t):  
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)  

            Pf_tn_in = sum(model.Pf_tn[l, t] for l in Set_bch_td  
                           if self.data.net.bch[l - 1][1] == b)
            Pf_dn_in = sum(model.Pf_dn[l, t] for l in Set_bch_dn  
                           if self.data.net.bch[l - 1][1] == b)
            Pf_dn_out = sum(model.Pf_dn[l, t] for l in Set_bch_dn  
                            if self.data.net.bch[l - 1][0] == b)

            Pess = 0
            if Set_ess:
                Pess = sum(model.Pess_discharge[e, t] - model.Pess_charge[e, t]
                           for e in model.Set_ess if self.data.net.ess[e - 1] == b)

            return Pg + Pess + Pf_tn_in + Pf_dn_in - Pf_dn_out == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_PowerBalance_DN_P = pyo.Constraint(model.Set_bt_dn, rule=power_balance_dn_P_rule)

        # 5.5) Power balance - DN (reactive)
        if has_reactive_demand:
            def power_balance_dn_Q_rule(model, b, t):  
                Qg = sum(model.Qg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
                Qf_in = sum(model.Qf_dn[l, t] for l in Set_bch_dn
                            if self.data.net.bch[l - 1][1] == b)
                Qf_out = sum(model.Qf_dn[l, t] for l in Set_bch_dn
                             if self.data.net.bch[l - 1][0] == b)

                return Qg + Qf_in - Qf_out == model.Qd[b, t] - model.Qc[b, t]

            model.Constraint_PowerBalance_DN_Q = pyo.Constraint(model.Set_bt_dn, rule=power_balance_dn_Q_rule)

        # 5.6) Thermal limits with line status - TN
        def thermal_limit_tn_pos_rule(model, l, t):
            return model.Pf_tn[l, t] <= model.Pmax_tn[l] * model.bch_status[l, t]

        def thermal_limit_tn_neg_rule(model, l, t):
            return model.Pf_tn[l, t] >= -model.Pmax_tn[l] * model.bch_status[l, t]

        model.Constraint_ThermalLimit_TN_pos = pyo.Constraint(model.Set_lt_tn_lines, rule=thermal_limit_tn_pos_rule)
        model.Constraint_ThermalLimit_TN_neg = pyo.Constraint(model.Set_lt_tn_lines, rule=thermal_limit_tn_neg_rule)

        # 5.7) Thermal limits - DN (linearized)
        def thermal_limit_dn_pos_rule(model, l, t):
            return model.Pf_dn[l, t] <= model.Pmax_dn[l] * model.bch_status[l, t]

        def thermal_limit_dn_neg_rule(model, l, t):
            return model.Pf_dn[l, t] >= -model.Pmax_dn[l] * model.bch_status[l, t]

        model.Constraint_ThermalLimit_DN_pos = pyo.Constraint(model.Set_lt_tn_lines, rule=thermal_limit_dn_pos_rule)
        model.Constraint_ThermalLimit_DN_neg = pyo.Constraint(model.Set_lt_tn_lines, rule=thermal_limit_dn_neg_rule)

        if has_reactive_demand:
            def thermal_limit_dn_Q_pos_rule(model, l, t):  
                return model.Qf_dn[l, t] <= model.Smax_dn[l] * model.bch_status[l, t]  

            def thermal_limit_dn_Q_neg_rule(model, l, t):  
                return model.Qf_dn[l, t] >= -model.Smax_dn[l] * model.bch_status[l, t]  

            def thermal_limit_dn_PQ_sum_rule(model, l, t):  
                return model.Pf_dn[l, t] + model.Qf_dn[l, t] <= np.sqrt(2) * model.Smax_dn[l] * model.bch_status[
                    l, t]  

            def thermal_limit_dn_PQ_diff_rule(model, l, t):  
                return model.Pf_dn[l, t] - model.Qf_dn[l, t] <= np.sqrt(2) * model.Smax_dn[l] * model.bch_status[
                    l, t]  

            model.Constraint_ThermalLimit_DN_Q_pos = pyo.Constraint(model.Set_lt_dn_lines,
                                                                    rule=thermal_limit_dn_Q_pos_rule)
            model.Constraint_ThermalLimit_DN_Q_neg = pyo.Constraint(model.Set_lt_dn_lines,
                                                                    rule=thermal_limit_dn_Q_neg_rule)
            model.Constraint_ThermalLimit_DN_PQ_sum = pyo.Constraint(model.Set_lt_dn_lines,
                                                                     rule=thermal_limit_dn_PQ_sum_rule)
            model.Constraint_ThermalLimit_DN_PQ_diff = pyo.Constraint(model.Set_lt_dn_lines,
                                                                      rule=thermal_limit_dn_PQ_diff_rule)

        # 5.8) Generation limits
        def Pg_upper_limit_rule(model, g, t):  
            if has_renewables and g in model.Set_gen_ren:
                return model.Pg[g, t] <= model.Pg_renewable_avail[g, t]  
            else:
                return model.Pg[g, t] <= model.Pg_max[g]  

        def Pg_lower_limit_rule(model, g, t):  
            return model.Pg[g, t] >= model.Pg_min[g]  

        model.Constraint_Pg_upper = pyo.Constraint(model.Set_gt, rule=Pg_upper_limit_rule)
        model.Constraint_Pg_lower = pyo.Constraint(model.Set_gt, rule=Pg_lower_limit_rule)

        if has_reactive_demand:
            def Qg_upper_limit_rule(model, g, t):  
                return model.Qg[g, t] <= model.Qg_max[g]  

            def Qg_lower_limit_rule(model, g, t):  
                return model.Qg[g, t] >= model.Qg_min[g]  

            model.Constraint_Qg_upper = pyo.Constraint(model.Set_gt, rule=Qg_upper_limit_rule)
            model.Constraint_Qg_lower = pyo.Constraint(model.Set_gt, rule=Qg_lower_limit_rule)

        # 5.9) ESS operation constraints
        if Set_ess:
            # Charge/discharge power limits with binary variables
            def ess_charge_limit_rule(model, e, t):
                return model.Pess_charge[e, t] <= model.Pess_max[e] * model.ess_charge_binary[e, t]

            def ess_discharge_limit_rule(model, e, t):
                return model.Pess_discharge[e, t] <= model.Pess_max[e] * model.ess_discharge_binary[e, t]

            model.Constraint_ESS_charge_limit = pyo.Constraint(model.Set_et, rule=ess_charge_limit_rule)
            model.Constraint_ESS_discharge_limit = pyo.Constraint(model.Set_et, rule=ess_discharge_limit_rule)

            # Charge/discharge exclusivity
            def ess_exclusivity_rule(model, e, t):
                return model.ess_charge_binary[e, t] + model.ess_discharge_binary[e, t] <= 1

            model.Constraint_ESS_exclusivity = pyo.Constraint(model.Set_et, rule=ess_exclusivity_rule)

            # SOC dynamics
            def ess_dynamics_rule(model, e, t):
                ts_list = sorted(list(model.Set_ts))

                if t == ts_list[0]:  # First timestep
                    E_prev = model.Eess_max[e] * model.initial_SOC[e]
                else:
                    prev_t = ts_list[ts_list.index(t) - 1]
                    E_prev = model.Eess[e, prev_t]

                # Assuming dt = 1 hour for simplicity
                dt = 1.0
                return model.Eess[e, t] == E_prev + dt * (
                        model.Pess_charge[e, t] * model.eff_ch[e] -
                        model.Pess_discharge[e, t] / model.eff_dis[e]
                )

            model.Constraint_ESS_dynamics = pyo.Constraint(model.Set_et, rule=ess_dynamics_rule)

            # SOC definition
            def soc_definition_rule(model, e, t):
                return model.SOC[e, t] == model.Eess[e, t] / model.Eess_max[e]

            model.Constraint_SOC_definition = pyo.Constraint(model.Set_et, rule=soc_definition_rule)

            # SOC limits
            def soc_min_rule(model, e, t):
                return model.SOC[e, t] >= model.SOC_min[e]

            def soc_max_rule(model, e, t):
                return model.SOC[e, t] <= model.SOC_max[e]

            model.Constraint_SOC_min = pyo.Constraint(model.Set_et, rule=soc_min_rule)
            model.Constraint_SOC_max = pyo.Constraint(model.Set_et, rule=soc_max_rule)

        # 5.10) Line failure and repair constraints
        # Fragility curve (piecewise linear)
        def fragility_curve_rule(model, l, t):  
            v = model.wind_gust_speed[t]  
            mu = model.frg_mu[l]
            sigma = model.frg_sigma[l]
            thrd_1 = model.frg_thrd_1[l]
            thrd_2 = model.frg_thrd_2[l]

            if v <= thrd_1:
                return model.fail_prob[l, t] == 0  
            elif v >= thrd_2:
                return model.fail_prob[l, t] == 1  
            else:
                # Linear interpolation between thresholds
                p1 = lognorm.cdf(thrd_1, s=sigma, scale=mu)
                p2 = lognorm.cdf(thrd_2, s=sigma, scale=mu)
                slope = (p2 - p1) / (thrd_2 - thrd_1)
                return model.fail_prob[l, t] == p1 + slope * (v - thrd_1)  

        model.Constraint_FragilityCurve = pyo.Constraint(model.Set_lt_lines, rule=fragility_curve_rule)

        # Failure conditions
        def fail_condition_1_rule(model, l, t):  
            return model.fail_prob[l, t] - model.rand_num[l, t] <= model.alpha[l, t]  

        def fail_condition_2_rule(model, l, t):  
            return model.fail_prob[l, t] - model.rand_num[l, t] >= -1 + model.alpha[l, t]  

        model.Constraint_FailCondition1 = pyo.Constraint(model.Set_lt_lines, rule=fail_condition_1_rule)
        model.Constraint_FailCondition2 = pyo.Constraint(model.Set_lt_lines, rule=fail_condition_2_rule)

        # Beta indicator (failure + impact)
        def beta_indicator_1_rule(model, l, t):  
            return model.beta[l, t] <= model.alpha[l, t]  

        def beta_indicator_2_rule(model, l, t):  
            return model.beta[l, t] <= model.impact_flag[l, t]  

        def beta_indicator_3_rule(model, l, t):  
            return model.beta[l, t] >= model.alpha[l, t] + model.impact_flag[l, t] - 1  

        model.Constraint_BetaIndicator1 = pyo.Constraint(model.Set_lt_lines, rule=beta_indicator_1_rule)
        model.Constraint_BetaIndicator2 = pyo.Constraint(model.Set_lt_lines, rule=beta_indicator_2_rule)
        model.Constraint_BetaIndicator3 = pyo.Constraint(model.Set_lt_lines, rule=beta_indicator_3_rule)

        # Failure occurrence
        def fail_occurs_1_rule(model, l, t):
            t_idx = Set_ts.index(t)
            if t_idx == 0:
                return model.fail_occurs[l, t] <= model.beta[l, t]
            else:
                t_prev = Set_ts[t_idx - 1]
                return model.fail_occurs[l, t] <= model.beta[l, t]

        def fail_occurs_2_rule(model, l, t):
            t_idx = Set_ts.index(t)
            if t_idx == 0:
                return model.fail_occurs[l, t] <= 1  # Line starts operational
            else:
                t_prev = Set_ts[t_idx - 1]
                return model.fail_occurs[l, t] <= model.bch_status[l, t_prev]

        def fail_occurs_3_rule(model, l, t):
            t_idx = Set_ts.index(t)
            if t_idx == 0:
                return model.fail_occurs[l, t] >= model.beta[l, t] + 1 - 1
            else:
                t_prev = Set_ts[t_idx - 1]
                return model.fail_occurs[l, t] >= model.beta[l, t] + model.bch_status[l, t_prev] - 1

        model.Constraint_FailOccurs1 = pyo.Constraint(model.Set_lt_lines, rule=fail_occurs_1_rule)
        model.Constraint_FailOccurs2 = pyo.Constraint(model.Set_lt_lines, rule=fail_occurs_2_rule)
        model.Constraint_FailOccurs3 = pyo.Constraint(model.Set_lt_lines, rule=fail_occurs_3_rule)

        # Line status and repair
        def line_status_rule(model, l, t):  
            return model.bch_status[l, t] <= 1 - model.fail_occurs[l, t]  

        model.Constraint_LineStatus = pyo.Constraint(model.Set_lt_lines, rule=line_status_rule)

        def line_repair_rule(model, l, t):
            # Find if repair should occur (TTR timesteps after failure)
            t_idx = Set_ts.index(t)
            ttr_hours = model.ttr[l]

            # Look back TTR hours for a failure
            repair_sum = 0
            for i in range(max(0, t_idx - ttr_hours + 1), t_idx + 1):
                if i <= t_idx - ttr_hours:
                    continue
                t_fail = Set_ts[i]
                if (t - t_fail) == ttr_hours:
                    repair_sum = model.fail_occurs[l, t_fail]
                    break

            return model.repair_applies[l, t] == repair_sum

        model.Constraint_LineRepair = pyo.Constraint(model.Set_lt_lines, rule=line_repair_rule)

        # ------------------------------------------------------------------
        # 6. Objective
        # ------------------------------------------------------------------
        def objective_rule(model):
            # Minimize operational costs (mainly load shedding)
            gen_cost = sum(
                (model.gen_cost_coef[g, 0] + model.gen_cost_coef[g, 1] * model.Pg[g, t])  
                for (g, t) in model.Set_gt  # USE Set_gt
            )

            ls_cost = sum(
                model.Pc_cost[b] * model.Pc[b, t]  
                for (b, t) in model.Set_bt  # USE Set_bt
            )

            if has_reactive_demand:
                ls_cost += sum(
                    model.Qc_cost[b] * model.Qc[b, t]  
                    for (b, t) in model.Set_bt_dn  # USE Set_bt_dn
                )

            rep_cost = sum(
                model.rep_cost[l] * model.repair_applies[l, t]  
                for (l, t) in model.Set_lt_lines  # USE Set_lt_lines
            )

            return gen_cost + ls_cost + rep_cost

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Add expression for EENS calculation
        def eens_dn_rule(model):
            return sum(model.Pc[b, t] for (b, t) in model.Set_bt_dn)

        model.eens_dn_expr = pyo.Expression(rule=eens_dn_rule)

        return model

    def solve_combined_opf_model_under_ws_scenarios(self,
                                                    model,
                                                    solver_name: str = "gurobi",
                                                    mip_gap: float = 1e-3,
                                                    time_limit: int = 60):
        """
        Solve the OPF model and return total EENS at DN level.

        Args:
            model: Pyomo model from build_combined_opf_model_under_ws_scenarios
            solver_name: Solver to use
            mip_gap: MIP gap tolerance
            time_limit: Time limit in seconds

        Returns:
            float: Total EENS at DN level (MWh)
        """

        print(f"Solving OPF model with {solver_name}...")

        # Configure solver
        solver = SolverFactory(solver_name)

        if solver_name == "gurobi":
            solver.options['MIPGap'] = mip_gap
            solver.options['TimeLimit'] = time_limit
            solver.options['OutputFlag'] = 0  # Suppress output
        elif solver_name == "cplex":
            solver.options['mipgap'] = mip_gap
            solver.options['timelimit'] = time_limit

        # Solve
        results = solver.solve(model, tee=False)

        # Check solution status
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"Warning: Solver terminated with status {results.solver.termination_condition}")
            if results.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
                print("Model is infeasible or unbounded - writing LP file for debugging")

                # Write LP file for debugging
                model.write("debug_infeasible_model.lp", io_options={'symbolic_solver_labels': True})
                print("Model written to: debug_infeasible_model.lp")

                # Also compute IIS if using Gurobi
                if solver_name == "gurobi":
                    print("Computing IIS (Irreducible Infeasible Subset)...")
                    import gurobipy as gp

                    # Read the LP file with Gurobi directly
                    m = gp.read("debug_infeasible_model.lp")
                    m.computeIIS()
                    m.write("debug_infeasible_model.ilp")
                    print("IIS written to: debug_infeasible_model.ilp")

                    # Print IIS summary
                    print("\nConstraints in IIS:")
                    for c in m.getConstrs():
                        if c.IISConstr:
                            print(f"  {c.ConstrName}")

                    print("\nVariable bounds in IIS:")
                    for v in m.getVars():
                        if v.IISLB > 0:
                            print(f"  {v.VarName} lower bound")
                        if v.IISUB > 0:
                            print(f"  {v.VarName} upper bound")

        # Extract EENS at DN level
        total_eens_dn = pyo.value(model.eens_dn_expr)

        print(f"Total EENS at DN level: {total_eens_dn:.4f} MWh")

        return total_eens_dn

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

    def set_scaled_demand_profile_for_buses(self, normalized_profile):
        """
        Set demand profiles (both Pd and Qd) scaled from a normalized profile.

        Only sets profiles for buses that don't already have profiles (i.e., None entries).
        This allows bus-specific profiles to be preserved if already set in network_factory.

        Note: If Pd_min and Qd_min are provided, do a linear stretch between min and max values;
              otherwise scale by max values only.
        """
        Pd_max = self.data.net.Pd_max
        Pd_min = self.data.net.Pd_min
        Qd_max = self.data.net.Qd_max
        Qd_min = self.data.net.Qd_min

        # Check if profile_Pd already exists and is a list
        if hasattr(self.data.net, 'profile_Pd') and isinstance(self.data.net.profile_Pd, list):
            # Profile_Pd exists - only fill None entries
            profile_Pd = self.data.net.profile_Pd
            profile_Qd = self.data.net.profile_Qd if hasattr(self.data.net, 'profile_Qd') else []

            # Ensure profile_Qd has the same length as profile_Pd
            while len(profile_Qd) < len(profile_Pd):
                profile_Qd.append(None)

            # Fill in None entries with scaled profiles
            for i in range(len(profile_Pd)):
                if profile_Pd[i] is None:
                    # This bus needs a scaled profile
                    if Pd_min is not None and Qd_min is not None:
                        # Linear stretch between min and max
                        pd_profile = [Pd_min[i] + v * (Pd_max[i] - Pd_min[i]) for v in normalized_profile]
                        qd_profile = [Qd_min[i] + v * (Qd_max[i] - Qd_min[i]) for v in normalized_profile]
                    else:
                        # Scale by max only
                        pd_profile = [v * Pd_max[i] for v in normalized_profile]
                        qd_profile = [v * Qd_max[i] for v in normalized_profile]

                    profile_Pd[i] = pd_profile
                    profile_Qd[i] = qd_profile

        else:
            # No existing profiles - create new ones for all buses
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

    def read_renewable_profile(self, file_path):
        """Read normalized renewable generation profile from given file path into a Python list"""
        try:
            # Read the Excel file
            df = pd.read_excel(file_path, header=0)
            # Extract the generation column (column B, index 1) and convert to list
            # Assuming 8760 hourly values for a year
            renewable_profile = df.iloc[:, 1].tolist()
            return renewable_profile
        except Exception as e:
            print(f"Error reading renewable profile from {file_path}: {str(e)}")
            return None

    def set_renewable_generation_profiles(self):
        """
        Set renewable generation profiles for wind and PV generators.
        Profiles are scaled versions of normalized profiles based on generator capacity.
        """
        # Check if we have renewable generators
        if not hasattr(self.data.net, 'gen_type'):
            print("No generator type information available. Skipping renewable profiles.")
            return

        # Get the time period from windstorm settings
        ws = WindClass()

        # Initialize profile variables to None
        wind_profile_normalized = None
        pv_profile_normalized = None

        # Determine profile length based on period
        if ws.data.MC.lng_prd == 'year':
            default_profile_length = 8760
        elif ws.data.MC.lng_prd == 'month':
            default_profile_length = 720
        elif ws.data.MC.lng_prd == 'week':
            default_profile_length = 168
        else:
            default_profile_length = 8760

        # Read wind profile if we have wind generators
        has_wind = 'wind' in self.data.net.gen_type
        if has_wind:
            if hasattr(self.data.net, 'wind_profile_path'):
                wind_profile_full = self.read_renewable_profile(self.data.net.wind_profile_path)
                if wind_profile_full:
                    # Trim profile based on time period
                    if ws.data.MC.lng_prd == 'year':
                        wind_profile_normalized = wind_profile_full[:8760]
                    elif ws.data.MC.lng_prd == 'month':
                        wind_profile_normalized = wind_profile_full[:720]
                    elif ws.data.MC.lng_prd == 'week':
                        wind_profile_normalized = wind_profile_full[:168]
                    else:
                        wind_profile_normalized = wind_profile_full
                else:
                    # Default to constant profile if file reading fails
                    print("Using default constant wind profile")
                    wind_profile_normalized = [0.3] * default_profile_length
            else:
                # No profile path provided, use default
                print("No wind profile path provided. Using default wind profile")
                wind_profile_normalized = [0.3] * default_profile_length

        # Read PV profile if we have PV generators
        has_pv = 'pv' in self.data.net.gen_type
        if has_pv:
            if hasattr(self.data.net, 'pv_profile_path'):
                pv_profile_full = self.read_renewable_profile(self.data.net.pv_profile_path)
                if pv_profile_full:
                    # Trim profile based on time period
                    if ws.data.MC.lng_prd == 'year':
                        pv_profile_normalized = pv_profile_full[:8760]
                    elif ws.data.MC.lng_prd == 'month':
                        pv_profile_normalized = pv_profile_full[:720]
                    elif ws.data.MC.lng_prd == 'week':
                        pv_profile_normalized = pv_profile_full[:168]
                    else:
                        pv_profile_normalized = pv_profile_full
                else:
                    # Default sinusoidal pattern
                    print("Using default PV profile")
                    pv_profile_normalized = []
                    for hour in range(default_profile_length):
                        hour_of_day = hour % 24
                        if 6 <= hour_of_day <= 18:
                            value = 0.2 * math.sin(math.pi * (hour_of_day - 6) / 12)
                        else:
                            value = 0
                        pv_profile_normalized.append(value)
            else:
                # No profile path provided, use default
                print("No PV profile path provided. Using default PV profile")
                pv_profile_normalized = []
                for hour in range(default_profile_length):
                    hour_of_day = hour % 24
                    if 6 <= hour_of_day <= 18:
                        value = 0.2 * math.sin(math.pi * (hour_of_day - 6) / 12)
                    else:
                        value = 0
                    pv_profile_normalized.append(value)

        # Create scaled profiles for each generator - one profile per generator
        profile_Pg_wind = []
        profile_Pg_pv = []
        profile_Pg_gas = []

        # Create a profile for EVERY generator based on its type
        for i, (gen_bus, gen_type, pg_max) in enumerate(zip(self.data.net.gen,
                                                            self.data.net.gen_type,
                                                            self.data.net.Pg_max)):
            if gen_type == 'wind':
                if wind_profile_normalized:
                    scaled_profile = [v * pg_max for v in wind_profile_normalized]
                else:
                    scaled_profile = [0.3 * pg_max] * default_profile_length
                profile_Pg_wind.append(scaled_profile)

            elif gen_type == 'pv':
                if pv_profile_normalized:
                    scaled_profile = [v * pg_max for v in pv_profile_normalized]
                else:
                    scaled_profile = [0.1 * pg_max] * default_profile_length
                profile_Pg_pv.append(scaled_profile)

            elif gen_type == 'gas':
                profile_length = len(wind_profile_normalized) if wind_profile_normalized else \
                    len(pv_profile_normalized) if pv_profile_normalized else \
                        default_profile_length
                profile_Pg_gas.append([pg_max] * profile_length)

        # Store profiles in data structure
        self.data.net.profile_Pg_wind = profile_Pg_wind if profile_Pg_wind else None
        self.data.net.profile_Pg_pv = profile_Pg_pv if profile_Pg_pv else None
        self.data.net.profile_Pg_gas = profile_Pg_gas if profile_Pg_gas else None

        # Create a combined renewable profile list aligned with generator indices
        profile_Pg_renewable = []
        wind_idx = 0
        pv_idx = 0
        gas_idx = 0

        for gen_type in self.data.net.gen_type:
            if gen_type == 'wind':
                if profile_Pg_wind and wind_idx < len(profile_Pg_wind):
                    profile_Pg_renewable.append(profile_Pg_wind[wind_idx])
                    wind_idx += 1
                else:
                    profile_Pg_renewable.append(None)
            elif gen_type == 'pv':
                if profile_Pg_pv and pv_idx < len(profile_Pg_pv):
                    profile_Pg_renewable.append(profile_Pg_pv[pv_idx])
                    pv_idx += 1
                else:
                    profile_Pg_renewable.append(None)
            elif gen_type == 'gas':
                profile_Pg_renewable.append(None)  # Gas is dispatchable
                if profile_Pg_gas:
                    gas_idx += 1
            else:
                profile_Pg_renewable.append(None)

        self.data.net.profile_Pg_renewable = profile_Pg_renewable

        # print(f"Renewable generation profiles set successfully:")
        # print(f"  - Wind generators: {sum(1 for t in self.data.net.gen_type if t == 'wind')}")
        # print(f"  - PV generators: {sum(1 for t in self.data.net.gen_type if t == 'pv')}")
        # print(f"  - Gas generators: {sum(1 for t in self.data.net.gen_type if t == 'gas')}")