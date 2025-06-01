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
        model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch)+1))
        model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen)+1))

        # 2. Parameters:
        model.Pd = pyo.Param(model.Set_bus, initialize={b + 1: self.data.net.Pd_max[b]
                                                            for b in range(len(self.data.net.Pd_max))})

        model.Pg_max = pyo.Param(model.Set_gen, initialize={i+1: g for i, g in
                                                                            enumerate(self.data.net.Pg_max)})

        model.Pg_min = pyo.Param(model.Set_gen, initialize={i+1: g for i, g in
                                                                            enumerate(self.data.net.Pg_min)})


        model.bch_Pmax = pyo.Param(model.Set_bch, initialize={i+1: bc for i, bc in
                                                                     enumerate(self.data.net.bch_Pmax)})

        # calculate susceptance B for each line (under the assumption of DC power flow)
        model.bch_B = pyo.Param(model.Set_bch, initialize={i+1: 1/X for i, X in
                                                                   enumerate(self.data.net.bch_X)})

        # model.gen_cost_model = pyo.Param(model.Set_gen, initialize=self.data.net.gen_cost_model)

        model.gen_cost_coef = pyo.Param(model.Set_gen, range(len(self.data.net.gen_cost_coef[0])),
                                        initialize={
                                            (i + 1, j): self.data.net.gen_cost_coef[i][j]
                                            for i in range(len(self.data.net.gen_cost_coef))
                                            for j in range(len(self.data.net.gen_cost_coef[i]))})

        model.Pext_cost = pyo.Param(initialize=self.data.net.Pext_cost)
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

        model.Pext = pyo.Var(within=pyo.NonNegativeReals)  # cost of grid import/export


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

            # if this is the slack bus, add the grid import/export 'Pext'
            if bus_idx == self.data.net.slack_bus:
                Pg_sum = Pg_sum + model.Pext

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
            return Pg_sum + Pf_in_sum - Pf_out_sum == model.Pd[bus_idx] - model.Pc[bus_idx]
        model.Constraint_PowerBalance = pyo.Constraint(model.Set_bus, rule=power_balance_rule)


        # 2) Line flow constraints
        def line_flow_rule(model, line_idx):
            i, j = self.data.net.bch[line_idx-1]
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

            grid_exit_cost = model.Pext_cost * model.Pext

            total_ls_cost = sum(model.Pc_cost[b] * model.Pc[b] for b in model.Set_bus)

            return total_gen_cost + grid_exit_cost + total_ls_cost

        model.Objective_MinimiseTotalCost = pyo.Objective(rule=objective_rule, sense=1)

        return model


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

        # - cost for active load shedding
        model.Pc_cost = pyo.Param(model.Set_bus,
                             initialize={b + 1: self.data.net.Pc_cost[b] for b in range(len(self.data.net.Pc_cost))})


        # 3) Variables
        model.Pg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.NonNegativeReals)  # active generation
        model.Qg = pyo.Var(model.Set_gen, model.Set_ts, within=pyo.Reals)  # reactive generation
        model.Pf = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Reals)  # active power flow
        model.Qf = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.Reals)  # reactive power flow
        model.Pc = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # curtailed active demand
        model.Qc = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # curtailed reactive demand
        model.V2 = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)  # squared voltage
        model.I2 = pyo.Var(model.Set_bch, model.Set_ts, within=pyo.NonNegativeReals)  # squared current


        # 4) Constraints
        # 4.1) Active power balance at each bus
        def P_balance_at_bus_rule(model, b, t):
            Pg = sum(model.Pg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            Pf_in = sum(model.Pf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][1] == b)
            Pf_out = sum(model.Pf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][0] == b)
            return Pf_in - Pf_out + Pg == model.Pd[b, t] - model.Pc[b, t]

        model.Constraint_ActivePowerBalance = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                             rule=P_balance_at_bus_rule)

        # 4.2) Reactive‐power balance at each bus
        def Q_balance_at_bus_rule(model, b, t):
            Qg = sum(model.Qg[g, t] for g in model.Set_gen if self.data.net.gen[g - 1] == b)
            Qf_in = sum(model.Qf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][1] == b)
            Qf_out = sum(model.Qf[l, t] for l in model.Set_bch if self.data.net.bch[l - 1][0] == b)
            return Qf_in - Qf_out + Qg == model.Qd[b, t] - model.Qc[b, t]

        model.Constraint_ReactivePowerBalance = pyo.Constraint(model.Set_bus, model.Set_ts,
                                                               rule=Q_balance_at_bus_rule)

        # 4.3) Voltage drop constraints
        def volt_drop_rule(model, l, t):
            i, j = self.data.net.bch[l - 1]
            return (model.V2[i, t] - model.V2[j, t]
                    == 2 * (model.R[l] * model.Pf[l, t] + model.X[l] * model.Qf[l, t])
                    - (model.R[l] ** 2 + model.X[l] ** 2) * model.I2[l, t])

        model.Constraint_VoltageDrop = pyo.Constraint(model.Set_bch, model.Set_ts, rule=volt_drop_rule)

        # 4.4) Current‐flow relation
        def current_flow_rule(model, l, t): 
            i, j = self.data.net.bch[l - 1]
            # Pf[l,t]^2 + Qf[l,t]^2 == I2[l,t] * V2[i,t]
            return model.Pf[l, t] ** 2 + model.Qf[l, t] ** 2 == model.I2[l, t] * model.V2[i, t]

        model.Constraint_CurrentFlow = pyo.Constraint(
            model.Set_bch, model.Set_ts,
            rule=current_flow_rule
        )

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


        # # 5) Objective – Minimizing total generation cost
        # def objective_rule(model):  # --> deprecated as no penalty for reactive load shedding
        #     total_gen_cost = sum(
        #         sum(model.gen_cost_coef[g, c] * (model.Pg[g, t] ** c)
        #             for c in range(len(self.data.net.gen_cost_coef[0])))
        #         for g in model.Set_gen
        #         for t in model.Set_ts
        #     )
        #
        #     total_ls_cost = sum(
        #         model.Pc_cost[b] * model.Pc[b, t]
        #         for b in model.Set_bus
        #         for t in model.Set_ts
        #     )
        #
        #     return total_gen_cost + total_ls_cost
        #
        # model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


        # ==================================================================
        # 5) Objective – Minimizing total generation cost

        # Objective #1: minimize total reactive load shedding (to ensure no excessive reactive load shedding)
        model.Obj1_min_Qc = pyo.Objective(
            expr=sum(model.Qc[b, t]
                     for b in model.Set_bus
                     for t in model.Set_ts),
            sense=pyo.minimize
        )
        model.Obj1_min_Qc.deactivate()

        # Objective #2: minimize generation + active‐shed cost
        model.Obj2_min_total_cost = pyo.Objective(
            expr=sum(
                sum(model.gen_cost_coef[g, c] * (model.Pg[g, t] ** c)
                    for c in range(len(self.data.net.gen_cost_coef[0])))
                for g in model.Set_gen for t in model.Set_ts
            )
                 + sum(model.Pc_cost[b] * model.Pc[b, t]
                       for b in model.Set_bus for t in model.Set_ts),
            sense=pyo.minimize
        )
        model.Obj2_min_total_cost.deactivate()

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


    def solve_linearized_ac_opf(self, model, solver='gurobi'):
        """Solve the linearized AC OPF model"""
        solver = SolverFactory(solver)

        # 1) Solve the OPF
        # Note: two-stage solving is used to avoid unrealistic reactive load sheddings while keep the objective values
        # meaningful
        # 1.1) Phase-1 solve: minimize Qc
        model.Obj1_min_Qc.activate()
        results = solver.solve(model, tee=False)
        Qc_min = sum(pyo.value(model.Qc[b, t])
                     for b in model.Set_bus
                     for t in model.Set_ts)

        # 1.2) Fix reactive‐shed total to its minimum
        model.Constraint_QcFix = pyo.Constraint(
            expr=sum(model.Qc[b, t] for b in model.Set_bus for t in model.Set_ts)
                 == Qc_min
        )

        # 1.3) Phase-2 solve: minimize actual cost
        model.Obj1_min_Qc.deactivate()
        model.Obj2_min_total_cost.activate()

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


    def read_normalized_profile(self, file_path):
        """Read normalized demand profile from given file path into a Python list"""
        # read into a pandas dataframe
        demand_profile_df = pd.read_excel(file_path)
        # convert it into a python list
        demand_profile = demand_profile_df.iloc[:, 0].tolist()

        return demand_profile


    def set_scaled_profile_for_buses(self, normalized_profile):
        """Set scaled demand profiles for each bus based on the normalized profile"""
        max_demands = self.data.net.Pd_max
        bus_profiles = []
        # Loop over each bus and scale the profile
        for _, max_demand in enumerate(max_demands):
            # Scale the normalized profile by the maximum demand
            scaled_profile = [value * max_demand for value in normalized_profile]
            bus_profiles.append(scaled_profile)
        # Set both active and reactive demand profiles into the instance
        # (assumes identical profiles for both Pd and Qd)
        self.data.net.profile_Pd = bus_profiles
        self.data.net.profile_Qd = bus_profiles


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
            gis_bgn.append( self.data.net.all_bus_coords_in_tuple[ bch[b][0] - 1 ] )
            gis_end.append( self.data.net.all_bus_coords_in_tuple[ bch[b][1] - 1 ] )

        self.data.net.bch_gis_bgn = gis_bgn
        self.data.net.bch_gis_end = gis_end


    def _get_bch_gis_bgn(self):
        """Get bch_gis_bgn"""
        return self.data.net.bch_gis_bgn

    def _get_bch_gis_end(self):
        """Get bch_gis_end"""
        return self.data.net.bch_gis_end