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
        DP_file_path = "Demand_Profile/normalized_hourly_demand_profile_year.xlsx"
        df = pd.read_excel(DP_file_path, header=0)  # ignore any header row
        if ws.data.MC.lng_prd == 'year':
            normalized_profile = df.iloc[:, 0].tolist()  # Extract the whole column (1 year) and convert to list
        elif ws.data.MC.lng_prd == 'month':
            normalized_profile = df.iloc[0:720, 0].tolist()  # Extract the first month
        elif ws.data.MC.lng_prd == 'week':
            normalized_profile = df.iloc[0:168, 0].tolist()  # Extract the first week
        self.set_scaled_profile_for_buses(normalized_profile)


    def build_dc_opf_model(self):
        """The model that calculates DC optimal power flow based on the pyomo optimisation package"""
        # Define a concrete model
        model = pyo.ConcreteModel()

        # 1. Sets (indices for buses, branches, generators):
        model.Set_bus = pyo.Set(initialize=self.data.net.bus)
        model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch)+1))
        model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen)+1))

        ws = WindClass()
        model.Set_ts = pyo.Set(initialize=range(ws._get_num_hrs_prd()))  # Timesteps in a period

        # 2. Parameters:
        model.demand = pyo.Param(model.Set_bus, initialize={b + 1: self.data.net.max_demand_active[b]
                                                            for b in range(len(self.data.net.max_demand_active))})

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
            return sum(
                sum(model.gen_cost_coef[g, c] * (model.P_gen[g] ** c)
                    for c in range(len(self.data.net.gen_cost_coef[0])))
                for g in model.Set_gen
            )

        model.Objective_MinimiseTotalGenCost = pyo.Objective(rule=Objective_rule, sense=1)

        return model


    def solve_dc_opf(self, model, solver='glpk'):
        """Solve the DC OPF model"""
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


    def read_normalized_profile(self, file_path):
        """Read normalized demand profile from given file path into a Python list"""
        # read into a pandas dataframe
        demand_profile_df = pd.read_excel(file_path)
        # convert it into a python list
        demand_profile = demand_profile_df.iloc[:, 0].tolist()

        return demand_profile


    def set_scaled_profile_for_buses(self, normalized_profile):
        """Set scaled demand profiles for each bus based on the normalized profile"""
        max_demands = self.data.net.max_demand_active
        bus_profiles = []
        # Loop over each bus and scale the profile
        for _, max_demand in enumerate(max_demands):
            # Scale the normalized profile by the maximum demand
            scaled_profile = [value * max_demand for value in normalized_profile]
            bus_profiles.append(scaled_profile)
        # Set profiles into the instance
        self.data.net.demand_profile_active = bus_profiles


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