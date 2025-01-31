

import pyomo.environ as pyo
from config import *
from windstorm import *
from network_linear import *


# Create instance for the windstorm and network class
ws = WindClass()
net = NetworkClass()

# Define a concrete model
model = pyo.ConcreteModel()

# Define sets:
model.Set_bus = pyo.Set(initialize=net.data.net.bus)
model.Set_bch = pyo.Set(initialize=range(1, len(net.data.net.bch) + 1))
model.Set_gen = pyo.Set(initialize=range(1, len(net.data.net.gen) + 1))
model.Set_timestep = pyo.Set(initialize=ws._get_num_hrs_prd())

# Load scenario data:


# Define parameters:
model.Demand = pyo.Param(model.Buses, model.T, initialize=DemandData)  # Load profiles
model.GenerationCost = pyo.Param(model.Buses, initialize=GenCost)  # Gen cost at buses
model.InvestmentCost = pyo.Param(model.Branches, initialize=InvCost)  # Line hardening cost
model.WindstormScenarios = pyo.Param(model.Branches, model.T, initialize=WindstormData)
model.RandomNumbers = pyo.Param(model.Branches, model.T, initialize=RandomData)
