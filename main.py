

import pyomo as pyo
from config import *
from windstorm import *
from network_linear import *



# Define a concrete model
model = pyo.ConcreteModel()

# Define sets
model.Set_bus = pyo.Set(initialize=self.data.net.bus)
model.Set_bch = pyo.Set(initialize=range(1, len(self.data.net.bch) + 1))
model.Set_gen = pyo.Set(initialize=range(1, len(self.data.net.gen) + 1))