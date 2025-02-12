

from config import *
from utils import *
from visualization import *
from windstorm import *
from network_linear import *
from investment_model import *


wcon = WindConfig()
ws = WindClass(wcon)

ncon = NetConfig()
net = NetworkClass(ncon)

icon = InvestmentConfig()
inv = InvestmentClass(icon)

model = inv.build_investment_model()
inv.solve_investment_model(model, solver='gurobi')