# This script is to test power network features (e.g., network modelling and DC power flows)

from core.config import NetConfig
from core.network import NetworkClass
from factories.network_factory import make_network

net = make_network("29_bus_GB_transmission_network_with_Kearsley_GSP_group")

model = net.build_dc_opf_model()
results = net.solve_dc_opf(model, solver='gurobi')








