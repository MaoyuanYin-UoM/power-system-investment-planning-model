# This script is to test power network features (e.g., network modelling and DC power flows)

from core.config import NetConfig
from core.network import NetworkClass
from factories.network_factory import make_network

net = make_network("29_bus_GB_transmission_network_with_Kearsley_GSP_group")

model = net.build_combined_dc_linearized_ac_opf_model(timesteps=list(range(24)))
results = net.solve_combined_dc_linearized_ac_opf(model,
                                                  solver='gurobi',
                                                  mip_gap=1e-4,
                                                  mip_gap_abs=1e3
                                                  )








