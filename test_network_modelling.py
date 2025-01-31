# This script is to test power network features (e.g., network modelling and DC power flows)

import numpy as np

from config import NetConfig
from network_linear import NetworkClass

net_config = NetConfig()

net = NetworkClass(net_config)

model = net.build_dc_opf_model()
results = net.solve_dc_opf(model)








