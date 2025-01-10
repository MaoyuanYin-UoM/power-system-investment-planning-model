# This script is to test features (e.g., network modelling and power flows) from the mes-tutorial project

import numpy as np
from mes_features import Integrated_networks_modelling_Electricity
from mes_features.Multi_energy_districts_Model import Elec_Model

# Electric network model from MATPOWER case5
Net = {}
Net['Connectivity'] = np.array([[1,2],[1,4],[1,5],[2,3],[3,4],[4,5]])
Net['R'] = [0.00281,0.00304,0.00064,0.00108,0.00297,0.00297]
Net['X'] = [0.00281,0.00304,0.00064,0.00108,0.00297,0.00297]

Net['Demand_Active'] = [0, 0, 0, 0, 0]
Net['Demand_Reactive'] = [0, 0, 0, 0, 0]
Net['Generation_Active'] = [0, 0, 0, 0, 0]
Net['Generation_Reactive'] = [0, 0, 0, 0, 0]

Net['Generation_Capacity_Active'] = [40, 170, 323.49, 0, 466.51]
Net['Generation_Capacity_Reactive'] = [0, 0, 0, 0, 0]
# the format of generation cost aligns with the polynomial cost model in MATPOWER case struct
Net['Generation_Cost'] = [[2, 14, 0], [2, 15, 0], [2, 30, 0], [2, 40, 0], [2, 10, 0]]

Net['Slack_Bus'] = 4

model = Elec_Model(Net)
model.run()
model.display()