
from config import NetConfig
from network_linear import NetworkClass

def make_network(name: str) -> NetworkClass:

    # build the right config object
    ncon = NetConfig()  # initialize the 'ncon' objective as the default

    if name == 'matpower_case22':
        # 1) Base values
        ncon.data.net.base_MVA = 1.0           # mpc.baseMVA
        ncon.data.net.base_kV  = 11.0          # mpc.bus(:,BASE_KV)

        # 2) Bus data
        ncon.data.net.bus = list(range(1, 23))  # bus indices
        ncon.data.net.slack_bus = 1  # slack bus

        ncon.data.net.bus_lon = [0, 1, 1, 2, 2, 2,  2,  2, 3, 3, 4, 4, 4,  4,  4,  5, 5, 5, 6, 7,  7,  7]
        ncon.data.net.bus_lat = [2, 2, 1, 2, 1, 0, -1, -2, 2, 1, 2, 1, 0, -1, -2, -1, 0, 1, 0, 0, -1, -2]

        # 3) Active/Reactive Demand (Pd/Qd in MW / MVAr)
        ncon.data.net.Pd_max = [
            0.0,
            16.78/1000, 16.78/1000, 33.8/1000, 14.56/1000, 10.49/1000,
            8.821/1000, 14.35/1000, 19.31/1000, 14.35/1000, 16.27/1000,
            16.27/1000, 82.13/1000, 34.71/1000, 34.71/1000, 80.31/1000,
            49.62/1000, 49.62/1000, 43.77/1000, 37.32/1000, 37.32/1000,
            31.02/1000
        ]
        ncon.data.net.Qd_max = [
            0.0,
            20.91/1000, 20.91/1000, 37.32/1000, 12.52/1000, 14.21/1000,
            11.66/1000, 18.59/1000, 25.87/1000, 18.59/1000, 19.48/1000,
            19.48/1000, 71.65/1000, 30.12/1000, 30.12/1000, 70.12/1000,
            47.82/1000, 47.82/1000, 38.93/1000, 35.96/1000, 35.96/1000,
            29.36/1000
        ]

        # 4) Gen data
        ncon.data.net.gen_bus  = [1]
        ncon.data.net.Pg_min   = [0]
        ncon.data.net.Pg_max   = [10]
        ncon.data.net.Qg_min   = [-10]
        ncon.data.net.Qg_max   = [10]
        ncon.data.net.gen_cost_coef  = [[0, 20]]

        # 5) Branch data (p.u.)
        #    orig. Ohm→p.u. by dividing by Zbase=(11kV)^2/1MVA=121 Ω
        ncon.data.net.bch = [
            [1,2], [2,3], [2,4], [4,5], [4,9],
            [5,6], [6,7], [6,8], [9,10],[9,11],
            [11,12],[11,13],[13,14],[14,15],[14,16],
            [16,17],[17,18],[17,19],[19,20],[20,21],
            [20,22]
        ]
        ncon.data.net.bch_R = [
            0.3664/121, 0.0547/121, 0.5416/121, 0.1930/121, 0.7431/121,
            1.3110/121, 0.0598/121, 0.2905/121, 0.0547/121, 0.6750/121,
            0.0547/121, 0.3942/121, 1.0460/121, 0.0220/121, 0.0547/121,
            0.3212/121, 0.0949/121, 0.5740/121, 0.1292/121, 0.0871/121,
            0.5329/121
        ]
        ncon.data.net.bch_X = [
            0.1807/121, 0.0282/121, 0.2789/121, 0.0990/121, 0.3827/121,
            0.6752/121, 0.0308/121, 0.1496/121, 0.0282/121, 0.3481/121,
            0.0282/121, 0.2030/121, 0.5388/121, 0.0116/121, 0.0282/121,
            0.1654/121, 0.0488/121, 0.2959/121, 0.0660/121, 0.0450/121,
            0.2744/121
        ]

        # thermal (apparent‐power) limits [MW] – set to 1 MVA per branch by default
        # Note that no branch power rating values are specified, 1 MVA is large enough that is guaranteed never to bind
        ncon.data.net.bch_Smax = [1] * len(ncon.data.net.bch)

        # 6) Voltage limits (squared, for DistFlow‐style v = V^2)
        V1 = 1.0**2
        ncon.data.net.V_min = [V1] + [0.9**2] * 21
        ncon.data.net.V_max = [V1] + [1.1**2] * 21

    else:
        raise ValueError(f"Unknown network preset '{name}'")

    return NetworkClass(ncon)