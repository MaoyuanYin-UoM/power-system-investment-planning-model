# This script contains constants, filepaths etc.

import numpy as np

# Create an simple object class as a placeholder (will be used later for storing data)
class Object(object):
    pass

# -------------------- Power System Configurations --------------------

# upper and lower limits of longitude and latitude for randomly generated substation coordinates
LIM_LON1 = -5
LIM_LON2 = -1
LIM_LAT1 = 50
LIM_LAT2 = 55

class NetConfig:
    # Define the parameters for the network model
    def __init__(self) -> object:
        # create object for data storage
        self.data = Object()

        # data.net stores network model parameters
        self.data.net = Object()

        # 1) bus data
        self.data.net.bus = [1, 2, 3, 4, 5]  # bus No.
        self.data.net.slack_bus = 3  # select slack bus
        self.data.net.bus_coords = None
        self.data.net.demand_active = [300, 200, 0, 150, 0]
        self.data.net.demand_reactive = [0, 0, 0, 0, 0]  # unused for DC power flow

        # 2) branch data
        self.data.net.bch = [[1,2],[1,4],[1,5],[2,3],[3,4],[4,5]]  # branch indicated by its start and end bus
        self.data.net.bch_R = [0.00281,0.00304,0.00064,0.00108,0.00297,0.00297]
        self.data.net.bch_X = [0.00281,0.00304,0.00064,0.00108,0.00297,0.00297]
        self.data.net.bch_cap = [400, 400, 400, 400, 400, 400]

        # 3) generator data
        self.data.net.gen = [1, 2, 3, 4, 5]  # which bus the gen is attached to
        self.data.net.gen_active_max = [200, 150, 300, 100, 250]
        self.data.net.gen_active_min = [0, 0, 0, 0, 0]
        self.data.net.gen_reactive_max = [0, 0, 0, 0, 0]  # unused for DC power flow
        self.data.net.gen_reactive_min = [0, 0, 0, 0, 0]  # unused for DC power flow
        # self.data.net.gen_cost_model = 2
        self.data.net.gen_cost_coef = [[10, 0], [15, 0], [20, 0], [25, 0], [5, 0]]  # coefficients for the generation
                                                                                     # cost function
                                                                                     # e.g., for coefficient [a, b, c]:
                                                                                     # gen_cost = a*x^2 + b*x + c


# -------------------- Windstorm Configurations --------------------

class WindConfig:
    # Define parameters for windstorm generation
    def __init__(self):
        # create data as an object for data storage (and same below)
        self.data = Object()

        self.data.num_hr_year = 8760  # set constants
        self.data.num_hr_week = 144
        self.data.num_hr_day = 24

        # data.WS stores parameters of the windstorm
        self.data.WS = Object()

        self.data.WS.event = Object()
        self.data.WS.event.max_num_year = 3  # maximum number of windstorms per year
        self.data.WS.event.max_v = [20, 55]  # lower and upper bounds for peak gust speed
        self.data.WS.event.min_v = [15, 20]
        self.data.WS.event.lng = [4, 48]  # lower and upper bounds for windstorm duration
        self.data.WS.event.ttr = [24, 168]  # lower and upper bounds for line repair (time to restoration)

        self.data.WS.contour = Object()
        self.data.WS.contour.start_lon = [-2.0, -3.3, -3.3, -4.8, -4.8, -3.2, -2.2,
                                     -5.4, -3.2, -5.4, -5.4, 0.4]  # Longitude for starting-point contour
        self.data.WS.contour.start_lat = [55.8, 55.0, 53.5, 53.5, 52.8, 52.8, 52.1,
                                     51.8, 51.3, 50.6, 49.9, 50.4]  # Latitude for starting-point contour
        self.data.WS.contour.start_connectivity = \
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
             [9, 10], [10, 11], [11, 12]]  # Connectivity for starting contour plot

        self.data.WS.contour.end_lon = [-1.5, 2.1]  # Longitude for ending-point contour
        self.data.WS.contour.end_lat_coef = [-17/18, 54.183333]  # Coefficients for a linear relation between
                                                            # end_lon and end_lat (e.g., y = ax + b)
                                                            # --> end_lat = coef[0] * end_lon + coef[1]
        # data.MC stores parameters of Monte Carlo
        self.data.MC = Object()
        self.data.MC.num_trials = 1

        # data.frg stores fragility curve information
        self.data.frg = Object()
        self.data.frg.mu = 3.8
        self.data.frg.sigma = 0.122
        self.data.frg.thrd_1 = 20
        self.data.frg.thrd_2 = 90
        self.data.frg.shift_f = 0


