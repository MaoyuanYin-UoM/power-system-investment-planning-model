# This script contains constants, filepaths etc.

import numpy as np

# Create an simple object class as a placeholder (will be used later for storing data)
class Object(object):
    pass

# -------------------- Power Network Configurations --------------------

class NetConfig:
    # Define the parameters for the network model
    def __init__(self) -> object:
        # create object for data storage
        self.data = Object()

        # data.net stores network model parameters
        self.data.net = Object()

        # 1) Base values
        self.data.net.base_MVA = None
        self.data.net.base_kV = None

        # 2) Bus data
        self.data.net.bus = [1, 2, 3, 4, 5]  # bus No.
        self.data.net.slack_bus = 4  # select slack bus
        self.data.net.theta_limits = [-0.5, 0.5]  # allowed lower and upper limits for voltage angle
        self.data.net.V_min = None
        self.data.net.V_min = None
        self.data.net.Pd_max = [500, 400, 300, 350, 500]  # maximum active demand at each bus (in MW / MVAr)
        self.data.net.Qd_max = [0, 0, 0, 0, 0]  # maximum reactive demand at each bus (in MW / MVAr)
        self.data.net.Pc_cost = None
        self.data.net.Qc_cost = None

        self.data.net.profile_Pd = None
        self.data.net.profile_Qd = None
        self.data.net.bus_lon = [-2, -3, -1, -3, -1]  # longitudes of buses
        self.data.net.bus_lat = [55, 53, 54, 51, 51]  # latitudes of buses
        self.data.net.all_bus_coords_in_tuple = None
        self.data.net.bch_gis_bgn = None
        self.data.net.bch_gis_end = None

        # 3) Branch data
        self.data.net.bch = [[1,2],[1,3],[2,3],[2,5],[3,5],[4,5]]  # branch indicated by its start and end bus
        self.data.net.bch_R = [0.00281,0.00304,0.00064,0.00108,0.00297,0.00297]
        self.data.net.bch_X = [0.00281,0.00304,0.00064,0.00108,0.00297,0.00297]
        self.data.net.bch_Smax = None
        self.data.net.bch_Pmax = [400, 400, 400, 400, 400, 400]  # active power flow capacity of each branch (in MW / MVAr)


        # 4) Generator data
        self.data.net.gen = [1, 2, 3, 4, 5]  # which bus the gen is attached to
        self.data.net.Pg_max = [400, 300, 300, 350, 500]  # (Pg/Qg in MW / MVAr)
        self.data.net.Pg_min = [0, 0, 0, 0, 0]
        self.data.net.Qg_max = [0, 0, 0, 0, 0]
        self.data.net.Qg_min = [0, 0, 0, 0, 0]
        # self.data.net.gen_cost_model = 2
        self.data.net.gen_cost_coef = [[0, 10], [0, 15], [0, 20], [0, 25], [0, 15]]
            # coefficients for the generation cost function e.g., for coefficient [a, b, c]: gen_cost = a + b*x + c*x^2
            # Note that all elements (lists) contained in "gen_cost_coef" should have same length

        # 5) Fragility data
        self.data.frg = Object()
        self.data.frg.mu = 3.8
        self.data.frg.sigma = 0.122
        self.data.frg.thrd_1 = 20
        self.data.frg.thrd_2 = 90
        self.data.frg.shift_f = 0



# -------------------- Windstorm Configurations --------------------

class WindConfig:
    # Define parameters for windstorm generation
    def __init__(self):
        # # create object for data storage
        self.data = Object()

        self.data.num_hrs_year = 8760  # set constants
        self.data.num_hrs_week = 144
        self.data.num_hrs_day = 24


        # 1. data.WS stores parameters related to windstorms
        self.data.WS = Object()

        # 1) parameters of windstorm event generation:
        self.data.WS.event = Object()
        self.data.WS.event.max_num_ws_prd = 1  # maximum number of windstorms per period
        self.data.WS.event.max_v = [40, 60]  # upper and lower bounds for initial gust speed
        self.data.WS.event.min_v = [25, 35]  # upper and lower bounds for final gust speed
        self.data.WS.event.max_r = [20, 25]  # upper and lower bounds for initial radius
        self.data.WS.event.min_r = [15, 10]  # upper and lower bounds for final radius
        self.data.WS.event.max_prop_v = [22, 26]  # upper and lower bounds for initial windstorm propagation speed
        self.data.WS.event.min_prop_v = [8, 10]  # upper and lower bounds for final windstorm propagation speed

        self.data.WS.event.lng = [12, 48]  # lower and upper bounds for windstorm duration
        self.data.WS.event.ttr = [24, 120]  # lower and upper bounds for line repair (time to restoration)

        # if the 'constant_weibull' gust speed model is used
        self.data.WS.event.gust_model = 'constant_weibull'
        self.data.WS.event.gust_weibull_shape = 2.0  # k
        self.data.WS.event.gust_weibull_scale = 30.0  # λ (m/s)

        # if the 'constant_uniform' radius model is used
        self.data.WS.event.r_model = 'constant_uniform'
        self.data.WS.event.r_uniform_bounds = [20, 25]  # km

        # if the 'constant_uniform' radius model is used
        self.data.WS.event.prop_model = 'constant_uniform'
        self.data.WS.event.prop_uniform_bounds = [22, 26]  # km/h

        # 2) define the contours where windstorms start and end (it impacts windstorms' path):
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
        # 3) define the properties of the windstorm
        self.data.WS.init_radius = 20  # radius in km
        self.data.WS.final_radius = 10  # radius in km
        self.data.WS.init_propagation_speed = 24  # speed in km/h
        self.data.WS.final_propagation_speed = 8  # speed in km/h


        # 2. data.MC stores parameters related to Monte Carlo simulation
        self.data.MC = Object()

        self.data.MC.num_prds = 10  #  number of periods (i.e., number of monte carlo simulations)
        self.data.MC.lng_prd = 'year'  # select which period (year, month, week) to be used for MC simulation

        self.data.MC.prd_to_hrs = {  # define the mapping between periods name to the number of hours
            "year": 8760,
            "month": 720,
            "week": 168
        }



# -------------------- Investment Model Configurations --------------------

class InvestmentConfig:
    def __init__(self):
        # create object for data storage
        self.data = Object()

        # upper and lower limit for line hardening amount (in m/s)
        self.data.bch_hrdn_limits = [0, 20]

        # hardening cost for each branch to shift the fragility curve for per m/s wind speed
        self.data.cost_bch_hrdn = [500, 500, 500, 500, 500, 500]

        # repair cost for each branch
        self.data.cost_bch_rep = [1000, 1000, 1000, 1000, 1000, 1000]

        # total budget for line hardening
        self.data.budget_bch_hrdn = 20000

        # load shedding cost per kW demand per hour at each bus
        self.data.cost_bus_ls = [100, 100, 100, 100, 100]