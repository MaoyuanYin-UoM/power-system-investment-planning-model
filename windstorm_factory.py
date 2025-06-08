
from config import WindConfig
from windstorm import WindClass

def make_windstorm(name: str) -> WindClass:

    wcon = WindConfig()

    if name == 'windstorm_UK_transmission_network':
        # starting-point contour:
        wcon.data.WS.contour.start_lon = [-2.0, -3.3, -3.3, -4.8, -4.8, -3.2, -2.2,
                                          -5.4, -3.2, -5.4, -5.4, 0.4]
        wcon.data.WS.contour.start_lat = [55.8, 55.0, 53.5, 53.5, 52.8, 52.8, 52.1,
                                          51.8, 51.3, 50.6, 49.9, 50.4]
        wcon.data.WS.contour.start_connectivity = \
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
             [9, 10], [10, 11], [11, 12]]

        # ending-point contour:
        wcon.data.WS.contour.end_lon = [-1.5, 2.1]
        wcon.data.WS.contour.end_lat_coef = [-17 / 18, 54.183333]

        # windstorm parameters:
        wcon.data.MC.num_prds = 3  # number of periods (i.e., simulations)
        wcon.data.WS.event.max_num_ws_prd = 1  # maximum number of windstorms per period

        wcon.data.WS.event.max_v = [40, 60]  # upper and lower bounds for initial gust speed
        wcon.data.WS.event.min_v = [25, 35]  # upper and lower bounds for final gust speed
        wcon.data.WS.event.max_r = [20, 25]  # upper and lower bounds for initial radius
        wcon.data.WS.event.min_r = [15, 10]  # upper and lower bounds for final radius
        wcon.data.WS.event.max_prop_v = [22, 26]  # upper and lower bounds for initial windstorm propagation speed
        wcon.data.WS.event.min_prop_v = [8, 10]  # upper and lower bounds for final windstorm propagation speed
        wcon.data.WS.event.lng = [12, 48]  # lower and upper bounds for windstorm duration
        wcon.data.WS.event.ttr = [24, 120]  # lower and upper bounds for line repair (time to restoration)

        # # fragility modelling:  (deprecated) (fragility data moved to NetConfig)
        # wcon.data.frg.mu = 3.8
        # wcon.data.frg.sigma = 0.122
        # wcon.data.frg.thrd_1 = 20
        # wcon.data.frg.thrd_2 = 90
        # wcon.data.frg.shift_f = 0


    elif name == "windstorm_1_matpower_case22":

        # starting-point contour:
        wcon.data.WS.contour.start_lon = [-2, 0]
        wcon.data.WS.contour.start_lat = [0, 5]
        wcon.data.WS.contour.start_connectivity = [
            [1, 2]
        ]

        # ending-point contour:
        wcon.data.WS.contour.end_lon = [8, 10]
        wcon.data.WS.contour.end_lat_coef = [3, -28.5]

        # windstorm parameters:
        wcon.data.MC.num_prds = 10  # number of periods (i.e., simulations)
        wcon.data.WS.event.max_num_ws_prd = 1  # maximum number of windstorms per period

        wcon.data.WS.event.max_v = [40, 60]  # upper and lower bounds for initial gust speed
        wcon.data.WS.event.min_v = [25, 35]  # upper and lower bounds for final gust speed
        wcon.data.WS.event.max_r = [20, 25]  # upper and lower bounds for initial radius
        wcon.data.WS.event.min_r = [15, 10]  # upper and lower bounds for final radius
        wcon.data.WS.event.max_prop_v = [22, 26]  # upper and lower bounds for initial windstorm propagation speed
        wcon.data.WS.event.min_prop_v = [8, 10]  # upper and lower bounds for final windstorm propagation speed
        wcon.data.WS.event.lng = [12, 48]  # lower and upper bounds for windstorm duration
        wcon.data.WS.event.ttr = [24, 120]  # lower and upper bounds for line repair (time to restoration)

        # fragility modelling:
        wcon.data.frg.mu = 3.8
        wcon.data.frg.sigma = 0.122
        wcon.data.frg.thrd_1 = 20
        wcon.data.frg.thrd_2 = 90
        wcon.data.frg.shift_f = 0


    else:
        raise ValueError(f"Unknown windstorm preset '{name}'")


    return WindClass(wcon)

