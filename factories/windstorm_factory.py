
from core.config import WindConfig
from core.windstorm import WindClass, Object


def make_windstorm(name: str) -> WindClass:

    wcon = WindConfig()

    if name == 'windstorm_GB_transmission_network':
        # starting-point contour:
        wcon.data.WS.contour.start_lon = [-2.0, -3.3, -3.3, -4.8, -4.8, -3.2, -2.2, -5.4, -3.4, -5.4, -5.4, -2.0]
        wcon.data.WS.contour.start_lat = [55.6, 55.0, 54, 53.5, 52.8, 52.8, 52.1, 51.8, 51.2, 50.6, 49.9, 49.7]
        wcon.data.WS.contour.start_connectivity = \
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]

        # ending-point contour:
        wcon.data.WS.contour.end_lon = [-1, 2.1]
        wcon.data.WS.contour.end_lat_coef = [-1, 54.8]

        # windstorm parameters:
        wcon.data.MC.num_prds = 3  # number of periods (i.e., simulations)
        wcon.data.WS.event.max_num_ws_prd = 1  # maximum number of windstorms per period

        # wcon.data.WS.event.max_v = [30, 50]  # upper and lower bounds for initial gust speed (m/s)
        # wcon.data.WS.event.min_v = [10, 30]  # upper and lower bounds for final gust speed (m/s)

        # wcon.data.WS.event.max_v = [70, 110]  # upper and lower bounds for initial gust speed (mph)
        # wcon.data.WS.event.min_v = [20, 60]  # upper and lower bounds for final gust speed (mph)

        wcon.data.WS.event.max_v = [40, 50]  # upper and lower bounds for initial gust speed (m/s)
        wcon.data.WS.event.min_v = [30, 40]  # upper and lower bounds for final gust speed (m/s)

        # wcon.data.WS.event.max_r = [20, 25]  # upper and lower bounds for initial radius (km)
        # wcon.data.WS.event.min_r = [15, 10]  # upper and lower bounds for final radius (km)

        wcon.data.WS.event.max_r = [25, 25]  # upper and lower bounds for initial radius (km)
        wcon.data.WS.event.min_r = [25, 25]  # upper and lower bounds for final radius (km)

        # wcon.data.WS.event.max_prop_v = [20, 25]  # upper and lower bounds for initial windstorm propagation speed (km/h)
        # wcon.data.WS.event.min_prop_v = [15, 20]  # upper and lower bounds for final windstorm propagation speed (km/h)

        wcon.data.WS.event.max_prop_v = [25, 25]  # upper and lower bounds for initial windstorm propagation speed (km/h)
        wcon.data.WS.event.min_prop_v = [25, 25]  # upper and lower bounds for final windstorm propagation speed (km/h)

        wcon.data.WS.event.lng = [12, 48]  # lower and upper bounds for windstorm duration (h)
        wcon.data.WS.event.ttr = [24, 120]  # lower and upper bounds for line repair (time to restoration) (h)

        # fragility modelling:  (deprecated) (fragility data moved to NetConfig)
        wcon.data.frg = Object()
        wcon.data.frg.mu = 3.8
        wcon.data.frg.sigma = 0.122
        wcon.data.frg.thrd_1 = 20
        wcon.data.frg.thrd_2 = 90
        wcon.data.frg.shift_f = 0


    elif name == 'windstorm_29_bus_GB_transmission_network':
        # starting-point contour:
        wcon.data.WS.contour.start_lon = [-5.643, -7.281, -6.588, -5.486, -5.076, -6.604, 0.120]
        wcon.data.WS.contour.start_lat = [58.445, 56.637, 55.450, 54.956, 53.893, 49.660, 50.443]
        wcon.data.WS.contour.start_connectivity = \
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]

        # ending-point contour:
        wcon.data.WS.contour.end_lon = [-1.1, 1.8]
        wcon.data.WS.contour.end_lat_coef = [-2, 56]

        # windstorm parameters:
        wcon.data.MC.num_prds = 3  # number of periods (i.e., simulations)
        wcon.data.WS.event.max_num_ws_prd = 1  # maximum number of windstorms per period

        # wcon.data.WS.event.max_v = [40, 60]  # upper and lower bounds for initial gust speed
        # wcon.data.WS.event.min_v = [25, 35]  # upper and lower bounds for final gust speed

        wcon.data.WS.event.max_v = [55, 75]  # upper and lower bounds for initial gust speed
        wcon.data.WS.event.min_v = [35, 45]  # upper and lower bounds for final gust speed

        # wcon.data.WS.event.max_v = [70, 90]  # upper and lower bounds for initial gust speed
        # wcon.data.WS.event.min_v = [45, 55]  # upper and lower bounds for final gust speed

        wcon.data.WS.event.max_r = [20, 25]  # upper and lower bounds for initial radius
        wcon.data.WS.event.min_r = [15, 10]  # upper and lower bounds for final radius
        wcon.data.WS.event.max_prop_v = [22, 26]  # upper and lower bounds for initial windstorm propagation speed
        wcon.data.WS.event.min_prop_v = [8, 10]  # upper and lower bounds for final windstorm propagation speed
        wcon.data.WS.event.lng = [12, 48]  # lower and upper bounds for windstorm duration
        wcon.data.WS.event.ttr = [24, 120]  # lower and upper bounds for line repair (time to restoration)

        # fragility modelling:  (deprecated) (fragility data moved to NetConfig)
        wcon.data.frg = Object()
        wcon.data.frg.mu = 3.8
        wcon.data.frg.sigma = 0.122
        wcon.data.frg.thrd_1 = 20
        wcon.data.frg.thrd_2 = 90
        wcon.data.frg.shift_f = 0


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
        wcon.data.WS.event.min_r = [10, 12]  # upper and lower bounds for final radius
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

