
from config import WindConfig
from windstorm import WindClass

def make_windstorm(name: str) -> WindClass:

    wcon = WindConfig()

    if name == "windstorm_1_matpower_case22":

        # starting-point contour:
        wcon.data.WS.contour.start_lon = [-2, 0]
        wcon.data.WS.contour.start_lat = [0, 5]
        wcon.data.WS.contour.start_connectivity = [
            [1, 2]
        ]

        # ending-point contour:
        wcon.data.WS.contour.end_lon = [8, 10]
        wcon.data.WS.contour.end_lat_coef = [3, -28.5]



    return WindClass(wcon)

