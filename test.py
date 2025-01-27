

from config import *
from utils import *
from visualization import *
from windstorm import *
from network_linear import *


wcon = WindConfig()
ws = WindClass(wcon)

ncon = NetConfig()
net = NetworkClass(ncon)

visualize_fragility_curve(wcon)
visualize_bch_and_ws_contour(wcon, ncon)