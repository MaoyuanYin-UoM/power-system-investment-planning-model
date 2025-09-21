# This script contains the windstorm engine imported (with minor changes) from pywellness

import math
import random
import numpy as np

from core.config import WindConfig
from utils import *

class Object(object):
    pass

class WindClass:
    # Contains functions for windstorm simulation
    def __init__(self, obj=None):

        # Get default values from config
        if obj == None:
            obj = WindConfig()

        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # Get parameters
        max_num_ws_prd = self._get_max_num_ws_prd()
        lim_max_v_ws = self._get_lim_max_v_ws()
        lim_min_v_ws = self._get_lim_min_v_ws()
        lim_lng_ws = self._get_lim_lng_ws()

        # Settings for Monte Carlo simulation
        self.MC = Object()

        # MC.WS samples and stores windstorm parameters
        self.MC.WS = Object()
        # 1) sample number of windstorms for each simulation (period)
        self.MC.WS.num_ws_prd = [random.randint(1, max_num_ws_prd) for i in range(self.data.MC.num_prds)]
        # total number of events
        self.MC.WS.num_ws_total = sum(self.MC.WS.num_ws_prd)
        # 2) sample max and min wind speed for each event
        self.MC.WS.lim_v_ws_all = \
            [[lim_max_v_ws[0] + random.random() *
              (lim_max_v_ws[1] - lim_max_v_ws[0]),
              lim_min_v_ws[0] + random.random() *
              (lim_min_v_ws[1] - lim_min_v_ws[0])]
              for i in range(self.MC.WS.num_ws_total)]
        # 3) Sample duration for each windstorm
        self.MC.WS.lng = \
            [random.randint(lim_lng_ws[0], lim_lng_ws[1])
             for i in range(self.MC.WS.num_ws_total)]


    def crt_bgn_hr(self):
        """Define hour when windstorm begins per year"""
        # Gets
        lim_lng_ws = self._get_lim_lng_ws()
        lim_ttr = self._get_lim_ttr()
        max_num_ws_prd = self._get_max_num_ws_prd()
        num_ws_prd = self._get_num_ws_prd()
        num_hrs_prd = self._get_num_hrs_prd()

        max_lng = max(lim_lng_ws) + max(lim_ttr)
        bgn_hrs_ws_prd = np.zeros((len(num_ws_prd), max_num_ws_prd))  # empty a space to store beginning hours
        
        for i in range(len(num_ws_prd)):
            for j in range(num_ws_prd[i]):
                # Find initial point in the range
                if j == 0:
                    rn1 = 1  # set the beginning hour of the first event in each period to be 1
                else:
                    rn1 = bgn_hrs_ws_prd[i][j-1] + max_lng
                # Find final point in the range
                rn2 = num_hrs_prd - (num_ws_prd[i] - j+1)*max_lng
                rn1 = int(rn1)
                rn2 = int(rn2)
                # Generate random hour
                bgn_hrs_ws_prd[i][j] = random.randint(rn1, rn2)

        self.MC.WS.bgn_hrs_ws_prd = bgn_hrs_ws_prd


    def get_distance(self, Lon1, Lat1, Lon2, Lat2):
        """Get distance between two coordinates [km]"""
        R = 6371000  # Earth Radious [m]
        L1 = Lat1 * math.pi / 180  # Radians
        L2 = Lat2 * math.pi / 180  # Radians
        DL = (Lat2 - Lat1) * math.pi / 180
        DN = (Lon2 - Lon1) * math.pi / 180

        a = math.sin(DL / 2) * math.sin(DL / 2) + math.cos(L1) * math.cos(L2) * \
            math.sin(DN / 2) * math.sin(DN / 2)
        c = 2 * math.atan(math.sqrt(a) / math.sqrt(1 - a))

        d = R * c / 1000  # [km]

        return d


    def get_bearing(self, Lon1, Lat1, Lon2, Lat2):
        """Get the geographical bearing between two coordinates in radian"""
        phi_1 = Lat1 * math.pi / 180  # Radians
        phi_2 = Lat2 * math.pi / 180  # Radians
        lambda_1 = Lon1 * math.pi / 180  # Radians
        lambda_2 = Lon2 * math.pi / 180  # Radians

        delta_lambda = lambda_2 - lambda_1
        y = math.sin(delta_lambda) * math.cos(phi_2)
        x = math.cos(phi_1) * math.sin(phi_2) - math.sin(phi_1) * \
            math.cos(phi_2) * math.cos(delta_lambda)

        theta = math.atan2(y, x)

        alpha = (theta + 2 * math.pi) % (2 * math.pi)

        return alpha  # Radians


    def get_destination(self, Lon1, Lat1, bearing, distance):
        """Get the destination's coordinates based on the starting point's coordinates,
        the travelling direction and distance [m]"""
        max_trials = 10
        tolerance = 1e-3

        # The ubiquitous WGS-84 is a geocentric datum, based on an ellipsoid with:
        a = 6378137.0  # metres - Semi-major axis
        b = 6356752.314245  # metres - Semi-minor axis
        f = (a - b) / a  # Flattening

        phi_1 = Lat1 * math.pi / 180  # Radians
        lambda_1 = Lon1 * math.pi / 180  # Radians
        alpha_1 = bearing

        s_alpha_1 = math.sin(alpha_1)
        c_alpha_1 = math.cos(alpha_1)
        t_u1 = (1 - f) * math.tan(phi_1)
        c_u1 = 1 / math.sqrt(1 + t_u1 * t_u1)
        s_u1 = t_u1 * c_u1
        sigma_1 = math.atan2(t_u1, c_alpha_1)
        s_alpha = c_u1 * s_alpha_1
        c_sq_alpha = 1 - s_alpha * s_alpha
        u_sq = c_sq_alpha * (a * a - b * b) / (b * b)
        A = 1 + (u_sq / 16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
        B = (u_sq / 1024) * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

        sigma = distance / (b * A)
        sigma_p = 0
        for _ in range(max_trials):
            c_2_sigma_m = math.cos(2 * sigma_1 + sigma)
            s_sigma = math.sin(sigma)
            c_sigma = math.cos(sigma)
            delta_sigma = B * s_sigma * (c_2_sigma_m + (B / 4) * (c_sigma * \
                                                                  (-1 + 2 * c_2_sigma_m * c_2_sigma_m) - (
                                                                              B / 6) * c_2_sigma_m * \
                                                                  (-3 + 4 * s_sigma * s_sigma) * (
                                                                              -3 + 4 * c_2_sigma_m * c_2_sigma_m)))
            sigma_p = sigma
            sigma = distance / (b * A) + delta_sigma
            if abs(sigma - sigma_p) < tolerance:
                break

        xx = s_u1 * s_sigma - c_u1 * c_sigma * c_alpha_1
        phi_d = math.atan2(s_u1 * c_sigma + c_u1 * s_sigma * c_alpha_1,
                           (1 - f) * math.sqrt(s_alpha * s_alpha + xx * xx))
        lambda_ = math.atan2(s_sigma * s_alpha_1, c_u1 * c_sigma - s_u1 * \
                             s_sigma * c_alpha_1)
        C = (f / 16) * c_sq_alpha * (4 + f * (4 - 3 * c_sq_alpha))
        L = lambda_ - (1 - C) * f * s_alpha * (sigma + C * s_sigma * \
                                               (c_2_sigma_m + C * c_sigma * (-1 + 2 * c_2_sigma_m * c_2_sigma_m)))
        lambda_d = lambda_1 + L

        d_lon_lat = [math.degrees(lambda_d), math.degrees(phi_d)]
        if _ == max_trials - 1:
            print('Warning: Not meeting tolerance')

        return d_lon_lat


    def init_ws_path0(self):
        """Preparations to define starting points of windstorms"""
        # Gets
        cp_start_connectivity = self._get_cp_start_connectivity()
        cp_lat = self._get_cp_start_lat()
        cp_lon = self._get_cp_start_lon()

        # Getting distances for each segment of the contour
        cp_num = len(cp_start_connectivity)
        dis = [0 for i in range(cp_num + 1)]

        for i in range(cp_num):
            f = cp_start_connectivity[i][0] - 1
            t = cp_start_connectivity[i][1] - 1
            d = self.get_distance(cp_lon[f], cp_lat[f], cp_lon[t], cp_lat[t])
            dis[i + 1] = dis[i] + d

        # Keep the segment endpoints in the same order used for 'dis'
        self.data.WS.contour.seg_from = [cp_start_connectivity[i][0] - 1 for i in range(cp_num)]
        self.data.WS.contour.seg_to = [cp_start_connectivity[i][1] - 1 for i in range(cp_num)]

        self.data.WS.contour.num = cp_num
        self.data.WS.contour.dis = [dis[i] / dis[cp_num] for i in range(cp_num + 1)]


    def init_ws_path(self, num_ws):
        """
        Defining starting point and direction (ending) point of the windstorm
        Note: The direction (ending) point is the ending point of the directional path (the path without damping),
              not the actual path. The actual path is generated by "crt_ws_path(self)"
        """
        # Gets
        cp_start_lon = self._get_cp_start_lon()
        cp_start_lat = self._get_cp_start_lat()
        cp_end_lon = self._get_cp_end_lon()
        cp_end_lat_coef = self._get_cp_end_lat_coef()
        cp_dis_aggregated = self._get_cp_dis_aggregated()
        rand_location = [random.random() for i in range(num_ws)]
        rand_direction = [random.random() for i in range(num_ws)]

        # Random starting point
        start_lon = [0 for i in range(num_ws)]
        start_lat = [0 for i in range(num_ws)]
        for i in range(num_ws):
            j = 1
            while rand_location[i] > cp_dis_aggregated[j]:
                j += 1

            aux = (rand_location[i] - cp_dis_aggregated[j - 1]) / \
                  (cp_dis_aggregated[j] - cp_dis_aggregated[j - 1])
            f = self.data.WS.contour.seg_from[j - 1]
            t = self.data.WS.contour.seg_to[j - 1]
            start_lon[i] = cp_start_lon[f] + aux * (cp_start_lon[t] - cp_start_lon[f])
            start_lat[i] = cp_start_lat[f] + aux * (cp_start_lat[t] - cp_start_lat[f])

        # Random direction
        end_lon = [0 for i in range(num_ws)]
        end_lat = [0 for i in range(num_ws)]
        for i in range(num_ws):
            end_lon[i] = cp_end_lon[1] - \
                       rand_direction[i] * (cp_end_lon[1] - cp_end_lon[0])
            end_lat[i] = end_lon[i] * cp_end_lat_coef[0] + cp_end_lat_coef[1]

        return start_lon, start_lat, end_lon, end_lat


    def linear_interpolate(self, start, end, num_points):
        """ Linearly interpolate between start and end """
        return [start + i * (end - start) / (num_points - 1) for i in range(num_points)]


    def cubic_interpolate(self, Lon1, Lat1, Lon2, Lat2):
        from scipy.interpolate import CubicSpline

        reverse = Lon1 > Lon2
        # Ensure x is always in increasing order for interpolation
        if reverse:
            Lon1, Lon2 = Lon2, Lon1
            Lat1, Lat2 = Lat2, Lat1

        cs = CubicSpline([Lon1, Lon2], [Lat1, Lat2], bc_type=((1, 0), (1, 0)))
        return cs


    def crt_ws_path(self, Lon1, Lat1, Lon2, Lat2, lng_ws):
        """Create the propagation path of a windstorm on an hourly basis"""
        # trajectory of windstorm
        dir_lon = self.linear_interpolate(Lon1, Lon2, lng_ws + 1)
        cs = self.cubic_interpolate(Lon1, Lat1, Lon2, Lat2)
        dir_lat = cs(dir_lon)

        path_ws = [[0, 0] for _ in range(lng_ws + 1)]
        path_ws[0] = [Lon1, Lat1]

        # Get bounds
        lim_max_prop_v_ws = self._get_lim_max_prop_v_ws()
        lim_min_prop_v_ws = self._get_lim_min_prop_v_ws()

        prop_model = getattr(self.data.WS.event, 'prop_model', None)

        if prop_model == 'constant_uniform':
            # Bounds for constant sampling (km/h)
            bounds = getattr(self.data.WS.event, 'prop_uniform_bounds', None)
            if bounds is None:
                # Envelope from configured min/max ranges
                low = min(lim_min_prop_v_ws[0], lim_min_prop_v_ws[1],
                          lim_max_prop_v_ws[0], lim_max_prop_v_ws[1])
                high = max(lim_min_prop_v_ws[0], lim_min_prop_v_ws[1],
                           lim_max_prop_v_ws[0], lim_max_prop_v_ws[1])
                bounds = [low, high]

            prop_v_const = random.uniform(bounds[0], bounds[1])

            # compute windstorm path with constant propagation speed
            for hr in range(1, lng_ws + 1):
                dist_hr = prop_v_const * 1e3  # km/h → m/h
                brg_hr = self.get_bearing(dir_lon[hr - 1], dir_lat[hr - 1], dir_lon[hr], dir_lat[hr])
                path_ws[hr] = self.get_destination(path_ws[hr - 1][0], path_ws[hr - 1][1], brg_hr, dist_hr)

        elif prop_model == 'linear_decrease':
            # Legacy behavior: linearly decreasing propagation speed
            init_prop_v_ws = random.uniform(lim_max_prop_v_ws[0], lim_max_prop_v_ws[1])
            final_prop_v_ws = random.uniform(lim_min_prop_v_ws[0], lim_min_prop_v_ws[1])

            for hr in range(1, lng_ws + 1):
                # Linear decrease over the event duration
                v_hr = init_prop_v_ws + (final_prop_v_ws - init_prop_v_ws) * (hr - 1) / max(1, lng_ws - 1)
                dist_hr = v_hr * 1e3  # km/h → m/h
                brg_hr = self.get_bearing(dir_lon[hr - 1], dir_lat[hr - 1], dir_lon[hr], dir_lat[hr])
                path_ws[hr] = self.get_destination(path_ws[hr - 1][0], path_ws[hr - 1][1], brg_hr, dist_hr)

        else:
            raise ValueError(
                f"Unsupported prop_model '{prop_model}'. "
                "Expected 'constant_uniform' or 'linear_decrease'."
            )

        return path_ws

    def crt_ws_radius(self, lng_ws):
        """Create radius of a windstorm at each hour.

        Notes:
            - If event.r_model == 'constant_uniform': sample ONCE from a Uniform (low, high) and keep it constant across
            the event.
            - If event.r_model == 'linear_decrease': use the legacy linear decrease between sampled initial and final
            radii.
        """
        r_model = getattr(self.data.WS.event, 'r_model', None)

        if r_model == 'constant_uniform':
            # Bounds for constant sampling (km)
            # If not explicitly set, fall back to a sensible envelope from min/max bounds
            bounds = getattr(self.data.WS.event, 'r_uniform_bounds', None)
            if bounds is None:
                # Envelope from configured min/max ranges
                lim_max_r_ws = self._get_lim_max_r_ws()
                lim_min_r_ws = self._get_lim_min_r_ws()
                low = min(lim_min_r_ws[0], lim_min_r_ws[1], lim_max_r_ws[0], lim_max_r_ws[1])
                high = max(lim_min_r_ws[0], lim_min_r_ws[1], lim_max_r_ws[0], lim_max_r_ws[1])
                bounds = [low, high]

            r_const = random.uniform(bounds[0], bounds[1])
            return np.full(lng_ws + 1, r_const, dtype=float)

        elif r_model == 'linear_decrease':
            # Legacy behavior (kept for backward compatibility)
            # Assumption: radius decreases linearly with time
            lim_max_r_ws = self._get_lim_max_r_ws()
            lim_min_r_ws = self._get_lim_min_r_ws()
            init_r_ws = random.uniform(lim_max_r_ws[0], lim_max_r_ws[1])
            end_r_ws = random.uniform(lim_min_r_ws[0], lim_min_r_ws[1])
            radius = np.linspace(init_r_ws, end_r_ws, lng_ws + 1)
            return radius

        else:
            raise ValueError(
                f"Unsupported r_model '{r_model}'. "
                "Expected 'constant_uniform' or 'linear_decrease'."
            )

    def crt_ws_v(self, lim_v_ws, lng_ws):
        """Create gust speeds of a windstorm at each hour.

        Notes:
            - If event.gust_model == 'constant_weibull': sample once from a Weibull(k, λ) and keep it constant across
            the event.
            - If event.gust_model == 'log_linear_decrease': use the old log-linear model between lim_v_ws[0] and
            lim_v_ws[1].
        """
        # Read model name; default to the legacy log-linear model for backward compatibility
        gust_model = getattr(self.data.WS.event, 'gust_model', 'log_linear_decrease')

        if gust_model == 'constant_weibull':
            # Weibull parameters (shape k, scale λ), with sensible defaults
            k = (getattr(self.data.WS.event, 'gust_weibull_shape', None)
                 or getattr(self.data.WS.event, 'gust_weibull_k', None)
                 or 2.0)
            lam = getattr(self.data.WS.event, 'gust_weibull_scale', 30.0)

            # Sample ONCE → constant gust during the event
            v_const = float(np.random.weibull(k) * lam)
            return [v_const] * lng_ws

        elif gust_model == 'log_linear_decrease':
            # Legacy behavior (kept for backward compatibility)
            # Monotonic (log-linear) profile between min/max.
            # Protect against division by zero when lng_ws == 1.
            denom = max(1, (lng_ws - 1))

            # Guard against invalid ratios; if min == 0 or ratio <= 0, keep 'a' at 0.
            ratio = (lim_v_ws[1] / lim_v_ws[0]) if lim_v_ws[0] != 0 else 1.0
            a = math.log(ratio) / denom if ratio > 0 else 0.0

            v_ws = [lim_v_ws[0] * math.exp(a * i) for i in range(lng_ws)]
            return v_ws

        else:
            # Explicit error for unsupported model names
            raise ValueError(
                f"Unsupported gust_model '{gust_model}'. "
                "Expected 'constant_weibull' or 'log_linear_decrease'."
            )

    def compare_circle(self, epicentre, rad_ws, gis_bgn, gis_end, num_bch):
        """Identify whether an asset falls within the impact zone marked by a radius [km] around the epicentre."""

        Flgs = [0] * num_bch  # Ensure explicit 0s instead of False
        for xt in range(num_bch):
            if gis_bgn[xt][0] == gis_end[xt][0]:  # Special case, vertical line
                x = gis_bgn[xt][0]
                aux = max(gis_bgn[xt][1], gis_end[xt][1])
                if epicentre[1] > aux:
                    y = aux
                else:
                    aux = min(gis_bgn[xt][1], gis_end[xt][1])
                    if epicentre[1] < aux:
                        y = aux
                    else:
                        y = epicentre[1]
            else:
                b = (gis_bgn[xt][1] - gis_end[xt][1]) / (gis_bgn[xt][0] - gis_end[xt][0])
                a = gis_bgn[xt][1] - gis_bgn[xt][0] * b
                x = (b * (epicentre[1] - a) + epicentre[0]) / (b ** 2 + 1)
                aux = max(gis_bgn[xt][0], gis_end[xt][0])
                if x > aux:
                    x = aux
                else:
                    aux = min(gis_bgn[xt][0], gis_end[xt][0])
                    if x < aux:
                        x = aux
                y = a + b * x

            # Calculate distance and check if within radius
            if self.get_distance(epicentre[0], epicentre[1], x, y) < rad_ws:
                Flgs[xt] = 1  # Change from True to 1

        return Flgs

    def _km_to_deg_lat(self, km: float) -> float:
        """Approx: 1 degree latitude ≈ 111 km."""
        return km / 111.0

    @staticmethod
    def _seg_intersect(p1, p2, q1, q2):
        """Segment intersection (2D, robust enough for plotting scale)."""
        def orient(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

        def on_seg(a, b, c):
            return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                    min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)

        # General case
        if (o1*o2 < 0) and (o3*o4 < 0):
            return True

        # Collinear cases
        if o1 == 0 and on_seg(p1, p2, q1): return True
        if o2 == 0 and on_seg(p1, p2, q2): return True
        if o3 == 0 and on_seg(q1, q2, p1): return True
        if o4 == 0 and on_seg(q1, q2, p2): return True
        return False

    @staticmethod
    def _point_in_poly(pt, poly):
        """Ray casting point in polygon (poly = list[(x,y)])."""
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
                inside = not inside
        return inside

    @staticmethod
    def _unit_perp(p1, p2):
        """Unit normal of segment p1->p2 in degree space."""
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        nx, ny = -vy, vx
        nrm = math.hypot(nx, ny)
        if nrm == 0:
            return (0.0, 0.0)
        return (nx / nrm, ny / nrm)

    def compare_capsule_series(self,
                               epicentres,
                               # sequence of (lon, lat), length = duration (+1 if your radius array includes the end)
                               radii_km,  # sequence of radius values (km), same indexing as epicentres
                               gis_bgn, gis_end,
                               num_bch,
                               radius_mode: str = "t"):
        """
        Compute branch impact flags for each hour using a swept-band (capsule) per hour.

        For hour t (0..T-2):
          - Build oriented rectangle that covers the segment E_t -> E_{t+1}
            with half-width = r_deg where r is:
              radius_mode == 't'   -> r_t
              radius_mode == 'min' -> min(r_t, r_{t+1})
              radius_mode == 'max' -> max(r_t, r_{t+1})  [conservative]
          - Add a circle at E_t as an end-cap (full circle is ok & simple).

        For the final hour (t == T-1):
          - Circle only at E_{T-1}.

        Returns:
            flgs_impacted_bch : np.ndarray shape [num_bch, T] of 0/1.
        """
        import numpy as np

        # Accept both length T and T+1 epicentres; unify to T = duration
        T = len(epicentres)
        # If caller passes radius for T+1 points (as returned by crt_ws_radius), drop the extra tail.
        if len(radii_km) == T + 1:
            radii_km = radii_km[:T]
        elif len(radii_km) != T:
            raise ValueError("radii_km must have length T or T+1 to match epicentres.")

        epic = list(map(tuple, epicentres))
        flgs = np.zeros((num_bch, T), dtype=int)

        # Helper: test circle (reuse existing logic) → returns [num_bch] 0/1
        def circle_flags(center, r_km):
            return np.array(self.compare_circle(center, r_km, gis_bgn, gis_end, num_bch), dtype=int)

        # Loop hours
        for t in range(T):
            # Always include circle at E_t
            r_t = radii_km[t]
            hour_flags = circle_flags(epic[t], r_t)

            # For swept band between E_t and E_{t+1}
            if t < T - 1:
                # pick radius for this hour's band
                if radius_mode == "min":
                    r_use = min(radii_km[t], radii_km[t + 1])
                elif radius_mode == "max":
                    r_use = max(radii_km[t], radii_km[t + 1])
                else:  # "t"
                    r_use = radii_km[t]

                r_deg = self._km_to_deg_lat(r_use)

                p1 = epic[t]
                p2 = epic[t + 1]

                nx, ny = self._unit_perp(p1, p2)
                if not (nx == 0.0 and ny == 0.0):
                    # Build oriented rectangle of width 2*r_deg around the segment
                    c1 = (p1[0] + nx * r_deg, p1[1] + ny * r_deg)
                    c2 = (p2[0] + nx * r_deg, p2[1] + ny * r_deg)
                    c3 = (p2[0] - nx * r_deg, p2[1] - ny * r_deg)
                    c4 = (p1[0] - nx * r_deg, p1[1] - ny * r_deg)
                    rect = [c1, c2, c3, c4]

                    # For each branch, set impact if:
                    #   - either endpoint of the branch is inside the rect, OR
                    #   - the branch segment intersects any rect edge.
                    for b in range(num_bch):
                        a1 = gis_bgn[b]
                        a2 = gis_end[b]

                        # Fast inside check
                        inside = self._point_in_poly(a1, rect) or self._point_in_poly(a2, rect)
                        if inside:
                            hour_flags[b] = 1
                            continue

                        # Edge intersections
                        hit = False
                        for i in range(4):
                            e1 = rect[i]
                            e2 = rect[(i + 1) % 4]
                            if self._seg_intersect(a1, a2, e1, e2):
                                hit = True
                                break
                        if hit:
                            hour_flags[b] = 1

            flgs[:, t] = hour_flags

        return flgs

    def _fragility_curve(self, hzd_int):
        """Calculate the asset's probability of failure based on a fragility curve"""

        from scipy.stats import lognorm

        # Gets
        mu = self._get_frg_mu()
        sigma = self._get_frg_sigma()
        thrd_1 = self._get_frg_thrd_1()
        thrd_2 = self._get_frg_thrd_2()
        shift_f = self._get_frg_shift_f()

        f_hzd_int = hzd_int - shift_f

        if f_hzd_int < thrd_1:
            pof = 0
        elif f_hzd_int > thrd_2:
            pof = 1
        else:
            # Convert mu and sigma for lognormal distribution
            shape = sigma
            scale = np.exp(mu)
            # Calculate the cumulative distribution function (CDF) of the lognormal distribution
            pof = lognorm.cdf(f_hzd_int, s=shape, scale=scale)

        return pof


    # this function is deprecated:
    # def sample_bch_failure(self, timestep, flgs_bch_status, flgs_impacted_bch, wind_speed):
    #     """
    #     Sample if the impacted bchs fail under the wind speed.
    #     For each failed branch, sample the time to repair and set the corresponding branch-status flags to False
    #     """
    #
    #     import random
    #
    #     ttr = self._get_ttr()  # get lower and upper bounds for sampling the time to repair
    #     ts = timestep
    #
    #     for b in range(len(flgs_bch_status)):  # loop over each branch
    #
    #          if flgs_impacted_bch[b] & flgs_bch_status[b][ts]:  # check if the branch is both operating and impacted at this timestep
    #             # get failure probability from the fragility curve
    #             pof = self._fragility_curve(wind_speed)
    #             if random.random() < pof:  # sample if it fails
    #                 # if it fails, sample the time to repair
    #                 time_to_repair = random.randint(ttr[0], ttr[1])
    #                 for t in range(ts, min(ts + time_to_repair, len(flgs_bch_status[0]))):  # loop over each hour before being repaired
    #                     # set branch status to 0 (False) for each hour until it's repaired
    #                     flgs_bch_status[b][t] = False
    #
    #     return flgs_bch_status






    # Gets:
    def _get_bgn_hrs_ws_prd(self):
        """Get bgn_hrs_ws_prd"""
        return self.MC.WS.bgn_hrs_ws_prd

    def _get_cp_end_lat_coef(self):
        """Get end_lat_coef"""
        return self.data.WS.contour.end_lat_coef

    def _get_cp_end_lon(self):
        """Get cp_end_lon"""
        return self.data.WS.contour.end_lon

    def _get_lim_lng_ws(self):
        """Get lim_lng_ws"""
        return self.data.WS.event.lng

    def _get_lim_max_v_ws(self):
        """Get lim_max_v_ws"""
        return self.data.WS.event.max_v

    def _get_lim_min_v_ws(self):
        """Get lim_min_v_ws"""
        return self.data.WS.event.min_v

    def _get_lim_max_r_ws(self):
        """Get lim_min_v_ws"""
        return self.data.WS.event.max_r

    def _get_lim_min_r_ws(self):
        """Get lim_min_v_ws"""
        return self.data.WS.event.min_r

    def _get_lim_max_prop_v_ws(self):
        """Get lim_max_prop_v_ws"""
        return self.data.WS.event.max_prop_v

    def _get_lim_min_prop_v_ws(self):
        """Get lim_min_prop_v_ws"""
        return self.data.WS.event.min_prop_v

    def _get_lim_ttr(self):
        """Get lim_ttr"""
        return self.data.WS.event.ttr

    def _get_lng_ws(self):
        """Get lng_ws"""
        return self.MC.WS.lng

    def _get_cp_dis_aggregated(self):
        """Get cp_dis_aggregated"""
        return self.data.WS.contour.dis

    def _get_cp_start_connectivity(self):
        """Get cp_start_connectivity"""
        return self.data.WS.contour.start_connectivity

    def _get_cp_start_lat(self):
        """Get cp_start_lat"""
        return self.data.WS.contour.start_lat

    def _get_cp_start_lon(self):
        """Get cp_start_lon"""
        return self.data.WS.contour.start_lon

    def _get_cp_num(self):
        """Get cp_num"""
        return self.data.WS.contour.num

    def _get_lim_v_ws_all(self):
        """Get lim_v_ws"""
        return self.MC.WS.lim_v_ws_all

    def _get_max_num_ws_prd(self):
        """Get max_num_ws_prd"""
        return self.data.WS.event.max_num_ws_prd

    def _get_num_hrs_prd(self):
        """Get the number of hours in the selected simulation period"""
        return self.data.MC.prd_to_hrs[self.data.MC.lng_prd]

    def _get_num_ws_prd(self):
        """Get num_ws_prd"""
        return self.MC.WS.num_ws_prd

    def _get_num_ws_total(self):
        """Get num_ws_total"""
        return self.MC.WS.num_ws_total

    def _get_mcs_prd(self):
        """Get num_prds"""
        return self.data.MC.num_prds

    def _get_frg_mu(self):
        """Get frg_mu"""
        return self.data.frg.mu

    def _get_frg_sigma(self):
        """Get frg_sigma"""
        return self.data.frg.sigma

    def _get_frg_thrd_1(self):
        """Get frg_thrd_1"""
        return self.data.frg.thrd_1

    def _get_frg_thrd_2(self):
        """Get frg_thrd_2"""
        return self.data.frg.thrd_2

    def _get_frg_shift_f(self):
        """Get frg_shift_f"""
        return self.data.frg.shift_f

    def _get_ttr(self):
        """Get ttr"""
        return self.data.WS.event.ttr
