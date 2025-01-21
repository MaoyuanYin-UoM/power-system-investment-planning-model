# This script contains the windstorm engine imported (with minor changes) from pywellness

import math
import random

from config import WindConfig
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
        max_ws_yr = self._get_max_ws_yr()
        lim_max_v_ws = self._get_lim_max_v_ws()
        lim_min_v_ws = self._get_lim_min_v_ws()
        lim_lng_ws = self._get_lim_lng_ws()

        # Store Monte Carlo parameters
        self.MC = Object()

        # MC.WS samples and stores windstorm parameters
        self.MC.WS = Object()
        # 1) sample number of windstorms per year
        self.MC.WS.num_yr = [random.randint(1, max_ws_yr) for i in range(self.data.MC.num_trials)]
        # total number of events
        self.MC.WS.num_yr_total = sum(self.MC.WS.num_yr)
        # 2) sample max and min wind speed for each event
        self.MC.WS.v = \
            [[lim_max_v_ws[0] + random.random() *
              (lim_max_v_ws[1] - lim_max_v_ws[0])
              for i in range(self.MC.WS.num_yr_total)],
             [lim_min_v_ws[0] + random.random() *
              (lim_min_v_ws[1] - lim_min_v_ws[0])
              for i in range(self.MC.WS.num_yr_total)]]
        # 3) Sample duration for each windstorm
        self.MC.WS.lng = \
            [random.randint(lim_lng_ws[0], lim_lng_ws[1])
             for i in range(self.MC.WS.num_yr_total)]


    def _init_ws_path0(self):
        '''Preparations to define starting point of wind storm'''
        # Gets
        cp_start_connectivity = self._get_cp_start_connectivity()
        cp_lat = self._get_cp_lat()
        cp_lon = self._get_cp_lon()

        # Getting distances for each segment of the contour
        cp_num = len(cp_start_connectivity)
        dis = [0 for i in range(cp_num + 1)]

        for i in range(cp_num):
            f = cp_start_connectivity[i][0] - 1
            t = cp_start_connectivity[i][1] - 1
            d = self._getDistance(cp_lon[f], cp_lat[f], cp_lon[t], cp_lat[t])
            dis[i + 1] = dis[i] + d
        self.data.WS.contour.num = cp_num
        self.data.WS.contour.dis = [dis[i] / dis[cp_num] for i in range(cp_num + 1)]


    def _init_ws_path(self, NumWS):
        '''Defining starting point and direction of wind storm'''
        # Gets
        cp_start_lon = self._get_cp_start_lon()
        cp_start_lat = self._get_cp_start_lat()
        cp_end_lon = self._get_cp_end_lon()
        cp_end_lat_coef = self._get_cp_end_lat_coef()
        cp_dis_aggregated = self._get_cp_dis_aggregated()
        rand_location = [random.random() for i in range(NumWS)]
        rand_direction = [random.random() for i in range(NumWS)]

        # Random starting point
        start_lon = [0 for i in range(NumWS)]
        start_lat = [0 for i in range(NumWS)]
        for i in range(NumWS):
            j = 1
            while rand_location[i] > cp_dis_aggregated[j]:
                j += 1

            aux = (rand_location[i] - cp_dis_aggregated[j - 1]) / \
                  (cp_dis_aggregated[j] - cp_dis_aggregated[j - 1])
            start_lon[i] = cp_start_lon[j - 1] + aux * (cp_start_lon[j] - cp_start_lon[j - 1])
            start_lat[i] = cp_start_lat[j - 1] + aux * (cp_start_lat[j] - cp_start_lat[j - 1])

        # Random direction
        end_lon = [0 for i in range(NumWS)]
        end_lat = [0 for i in range(NumWS)]
        for i in range(NumWS):
            end_lon[i] = cp_end_lon[1] - \
                       rand_direction[i] * (cp_end_lon[1] - cp_end_lon[0])
            end_lat[i] = end_lon[i] * cp_end_lat_coef[0] + cp_end_lat_coef[1]

        return start_lon, start_lat, end_lon, end_lat


    def _getDistance(self, Lon1, Lat1, Lon2, Lat2):
        '''Get distance between two coordinates [km]'''
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


    def _getBearing(self, Lon1, Lat1, Lon2, Lat2):
        '''Get the geographical bearing between two coordinates in radian'''
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


    def _getDestination(self, Lon1, Lat1, bearing, distance):
        '''Get the destination's coordinates based on the starting point's coordinates,
        the travelling direction and distance [m]'''
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


    def linear_interpolate(self, start, end, num_points):
        ''' Linearly interpolate between start and end '''
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


    def _crt_ws_path(self, Lon1, Lat1, Lon2, Lat2, Num_hrs):
        '''Create the propagation path of a windstorm on an hourly basis'''
        # trajectory of windstorm
        dir_lon = self.linear_interpolate(Lon1, Lon2, Num_hrs + 1)
        cs = self.cubic_interpolate(Lon1, Lat1, Lon2, Lat2)
        dir_lat = cs(dir_lon)

        path_ws = [[0, 0] for _ in range(Num_hrs + 1)]
        path_ws[0] = [Lon1, Lat1]

        for hr in range(1, Num_hrs + 1):
            dist_hr = 24000 - 8000 * (hr - 1) / Num_hrs
            brg_hr = self._getBearing(dir_lon[hr - 1], dir_lat[hr - 1], dir_lon[hr], dir_lat[hr])
            path_ws[hr] = self._getDestination(path_ws[hr - 1][0], path_ws[hr - 1][1], brg_hr, dist_hr)

        return path_ws


    def _crt_ws_v(self, lim_v_ws, Num_hrs):
        '''wind gust speeds of a wind storm at each hour'''
        a = math.log(lim_v_ws[1] / lim_v_ws[0]) / (Num_hrs - 1)

        v_ws = [lim_v_ws[0] * math.exp(a * i) for i in range(Num_hrs)]

        return v_ws


    def _compare_circle(self, epicentre, rad_ws, gis_bgn, gis_end, Num_bch):
        '''identify whether an asset falls within the impact zone
        marked by a radius [km] around the epicentre'''

        Flgs = [False] * Num_bch
        for xt in range(Num_bch):
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
            if self._getDistance(epicentre[0], epicentre[1], x, y) < rad_ws:
                Flgs[xt] = True
        return Flgs


    def _crt_envelope(self, epicentre, epicentre1, rad_ws):
        '''Create an envelope of coordinates around a path defined by two points'''
        evlp_pts = []

        # Calculate initial bearing from the epicentre to epicentre1
        alpha_0 = self._getBearing(epicentre[0], epicentre[1], epicentre1[0], epicentre1[1])

        # Calculate envelope points
        rad_ws = rad_ws * 1000  # change to [meters]
        evlp_pts.append(self._getDestination(epicentre[0], epicentre[1], alpha_0 - 90 * math.pi / 180, rad_ws))
        evlp_pts.append(self._getDestination(epicentre[0], epicentre[1], alpha_0 + 90 * math.pi / 180, rad_ws))
        evlp_pts.append(self._getDestination(epicentre1[0], epicentre1[1], alpha_0 - 90 * math.pi / 180, rad_ws))
        evlp_pts.append(self._getDestination(epicentre1[0], epicentre1[1], alpha_0 + 90 * math.pi / 180, rad_ws))

        return evlp_pts


    def _compare_envelope(self, evlp, gis_bgn, gis_end, Num_bch):
        '''Identify groups of lines that are within the envelope'''

        # Assumed sequence
        evlp_sequence = [0, 2, 3, 1, 0]
        evlp = np.array(evlp)

        x = evlp[evlp_sequence, 0]
        y = evlp[evlp_sequence, 1]

        # Get line equations and angle between lines
        aux = np.array([[1, 0], [2, 1], [3, 2], [4, 3]])
        for i in np.where(x[1:] == x[:-1])[0]:
            x[aux[i, 0]] -= 0.000001
            x[aux[i, 1]] += 0.000001

        for i in np.where(y[1:] == x[:-1])[0]:
            y[aux[i, 0]] -= 0.000001
            y[aux[i, 1]] += 0.000001

        b = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        a = y[:-1] - b * x[:-1]

        # XY range of each line
        aux1 = np.column_stack((x, np.roll(x, -1)))
        aux2 = np.column_stack((y, np.roll(y, -1)))
        xy_range = np.column_stack(
            (np.min(aux1, axis=1), np.max(aux1, axis=1), np.min(aux2, axis=1), np.max(aux2, axis=1)))

        flgs = np.zeros(Num_bch, dtype=bool)
        for i in range(Num_bch):
            flgs[i] = self._compare_envelope_line(xy_range, a, b, gis_bgn[i], gis_end[i])

        return flgs


    def _compare_envelope_line(self, xy_range, a, b, bgn, end):
        '''Auxiliary function to identify if a line is within an envelope'''

        bgn = list(bgn)
        end = list(end)

        # Adjusting values to avoid division by zero
        if end[0] == bgn[0]:
            end[0] = end[0] + 0.00001
            bgn[0] = bgn[0] - 0.00001

        d = (end[1] - bgn[1]) / (end[0] - bgn[0])
        c = end[1] - d * end[0]

        # Intersection points
        x = (a - c) / (d - b)
        y = a + b * x

        flg = np.zeros((6, 4))
        xy_line_range = [np.min([bgn[0], end[0]]),  # Min longitude
                         np.max([bgn[0], end[0]]),  # Max longitude
                         np.min([bgn[1], end[1]]),  # Min latitude
                         np.max([bgn[1], end[1]])]  # Max latitude

        for i in range(4):
            if xy_range[i, 0] <= x[i] <= xy_range[i, 1]:
                flg[2, i] = 1
                if xy_line_range[0] <= x[i] <= xy_line_range[1]:
                    flg[0, i] = 1
                elif x[i] >= xy_line_range[1]:
                    flg[4, i] = 1
                elif x[i] <= xy_line_range[0]:
                    flg[4, i] = -1

            if xy_range[i, 2] <= y[i] <= xy_range[i, 3]:
                flg[3, i] = 1
                if xy_line_range[2] <= y[i] <= xy_line_range[3]:
                    flg[1, i] = 1
                elif y[i] >= xy_line_range[3]:
                    flg[5, i] = 1
                elif y[i] <= xy_line_range[2]:
                    flg[5, i] = -1

            in_out = False
            if np.sum(flg[0:2, :]) > 0:
                in_out = True
            elif np.sum(flg[2, :]) > 0:
                if np.min(flg[4, flg[2, :] == 1]) == -1 and np.max(flg[4, flg[2, :] == 1]) == 1:
                    in_out = True
            elif np.sum(flg[4, :]) > 0:
                if np.min(flg[5, flg[3, :] == 1]) == -1 and np.max(flg[5, flg[3, :] == 1]) == 1:
                    in_out = True

            return in_out

    # Gets:
    def _get_bgn_hr_ws_yr(self):
        '''Get bgn_hr_ws_yr'''
        return self.MC.WS.hrs_yr

    def _get_cp_end_lat_coef(self):
        '''Get cp_lat_n'''
        return self.data.WS.contour.end_lat_coef

    def _get_cp_end_lon(self):
        '''Get cp_lon_n'''
        return self.data.WS.contour.end_lon

    def _get_lim_lng_ws(self):
        '''Get lim_lng_ws'''
        return self.data.WS.event.lng

    def _get_lim_max_v_ws(self):
        '''Get lim_max_v_ws'''
        return self.data.WS.event.max_v

    def _get_lim_min_v_ws(self):
        '''Get lim_max_v_ws'''
        return self.data.WS.event.min_v

    def _get_lim_ttr(self):
        '''Get lim_ttr'''
        return self.data.WS.event.ttr

    def _get_lng_ws(self):
        '''Get lng_ws'''
        return self.MC.WS.lng

    def _get_cp_dis_aggregated(self):
        '''Get cp_dis_aggregated'''
        return self.data.WS.contour.dis

    def _get_cp_start_connectivity(self):
        '''Get cp_from_to'''
        return self.data.WS.contour.start_connectivity

    def _get_cp_start_lat(self):
        '''Get cp_lat'''
        return self.data.WS.contour.start_lat

    def _get_cp_start_lon(self):
        '''Get cp_lon'''
        return self.data.WS.contour.start_lon

    def _get_cp_num(self):
        '''Get cp_num'''
        return self.data.WS.contour.num

    def _get_lim_v_ws(self):
        '''Get lim_v_ws'''
        return self.MC.WS.v

    def _get_max_ws_yr(self):
        '''Get max_ws_yr'''
        return self.data.WS.event.max_yr

    def _get_num_hrs_yr(self):
        '''Get num_hrs_yr'''
        return self.data.num_hrs_yr

    def _get_NumWS_total(self):
        '''Get NumWS_total'''
        return self.MC.WS.total

    def _get_NumWS_yr(self):
        '''Get NumWS_yr'''
        return self.MC.WS.num_yr

    def _get_mcs_yr(self):
        '''Get num_mcs_yr'''
        return self.data.MC.trials

    def _get_lng_ws(self):
        '''Get lng_ws'''
        return self.MC.WS.lng

    def _get_frg_mu(self):
        '''Get frg_mu'''
        return self.data.frg.mu

    def _get_frg_sigma(self):
        '''Get frg_sigma'''
        return self.data.frg.sigma

    def _get_frg_thrd_1(self):
        '''Get frg_thrd_1'''
        return self.data.frg.thrd_1

    def _get_frg_thrd_2(self):
        '''Get frg_thrd_2'''
        return self.data.frg.thrd_2

    def _get_frg_shift_f(self):
        '''Get frg_shift_f'''
        return self.data.frg.shift_f



