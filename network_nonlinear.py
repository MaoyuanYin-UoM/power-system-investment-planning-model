# This script contains the network engine using non-linear power flow models (unfinished)
# Some codes are imported from the MES project with minor modifications


import numpy as np
import cmath, math
import scipy.sparse

from config import NetConfig

class Object(object):
    pass


class NetworkClass:
    def __init__(self, obj=None):

        # Get default values from config
        if obj == None:
            obj = NetConfig()

        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))


        self.data.net.bch_G, self.data.net.bch_B = self._get_admittance()
        self.data.net.Ybus = self._get_Ybus()


    def _get_admittance(self):
        """Convert impedance R, X to admittance G, B"""
        # Get R and X values
        bch_R = self._get_resistance()
        bch_X = self._get_reactance()
        
        # compute G and B values for all branches
        bch_G = []
        bch_B = []
        for i in range(len(bch_R)):
            val = (bch_R[i] ** 2 + bch_X[i] ** 2) ** 0.5
            ang = bch_R[i] / val
            bch_G.append(bch_R[i] / val / val)
            bch_B.append(-bch_X[i] / val / val)

        return bch_G, bch_B


    def _get_Ybus(self):
        """Calculate the Ybus matrix using branch admittances."""
        # Ensure admittance values are computed
        if not self.data.net.bch_G or not self.data.net.bch_B:
            self._get_admittance()

        # Number of buses and lines
        num_buses = len(self.data.net.bus)
        num_lines = len(self.data.net.bch)

        # Initialize Ybus matrix
        Ybus = np.zeros((num_buses, num_buses), dtype=complex)

        # Populate Ybus
        for b in range(num_lines):
            start_bus = self.data.net.bch[b][0] - 1  # Start bus (zero-indexed)
            end_bus = self.data.net.bch[b][1] - 1  # End bus (zero-indexed)

            # Admittance values
            bch_G = self.data.net.bch_G[b]
            bch_B = self.data.net.bch_B[b]
            admittance = bch_G + bch_B * 1j

            # Off-diagonal elements
            Ybus[start_bus, end_bus] -= admittance
            Ybus[end_bus, start_bus] -= admittance

            # Diagonal elements
            Ybus[start_bus, start_bus] += admittance
            Ybus[end_bus, end_bus] += admittance

            return Ybus


    def _get_polar(self):
        """Get data in polar form."""
        # Number of buses
        num_buses = len(self.data.net.bus)

        # Slack bus
        slack = self.data.net.slack_bus[0] - 1  # Adjusted for zero-based indexing

        # Power injections (active and reactive)
        gen_active = self.data.net.gen_active
        gen_reactive = self.data.net.gen_reactive
        demand_active = self.data.net.demand_active
        demand_reactive = self.data.net.demand_reactive
        base = self.data.net.base

        P = [(gen_active[i] - demand_active[i]) / base for i in range(num_buses)]
        Q = [(gen_reactive[i] - demand_reactive[i]) / base for i in range(num_buses)]

        # Ybus
        Ybus = self.data.net.Ybus
        ang_Y = np.angle(Ybus)
        mag_Y = np.abs(Ybus)

        # Voltages - Assuming flat start
        voltage_magnitude = np.ones(num_buses)
        voltage_angle = np.zeros(num_buses)
        voltage_magnitude[slack] = self.data.net.slack_voltage

        # Update data object
        self.data.net.voltage_magnitude = voltage_magnitude
        self.data.net.voltage_angle = voltage_angle

        return P, Q, ang_Y, mag_Y, voltage_magnitude, voltage_angle, num_buses, slack
        

    # Gets:
    def _get_resistance(self):
        return self.data.net.bch_R

    def _get_reactance(self):
        return self.data.net.bch_X