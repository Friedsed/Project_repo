# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

Aircraft class:
    Aircraft data without the polar
    m, W, S, lambda, b, e, k,
    v, q, w/S

Flight class :
    flight conditions

@author: Christophe Airiau
"""
import numpy as np
from scipy import constants
from flight.atmosphere_std import SimpleAtmosphere, TZERO, RHOZERO
from flight.aerodynamics import sound_velocity


__G__ = constants.g


class Flight(object):
    """
    flight conditions :
        h : altitude
        mach : range of Mach number
        v : range of velocity
        rho : air density at altitude h
        sigma : rho  / rho at sea level
    """
    def __init__(self, d=None):
        if d is None:
            # d = dict(h=np.linspace(0, 15000, 5), mach=np.linspace(0.2, 0.8, 5))
            self.h = self.mach = self.v = None
            self.rho = self.sigma = None
            print("empty Flight instanciation")
        else:
            self.h = d["h"]
            self.mach = d["mach"]
            self.v = np.zeros((len(self.mach), len(self.h)), dtype=float)
            rho, sigma = [], []
            for i, h_tmp in enumerate(self.h):
                (sigma_h, delta, theta) = SimpleAtmosphere(h_tmp/1000)
                rho.append(sigma_h * RHOZERO)
                sigma.append(sigma_h)
                self.v[:, i] = self.mach * sound_velocity(theta * TZERO)
            self.rho = np.array(rho)
            self.sigma = np.array(sigma)


class Aircraft(object):
    """
    aircraft structure characteristics (class attributes) :
    m : mass [kg]
    s : wing surface [m^2]
    ar : aspect ratio
    b : span [m]
    e : Oswald coefficient
    rho : air density at the flight level
    v : range of flight velocity
    w_s : W/S [N/m^2]
    w : weight : m g
    k : 1 (pi AR e)
    q : dynamic pressure [N]
    one of the parameters in (s, b, ar) is calculated from the other values
    """
    def __init__(self, d=None):
        if d is None:
            d = {"name": "plane", "m": 0, "s": 0, "AR": 0, "b": 0, "e": 0, "rho": 0, "v": np.linspace(10, 100)}
        self.name = d["name"]
        self.m = d["m"]
        self.w = d["m"] * __G__
        self.e = d["e"]
        self.rho = d["rho"]
        if d["s"] == 0:
            self.b, self.ar = d["b"], d["AR"]
            self.s = self.b**2 / self.ar
        elif d["b"] == 0:
            self.s, self.ar = d["s"], d["AR"]
            self.b = np.sqrt(self.s * self.ar)
        else:
            self.s, self.b = d["s"], d["b"]
            self.ar = self.b ** 2 / self.s
        self.w_s = self.w / self.s
        self.k = 1 / (np.pi * self.e * self.ar)
        self.v = d["v"]
        self.q = 1/2 * self.rho * self.v ** 2
        if 0 in self.q:
            print("q=0")
            print(self.q[:10])
        else:
            self.cl = self.w_s / self.q
            self.cl_max = max(self.cl)

    def __repr__(self):
        n = 80
        print("=" * n)
        print("#  " + self.name)
        print("=" * n)
        column_names = "  W       S      W/S      lambda    b      k     rho       v_min   v_max    m"
        column_unit =  "  kN      m^2    N/m^2      -       m      %     kg/m^3    km/h    km/h     t"
        print(column_names)
        print(column_unit)
        form = "%6.2f  %6.2f   %6.1f  %6.2f  %6.2f %6.3f  %6.3f    %6.1f  %6.1f  %7.3f"
        print("-" * n)
        print(form % (self.w / 1000, self.s, self.w_s, self.ar, self.b, self.k * 100,
                      self.rho, self.v[0] * 3.6, self.v[-1] * 3.6, self.m / 1000 ))
        print("=" * n)
        return " "

