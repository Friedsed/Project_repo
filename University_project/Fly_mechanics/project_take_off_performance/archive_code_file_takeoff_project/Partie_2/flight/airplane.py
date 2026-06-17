# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

Airplane class:
     structure
     polar (aerodynamics)
     performances

@author: Christophe Airiau
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import copy
from numpy.polynomial import polynomial
from flight.aircraft import Flight, Aircraft
from flight.engine import engine_thrust_power
from flight.general import set_local_zeros_mat, m2ft, ft2m,  COLORS, watt2hp
from flight.solution_p3 import poly_roots


class Airplane(object):
    def __init__(self):
        print("new instance of Airplane")
        self.str = None
        self.flight = Flight()
        self.polar = None
        self.pr_coeffs = None
        self.pr, self.drag, self.pa, self.thrust = None, None, None, None
        self.excess_power, self.sin_theta, self.vc = None, None, None
        self.theta_c = None
        self.vc_min = 0       # for plot, minimal value of Vc
        self.vc_poly_deg = 3  # degree n of polynomial in V_c = P_n(h)
        # self.engine = dict(type=None, model=None, tr=None, high_bpr=None, hc=None,
        #                    mil=None, pa_sl=None, thrust_sl=None, correction=1)
        self.engine = None
        self.vc_data = None
        self.vc_0 = None
        self.max_perfo = dict()
        self.vc_model_lin, self.vc_model_poly = None, None
        self.ceiling = dict()

    def set_structure_parameters(self, aircraft_str):
        """ enter the structure parameters dictionary """
        self.str = aircraft_str

    def set_flight_parameters(self, flight):
        """ enter the flight parameters dictionary """
        self.flight = flight

    def set_engine(self, engine):
        """ enter the engine dictionary """
        self.engine = engine

    def set_polar(self, polar):
        """ enter the polar dictionary """
        self.polar = polar
        self.polar.set_flight_data(m=self.str.m, s=self.str.s, rho=self.str.rho)

    def set_available_power(self):
        """
        return the available power from the engine model
        """
        nh = len(self.flight.h)
        nv = len(self.flight.mach)
        self.pa = np.zeros((nv, nh), dtype=float)
        self.thrust = np.zeros((nv, nh), dtype=float)
        print("self.engine : ", self.engine)
        if self.engine["type"] == "piston":
            for j in range(nh):
                for i in range(nv):
                    self.pa[i, j] = engine_thrust_power(model=self.engine["type"], h=self.flight.h[j], thrust_sl=1,
                                                        power_sl=self.engine["pa_sl"], mach=self.flight.mach[i],
                                                        delta_oat=0, params=self.engine)
                    self.thrust[i, j] = self.pa[i, j] / self.flight.v[i, j]
        else:
            for j in range(nh):
                for i in range(nv):
                    self.thrust[i, j], self.pa[i, j], _ = engine_thrust_power(model=self.engine["type"],
                                                                              h=self.flight.h[j],
                                                                              thrust_sl=self.engine["thrust_sl"],
                                                                              mach=self.flight.mach[i],
                                                                              delta_oat=0,
                                                                              params=self.engine)

    def set_required_power_coeffs(self):
        """
        set polynomial coefficients of the required power from Polar class
        """
        self.pr_coeffs = self.polar.set_power_coefficients(unit_density=True)

    def set_required_power(self):
        self.set_required_power_coeffs()
        nh = len(self.flight.h)
        nv = len(self.flight.mach)
        self.pr = np.zeros((nv, nh), dtype=float)
        self.drag = np.zeros((nv, nh), dtype=float)
        a1, a2, a3 = self.pr_coeffs
        for j in range(nh):
            v, rho = self.flight.v[:, j], self.flight.rho[j]
            self.pr[:, j] = rho * a1 * v**3 + a2 / (rho * v) + a3 * v
            self.drag[:, j] = self.pr[:, j] / v
    #
    #  Climbing problem
    #

    def set_climbing_performance(self, vb=True):
        """
        return Rate of Climb, Climbing velocity, Excess power, climbing angle, required power
        """
        self.excess_power = self.pa - self.pr
        self.vc = self.excess_power / self.str.w * 60
        self.sin_theta = (self.thrust - self.drag) / self.str.w
        self.theta_c = np.zeros_like(self.sin_theta)
        im = np.where(abs(self.sin_theta) < 1, self.sin_theta, -1)
        self.theta_c = np.rad2deg(np.arcsin(im))
        self.max_perfo["sin_theta"] = np.max(self.sin_theta)
        self.max_perfo["vc"] = np.max(self.vc)
        self.max_perfo["Pe"] = np.max(self.excess_power)
        self.max_perfo["theta_c"] = np.max(self.theta_c)
        self.max_perfo["pr"] = np.max(self.pr)
        self.min_vc(vb=vb, plot=False)
        print(self.max_perfo) # to improve

    def min_vc(self, vb=True, plot=False):
        """
        returns
        h     : m
        V_c   : m/mn
        V_min : km/h
        V_max : km/h
        """
        ht, v_min, v_max, vc_max, counter = [], [], [], [], 0
        zeros = set_local_zeros_mat(self.flight.v, self.vc)
        if vb:
            self.get_zeros(zeros)
        for zero, h, vc in zip(zeros, self.flight.h, self.vc.T):
            vc_m = max(vc)
            if len(zero) > 2:
                print("more than 2 zeros for h = %.2f " % h, zero)
            if vc_m >= self.vc_min:
                vc_max.append(vc_m)
                ht.append(h)
                if len(zero) >= 2:
                    v_min.append(zero[0] * 3.6)
                    v_max.append(zero[-1] * 3.6)
                    counter += 1
                elif len(zero) == 1:
                    v_min.append(zero[0] * 3.6)
                    v_max.append(zero[0] * 3.6)
                else:
                    mean = (v_min[-1] + v_max[-1]) / 2
                    v_min.append(mean)
                    v_max.append(mean)
        self.vc_data = [ht, vc_max, v_min, v_max]
        self.set_vc_0(vb=vb)
        self.set_vmax_at_ceiling(n=counter, vb=vb, plot=plot)

    def set_vc_0(self, vb=False):
        """
        calculate a linear regression and a polynomial regression of the climbing velocity in m/mn Vc(h)
        return maximum ceiling and service ceiling
        """
        # 1 - Linear regression
        slope, intercept, r, p, se = linregress(self.vc_data[0], self.vc_data[1]) # in m, m/mn
        self.ceiling["h_max"] = m2ft(-intercept / slope)
        self.vc_0 = intercept
        self.vc_model_lin = lambda x: slope * x + intercept
        self.ceiling["service_h"] = m2ft((ft2m(100) - intercept) / slope)
        # 2 - Polynomial regression
        p2_h = np.polyfit(self.vc_data[0], self.vc_data[1], self.vc_poly_deg)
        self.vc_model_poly = p2_h
        roots = poly_roots(p2_h, vb=False)
        self.ceiling["h_max_poly"] = m2ft(roots[0].real)
        p3_h = copy.deepcopy(p2_h)
        p3_h[-1] -= ft2m(100)
        roots1 = poly_roots(p3_h, vb=False)
        self.ceiling["service_h_poly"] = m2ft(roots1[0].real)
        if vb:
            self.get_ceiling_data_from_vc(slope, intercept, r)

    def set_vmax_at_ceiling(self, n=0, k_pts=2, vb=False, plot=True):
        """
        calculate the velocity and altitude at the ceiling
        by quadratic polynomial interpolation.
        """
        (h, vc, vmin, vmax) = self.vc_data
        v = np.hstack([vmin[n-k_pts:n], np.flip(vmax[n-k_pts:n])])
        y_ = h[n-k_pts:n] + list(reversed(h[n-k_pts:n]))
        h_ft = [m2ft(eta) for eta in y_]  # altitude in feets
        p2 = np.polyfit(v, h_ft, 2)
        v_ceiling = -0.5 * p2[1] / p2[0]
        self.ceiling["abs_h"] = np.polyval(p2, v_ceiling)
        self.ceiling["V"] = ((vmin[-1] + vmax[-1]) / 2, v_ceiling)
        if vb:
            self.get_ceiling_data_from_v(v, h_ft)
        if plot:
            self.plot_ceiling_v_h(v, h_ft, p2)

    # ***********************************************************
    # GETTERS
    # ***********************************************************
    def get_zeros(self, zeros):
        """
        nice table of the zeros of Vc
        """
        ns = 60
        print()
        print("=" * ns)
        print(" h [m]     h [ft]   v_min [km/h]  v_max [km/h]   vc_max [m/mn]")
        print("-" * ns)
        form = "%7.1f   %6d     %6.1f        %6.1f        %6.1f       %d"
        for zero, h, vc in zip(zeros, self.flight.h, self.vc.T):
            vc_m = max(vc)
            if zero:
                if len(zero) >= 2:
                    print(form % (h, int(m2ft(h)), zero[0] * 3.6, zero[1] * 3.6, vc_m, len(zero)))
                elif len(zero) == 1:
                    print(form % (h, int(m2ft(h)), zero[0] * 3.6, 0, vc_m, len(zero)))
            else:
                print(form % (h, m2ft(h), 0, 0, vc_m,  len(zero)))
        print("=" * ns)
        print()

    def get_ceiling_data_from_vc(self, slope, intercept, r):
        """
        display data solved from Vc(h)
        """
        print("vc(h) polynomial regression:")
        print("h_max (poly)       : %d ft = %1.f m" % (self.ceiling["h_max_poly"], ft2m(self.ceiling["h_max_poly"])))
        print("h_service (poly)   : %d ft = %.1f m " % (self.ceiling["service_h_poly"],
                                                        ft2m(self.ceiling["service_h_poly"])))

        print("vc(h) linear regression:")
        print("slope, intercept, r ", slope, intercept, r)
        print("vc_0               : %.3f m/mn (lin. reg)" % self.vc_0)
        print("vc_0               : %.3f m/mn (calc)" % self.vc_data[1][0])
        print("h_max              : %.3f m(absolute ceiling from lin. reg.)" % ft2m(self.ceiling["h_max"]))
        print("h_max              : %.d ft (absolute ceiling from lin. reg.)" % self.ceiling["h_max"])

    def get_ceiling_data_from_v(self, v, h_ft):
        """
        display data solved from V(h)
        """
        print("Vc(V): quadratic interpolation")
        # print("v [km/h] : ", v)
        # print("h [ft]   : ", h_ft)
        print("Velocity at ceiling (mean) : %.3f km/h" % self.ceiling["V"][0])
        print("Velocity at ceiling (P_2)  : %.3f km/h " % self.ceiling["V"][1])
        print("Absolute ceiling  (P_2)    : %.d ft" % self.ceiling["abs_h"])
        print("Service ceiling            : %.d ft" % self.ceiling["service_h"])

    # ***********************************************************
    # PLOTS
    # ***********************************************************
    @staticmethod
    def plot_ceiling_v_h(v, h_ft, p2):
        """
        a plot closed to the maximum ceiling of regression laws
        """
        x_tmp = np.linspace(v[0], v[-1], 51)
        plt.figure()
        plt.title("Interpolation for the velocity at the ceiling")
        plt.plot(v, h_ft)
        plt.plot(x_tmp, np.polyval(p2, x_tmp))
        plt.xlabel("v [km/h]")
        plt.ylabel("h [ft]")

    def plot_climbing_performance(self, mach=True, options=None):
        """
        Plots of Vc, Pe, ROC, and theta_c w.r.t. velocity or Mach number
        """
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(10, 8))
        # Trac√© des graphiques
        # for i, h in enumerate(self.flight.h):
        # colors = [ "blue", "orange", "green", "red", "purple",  "brown", "pink", "gray", "olive", "cyan",
        #           "black", "magenta"]
        leg = ("V[km/h]", "Mach")
        x_label = leg[0]
        if options is None:
            options = dict(h_ft=False)
        elif "h_ft" not in options.keys():
            options["h_ft"] = False

        for i in range(len(self.flight.h)):
            h_ = self.flight.h[i]
            if options["h_ft"]:
                h = m2ft(h_)
            else:
                h = h_
            if mach:
                x = self.flight.mach
                x_label = leg[1]
            else:
                x = self.flight.v[:, i] * 3.6
            axs[0, 0].plot(x, self.sin_theta[:, i], label="h = %.0f" % h)
            axs[0, 1].plot(x, self.vc[:, i], label="h = %.0f" % h)
            axs[1, 0].plot(x, self.theta_c[:, i], label="h = %.0f" % h)
            axs[1, 1].plot(x, self.excess_power[:, i] / 1e6, label="h = %.0f" % h)
            axs[1, 1].fill_between(x, self.excess_power[:, i] / 1e6, 0, alpha=0.5)

        axs[0, 0].set_ylim(-0.01, self.max_perfo["sin_theta"] * 1.1)
        axs[0, 0].set_ylabel("(T - D) / W")
        axs[0, 0].legend()
        axs[0, 1].set_ylim(-10, self.max_perfo["vc"] * 1.1)
        axs[0, 1].set_ylabel(r"$V_c$ [m/mn]")
        axs[1, 0].set_ylim(-1, self.max_perfo["theta_c"] * 1.1)
        axs[1, 0].set_ylabel(r"$\theta_c$ [∞]")
        axs[1, 1].set_ylim(0, self.max_perfo["Pe"]/1e6 * 1.1)
        axs[1, 1].set_ylabel(r"$P_a - P_r$ [MW]")
        axs[1, 0].set_xlabel(x_label)
        axs[1, 1].set_xlabel(x_label)

    def plot_vc_data(self, reg=False):
        """
        reg: regression flag for V_min and V_max
        """
        fig, axs = plt.subplots(2, sharex=True, figsize=(10, 8))
        axs[0].plot(self.vc_data[0], self.vc_data[1], label= r"$Vc_{\max}$ (calc.)")
        axs[0].plot(self.vc_data[0], self.vc_model_lin(np.array(self.vc_data[0])), "ko",
                    markersize=3, label="Linear reg.")
        axs[0].plot(self.vc_data[0], np.polyval(self.vc_model_poly, np.array(self.vc_data[0])), "rs",
                    markersize=3, label="poly reg., deg %d" % self.vc_poly_deg)
        axs[1].plot(self.vc_data[0], self.vc_data[2], "r-o", markersize=3, label="V min")
        axs[1].plot(self.vc_data[0], self.vc_data[3], "b-o", markersize=3, label="V max")
        if reg:
            iy = 0
            p, x, y = [], [], []
            for ix in [2, 3]:
                p.append(polynomial.polyfit(self.vc_data[ix], self.vc_data[iy], 5))
                x_tmp = np.linspace(self.vc_data[ix][0], self.vc_data[ix][-1], 201)
                y_tmp = np.polyval(np.flip(p[-1]), x_tmp)
                x.append(x_tmp)
                y.append(y_tmp)
            for ix in range(2):
                axs[1].plot(y[ix], x[ix],  label="reg")
        axs[0].set_ylabel(r"$Vc_{\max}$ [m/mn]")
        axs[0].legend()
        axs[1].set_xlabel("h [m]")
        axs[1].legend()

    def plot_drag_thrust(self, mach=True, maxval=None):
        """
        D and T versus velocity or Mach number
        """
        plt.figure()
        plt.title(r"Drag ($D$) and Thrust (dashed)")
        leg = ("V[km/h]", "Mach")
        x_label = leg[0]
        for i in range(len(self.flight.h)):
            h = self.flight.h[i]
            if mach:
                x = self.flight.mach
                x_label = leg[1]
            else:
                x = self.flight.v[:, i] * 3.6
            plt.plot(x, self.drag[:, i] / 1000, color=COLORS[i], label=r"$D$, h = %.0f ft" % m2ft(h))
            plt.plot(x, self.thrust[:, i] / 1000, color=COLORS[i],  linestyle="dashed", lw=2)
        plt.xlabel(x_label)
        if maxval is None:
            maxval = np.amax(self.thrust) / 1000 * 1.2
        plt.ylim(0, maxval)
        plt.ylabel(r"$D, T$ [kN]")
        plt.legend(loc="best")

    def finess_propulsion_rate(self, mach=True):
        """
        W/D and T/W versus velocity or Mach number
        """
        fig, axs = plt.subplots(2, sharex=True, figsize=(10, 8))
        plt.suptitle(r"W/D and T/W ")
        leg = ("V[km/h]", "Mach")
        x_label = leg[0]
        for i in range(len(self.flight.h)):
            h = self.flight.h[i]
            if mach:
                x = self.flight.mach
                x_label = leg[1]
            else:
                x = self.flight.v[:, i] * 3.6
            axs[0].plot(x, self.str.w / self.drag[:, i], color=COLORS[i], label=r"$f$, h = %.0f ft" % m2ft(h))
            axs[1].plot(x, self.thrust[:, i] / self.str.w, color=COLORS[i],  linestyle="dashed", lw=2)
        axs[1].set_xlabel(x_label)
        axs[0].set_ylabel(r"$f=W/D$")
        axs[1].set_ylabel(r"$f=T/W$")
        axs[0].legend(loc="best")

    def plot_drag(self, mach=True, maxval=None):
        """
        D and T versus velocity or Mach number
        """
        plt.figure()
        plt.title(r"Drag ($D$)")
        leg = ("V[km/h]", "Mach")
        x_label = leg[0]
        for i in range(len(self.flight.h)):
            h = self.flight.h[i]
            if mach:
                x = self.flight.mach
                x_label = leg[1]
            else:
                x = self.flight.v[:, i] * 3.6
            plt.plot(x, self.drag[:, i] / 1000, color=COLORS[i], label=r"$D$, h = %.0f ft" % m2ft(h))
        plt.xlabel(x_label)
        if maxval is None:
            maxval = np.amax(self.drag[:, i] / 1000) * 0.8
        plt.ylim(0, maxval)
        plt.ylabel(r"$D, T$ [kN]")
        plt.legend(loc="best")

    def plot_power(self, mach=True, option=None):
        """
        Pr and Pa versus velocity or Mach number
        """
        if option is None:
            option = dict(power_unit="kW")
        elif "power_unit" not in option.keys():
            option["power_unit"] = "kW"
        plt.figure()
        if option["power_unit"] == "kW":
            scale = 1000
        elif option["power_unit"] == "hp":
            scale = 745.7
        else:
            raise ValueError("bad unit in plot_power")
        leg = ("V[km/h]", "Mach")
        x_label = leg[0]
        for i in range(len(self.flight.h)):
            h = self.flight.h[i]
            if mach:
                x = self.flight.mach
                x_label = leg[1]
            else:
                x = self.flight.v[:, i] * 3.6
            plt.plot(x, self.pr[:, i] / scale, color=COLORS[i], label=r"$P_r$, h = %.0f ft" % m2ft(h))
            plt.plot(x, self.pa[:, i] / scale , color=COLORS[i],  linestyle="dashed", lw=2)
        plt.xlabel(x_label)

        plt.title(r"Required Power ($P_r$) and Available Powers $P_a$ (dashed)")
        pmax = np.amax(self.pa) / scale
        pmin = np.amin(self.pr) / scale
        plt.ylim(pmin * 0.8, pmax * 1.1)
        plt.ylabel(r"$P_r, P_a$ [%s]" % option["power_unit"])
        plt.legend(loc="best")

    def plot(self, choice=None, mach=False):
        """
        various plots, main method for plotting
        """
        if choice is None:
            choice = dict(perfo=False, force=False, power=True, vc=False, power_unit="kW", h_ft=False)
        if choice["perfo"]:
            self. plot_climbing_performance(mach=mach, options=choice)
        if choice["force"]:
            self.plot_drag_thrust(mach=mach)
        if choice["power"]:
            self.plot_power(mach=mach, option=choice)
        if choice["vc"]:
            self.plot_vc_data()
        # if choice["f_tau"]:
        #     self.finess_propulsion_rate(mach=mach)







