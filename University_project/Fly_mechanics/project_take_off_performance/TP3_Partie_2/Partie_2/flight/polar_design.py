# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

polar design module :
    define generic polar with bezier interpolation
    any symmetrical polar
    get approximate quadratic polar

    when the polar is defined points, translation and scaling are possible to modify it
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import constants
from copy import deepcopy
# from flight.solution_p3 import Resolution, print_roots_
from flight.bezier import evaluate_bezier, get_cubic, get_bezier_coef, get_bezier_cubic, \
    get_points_from_curves
from flight.quadratic_polar import QuadraticPolar
from flight.aircraft import Aircraft
from flight.aero_fields import AeroFields

plt.style.use("seaborn")
INCH = constants.inch * 100
__FIGSIZE__ = (30 / INCH, 20 / INCH)
__G__ = constants.g
__LW__ = 3


def polar_unit_model(case=0):
    """
    different model can be implemented
    only a positive part, symmetrical
    case = 0 : parabola
           1 : first example
    """
    if case == 0:
        x = np.linspace(0, 1, 11)
        data = []
        for eta in x:
            data.append([eta, eta**2])
    elif case == 1:
        data = [[0, 0.04], [0.02, 0.041], [0.05, 0.045], [0.1, 0.05], [0.15, 0.06], [0.18, 0.08], [0.38, 0.10],
                [0.5, 0.2], [0.65, 0.4225], [0.8, 0.64], [0.9, 0.81], [1, 1]]
    else:
        raise ValueError("bad choice in polar_unit_model")
    return data


class Polar(object):
    """
    Class to generate a polar from various ways
    polar : C_D = f(C_L)
    index 0 : problem with max (C_L/C_D)
          1 : problem with min (Pr)
    """
    def __init__(self, data=None):
        """
        class initialization
        """
        self.print_title()
        self.polar_points = data
        self.oswald_coef = 0.85
        self.data = None
        self.aircraft = None
        self.s = [0, 0]             # for scaling
        self.t = [0, 0]             # for translation
        self.pt_test = [0.4, 0.6]   # point tests
        self.npt = 51              # number of points along C_L axis
        self.n = 21                 # number of points per panel
        self.eta = None
        self.points = None
        self.polar = None           # Bézier curves
        self.fig_polar = None
        self.fig_test = None
        self.cs = self.p2 = None   # for cubic spline and P2 interpolations
        self.f, self.pr = None, None              # aerodynamic efficiency
        self.pr_coef_square = None     # |(C_D / C_L^{3/2})^2|

        self.local = self.para = None
        self.q_polar = None   # quadratic polar approximation
        self.cl_lim = [0, 0]
        self.cd_lim = [0, 0]
        self.cl_opt, self.cd_opt, self.f_opt = [0, 0], [0, 0], [0, 0]
        self.v_opt, self.pr_opt, self.q_opt = [0, 0], [0, 0], [0, 0]

    def run(self):
        """
        main methods
        """
        self.set_symmetrization()

    # =========================================
    # setters
    # =========================================

    def set_aircraft(self, aircraft=None):
        """
        enter data related to the structure and flight
        """
        if aircraft is None:
            aircraft = {"s": 50,
                        "m": 7800,
                        "rho": 0.7,
                        "v": np.linspace(240, 400, 200) / 3.6}
            self.aircraft = Aircraft(aircraft)
        else:
            self.aircraft = aircraft

    def set_oswald_coefficient(self, e=1):
        """ """
        self.oswald_coef = e

    def polar_transformation(self, s=(1, 1), t=(0, 0)):
        """
        symmetrization, scaling along C_L and C_D axes, translation.
        """
        self.set_symmetrization()
        self.data = self.set_scale(s)
        self.data = self.set_translation(t)

    def set_polar_with_bezier_curves(self, plot=False):
        """
        Define the polar with Bézier curves.
        """
        self.eta = np.linspace(self.data[0][0], self.data[-1][0], self.npt)
        self.points = np.array(self.data)
        self.polar = evaluate_bezier(self.points, self.n)
        # x, y = points[:, 0], points[:, 1]
        # px, py = path[:, 0], path[:, 1]
        self.set_polar_limits()
        if plot:
            self.plot_polar()

    def set_polar_limits(self):
        """
        return the limit in C_L and C_D of the given polar.
        Simulations can not be outside the limits for safety
        """
        self.cl_lim = [self.polar[0, 0], self.polar[-1, 0]]
        self.cd_lim[0] = min(self.polar[:, 1])
        self.cd_lim[1] = max(self.polar[:, 1])

    def get_polar_limits(self):
        """
        Nice table of the real polar limits
        """
        nc = 45
        print("=" * nc )
        print(" " * 7, "C_L", " " * 14, "C_D x 100" )
        print("   min       max " + " " * 6 + "   min       max ")
        print("-" * nc)
        form = " %6.3f    %6.3f       %6.3f    %6.3f "
        print(form % (self.cl_lim[0], self.cl_lim[1], self.cd_lim[0] * 100, self.cd_lim[1] * 100))
        print("=" * nc)

    def set_data(self, data):
        self.polar_points = data

    def set_symmetrization(self):
        """
        symmetrization
        """
        data_neg = deepcopy(self.polar_points[1:])
        data_neg.reverse()
        for k in range(len(data_neg)):
            data_neg[k][0] *= -1
        self.data = data_neg + self.polar_points

    def set_translation(self, t=(0, 0)):
        """ translation in C_L and C_D"""
        out = []
        for d_ in self.data:
            out.append([d_[0] + t[0], d_[1] + t[1]])
        return out

    def set_scale(self, s=(0, 0)):
        """ scaling in C_L and C_D"""
        out = []
        for d_ in self.data:
            out.append([d_[0] * s[0], d_[1] * s[1]])
        return out

    def set_parabola_approximation(self, plot=False):
        """
        return a parabola model from the true polar
        """
        self.p2 = np.polyfit(self.points[:, 0], self.points[:, 1], 2)
        print("Parabola :")
        print("Polynomial   : ", self.p2)
        self.q_polar = QuadraticPolar(d=dict(p_2=self.p2))
        self.q_polar.set_aspect_ratio(e=self.oswald_coef)
        print(self.q_polar)
        self.q_polar.solve(m=self.aircraft.m, s=self.aircraft.s, rho=self.aircraft.rho)
        if plot:
            self.plot_parabola()
        cd = np.polyval(self.p2, self.aircraft.cl)
        self.para = AeroFields("para", self.aircraft, cd)

    def display_aerodynamic_efficiency(self):
        """ """
        print()
        n = 60
        print("=" * n)
        print("# Aerodynamic efficiency (finess) ")
        print("=" * n)
        print(" f_max         CL     10 000 x CD     V [km/h]     q [Pa]     Pr [kW]")
        print("-" * n)
        form = " %5.2f       %5.3f       %5.1f         %5.1f        %5.1f      %5.1f"
        print(form % (self.ae["f_max"], self.ae['cl'], self.ae["cd"] * 10000, self.ae["v"] * 3.6, self.ae["q"],
                      self.ae["Pr"] / 1000))
        print("=" * n)

    def set_cubic_splines(self, plot=False):
        """ """
        self.cs = CubicSpline(self.points[:, 0], self.points[:, 1])
        if plot:
            self.plot_polar_with_splines()

    def test_point(self, point, var="x"):
        """ """
        ind, t_value, x_value, y_value = get_points_from_curves(self.points, point, var=var)
        self.plot_portion(ind, [x_value, y_value])
        print("panel for %s = %f    : %d" % (var, point[var], ind))
        print("points :  t = %f, x = %f, y = %f" % (t_value, x_value, y_value))

    def set_aero_efficiency(self, plot=False):
        """ """
        index = [0, 0]
        self.f = self.polar[:, 0] / self.polar[:, 1]
        self.pr_coef_square = abs(self.polar[:, 1]**2 / self.polar[:, 0]**3)
        index[0] = np.argmax(self.f)
        index[1] = np.argmin(self.pr_coef_square)
        for k, ind in enumerate(index):
            self.cl_opt[k] = self.polar[ind, 0]
            self.cd_opt[k] = self.polar[ind, 1]
            self.f_opt[k] = self.cl_opt[k] / self.cd_opt[k]
            self.q_opt[k] = self.aircraft.w / (self.aircraft.s * self.cl_opt[k])
            self.v_opt[k] = np.sqrt(2 * self.q_opt[k] / self.aircraft.rho)
            self.pr_opt[k] = self.q_opt[k] * self.aircraft.s * self.cd_opt[k] * self.v_opt[k]
        if plot:
            self.plot_cl_cd_funs()

    def set_required_power(self, plot=False):
        """
        return the required power in a range of velocity
        """
        cl, v, q = self.aircraft.cl, self.aircraft.v, self.aircraft.q
        if self.aircraft.cl_max > self.cl_lim[1]:
            raise ValueError("Cl_max outside the polar")
        print("cl_max  of the considered flight  :  %.3f" % self.aircraft.cl_max)
        cd = np.interp(cl, self.polar[:, 0], self.polar[:, 1])   # first order interpolation from Béziers
        self.local = AeroFields("local", self.aircraft, cd)
        print(self.local)
        if plot:
            self.plot_power()

    def plot_power(self, unit="km/h"):
        fig, axs = plt.subplots(2, 2, figsize=__FIGSIZE__)
        fig.suptitle("Designed polar (red) and parabola model (blue)", fontsize=16)
        v = self.local.v * self.set_coef_unit(unit)
        axs[0, 0].plot(v, self.local.cl, "r-")
        axs[0, 0].set_xlabel(r"$v$ [%s]" % unit)
        axs[0, 0].set_ylabel(r"$C_L$")
        axs[0, 1].set_xlabel(r"$v$ [%s]" % unit)
        axs[0, 1].set_ylabel(r"$C_L$")
        axs[0, 1].plot(v, self.local.f, "r-", lw=__LW__)
        axs[0, 1].set_xlabel(r"$v$ [%s]" % unit)
        axs[0, 1].set_ylabel(r"$f=C_L/C_D$")
        axs[1, 0].plot(v, self.local.pr/1000, "r-", lw=__LW__)
        axs[1, 0].set_xlabel(r"$v$ [%s]" % unit)
        axs[1, 0].set_ylabel(r"$P_r$ [kW]")
        axs[1, 1].plot(self.local.cl, self.local.cd, "r-", lw=__LW__, label="present")
        axs[1, 1].set_xlabel(r"$C_L$")
        axs[1, 1].set_ylabel(r"$C_D$")
        if self.para:
            axs[0, 1].plot(v, self.para.f, "b--", lw=__LW__)
            axs[1, 1].plot(self.para.cl, self.para.cd, "b--", lw=__LW__, label="parabola")
            axs[1, 0].plot(v, self.para.pr/1000, "b--", lw=__LW__)
        axs[1, 1].legend()

    @staticmethod
    def set_coef_unit(unit):
        d = {"km/h": 3.6, "m/s": 1}
        if unit in d.keys():
            return d[unit]
        else:
            raise ValueError("bad unit")

    # =========================================
    # getters
    # =========================================

    def get_data(self):
        """
        data displayed in a nice table
        """
        title = ("aerodynamic efficiency", "required power")
        column_title = " CL     CD x 10 000       f       V [km/h]    P [kW]    q [Pa] \t optimal criteria"
        form = " %6.3f   %6.1f      %6.2f     %6.1f      %6.2f     %6.1f \t %s"
        n = 100
        print("=" * n)
        print(column_title)
        print("-" * n)
        for i in range(2):
            print(form % (self.cl_opt[i], self.cd_opt[i]*1e4, self.f_opt[i], self.v_opt[i]*3.6, self.pr_opt[i]/1000,
                          self.q_opt[i], title[i]))
        # print("Ratios state 2 / state 1")
        print("-" * n)
        form1 = " %6.4f   %6.4f      %6.4f     %6.4f      %6.4f     %6.4f \t %s "
        print(form1 % (self.cl_opt[1] / self.cl_opt[0], self.cd_opt[1] / self.cd_opt[0], self.f_opt[1] / self.f_opt[0],
                       self.v_opt[1] / self.v_opt[0], self.pr_opt[1] / self.pr_opt[0], self.q_opt[1] / self.q_opt[0],
                       "Ratios state 2 / state 1"))
        print("=" * n)

    @staticmethod
    def fun_cl_cd(cl, cd, opt=0):
        """
        opt = 0 : return C_L / C_D
            = 1 : return C_D / C_L^{3/2}
        """
        if opt == 0:
            return cl / cd
        else:
            return cd / pow(abs(cl), 3 / 2)

    @staticmethod
    def get_cd_scale_from_aspect_ratio(ae, e=0.85, cl_scale=1.):
        """
        return the scale to set on the C_D to get a given wing aspect ratio with a given scale on C_L
        """
        print("C_L and aspect ratio fixed")
        k = 1 / (np.pi * e * ae)
        cd_scale = k * cl_scale ** 2
        print("C_L scale           : %.3f " % cl_scale)
        print("k (coeff in C_L^2)  : %.3f / 100" % (k * 100))
        print("C_D scale to set    : %.3f" % cd_scale)

    # =========================================
    # Plots
    # =========================================

    def plot_cl_cd_funs(self):
        """ """
        scale = (1, 100)
        legy = (r"$C_L~/~C_D$", r"$100 \times C_D~/~C_L^{3/2}$")
        title = ("Aerodynamic efficiency", "Required Power Coefficient")
        for k in range(2):
            plt.figure(figsize=__FIGSIZE__)
            c = scale[k]
            plt.plot(self.polar[:, 0], c * self.fun_cl_cd(self.polar[:, 0], self.polar[:, 1], opt=k),
                     'b-', lw=__LW__, label="Béziers")
            if self.p2 is not None:
                plt.plot(self.eta, c * self.fun_cl_cd(self.eta, np.polyval(self.p2, self.eta), opt=k),
                         lw=__LW__, color="orange", label="Parabola")
            if self.cs:
                plt.plot(self.eta, c * self.fun_cl_cd(self.eta,  self.cs(self.eta), opt=k), 'k--',
                         lw=__LW__, alpha=0.5, label="Cubic Spline")
            plt.plot(self.points[:, 0], c * self.fun_cl_cd(self.points[:, 0], self.points[:, 1], opt=k) , 'ro',
                     label="data", markersize=8)
            plt.legend(loc="best")
            plt.xlabel(r"$C_L$")
            plt.ylabel(legy[k])
            plt.title(title[k], fontsize=16)
            if k == 1:
                plt.xlim(0.2, 1.5)
                plt.ylim(2, 10)

    def plot_polar(self):
        """ """
        print("plot polar")
        self.fig_polar = plt.figure(figsize=__FIGSIZE__)
        plt.plot(self.polar[:, 0], self.polar[:, 1], 'b-', label="Béziers")
        plt.plot(self.points[:, 0], self.points[:, 1], 'ro', label="data", markersize=5)
        plt.legend(loc="best")
        plt.xlabel(r"$C_L$")
        plt.ylabel(r"$C_D$")
        plt.title("Polar")

    def plot_parabola(self):
        """  """
        plt.figure(self.fig_polar)
        plt.plot(self.eta, np.polyval(self.p2, self.eta), color="orange", label="Parabola")
        plt.legend(loc="best")

    def plot_polar_with_splines(self):
        """ """
        plt.figure(self.fig_polar)
        plt.plot(self.eta, self.cs(self.eta), 'k--', alpha=0.5, label="Cubic Spline")
        plt.legend(loc="best")

    def plot_portion(self, i, pt=None):
        """
        """
        print(" %f <= x <= %f " % (self.points[i][0], self.points[i + 1][0]))
        t = np.linspace(0, 1, self.n)
        curves = get_bezier_cubic(self.points)
        u = np.array([curves[i](t) for t in t])
        if self.fig_test is None:
            self.fig_test = plt.figure()
        else:
            plt.figure(self.fig_test)
        plt.plot(u[:, 0], u[:, 1], 'r-', label="Béziers")
        if pt:
            plt.plot(pt[0], pt[1], "bs")

    @staticmethod
    def plot_cl_vs_alpha(cl_alpha=2 * np.pi):
        """ """
        alpha = np.linspace(0, 15, 31)
        cl = cl_alpha * np.deg2rad(alpha)
        plt.figure(figsize=(8, 6))
        plt.title(r"linear law, $C_L = C_{L_\alpha} \alpha$,     $C_{L_\alpha} =$ %.2f" % cl_alpha)
        plt.plot(alpha, cl)
        plt.xlabel(r"$alpha$ [°]")
        plt.ylabel(r"$C_L$")

    @staticmethod
    def print_title():
        """ """
        title = """
        # *******************************
            Polar for Wing or Aircraft
               C. Airiau, March 2023 
        # *******************************
        """
        print(title + "\n")