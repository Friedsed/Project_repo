# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

design module :
    functions to calculate v_max at sea level from max turboprop power or thrust_sl known
    analysis of delta_0, theta_0, theta-break point
@author: Christophe Airiau

"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from flight.atmosphere_std import Atmosphere
from flight.atmosphere_11000 import atmos, temp_0, rho_0, p_0, a_0
from flight.general import ft2m, m2ft, k2r, r2k, HP, FT, G_REF, GAMMA
from flight.engine import set_theta_0, set_delta_0, set_mach_from_theta_0, set_pi_c, set_c1
from flight.engine import plot_theta0_Mach, plot_map_mach_h_theta_0, plot_pic_vs_theta0
from flight.aerodynamics import ti_over_t, pi_over_p, sound_velocity
from flight.solution_p3 import poly_roots


# use of constant values at sea level
def turboprop_thrust_ratio(v, gamma=1.4):
    """ T/T_sl"""
    mach = v / a_0
    delta_0 = pi_over_p(mach, gamma)
    return (1 - 0.96 * pow(mach - 0.1, 0.25)) * delta_0


def power_constante_poly(cd_0, cl_0, k, w, s, rho, pa_max):
    """
    a root of the polynomial will provide the maximum speed at s.l.
    Pa_max is given and imposed
    for any aircraft, just related to polar
    """
    scale = 1e-6  # just to have small figures
    a_4 = rho * s / 2 * (cd_0 + k * cl_0 ** 2) * scale
    a_2 = -2 * k * w * cl_0 * scale
    a_1 = - pa_max * scale
    a_0 = 2 * k * w ** 2 / (rho * s) * scale
    return [a_4, 0, a_2, a_1, a_0]


def thrust_given_fun(v, cd_0, cl_0, k, w, s, rho, thrust_sl):
    """
    the root of this function returns the maximum speed at s.l.
    The reference thrust at s.l. is imposed
    only valid for a turboprop, otherwise modify the funnction
    """
    a_4 = rho * s / 2 * (cd_0 + k * cl_0 ** 2)
    a_2 = -2 * k * w * cl_0
    a_1 = - turboprop_thrust_ratio(v) * thrust_sl
    a_0 = 2 * k * w ** 2 / (rho * s)
    # print(a_2/a_1)
    return (a_4 * v ** 4 + (a_2 + a_1) * v ** 2 + a_0) * 1e-6


def eval_turboprop_thrust_sl(v_max=550, pa_max=747 * 746, gamma=1.4, vb=True):
    """
    for a turboprop
    considering h = 0, maximun thrust
    Inputs:
        v_max  : maximum velocity [km/h]
        pa_max : maximum available power [W]
    return T_sl [N]
    """
    v = v_max / 3.6
    mach, thrust = v / a_0, pa_max / v
    thrust_ratio = turboprop_thrust_ratio(v)
    thrust_sl = thrust / thrust_ratio
    if vb:
        print("M : %.3f, Pa: %.2f kW, F : %.3f N" % (mach, pa_max / 1000, thrust))
        print("F/F_SL : %.3f, F_sl = %.2f  N" % (thrust_ratio, thrust_sl))
    return thrust_sl


def plot_power_over_thrust_sl(v, par, d, f_ref):
    for key in par.keys():
        plt.figure()
        for val, data in zip(par[key], d[key]):
            print('parameter values :', val)
            plt.plot(v * 3.6, data, label=key + ": %.3f" % val)
        plt.plot([v[0] * 3.6, v[-1] * 3.6], [0, 0], lw=3)
        plt.plot(v * 3.6, f_ref, "ko", label="ref", lw=3)
        plt.legend()
        plt.title(r"Turboprop, TBM 700: $f(V) = 0$ or $P_4(V) = 0$")
        plt.ylim(-10, 20)

        plt.xlabel("v [km/h]")


def plot_available_power_turboprop(v, thrust_sl):
    plt.figure()
    plt.plot(v * 3.6, turboprop_thrust_ratio(v) * v / HP * thrust_sl, lw=3)
    plt.xlabel("v [km/h]")
    plt.ylabel("$P_a$ [hp]")
    plt.title("Available power")


def power_constant_max_speed(v, cd_0, cl_0, k, w, s, rho, pa_max):
    """
    """
    # keep a reference value with a constant available power pa_max
    p_ref = power_constante_poly(cd_0, cl_0, k, w, s, rho, pa_max)
    r = poly_roots(p_ref, vb=False)
    v_sol = r[1] * 3.6  # to verify if it is the right index, right solution
    f_ref = np.polyval(p_ref, v)  # function to solve with thrust sl given
    print("AVAILABLE POWER GIVEN : ")
    print("Polynomial_coefficient   : ", p_ref)
    print("Polynomial roots         :", r)
    print("Pa given :  v_max        :  km/h", v_sol)
    return f_ref


def set_parameters_dict(keys, values):
    par = dict()
    for key, value in zip(keys, values):
        par[key] = value
    print("parameters               : ", par)
    return par


def get_fmax(cd0, cl0, k):
    eps2 = k * cl0**2 / cd0
    print("f_max       : %.2f" %(0.5 / np.sqrt(k * cd0) * (np.sqrt(1+eps2) + np.sqrt(eps2))))


def constant_thrust_sl_max_speed(v, par, cd_0, cl_0, k, w, s, rho, pa_max, thrust_sl):
    """
    """
    print("THRUST AT SEA LEVEL GIVEN")
    # power_ = v * turboprop_thrust_ratio(v)        # T V  / T_sl
    d = dict.fromkeys(par.keys())
    for key in d.keys():
        d[key] = []
    for key in par.keys():
        if key == "T_sl":
            for value in par[key]:
                # print('%s : %.3f' % (key, value))
                d[key].append(thrust_given_fun(v, cd_0, cl_0, k, w, s, rho, value))
                get_fmax(cd_0, cl_0, k)
        elif key == "cd_0":
            for value in par[key]:
                # print('%s : %.3f' % (key, value))
                d[key].append(thrust_given_fun(v, value, cl_0, k, w, s, rho, thrust_sl))
                get_fmax(value, cl_0, k)
        elif key == "cl_0":
            for value in par[key]:
                # print('%s : %.3f' % (key, value))
                d[key].append(thrust_given_fun(v, cd_0, value, k, w, s, rho, thrust_sl))
                get_fmax(cd_0, value, k)
    return d


def tbm_700():
    pa_max = 747 * HP  # available power of TBM 700 (ESHS = 747)
    m, s, e = 2900, 18, 0.85  # mass, wing surface, Oswald coef (guess)
    w = m * G_REF
    rho = rho_0
    lambda_ = 9  # wing aspect ratio (calculated)
    k = 1 / (np.pi * lambda_ * e)
    cd_0, cl_0 = 0.023, 0.1  # aerodynamic polar parameters (guess)

    eval_turboprop_thrust_sl(v_max=550, pa_max=pa_max)

    thrust_sl = 13500  # engine parameter of PT6A, (guess)
    v = np.linspace(450, 550, 11) / 3.6  # range of speed tested, in m/s
    # Different values are tested and the function to find v_max is plotted
    # use of dictionnary
    keys = ["T_sl", "cd_0", "cl_0"]
    values = ([thrust_sl, 12000, 12500], [cd_0, 0.024, 0.025], [cl_0, 0.15, 0.05])
    par = set_parameters_dict(keys, values)

    f_ref = power_constant_max_speed(v, cd_0, cl_0, k, w, s, rho, pa_max)
    d = constant_thrust_sl_max_speed(v, par, cd_0, cl_0, k, w, s, rho, pa_max, thrust_sl)
    plot_power_over_thrust_sl(v, par, d, f_ref)
    plot_available_power_turboprop(v, thrust_sl)


def atr72_500():
    pa_max = 2252e3 * 2
    m, s, e = 22000, 61, 0.85  # mass, wing surface, Oswald coef (guess)
    w = m * G_REF
    rho = rho_0
    lambda_ = 12  # wing aspect ratio (calculated)
    k = 1 / (np.pi * lambda_ * e)
    cd_0, cl_0 = 0.0165, 0.0  # aerodynamic polar parameters (guess)

    eval_turboprop_thrust_sl(v_max=700, pa_max=pa_max)

    thrust_sl = 64000  # engine parameter of PT6A, (guess)
    v = np.linspace(400, 700, 11) / 3.6  # range of speed tested, in m/s
    # Different values are tested and the function to find v_max is plotted
    # use of dictionnary
    keys = ["T_sl", "cd_0", "cl_0"]
    values = ([thrust_sl, 60000, 70000], [cd_0, 0.018, 0.020], [cl_0, 0.10, 0.15])
    par = set_parameters_dict(keys, values)

    f_ref = power_constant_max_speed(v, cd_0, cl_0, k, w, s, rho, pa_max)
    d = constant_thrust_sl_max_speed(v, par, cd_0, cl_0, k, w, s, rho, pa_max, thrust_sl)
    plot_power_over_thrust_sl(v, par, d, f_ref)
    plot_available_power_turboprop(v, thrust_sl)