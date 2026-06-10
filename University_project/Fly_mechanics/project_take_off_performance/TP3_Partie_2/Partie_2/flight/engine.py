# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

engine module :
    engine models :
        piston
        turboprop
        turbojet
        turbofan

    theta-break point analysis

@author: Christophe Airiau

ref: Gundmundsson (2014), Mattingly et al (2003, 2005)
"""

import numpy as np
import matplotlib.pyplot as plt
from flight.atmosphere_std import Atmosphere
from flight.atmosphere_11000 import atmos, temp_0, rho_0, p_0
from flight.general import ft2m, m2ft, r2k, k2r, FIGSIZE
from flight.aerodynamics import ti_over_t, pi_over_p, sound_velocity


def engine_thrust_power(model="turboprop", h=5000, thrust_sl=10000, power_sl=1, mach=0.5,
                        delta_oat=0, params=None):
    """
    model:
        piston
            option: simple, GF (Gagg and Ferrar), turbo (hc, boost), correction (1, 2)
        turboprop:
            option: tr, mil
        turbojet:
            option: tr, mil
        turbofan:
            option: tr, mil, high_bpr
    h : altitude in meter
    power_sl : power at sea-level, for piston engine only
    trust_sl : thrust at sea-level in Newtons, or in percentage
    mach : Mach number
    delta_oat : variation of the outside air temperature from the standard value, in degrees

    return:
        thrust in Newtons or T/T_0
        available power in W or Pa / Pa_0
        v : velocity in m/s
    """
    if params is None:
        params = dict(tr=1.072, mil=False, high_bpr=True, turbo=False, model="simple", hc=1000, boost=0, correction=2)
    if model == "piston":
        return piston_engine(model=params["model"], h=h, delta_oat=delta_oat, power_sl=power_sl, params=params)
    else:
        tpc = temperature_pressure_correction(h, mach, delta_oat=delta_oat)
        v = mach * sound_velocity(tpc["T"])
        if model == "turboprop":
            thrust = turboprop_thrust(thrust_sl=thrust_sl, mach=mach, params=params, tpc=tpc)
        elif model == "turbojet":
            thrust = turbojet_thrust(thrust_sl=thrust_sl, mach=mach, params=params, tpc=tpc)
        elif model == "turbofan":
            thrust = turbofan_thrust(thrust_sl=thrust_sl, mach=mach, params=params, tpc=tpc)
        else:
            raise ValueError("bad choice of model in engine_thrust_power")
        power = thrust * v
    return thrust, power, v


def piston_engine(model="GF", h=10000, delta_oat=0, power_sl=100, params=None):
    """
    correction 1:
        P/P_std = sqrt(T/(T+th))
    correction 2 (more accurate):
        rho = p / (r T)
        then use the new rho
    params["correction"] : 1 or 2
    params["model"] : "Simple", "GF", "turbo"
    params["hc"] : ft2m(18000))
    params["boost"] : 0.1
    """
    if params is None:
        params = dict(models="simple", correction=2, model="simple", hc=ft2m(18000), boost=0)
    theta, delta, sigma = atmos(h)
    temp_std = temp_0 * theta
    temp_h = temp_std + delta_oat
    if params["correction"] == 1:
        pa = np.sqrt(temp_std / temp_h) * propeller_model(params["model"], pa_sl=power_sl, h=h,
                                                          sigma=sigma, params=params)
        # print("correction :  sqrt(T/(T+th)      : %.3f" % )
    else:
        sigma_h = temp_0 / (temp_std + delta_oat) * delta
        # print("sigma_h / sigma std             : %.3f" % (sigma_h / sigma))
        pa = propeller_model(params["model"], pa_sl=power_sl, h=h, sigma=sigma_h, params=params)
    return pa


def propeller_model(model, pa_sl=1e5, h=1000, sigma=0.5, params=None):
    if params is None:
        params = dict(hc=0, boost=0)

    if model == "simple":
        return pa_sl * sigma
    elif model == "GF":
        # Gagg and Ferrar
        return pa_sl *(sigma - (1-sigma) / 7.55)
    elif model == "turbo":
        if h <= params["hc"]:
            return pa_sl * (1 + params["boost"])
        else:
            k_T = -6.5e-3
            t0 = 288.15
            n = 5.255894
            kappa = k_T / t0
            sig = (1 + kappa * (h-params["hc"])) ** n
            return pa_sl * (sig - (1-sig) / 7.55) * (1 + params["boost"])
    else:
        raise ValueError("bad choice of propeller engine model")


def turboprop_thrust(thrust_sl=10000, mach=0.5, params=None, tpc=None):
    """
    thrust_sl : thrust at seal level, in Newtons
    mach: Mach number
    params["tr"] : throttle ratio (close to 1)
    tpc : temperature - pressure correction dictionary
    """
    if params is None:
        params = dict(tr=1.072)
    if mach <= 0.1:
        return tpc["delta_0"] * thrust_sl
    else:
        if tpc["theta_0"] <= params["tr"]:
            return thrust_sl * tpc["delta_0"] * (1 - 0.96 * pow(mach - 0.1, 1/4))
        else:
            return thrust_sl * tpc["delta_0"] * (1 - 0.96 * pow(mach - 0.1, 1 / 4)
                                          - 3 * (tpc["theta_0"] - params["tr"]) / (8.13 * (mach - 0.1)))


def turbojet_thrust(thrust_sl=10000, mach=0.5, params=None, tpc=None):
    """
    thrust_sl : thrust at seal level, in Newtons
    mach: Mach number
    params["tr"] : throttle ratio (close to 1)
    params["mil"] : boolean, military thrust
    tpc : temperature - pressure correction dictionary
    """
    if params is None:
        params = dict(tr=1.072, mil=False)
    if not params["mil"]:
        if tpc["theta_0"] <= params["tr"]:
            return thrust_sl * tpc["delta_0"] * (1 - 0.3 * (tpc["theta_0"] - 1) - 0.1 * np.sqrt(mach))
        else:
            return thrust_sl * tpc["delta_0"] * (1 - 0.3 * (tpc["theta_0"] - 1) - 0.1 * np.sqrt(mach)
                                               - 1.5 * (1 - params["tr"]/tpc["theta_0"]))
    else:
        if tpc["theta_0"] <= params["tr"]:
            return thrust_sl * tpc["delta_0"] * 0.8 * (1 - 0.16 * np.sqrt(mach))
        else:
            return thrust_sl * tpc["delta_0"] * 0.8 * (1 - 0.16 * np.sqrt(mach)
                                                     - 24 * (1 - params["tr"] / tpc["theta_0"]) / (9 + mach))


def turbofan_thrust(thrust_sl=10000, mach=0.5, params=None, tpc=None):
    """
    thrust_sl : thrust at seal level, in Newtons
    mach: Mach number
    params["tr"] : throttle ratio (close to 1)
    params["mil"] : boolean, military thrust
    params["high_bpr"] : boolean, high bypass ratio
    tpc : temperature - pressure correction dictionary
    """
    if params is None:
        params = dict(tr=1.072, mil=False, high_bpr=False)

    if params["high_bpr"]:
        if tpc["theta_0"] <= params["tr"]:
            return thrust_sl * tpc["delta_0"] * (1 - 0.49 * np.sqrt(mach))
        else:
            return thrust_sl * tpc["delta_0"] * (1 - 0.49 * np.sqrt(mach)
                                                 - 3 * (1 - params["tr"] / tpc["theta_0"]) / (1.5 + mach))
    elif params["mil"]:
        if tpc["theta_0"] <= params["tr"]:
            return thrust_sl * tpc["delta_0"] * 0.6
        else:
            return thrust_sl * tpc["delta_0"] * 0.6 * (1 - 3.8 * (1 - params["tr"] / tpc["theta_0"]))
    else:
        if tpc["theta_0"] <= params["tr"]:
            return thrust_sl * tpc["delta_0"]
        else:
            return thrust_sl * tpc["delta_0"] * (1 - 3.5 * (1 - params["tr"]/tpc["theta_0"]))


def temperature_pressure_correction(h, mach, delta_oat=0):
    (sigma, delta, theta) = Atmosphere(h / 1000)
    p, rho, temp = delta * p_0, sigma * rho_0, theta * temp_0
    temp += delta_oat
    theta_0 = (temp / temp_0) * ti_over_t(mach)
    delta_0 = delta * pi_over_p(mach)
    return {"p": p, "T": temp, "theta_0": theta_0, "delta_0": delta_0, "sigma_0": theta_0/delta_0}


def plot_engine_properties(mode=3, params=None):
    """
    engine mode =   0 : piston
                    1 : turboprop
                    2 : turbojet
                    3 : turbofan
    """
    model = ("piston", "turboprop", "turbojet", "turbofan")
    if params is None:
        params = {"tr": 1.072, "mil": False, "high_bpr": True}
    if mode > 0:
        nh, h_max = 5, (0, 20000, 40000, 40000)
        mach_max, npt = 1, 201
        thrust_sl = 1
        h_ft = np.linspace(0, h_max[mode], nh)
        mach = np.linspace(0.0, mach_max, npt)
        thrust = np.zeros((npt, nh), dtype=float)
        power = np.zeros((npt, nh), dtype=float)
        v = np.zeros((npt, nh), dtype=float)
        for j in range(nh):
            for i in range(npt):
                thrust[i, j], power[i, j], v[i, j] = engine_thrust_power(model=model[mode],
                                                                         h=ft2m(h_ft[j]),
                                                                         thrust_sl=thrust_sl,
                                                                         mach=mach[i],
                                                                         delta_oat=0,
                                                                         params=params)
        plot_thrust_power(model[mode], h_ft, mach, v, thrust, power)

    else:
        plot_propeller_power_model()


def plot_thrust_power(model, h_ft, mach, v, thrust, power):
    fig, axs = plt.subplots(2, sharex=False, sharey=False, figsize=(10, 10))
    for j in range(len(h_ft)):
        axs[0].plot(mach, thrust[:, j], label=" h : %.0f ft" % h_ft[j])
        axs[1].plot(v[:, j] * 3.6, power[:, j] / 1000, label=" h : %.0f ft" % h_ft[j])
    for ax in axs:
        # ax.set_xlim(0, 1.2)
        ax.legend(loc="best", fontsize=10)
        # ax.grid()
    plt.suptitle(model)
    axs[0].set_ylabel(r"$T / T_{sl}$")
    axs[0].set_xlabel("Mach number")
    axs[1].set_ylabel(r"$Pa [kW] $")
    axs[1].set_xlabel("velocity [km/h]")
    plt.savefig(model + ".png", format="png")


def legend(models, par):
    if models == 'turbo':
        return models + "  with boost : %.2f" % par["boost"]
    else:
        return models


def plot_propeller_power_model():
    """ plot for the three cases of propellers , with correction 2"""
    h1, h2 = np.linspace(0, 18000, 51), np.linspace(18000, m2ft(11000), 51)
    h = ft2m(np.hstack([h1, h2]))
    nh = len(h)
    models = ("simple", "GF", "turbo", "turbo")
    power = np.zeros((nh, len(models)), dtype=float)
    params = (dict(model=models[0], correction=2), dict(model=models[1], correction=2),
              dict(model=models[2], correction=2, hc=ft2m(18000), boost=0),
              dict(model=models[3], correction=2, hc=ft2m(18000), boost=0.1))
    for i in range(nh):
        for j in range(len(models)):
            power[i, j] = piston_engine(model=models[j], h=h[i], delta_oat=0, power_sl=100, params=params[j])
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    for j in range(len(models)):
        ax1.plot(power[:, j], h, label=legend(models[j], params[j]), lw=3)
    plt.suptitle("Piston engines")
    ax1.set_ylabel("h [m]")
    ax2.set_ylabel("h [ft]")
    ax1.set_xlabel(r"$Pa/Pa_{sl}$ [%]")
    ylim1 = ax1.get_ylim()
    ax2.set_ylim(m2ft(ylim1[0]), m2ft(ylim1[1]))
    ax2.grid(color="grey")
    ax1.legend()

# *****************************************************
# plot of Mattingly diagramms with theta_0 and pi_c
# *****************************************************


def set_theta_0(theta, mach, gamma=1.4):
    """
    theta_0 defined for engine modeling
    theta_0 = T_t / T_sl
    """
    return theta * (1 + (gamma-1) / 2 * mach**2)


def set_delta_0(delta, mach, gamma=1.4):
    """
    delta_0 = p_t  / p_sl
    """
    return delta * pow(1 + (gamma-1) / 2 * mach**2, gamma / (gamma-1))


def set_mach_from_theta_0(theta, theta_0, gamma=1.4):
    """
    Inverse relationship
    Mach = f(Theta(h), theta_0)
    theta_0 >= theta otherwise no solution and return -1 instead of a positive Mach
    """
    tmp = 2/(gamma-1) * (theta_0/theta - 1)
    if tmp < 0:
        return -1
    else:
        return np.sqrt(tmp)


def set_pi_c(theta_0, c_1, t_tot_4, pi_c_max, gamma=1.4):
    """
    return the compression rate for a turbojet, turbofan, turboprop
    c_1 : constant
    t_tot_4 : total temperature after the combustion chamber Ti_4 (Tt_4)
    pi_c_max : max(pi_c), maximun of the compression rate given by the design

    return compression rate pi_c
    """
    x = pow(1 + t_tot_4 / theta_0 * c_1, gamma / (gamma-1))
    return [t if t <= pi_c_max else pi_c_max for t in x]


def set_c1(pi_c, theta_0, ti_4_r, gamma=1.4):
    """
    return the value of the constant C_1 in the pi_c law (compression rate)
    pi_c: a point of the curve
    theta_0 : value at the same point
    ti_4_r : total temperature after the combustion chamber, in Rankine degrees.
    """
    return (pow(pi_c, (gamma-1)/gamma) - 1) * theta_0 / r2k(ti_4_r)


def plot_theta0_Mach():
    """
    2 plots: (theta_0, h),    parameter : Mach
             (Mach, theta_0), parameter : h
    """
    h = np.array([0, 3000, 5000, 7000, 9000, 10000, 11000, 15000], dtype=float)  # in meters
    npt_mach, mach_max = 11, 2.
    mach = np.linspace(0, mach_max, npt_mach)
    npt_h = len(h)
    theta = np.zeros(npt_h, dtype=float)
    for i in range(npt_h):
        (sigma_, delta_, theta[i]) = Atmosphere(h[i] / 1000)
    theta_0 = []
    for theta_ in theta:
        theta_0.append(set_theta_0(theta_, mach))
    t0 = np.array(theta_0)
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    for j in range(npt_mach):
        axs[0].plot(t0[:, j], h, label="Mach : %.2f" % mach[j])
    for i in range(npt_h):
        axs[1].plot(mach, t0[i, :], label="h : %d" % h[i])
    x_label, y_label = (r"$\theta_0$", r"$Mach$"), (r"$h$ [m]", r"$\theta_0$")
    for i in range(2):
        axs[i].legend()
        axs[i].set_xlabel(x_label[i])
        axs[i].set_ylabel(y_label[i])
    fig.suptitle(r"engine parameter: $\theta_0$ in the standard atmosphere")


def plot_map_mach_h_theta_0():
    """
    h versus Mach, as parameter: theta_0
    fig. in Mattingly and al 's book
    in that problem, we do not  have a solution for theta < theta_0
    """
    theta_0 = np.array([0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2])
    h_1, h_2 = np.linspace(0, 37000, 401), np.linspace(38000, 60000, 51)
    h = np.hstack([h_1, h_2])
    npt_h, nt = len(h), len(theta_0)
    theta = np.zeros(npt_h, dtype=float)
    for i in range(npt_h):
        (sigma_, delta_, theta[i]) = Atmosphere(ft2m(h[i]) / 1000)
    mach, h_ = [], []
    for t0_ in theta_0:
        r, h_tmp = [], []
        for j in range(npt_h):
            v = set_mach_from_theta_0(theta[j], t0_)
            if v > 0:
                r.append(v)
                h_tmp.append(h[j])
        mach.append(r)
        h_.append(h_tmp)
    plt.figure()
    for j in range(nt):
        plt.plot(mach[j], h_[j], lw=3, label=r"$\theta_0$: %.2f" % theta_0[j])
        plt.text(mach[j][-1] - 0.06, h_[j][-1] + 500, "%.2f" % theta_0[j])
    plt.legend()
    plt.xlabel(r"$Mach$")
    plt.ylabel(r"$h$ [ft]")
    plt.title(r"engine parameter: $\theta_0$ in the standard atmosphere")


def plot_pic_vs_theta0():
    """
    map (theta_0, pi_c), parameters: Tt_4
     C_1 = 8.122e-4    5 <pi_c< 35       2800 < Tt4 < 3300°R as example
    total temperature Tt_4 is given en Rankine degrees
    """
    pi_c_max, ti_4_max = 15, 3200
    c_1 =  set_c1(pi_c_max, 1, ti_4_max, gamma=1.4)
    print("constant c_1 in pi_c law  : %.4e" % c_1)
    theta_0_ = np.linspace(0.7, 1.25, 401)
    t_tot_4 = r2k(np.arange(2000, 3400, 200))
    n0, n4 = len(theta_0_), len(t_tot_4)
    pi_c = np.zeros((n0, n4), dtype=float)
    for i in range(n0):
        pi_c[i, :] = set_pi_c(theta_0_[i], c_1, t_tot_4, pi_c_max)
    plt.figure()
    for j in range(n4):
        plt.plot(theta_0_, pi_c[:, j], lw=3, label="°R : %d" % k2r(t_tot_4[j]))
        plt.text(1.26, pi_c[-1, j]-0.1, "%d" % k2r(t_tot_4[j]) )
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\pi_c$")
    plt.xlim(0.8, 1.3)
    plt.text(1.25, 10.5, "$Tt_4$ (°R)")
    plt.text(1.11, 14.1, "$c_1$ = %.4e" % c_1)
    plt.text(1.02, pi_c_max, "Theta-Break")
    plt.scatter(1, pi_c_max, s=70, marker="s", color="k")
    plt.text(0.9,  pi_c_max + 0.2, r"$\pi_{c~\max}$")


def theta_break():
    """ Plots to get the theta-break point"""
    plot_theta0_Mach()
    plot_map_mach_h_theta_0()
    plot_pic_vs_theta0()