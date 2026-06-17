# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

flight_physics module :
    velocities conversion : KTAS, KEAS, KIAS, KGS

@author: Christophe Airiau

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from flight.general import kt2mps, mps2kt, ft2m, m2ft, GAMMA
from flight.atmosphere_11000 import atmos, R_AIR, P_SL, temp_0
from flight.aerodynamics import q_from_mach, qc_from_mach, dynamic_pressure, display_state, display_dynamic_pressure
from flight.aerodynamics import sound_velocity

# *** VELOCITIES ***

    
def _pi_function_(q, p):
    """
    Pi function, used in TAS and EAS 
    """
    return pow(1 + q / p, (GAMMA - 1) / GAMMA) - 1


def eas2tas(eas, sigma=1):
    """
    EAS = TAS / sqrt(sigma)
    """
    return eas / np.sqrt(sigma)


def tas2eas(tas, sigma=1):
    """
    TAS = EAS    sqrt(sigma)
    """
    return tas * np.sqrt(sigma)
    

def cas2eas(q, p, p_0, cas):
    """
    solve CAS from EAS
    """
    return cas * np.sqrt(p * _pi_function_(q, p) / (p_0 * _pi_function_(q, p_0)))


def eas2cas(q, p, p_0, eas):
    """
    solve EAS from CAS
    """
    c = np.sqrt(p_0 * _pi_function_(q, p_0) / (p * _pi_function_(q, p)))
    print("CAS/EAS :  ", c)
    return eas * c


def cas2tas(q, p, p_0, theta, cas):
    """
    solve TAS from CAS
    """
    return cas * np.sqrt(theta * _pi_function_(q, p) / _pi_function_(q, p_0))


def display_velocities(kias, kcas, keas, ktas, kgs):
    """
    Table of the velocities in kt and in km/h
    """
    n = 70
    print()
    print("=" * n)
    print("   KIAS" + " "*8 + "KCAS" + " "*8 + "KEAS" + " "*8 + "KTAS" + " "*8
          + "KGS     [kt]")
    print("-" * n)
    form = "  %7.2f   " * 5
    print(form % (kias, kcas, keas, ktas, kgs))
    print("=" * n)
    print("    IAS" + " "*9 + "CAS" + " "*9 + "EAS" + " "*9 + "TAS" + " "*9
          + "GS     [km / h]")
    print("-" * n)
    print(form % (kt2mps(kias) * 3.6, kt2mps(kcas) * 3.6, kt2mps(keas) * 3.6,
                  kt2mps(ktas) * 3.6, kt2mps(kgs) * 3.6))
    print("=" * n)
    
    
def display_velocities_pressures(h, kias, kcas, keas, ktas, kgs, p, temp, rho, a, q, mach, qc):
    """ 
    display velocities, atmosphere quantities and dynamic pressures
    """
    display_state("Altitude  : %.2f m" % h, p, rho, temp, a)
    display_velocities(kias, kcas, keas, ktas, kgs)
    display_dynamic_pressure(mach, q, qc)
    

def tas_from_cas(h_ft=25000, kcas=303, atm0=None):
    """
    return the True Air Speed when the Calibrated Air Speed is known
    h : altitude en feets
    """
    if atm0 is None:
        atm0 = {'p': P_SL, 'T': temp_0}
    cas = kt2mps(kcas)
    atm0["rho"] = atm0["p"] / (R_AIR * atm0["T"])
    atm0["a"] = sound_velocity(atm0["T"])
    theta, delta, sigma = atmos(ft2m(h_ft))
    # rho = sigma * atm0["rho"]
    temp = theta * atm0["T"]
    p = delta * atm0["p"]
    a = sound_velocity(temp)
   
    qc = dynamic_pressure(p=atm0["p"], mach=cas/atm0["a"], pi=True)
    eas = cas2eas(qc, p, atm0["p"], cas)
    keas = mps2kt(eas)
    tas = eas / np.sqrt(sigma)
    # TAS = CAS2TAS(qc, p, atm0['p'], Theta, CAS)
    ktas = mps2kt(tas)
 
    mach = tas / a
    q = dynamic_pressure(p=p, mach=mach)
    return keas-kcas, ktas - kcas, mach, q


def velocities_from_keas(h_ft=25000, keas=292, kwind=20, delta_error=3, atm0=None):
    """
    return all velocities when KEAS is known
    h_ft : altitude en feets
    """
    if atm0 is None:
        atm0 = {'p': 101325, 'T': 288.15}
    print("*" * 50)
    print("Velocities from KEAS")
    print("*" * 50)
    atm0["rho"], atm0["a"]  = atm0["p"] / (R_AIR * atm0["T"]), sound_velocity(atm0["T"])
    theta, delta, sigma = atmos(ft2m(h_ft))
    rho, temp, p = sigma * atm0["rho"], theta * atm0["T"], delta * atm0["p"]
    a = sound_velocity(temp)
    eas = kt2mps(keas)
    tas = eas2tas(eas, sigma)
    ktas = mps2kt(tas)
    kgs = ktas - kwind
    mach = tas / a
    q, qc = q_from_mach(mach, p),  qc_from_mach(mach, p)
    cas = eas2cas(qc, p, atm0["p"], eas)
    kcas = mps2kt(cas)
    kias = kcas - delta_error
    display_velocities_pressures(ft2m(h_ft), kias, kcas, keas, ktas, kgs, p, temp, rho, a, q, mach, qc)


def velocities_from_ktas(h_ft=25000, ktas=436, kwind=20, delta_error=3, atm0=None):
    """
    return all velocities when KTAS is known
    h_ft : altitude en feets
    """
    if atm0 is None:
        atm0 = {'p': 101325, 'T': 288.15}
    print("*" * 50)
    print("Velocities from KTAS")
    print("*" * 50)
    atm0["rho"], atm0["a"] = atm0["p"] / (R_AIR * atm0["T"]), sound_velocity(atm0["T"])
    theta, delta, sigma = atmos(ft2m(h_ft))
    rho, temp, p = sigma * atm0["rho"], theta * atm0["T"], delta * atm0["p"]
    a = sound_velocity(temp)
    tas = kt2mps(ktas)
    keas = tas2eas(ktas, sigma)
    kgs = ktas - kwind
    mach = tas / a
    q, qc = q_from_mach(mach, p), qc_from_mach(mach, p)
    kcas = eas2cas(qc, p, atm0["p"], keas)
    kias = kcas - delta_error
    display_velocities_pressures(ft2m(h_ft), kias, kcas, keas, ktas, kgs, p, temp, rho, a, q, mach, qc)
    
    
def velocities_from_kias(h_ft=25000, kias=300, kwind=20, delta_error=3, atm0=None):
    """
    return all velocities when KIAS is known
    h_ft : altitude en feets
    """
    if atm0 is None:
        atm0 = {'p': 101325, 'T': 288.15}
    print("*" * 50)
    print("Velocities from KIAS")
    print("*" * 50)
    atm0["rho"] = atm0["p"] / (R_AIR * atm0["T"])
    atm0["a"] = sound_velocity(atm0["T"])
    theta, delta, sigma = atmos(ft2m(h_ft))
    rho, temp, p = sigma * atm0["rho"], theta * atm0["T"], delta * atm0["p"]
    a = sound_velocity(temp)
    kcas = kias + delta_error
    cas = kt2mps(kcas)
    qc = qc_from_mach(cas / atm0["a"], atm0["p"])
    keas = cas2eas(qc, p, atm0["p"], kcas)
    ktas = eas2tas(keas, sigma)
    kgs = ktas - kwind
    mach = kt2mps(ktas) / a
    q = q_from_mach(mach, p)
    display_velocities_pressures(ft2m(h_ft), kias, kcas, keas, ktas, kgs, p, temp, rho, a, q, mach, qc)

     
def solve(h_ft, kcas_table=np.linspace(100, 650, 501)):
    """
    main function to generate data of the diagrams of velocities

    h_ft        : table of altitude
    kcas_table  : table of KCAS
    """
    data = []
    for h in h_ft:
        kcas, dkeas, dktas, ma, qt = [], [], [], [], []
        for kcas_ in kcas_table:
            delta_v, delta_vt, mach, qd = tas_from_cas(h_ft=h, kcas=kcas_)
            if mach <= 1:
                kcas.append(kcas_)
                dkeas.append(delta_v)
                dktas.append(delta_vt)
                ma.append(mach)
                qt.append(qd)
            else:
                break
        data.append([h/1000, kcas, dkeas, dktas, ma, qt])
    return data


def plot_diagram(data, kc=2):
    """
    component is given by cas
    kc = 2 : KEAS - KCAS
       = 3 : KTAS - KCAS
       = 4 : Mach
       = 5 : q 
    """
    title = ("", "KCAS", r"$\Delta= KEAS - KCAS$", r"$\Delta=KTAS - KCAS$", "Mach", "q")
    sub = "    and h / 1000 in ft"
    locator =([], [50, 10], [2, 1], [50, 10],[0.1, 0.02], [10000, 2000])
    fig = plt.figure(figsize=(9, 7))
    plt.title(title[kc] + sub)
    plt.xlabel("KCAS")
    plt.ylabel(title[kc])
    for k in range(len(data)):
        cas = data[k][1]                 # Calibrated Air Speed in kt
        print(data[k][0], len(cas))
        plt.plot(cas, data[k][kc], lw=2)
        if kc == 1:
            plt.text(cas[-1]-1, data[k][kc][-1]-1, "%d" % data[k][0], rotation=-60)
        elif kc == 2:
            plt.text(cas[-1]+10, data[k][kc][-1], "%d" % data[k][0])
        else:
            plt.text(cas[-1]+10, data[k][kc][-1], "%d" % data[k][0])
    axs = fig.get_axes()
    for ax in axs:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(locator[1][0]))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(locator[1][1]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(locator[kc][0]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(locator[kc][1]))
    plt.grid(which="major", axis="both", color="black", alpha=0.5)
    plt.grid(which="minor", axis="both", color="grey", alpha=0.5)


def get_diagrams_cas_2_tas():
    """ 
    """
    h_max = m2ft(11000)
    print("hmax : %.1f ft" % h_max)
    kcas_table = np.linspace(100, 650, 501)
    h_ft = np.linspace(5000, 35000, 7)
    data = solve(h_ft, kcas_table)
    for k in range(2, 6):
        plot_diagram(data, k)
    
    