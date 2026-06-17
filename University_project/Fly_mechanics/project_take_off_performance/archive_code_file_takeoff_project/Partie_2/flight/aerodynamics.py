# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

aerodynamics module which contains many functions used in flight mechanics,
all are usual aerodynamics functions, among:

    state gas equation
    dynamic pressure
    other pressures
    isentropic state
    sound velocity
    Normal shock wave
    Pitot tube problem

@author: Christophe Airiau
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from flight.general import mps2kt, GAMMA_LETTER, GAMMA, T_REF
from flight.atmosphere_11000 import P_SL, R_AIR


class State(object):
    """
    state of an ideal gas
    Mach number, temperature, density, pressure
    sound velocity, dynamic pressure, compressible dynamic pressure,
    isentropic state : temperature, pressure (density not used)
    """

    def __init__(self, name=None):
        self.name = name
        self.m = None
        self.t = self.p = self.rho = None
        self.a = self.ti = self.pi = None
        self.u = self.q = self.qc = None

    def set_t_from_ti(self):
        """
        return T, a and u from Ti and Mach number
        """
        self.t = self.ti * t_ti_temperature_ratio(self.m)
        self.a = sound_velocity(self.t)
        self.u = self.a * self.m

    def set_q_from_mach(self):
        """
        return dynamic pressure and compressible dynamic pressure from the Mach number and the pressure
        """
        self.q = q_from_mach(self.m, self.p)
        self.qc = self.pi - self.q

    def __repr__(self):
        """
        State quantities are given in nice tables
        """
        display_state(msg=self.name, p=self.p, rho=self.rho, temp=self.t, a=self.a)
        display_state_plus(mach=self.m, temp_i=self.ti, p_i=self.pi, u=self.u)
        display_dynamic_pressure(mach=self.m, q=self.q, qc=self.qc)
        return "-" * 50 + "\n"

# *** PRESSURES ***


def display_pressure_exponents(gamma=GAMMA):
    """ 
    display exponents found in various equations related to pressure, function of gamma
    """
    g_1 = "(" + GAMMA_LETTER + "-1)"
    print()
    print("=" * 30)
    print("Pressure exponents")
    print("=" * 30)
    r = (1 / (gamma-1), 2 / (gamma - 1), (gamma-1) / gamma, gamma / (gamma-1))
    e = (
        "1 / " + g_1 + " : %.2f",
        "2 / " + g_1 + " : %.2f",
        g_1 + " / " + GAMMA_LETTER + " : %.5f",
        GAMMA_LETTER + " / " + g_1 + " : %.2f")
    for leg, x in zip(e, r):
        print(leg % x)


def dynamic_pressure(rho=0, mach=0, u=0, p=0, pi=False, gamma=GAMMA):
    """
    The returned dynamic pressure will depend on the argument used:
        rho, U
        p, M
    if pi = True return  q_c, compressible dynamic pressure
    """
    if rho * u > 0:
        return 1 / 2 * rho * u**2
    elif p * mach > 0:
        if pi:
            return p * (pow(1 + (gamma-1) / 2 * mach**2, gamma / (gamma - 1)) - 1)
        else:
            return gamma * p * mach**2 / 2
    else:
        raise ValueError("problem : an information is missing")


def pi_over_p(mach, gamma=GAMMA):
    """
    return isentropic to static pressure ratio
    p_i / p = f(Mach)
    """
    return pow(1 + (gamma - 1) / 2 * mach**2, gamma / (gamma - 1))


def ti_over_t(mach, gamma=GAMMA):
    """
    return isentropic to static temperature ratio
    T_i / T = f(Mach)
    """
    return 1 + (gamma - 1) / 2 * mach**2


def q_from_mach(mach, p, gamma=GAMMA):
    """
    dynamic pressure
    q = f(Mach, p)
    """
    return p * gamma / 2 * mach**2


def qc_from_mach(mach, p, gamma=GAMMA):
    """
    return compressible dynamic pressure
    q_c = p_i - p = f(Mach, p)
    """
    return (pi_over_p(mach, gamma=gamma) - 1) * p


def pt_over_p(mach, gamma=GAMMA):
    """
    p_t / p = 1 + gamma M^2 / 2
    """
    return 1 + gamma / 2 * mach**2


def plot_errors():
    """
    error on total pressure and dynamic pressure
    """
    mach = np.linspace(0, 0.9, 91)
    epsilon = 1 - pt_over_p(mach) / pi_over_p(mach)
    epsilon_q = qc_from_mach(mach[1:], 1) / q_from_mach(mach[1:], 1) - 1

    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    fig.suptitle('Pressure error')
    ax[0].plot(mach, epsilon * 100, 'ko', label=r"$\varepsilon$")
    ax[1].plot(mach[1:], epsilon_q * 100, 'r-', label=r"$\varepsilon_q$")
    ax[1].plot(mach, mach**2 / 4 * 100, 'b--', label=r"$M^2/4$")
    ax[0].set_title(r"$\dfrac{p_t-p_i}{p_i}$")
    ax[1].set_title(r"$\dfrac{q_c-q}{q}$")
    for i in range(2):
        ax[i].legend(loc="best")
        ax[i].grid()
        ax[i].set_xlabel("Mach")
        ax[i].set_ylabel("error in %")
    return mach, epsilon, epsilon_q


def display_dynamic_pressure(mach=0.5, q=15000, qc=17000, p_sl=P_SL):
    """
    table of values :  Mach, q [Pa], q_c [Pa], q / p_0, q_c / p_0
    """
    n = 65
    print()
    print("=" * n)
    print("  Mach" + " "*8 + "q [Pa]" + " "*8 + "q_c [Pa]" + " "*4 + "q / p_0",
          " "*4 + "q_c / p_0")
    print("-" * n)
    form = " %6.3f      %8.3f     %8.3f     %5.3f       %5.3f"
    print(form % (mach, q, qc, q / p_sl, qc / p_sl))
    print("=" * n)
    

# *** MACH NUMBER ***


def sound_velocity(temp, gamma=GAMMA):
    """ a from temperature """
    return np.sqrt(gamma * R_AIR * temp)


def mach_from_q(x, gamma=GAMMA):
    """ 
    x = p / q 
    """
    return np.sqrt(2 / (gamma * x))


def mach_from_pt(x, gamma=GAMMA):
    """ 
    x = p_t / q 
    """
    return np.sqrt(2 / gamma * (x - 1))


def mach_from_pi(x, gamma=GAMMA):
    """
    x = p_i / q : isentropic to dynamic pressure ratio
    """
    return np.sqrt(2 / (gamma - 1) * (pow(x, (gamma - 1) / gamma) - 1))


def set_mach_number_from_pressures():
    """
    Mach given from three different conditions
    """
    n = 50
    print()
    print("=" * n)
    print("   p / q    M from q   M from p_t  M from p_i ")
    print("-" * n)
    form = "  %6.2f     %6.2f     %6.2f     %6.2f  "
    data = []
    for ratio in [1.2, 10]:
        data.append([ratio, mach_from_q(ratio), mach_from_pt(ratio), mach_from_pi(ratio)])
    for line in data:
        print(form % (line[0], line[1], line[2], line[3]))
    print("=" * n)
    return data


# *** NORMAL SHOCK WAVES ***


def p2_p1_static(mach, gamma=GAMMA):
    """ 
    return pressure ratio across the shock wave, oblique or normal shock wave
    
    Args:
        * mach (real) : normal Mach number 
        * gamma (real) : :math:`C_p/C_v`
    
    Returns:
        real : downstream p / upstream p
    """
    return 2.0 * gamma / (gamma + 1.0) * mach ** 2 - (gamma - 1.0) / (gamma + 1.0)


def rho2_rho1(mach, gamma=GAMMA):
    """    
    return density ratio across the shock wave, oblique or normal shock wave
    
    Args:
        * mach (real) : Mach number 
        * gamma (real) : :math:`C_p/C_v`
    
    Returns:
        real : downstream rho / upstream rho   
    """
    return 1.0 / (2.0 / ((gamma + 1.0) * mach ** 2) + (gamma - 1.0) / (gamma + 1.0))


def pi2_pi1(mach, gamma=GAMMA):
    """
    return isentropic pressure ratio across the shock wave, oblique or normal shock wave
    
    Args:
        * mach (real) : normal Mach number 
        * gamma (real) : :math:`C_p/C_v`
    
    Returns:
        real : downstream p_i / upstream p_i
    """
    tmp = pow(rho2_rho1(mach, gamma=gamma), gamma / (gamma - 1))
    return pow(p2_p1_static(mach, gamma=gamma), -1 / (gamma - 1)) * tmp


def downstream_normal_mach(mach, gamma=GAMMA):
    """       
    return downstream normal Mach across a shock
    
    Args:
        * mach (real) : normal Mach number 
        * gamma (real) : :math:`C_p/C_v`
    
    Returns:
        real : downstream normal Mach number
    """
    return np.sqrt((1.0 + 0.5 * (gamma - 1.0) * mach ** 2) / (gamma * mach ** 2 - 0.5 * (gamma - 1.0)))


def upstream_normal_mach(mach, gamma=GAMMA):
    """       
    return upstream normal Mach across a shock when downstream Mach is known
    
    Args:
        * mach (real) : normal Mach number 
        * gamma (real) : :math:`C_p/C_v`
    
    Returns:
        real : upstream normal Mach number
    """
    return np.sqrt((1.0 + 0.5 * (gamma - 1.0) * mach ** 2) / (gamma * mach ** 2 - 0.5 * (gamma - 1.0)))


# ***  ISENTROPIC FLOWS ***

def mach_from_p_pi(r, gamma=GAMMA):
    """
    return the Mach number when the ratio static to isentropic pressure is given
    r = p/p_i
    Mach =f(r)
    """
    return np.sqrt(2 / (gamma - 1) * (pow(r, (1 - gamma) / gamma) - 1))


def p_pi(mach, gamma=GAMMA):
    """
    return ratio pressure/isentropic pressure function of Mach number

    Args:
        * mach (real) : Mach number
        * gamma (real) : :math:`C_p/C_v`

    Returns:
        real : p/p_i
    """
    return (t_ti_temperature_ratio(mach, gamma=1.4)) ** (gamma / (gamma - 1))


def rho_rhoi(mach, gamma=GAMMA):
    """
    rethurn ratio rho/isentropic rho function of Mach number
    
    Args:
        * mach (real) : Mach number
        * gamma (real) : :math:`C_p/C_v`

    Returns:
        real : rho/rho_i
    """
    return (t_ti_temperature_ratio(mach, gamma=gamma)) ** (1. / (gamma - 1))


def t_ti_temperature_ratio(mach, gamma=1.4):
    """
   return ratio Temperature/isentropic Temperature function of Mach number
    
    Args:
        * mach (real) : Mach number
        * gamma (real) : :math:`C_p/C_v`

    Returns:
        real : T/T_i
    """
    return 1. / (1 + (gamma - 1) / 2 * mach ** 2)


# ***  PITOT TUBE ***


def pitot_rayleigh(mach, rhs=0, gamma=GAMMA):
    """
    return pi_2  / p_1 used in the Pitot tube for supersonic flow
    pi_2 : total pressure found in the Pitot tube
    p_1 : pressure before the shock wave, at a given altitude.

    The function is used to find the upstream Mach number.
    """
    r_1, r_2 = - 1 / (gamma - 1), gamma / (gamma - 1)
    cst = pow((gamma - 1) / (1 + gamma), r_1) * pow((gamma+1) / 2, r_2)
    return cst * pow(2 * r_2 * mach ** 2 - 1, r_1) * pow(mach, 2 * r_2) - rhs


def plot_pitot_rayleigh(gamma=GAMMA, opt=0):
    """
    plot of the Pitot-Rayleigh function for supersonic flow wrt Mach number
    opt = 1 : Delta p / p_0 else : p_i2 / p_1
    """
    mach = np.linspace(1, 3, 51)
    yleg = (r"$p_{i_2} / p_1$", r"$\dfrac{\Delta p}{p_0}$")
    r = pitot_rayleigh(mach, gamma=gamma)
    if opt == 1 :
        r -= 1
    plt.figure()
    plt.title("Pitot tube, supersonic flow")
    plt.ylabel(yleg[opt])
    plt.xlabel("Upstream Mach")
    plt.plot(mach, r)
    plt.grid()
    plt.show()


def delta_p_pitot(mach, gamma=GAMMA):
    """
    Delta_p / p_0 for supersonic Pitot tube
    """
    return pitot_rayleigh(mach, gamma=gamma) - 1


def m1_from_delta_p(r=4, gamma=GAMMA):
    """
    r = delta_p / p_0
    return upstream Mach number of a supersonic Pitot tube
    """
    return fsolve(pitot_rayleigh, 1.1, args=(1 + r, gamma))


# **************************************************
# display data on screen
# **************************************************

def display_state(msg='state', p=0, rho=0, temp=0, a=0):
    """
    display the gas state in a table, part 1 : p, rho, a, T [K and °C]
    """
    n = 60
    print()
    print("=" * n)
    print("# \t " + msg)
    print("=" * n)
    title = "  p [Pa]     rho [kg/m^3]   a [m/s]   T [K]     theta [°]"
    print(title)
    print("-" * n)
    form = " %6.2f     %6.4f       %6.2f    %6.2f     %6.2f "
    print(form % (p, rho, a, temp, temp-T_REF))
    print("=" * n)


def display_state_plus(mach=0, temp_i=0, p_i=0, u=0):
    """
    display the gas state: part 2: Ti, Pi and u
    """
    n = 60
    print()
    print("=" * n)
    title = "  Mach      T_i [K ]        p_i [Pa]    u [km/h]    u [kt]"
    print(title)
    print("-" * n)
    form = " %6.2f     %6.4f       %6.2f    %6.2f      %6.2f"
    print(form % (mach, temp_i, p_i, u * 3.6, mps2kt(u)))
    print("=" * n)


def display_pitot_tube(mach, p_ratio, delta_p, v_cas):
    """
    data used or resuls of the supersonic Pitot tube: Mach, pi_2/pi_1, Delta p, CAS, KCAS
    """
    if mach >= 1:
        n = 65
        print()
        print("=" * n)
        print("Pitot tube (true for supersonic only)")
        print("=" * n)
        print("  Mach     p_i2 / p_1    Delta p [Pa]  CAS [km/h]   KCAS [kt]")
        print("-" * n)
        form = " %6.3f      %6.3f       %8.1f        %6.1f      %6.3f "
        print(form % (mach, p_ratio, delta_p, v_cas * 3.6, mps2kt(v_cas)))
        print("=" * n)    
    else:
        print("Not possible to display, M must be greater than 1")

