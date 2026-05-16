"""
Author: Friedly WOLI
Date: 27 April 2026
Modified: 13 May 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ============================================================
# Numerical integral for takeoff distance
# ============================================================

def distance_integral_direct(Vhw, koi, k1i, k2i, Vi, Vip1, g):

    f = lambda V: (V - Vhw) / (koi + k1i*V + k2i*V**2)

    I, error = quad(f, Vi, Vip1)

    return I / g


# ============================================================
# Plot distance versus speed
# ============================================================

def plot_dist_speed(Vhw, koi, k1i, k2i, Vi, Vip1, g):

    speed = np.linspace(Vi, Vip1, 100)

    S = np.array([
        distance_integral_direct(Vhw, koi, k1i, k2i, Vi, v, g)
        for v in speed
    ])

    plt.figure(figsize=(8, 5))
    plt.plot(speed, S)
    plt.xlabel("Velocity (ft/s)")
    plt.ylabel("Distance (ft)")
    plt.title("Distance vs Velocity")
    plt.grid(True)
    plt.show()


# ============================================================
# Aerodynamic functions
# ============================================================

def C_L_alpha_sect(alpha, alpha_0):
    return 2*np.pi*(alpha - alpha_0)


def C_L_alpha(CL_alpha_sect, Ra):
    return CL_alpha_sect / (1 + CL_alpha_sect/(np.pi*Ra))


def C_L_function(CL_alpha, alpha, alpha_0):
    return CL_alpha*(alpha-alpha_0)


def C_D_function(C_D0, C_D0l, C_L, hw, bw, e, Ra):
    return C_D0 + C_D0l*C_L + \
           ((16*hw/bw)**2 * C_L**2) / ((1+(16*hw/bw)**2)*np.pi*e*Ra)


# ============================================================
# Forces
# ============================================================

def lift(V, Sw, C_L):
    return 0.5*V**2*Sw*C_L


def drag(V, Sw, C_D):
    return 0.5*V**2*Sw*C_D


def thrust(T0, T1, V, T2):
    return T0 + T1*V + T2*V**2


def friction(W, L, mu):
    return mu*(W-L)


# ============================================================
# K coefficients
# ============================================================

def Koi(Toi, W, mu):
    return Toi/W - mu


def K1i(T1, W):
    return T1/W


def K2i(T2, W, rho, Cl, mu, Cd, Sw):
    return T2/W + rho/(2*W/Sw)*(Cl*mu - Cd)


def Kri(Koi, K1i, K2i):
    return 4*Koi*K2i - K1i**2


def fi_function(koi, k1i, k2i, Vi):
    return koi + k1i*Vi + k2i*Vi**2


def dfi_function(k1i, k2i, Vi):
    return k1i + 2*k2i*Vi


# ============================================================
# Lift-off speed
# ============================================================

def Vlo_function(clmax, W, Sw, rho):
    return 1.1*np.sqrt((2*W)/(clmax*Sw*rho))


# ============================================================
# Distance
# ============================================================

def distance(Kti, Vhw, Kwi, g):
    return (Kti - Vhw*Kwi)/g


# ============================================================
# Kt
# ============================================================

def Kti_function(Vip1, Vi, koi, k1i, k2i, kwi, fi, fip1):

    if k2i == 0 and k1i == 0:
        return (Vip1**2 - Vi**2)/(2*koi)

    elif k2i == 0 and k1i != 0:
        return (koi/k1i**2)*np.log(fi/fip1) + (Vip1-Vi)/k1i

    else:
        return (1/(2*k2i))*np.log(fip1/fi) - (k1i*kwi)/(2*k2i)


# ============================================================
# Kw
# ============================================================

def kwi_function(Vip1, Vi, koi, k2i, k1i, fi, fip1,
                 dfi, dfip1, kri):

    if k2i == 0 and k1i == 0:
        return (Vip1 - Vi)/koi

    elif k1i != 0 and k2i == 0:
        return (1/k1i)*np.log(fip1/fi)

    elif kri < 0:
        return (1/np.sqrt(-kri))*np.log(
            ((dfip1 - np.sqrt(-kri)) *
             (dfi + np.sqrt(-kri))) /
            ((dfip1 + np.sqrt(-kri)) *
             (dfi - np.sqrt(-kri)))
        )

    elif kri == 0:
        return 2/dfi - 2/dfip1

    else:
        return (2/np.sqrt(kri))*(
            np.arctan(dfip1/np.sqrt(kri))
            - np.arctan(dfi/np.sqrt(kri))
        )


# ============================================================
# Main function
# ============================================================

def calcule_distance(alpha, alpha_0,
                     Sw, bw, hw, rho, W, Ra,
                     Cdo, Cdol, e,
                     Clmax, Cl,
                     mu,
                     T0, T1, T2,
                     Vhw, g, Vi):

    Ko = Koi(T0, W, mu)
    K1 = K1i(T1, W)

    Cd = C_D_function(Cdo, Cdol, Cl, hw, bw, e, Ra)

    K2 = K2i(T2, W, rho, Cl, mu, Cd, Sw)

    Kr = Kri(Ko, K1, K2)

    Vip1 = Vlo_function(Clmax, W, Sw, rho)

    fi = fi_function(Ko, K1, K2, Vi)
    fip1 = fi_function(Ko, K1, K2, Vip1)

    dfi = dfi_function(K1, K2, Vi)
    dfip1 = dfi_function(K1, K2, Vip1)

    Kw = kwi_function(
        Vip1, Vi, Ko, K2, K1,
        fi, fip1,
        dfi, dfip1, Kr
    )

    Kt = Kti_function(
        Vip1, Vi, Ko, K1, K2,
        Kw, fi, fip1
    )

    dist = distance(Kt, Vhw, Kw, g)

    dist_integ = distance_integral_direct(
        Vhw, Ko, K1, K2,
        Vi, Vip1, g
    )

    print("Ko =", Ko)
    print("K1 =", K1)
    print("K2 =", K2)
    print("Kr =", Kr)
    print("Vip1 =", Vip1)
    print("Kw =", Kw)
    print("Kt =", Kt)
    print("Distance analytical =", dist)
    print("Distance integral =", dist_integ)

    plot_dist_speed(Vhw, Ko, K1, K2, Vi, Vip1, g)


# ============================================================
# Example
# ============================================================

Sw = 180
bw = 33
hw = 6
W = 2700
Ra = 6.05

Cdo = 0.036
Cdol = 0
e = 0.82
Clmax = 1.4
Cl = 0.3485

mu = 0.04

T0 = 1200
T1 = -4
T2 = 0

rho = 0.0023769
alpha = 0
alpha_0 = 0

Vhw = 29.33
g = 32.2
Vi = 29.33


calcule_distance(
    alpha, alpha_0,
    Sw, bw, hw, rho,
    W, Ra,
    Cdo, Cdol, e,
    Clmax, Cl,
    mu,
    T0, T1, T2,
    Vhw, g, Vi
)