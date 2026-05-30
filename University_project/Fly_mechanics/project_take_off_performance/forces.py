
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 27 May 2026

B2: Book used Mechanic of flight by Warren

--------------------------------
Goal of Main_model              | Compile all the model in on code 
--------------------------------

------------
Advantages 1| : easier to manage among the code 
Advantages 2| : Useful to solve some problem enconterred during the take-off 
------------

-----------
Asumption | V_lof is assumed to be 1.1* stalling speed  
----------

Seats
Cabin width
Cabin height
Cabin length
Tail height
Fuselage diameter
Baggage volume
Gross weight
Maximum takeoff weight
Maximum landing weight
Fuel capacity
Maximum payload
Maximum speed
Cruise speed
Approach speed
Range
Fuel burn
Ceiling
Rate of climb
Takeoff distance
Landing distance
Thrust


"""



import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
from data import *


#************************************************************************************************  Model1 book b1

# ========================================================
# Aerodynamic coefficients
# ========================================================

def C_L_alpha_sect1(alpha, alpha_0):
    return 2 * np.pi * (alpha - alpha_0)


def C_L_alpha1(alpha, alpha_0, Ra):
    cl_alpha_sect = C_L_alpha_sect1(alpha, alpha_0)
    return cl_alpha_sect / (1 + cl_alpha_sect / (np.pi * Ra))


def C_L_function1(alpha, alpha_0, Ra):
    return C_L_alpha1(alpha, alpha_0, Ra) * (alpha - alpha_0)


def C_D_function1(Cdo, Cdol, Cl, hw, bw, e, Ra):
    return (
        Cdo
        + Cdol * Cl
        + ((16 * hw / bw) ** 2 * Cl ** 2)
        / ((1 + (16 * hw / bw) ** 2) * np.pi * e * Ra)
    )


# ========================================================
# Forces
# ========================================================

def lift1(V, Sw, Cl):
    return 0.5 * V**2 * Sw * Cl


def drag1(V, Sw, Cd):
    return 0.5 * V**2 * Sw * Cd


def thrust1(V, T0, T1, T2):
    return T0 + T1 * V + T2 * V**2

#************************************************************************************************ ^^^ Model1




#************************************************************************************************  Model2 book b2


# =====================================================
# Lift at rotation speed
# =====================================================
def lift2(rho, Vr, Sw, Cl):
    return 0.5 * rho * (Vr / np.sqrt(2))**2 * Sw * Cl


# =====================================================
# Drag at rotation speed
# =====================================================
def drag2(rho, Vr, Sw, Cd):
    return 0.5 * rho * (Vr / np.sqrt(2))**2 * Sw * Cd


# =====================================================
# Thrust for piston engine
# =====================================================
def thrust_piston2(efficiency, P, Vr):
    return efficiency * 550 * P * np.sqrt(2) / Vr


# =====================================================
# Thrust for jet engine
# =====================================================
def thrust_jet_powered2(T):
    return T

#************************************************************************************************ ^^^ Model1

















# =====================================================
# Friction
# =====================================================
def friction(mu, W, lift_value):
    return mu * (W - lift_value)


