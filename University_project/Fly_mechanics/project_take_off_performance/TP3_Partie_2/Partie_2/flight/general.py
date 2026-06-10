# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

General module :
    constants
    conversion
    set_local_zeros (of a discrete function)

@author: Christophe Airiau 
""" 
from scipy import constants

from scipy.constants import convert_temperature
# convert_temperature(np.array([-40, 40]), 'Celsius', 'Kelvin')
#

# constants could be obtained from the scipy.constants module
R_GAS = constants.value('molar gas constant')
T_REF = constants.zero_Celsius   # 0 degrés Celcius en Kelvins
G_REF = constants.g              # standard acceleration of gravity m/s^2
GAMMA = 1.4
HP = constants.hp
FT = constants.foot
KT_2_KMPH = constants.nautical_mile / 1000    # 1852 / 1000  1kt in km
KT = constants.knot      # 1.852 / 3.6  1 kt in  m/s

# Greek symbols for plots and display
GAMMA_LETTER = u"\u03b3"
RHO_LETTER = u"\u03c1"
DELTA_LETTER = u"\u03b4"
SIGMA_LETTER = u"\u03c3"
THETA_LETTER = u"\u0398"
# molar gas constant 8.314462618 J mol^-1 K^-1
FIGSIZE = (12, 8)

COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan",
          "black", "magenta"] * 5

# ***  UNIT CONVERSION  ***


def kt2mps(v):
    """ knots to m/s"""
    return v * KT


def mps2kt(v):
    """ m/s to knots"""
    return v / KT


def ft2m(z):
    """ feets to meters"""
    return z * FT


def m2ft(z):
    """ meters to feets"""
    return z / FT


def watt2hp(p):
    """ horsepower in britannic system"""
    return p / HP      #  745.69987158227


def watt2cv(p):
    """ Horsepower in the old metric system"""
    return p / 735.5


def r2k(x):
    """ Temperature conversion: Rankine to Kelvin"""
    return convert_temperature(x, 'R', 'K')


def k2r(x):
    """ Temperature conversion: Kelvin to Rankine"""
    return convert_temperature(x, 'K', 'R')


def set_local_zeros_vec(x=None, y=None, vb=False):
    """
    find the zeros of a given discrete function z_k = f(x_k)
    """
    g = y[:-1] * y[1:]
    # dz, dx = np.diff(y), np.diff(x)
    zeros = []
    counter = 0
    for k in range(len(y) - 1):
        if g[k] <= 0:
            counter += 1
            zeros.append(x[k] - y[k] * (x[k+1] - x[k]) / (y[k+1] - y[k]))
    if abs(y[-1]) <= 1e-10:
        zeros.append(x[-1])
    if vb:
        print("number of zeros found : ", counter)
    return zeros


def set_local_zeros_mat(x=None, y=None, vb=False):
    """
    find the zeros of given discrete data x: abscissa, y: matrix of column functions
    """
    zeros = []
    n, m = y.shape
    for j in range(m):
        zeros.append(set_local_zeros_vec(x=x[:, j], y=y[:, j], vb=vb))
    return zeros

