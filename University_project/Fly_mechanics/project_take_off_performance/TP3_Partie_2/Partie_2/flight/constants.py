# -*- coding: utf-8 -*-
"""
Use of scipy to get constants

Tests only.

@author: C. Airiau
"""

from scipy import constants


def constants_def():
    r_gas = constants.value('molar gas constant')
    g = constants.value('standard acceleration of gravity')
    p_sl = constants.value('standard atmosphere')
    print("p_sl ", p_sl)
    print(r_gas, g)


def print_constants():
    print(constants.inch)              #0.0254
    print(constants.foot)              #0.30479999999999996
    print(constants.yard)              #0.9143999999999999
    print(constants.mile)              #1609.3439999999998
    print(constants.mil)               #2.5399999999999997e-05
    print(constants.pt)                #0.00035277777777777776
    print(constants.point)             #0.00035277777777777776
    print(constants.survey_foot)       #0.3048006096012192
    print(constants.survey_mile)       #1609.3472186944373
    print(constants.nautical_mile)     #1852.0

    print(constants.kmh)            #0.2777777777777778
    print(constants.mph)            #0.44703999999999994
    print(constants.mach)           #340.5
    print(constants.speed_of_sound) #340.5
    print(constants.knot)           #0.5144444444444445

    print(constants.zero_Celsius)      #273.15
    print(constants.degree_Fahrenheit) #0.5555555555555556

    print(constants.dyn)             #1e-05
    print(constants.dyne)            #1e-05
    print(constants.lbf)             #4.4482216152605
    print(constants.pound_force)     #4.4482216152605
    print(constants.kgf)             #9.80665
    print(constants.kilogram_force)  #9.80665

    print(constants.g)
    print(constants.hp)
    print(constants.lb)
    print(constants.knot)


if __name__ == '__main__':
    constants_def()
    print_constants()