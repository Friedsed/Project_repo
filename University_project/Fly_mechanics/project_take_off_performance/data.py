from conversion import *


# === Structured Data for Excel ===
"""

Data from this website: https://aircraftinvestigation.info/airplanes/Airbus_A320neo.html

dict0       <=>     Airbus A320 neo
dict1       <=>     Boeing 737 Max 200
dict2       <=>     Airbus A320 neo

data1      <=>     Model1
data2     <=>     Model1
data3     <=>     Model4
param4      <=>     exercise
param5      <=>     exercise
param6      <=>     exercise

"""


# ===================================================================================================================
# Saturday, May 30, after simulation, I found a takeoff distance with data1 for model1 of 327 ft,
# and for model4 of 1365.251 ft, as well as for model2 of 445.07118 ft.
#
# With the simulation website, I find a takeoff distance of about 784ft with the element of data1
# <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj
#
# Model1 gives a value close to that of the Example, which is 327 ft in book B1, Example 3.10.2, page 350.
# ===================================================================================================================
data1 = {
    "Unit": "US",   "name": "None",  "engine": "piston",
    "W": 2700,    "Sw": 180,    "bw": 33,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.4,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.036,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 1200,    "T0": 1200,    "T1": -4,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 29.33,    "Vi": 29.33,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":0
}


# ===================================================================================================================
# Saturday, May 30, after simulation, I found a takeoff distance with data2 of ..... ft for model1,
# of ...... ft for model4, and of ..... ft for model2.
#
# With the simulation website, I find a takeoff distance of about ..... ft with the element of data2
# <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj
#
# Model1 gives a value close to that of the exercise, which is 600 ft in book B1, page 349, example 3.10.1.
# ===================================================================================================================
data2 = {
    "Unit": "US",    "name": "None",    "engine": "jet",
    "W": 2700,    "Sw": 180,    "bw": 33,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.4,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.036,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 1200,    "T0": 1200,    "T1": -4,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":0
}


# ===================================================================================================================
# Saturday, May 30, after simulation, I found a takeoff distance with data3 of 1100.62 ft for model1,
# of 1111.04 ft for model4, and of 875.71 ft for model2.
#
# With the simulation website, I find a takeoff distance of about 1177ft with the element of data3
# <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj
#
# Model2 gives a value close to that of the exercise, which is 877 ft in book B2, page 798, example 18-6.
# ===================================================================================================================
data3 = {
    "Unit": "US",    "name": "Cirrus SR22",    "engine": "piston",
    "W": 3400,    "Sw": 144.9,    "bw": 33,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.69,    "Cl": 0.5,    "Cd": 0.0417,    "Cdo": 0.0417,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.2,    "mu": 0.04,    "rho": 0.002378,
    "P": 310,    "T": 1169,    "T0": 1200,    "T1": -4,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35, "k":0
}


# ===================================================================================================================
# Saturday, May 30, after simulation, I found a takeoff distance with data4 of 2244.0467 ft for model1,
# of 149.251 ft for model4, and of 2214.05 ft for model2.
#
# With the simulation website, I find a takeoff distance of about ..... ft with the element of data4
# <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj
#
# Model1 gives a value close to that of the exercise, which is 2434.4 ft in book B1, page 359, example 3.11.1.
# ===================================================================================================================
data4 = {
    "Unit": "US",    "name": "Executive business jet",    "engine": "jet",
    "W": 20000,    "Sw": 320,    "bw": 54,    "hw": 6,
    "Clmax": 1.6,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.74,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 6500,    "T0": 6500,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":0
}


# ===================================================================================================================
# Saturday, May 30, after simulation, I found a takeoff distance with data5 of .........ft for model1,
# of ...... ft for model4, and of ..... ft for model2.
#
# With the simulation website, I find a takeoff distance of about ..... ft with the element of data5
# <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj
#
# Model1 gives a value close to that of the exercise, which is .......ft Beluga XL
# https://www.airbus.com/en/newsroom/press-releases/2015-09-beluga-xl-programme-achieves-design-freeze
# or https://aircraft.fandom.com/wiki/Airbus_BelugaXL
# ===================================================================================================================
data5 = {
    "Unit": "SI",    "name": "Beluga XL",    "engine": "jet",
    "W": 227000 * 9.81,    "Sw": 361.6,    "bw": 60.3,    "hw": 5.1,
    "Clmax": 1.6,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.82,
    "g": 9.81,    "mu": 0.04,    "rho": 1.225,
    "P": 316000 * 98.6,    "T": 316000,    "T0": 316000,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":1.80
}


# ===================================================================================================================
# CESSNA AIRCRAFT; TYPE: Light aircraft; ENGINE: Lycoming IO-360-L2A
#
# Model4 gives a ground run of 920ft close to the right one
#
# According to the website, the takeoff distance is around 960 ft
# https://airfleetcapital.com/blog/cessna-172-specs-dimensions/
# ===================================================================================================================
data10 = {
    "Unit": "US",    "name": "Cessna 172 SP",    "engine": "piston",
    "W": 2550,    "Sw": 174,    "bw": 36,    "hw": 7,
    "Clmax": 1.5,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.75,
    "g": 32.2,    "mu": 0.02,    "rho": 0.0023769,
    "P": 180,    "T": 450,    "T0": 450,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":0.20
}


data6 = {
    "Unit": "US",    "name": "exercise",    "engine": "jet",
    "W": 56000,    "Sw": 1000,    "bw": 36,    "hw": 7,
    "Clmax": 2.4,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.024,    "Cdol": 0.0,    "e": 0.75,
    "g": 32.2,    "mu": 0.25,    "rho": 0.0023769,
    "P": 2400,    "T": 5900,    "T0": 5900,    "T1": 0,    "T2": 0,    "efficiency": 0.75,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":0
}


# ===================================================================================================================
# Lockheed Martin C-130J Super Hercules; Military aircraft; American 4 engine turboprop
#
# NOTE: 1 shp = 1 bhp
# https://www.airandspaceforces.com/weapons/c-130j/
# and Wikipedia
# ===================================================================================================================
data7 = {
    "Unit": "US",    "name": "Lockheed Martin C-130J",    "engine": "jet",
    "W": 155000,    "Sw": 1745,    "bw": 132,    "hw": 16,
    "Clmax": 2.1,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.8,
    "g": 32.2,    "mu": 0.02,    "rho": 0.0023769,
    "P": 4700 * 4,    "T": 52000,    "T0": 52000,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k": 4.2
}


# ===================================================================================================================
# DASSAULT RAFALE
#
# https://aviationsmilitaires.net/v3/kb/aircraft/show/9472/dassault-rafale-m#tab:tab-details
# ===================================================================================================================
data8 = {
    "Unit": "US",    "name": "Dassault Rafale",    "engine": "jet",
    "W": 54013,    "Sw": 491.911,    "bw": 35.63,    "hw": 4,
    "Clmax": 1.8,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.022,    "Cdol": 0.0,    "e": 0.8,
    "g": 32.2,    "mu": 0.02,    "rho": 0.0023769,
    "P": 14000,    "T": 16855 * 2,    "T0": 16855 * 2,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,    "tr": 1,    "hoc": 35,
    "alpha": 0,    "alpha_0": 0,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":0
}


data9 = {
    "Unit": "US",    "name": "Airbus A320-200 (CFM56-5B4)",    "engine": "jet",
    "W": 162040,    "Sw": 1317.6,    "bw": 111.9,    "hw": 40.8,
    "Clmax": 2.0,    "Cl": 0.50,    "Cd": 0.033,    "Cdo": 0.024,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.174,    "mu": 0.02,    "rho": 0.0023769,
    "P": 24000,    "T": 54000,    "T0": 54000,    "T1": 0,    "T2": 0,    "efficiency": 0.35,
    "Vhw": 0,    "Vi": 0,    "alpha": 0,    "alpha_0": 0,    "tr": 1,    "hoc": 35,
    "A1": 0,    "A2": 0,    "A3": 0,    "A4": 0,
    "n": 35, "k":0.1
}






# ===================================================================================================================
# Saturday, May 30, after simulation, I found a takeoff distance with data3 of 1100.62 ft for model1,
# of 1111.04 ft for model4, and of 875.71 ft for model2.
#
# Model2 gives a value close to that of the exercise, which is 877 ft in book B2, page 798, example 18-6.
# ===================================================================================================================
data = {
    # General
    "Unit": "US",    "name": "Falcon 8X",    "engine": "jet",
    # Weight and geometry
    "W": 73000,    "Sw": 761,    "bw": 86,    "hw": 6,    "Ra": 9.77,
    # Aerodynamics
    "Clmax": 1.69,    "Cl": 0,    "Cd": 0.022,    "Cdo": 0.022,    "Cdol": 0,    "e": 0.95,
    # Environment and ground
    "g": 32.2,    "mu": 0.03,    "rho": 0.002378,
    # Propulsion
    "P": 3100,    "T": 3 * 6721,    "T0": 3 * 6721,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    # Speeds
    "Vhw": 0,    "Vi": 0,
    # Angles
    "alpha": 0,    "alpha_0": 0,
    # Thrust polynomial coefficients
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    # Numerical settings
    "n": 35, "k":0.20
}




data1 = convert_units(data1, "SI", "US")
data2 = convert_units(data2, "SI", "US")
data3 = convert_units(data3, "SI", "US")
data4 = convert_units(data4, "SI", "US")
data5 = convert_units(data5, "SI", "US")
data6 = convert_units(data6, "SI", "US")
data7 = convert_units(data7, "SI", "US")
data8 = convert_units(data8, "SI", "US")
data9 = convert_units(data9, "SI", "US")
data10 = convert_units(data10, "SI", "US")

"""
SOME FORMULAS

Ra: the aspect ratio = bw^2 / Sw
Wing span squared divided by the wing area
"""