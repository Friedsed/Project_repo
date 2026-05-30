

from conversion import *


# === Données structurées pour l'Excel ===
""" 

Data from this website : https://aircraftinvestigation.info/airplanes/Airbus_A320neo.html

dict0       <==>     Aibus A320 neo
dict1       <==>     Boeing 737 Max 200
dict2       <==>     Aibus A320 neo


data1      <==>     Model1
data2     <==>     Model1 
data3     <==>     Model4
param4      <==>     exercice 
param5      <==>     exercice 
param6      <==>     exercice 

"""


# For model 1 simulation in :  meter    N      N

dict01 = {"Unit": "US" ,"name":" None ", "engine": "piston", "Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T0":120640 , "T1": 0, "T2": 0}
dict11 = {"Unit": "US" ,"name":" None ", "engine": "piston", "Sw": 127, "bw": 35.92, "hw": 12.29, "W": 82190*9.81,"Ra": 10.16, "e": 0.723, "T":  119200}
dict21 = {"Unit": "US" ,"name":" None ", "engine": "piston", "Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T": 0}






data1 = {
    "Unit": "US",   "name": "None",  "engine": "piston",   
    "W": 2700,    "Sw": 180,    "bw": 33,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.4,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.036,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 1200,    "T0": 1200,    "T1": -4,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 29.33,    "Vi": 29.33,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}

data2 = {
    "Unit": "US",    "name": "None",    "engine": "jet",
    "W": 2700,    "Sw": 180,    "bw": 33,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.4,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.036,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 1200,    "T0": 1200,    "T1": -4,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}

data3 = {
    "Unit": "US",    "name": "Cirrus SR22",    "engine": "piston",
    "W": 3400,    "Sw": 144.9,    "bw": 33,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.69,    "Cl": 0.5,    "Cd": 0.0417,    "Cdo": 0.0417,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.2,    "mu": 0.04,    "rho": 0.002378,
    "P": 310,    "T": 1169,    "T0": 1200,    "T1": -4,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}




data = {
# General
    "Unit": "US",    "name": "Flacon 8x ",    "engine": "jet",
# Weight and geometry
    "W": 73000,    "Sw": 761,    "bw": 86,    "hw": 6,    "Ra": 9.77,
# Aerodynamics
    "Clmax": 1.69,    "Cl": 0,    "Cd": 0.022,    "Cdo": 0.022,    "Cdol": 0,    "e": 0.95,
# Environment and ground
    "g": 32.2,    "mu": 0.03,    "rho": 0.002378,
# Propulsion
    "P": 3100,    "T": 3*6721,    "T0": 3*6721,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
# Speeds
    "Vhw": 0,    "Vi": 0,
 # Angles
    "alpha": 0,    "alpha_0": 0,
# Thrust polynomial coefficients
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
# Numerical settings
    "n": 35
}





