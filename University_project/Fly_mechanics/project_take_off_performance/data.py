

from conversion import *


# === Données structurées pour l'Excel ===
""" 

Data from this website : https://aircraftinvestigation.info/airplanes/Airbus_A320neo.html

dict0       <==>     Aibus A320 neo
dict1       <==>     Boeing 737 Max 200
dict2       <==>     Aibus A320 neo
dict3       <==>     Aibus A320 neo
dict4       <==>     Aibus A320 neo
dict5       <==>     Aibus A320 neo
dict6       <==>     Aibus A320 neo
dict7       <==>     Aibus A320 neo
dict8       <==>     Aibus A320 neo
dict9       <==>     Aibus A320 neo
dict21      <==>     Aibus A320 neo
param1      <==>     exercice 
param2      <==>     exercice 
param3      <==>     exercice 
param4      <==>     exercice 
param5      <==>     exercice 
param6      <==>     exercice 

"""


# For model 1 simulation in :  meter    N      N

dict0 = {"name":" None ", "engine": "piston", "Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T0":120640 , "T1": 0, "T2": 0}
dict1 = {"name":" None ", "engine": "piston", "Sw": 127, "bw": 35.92, "hw": 12.29, "W": 82190*9.81,"Ra": 10.16, "e": 0.723, "T":  119200}
dict2 = {"name":" None ", "engine": "piston", "Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T": 0}
dict3 = {"name":" None ", "engine": "piston", "Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T": 0}
dict4 = {"name":" None ", "engine": "piston", "Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T": 0}
param1 = {"name":" None ", "engine": "piston", "Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
param2 = {"name":" None ", "engine": "piston", "Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 0,"g": 32.2, "Vi": 0 }
param3 = {"name":" None ",  "engine": "piston", "W": 3400, "Sw": 144.9, "Clmax": 1.69, "Clto": 0.5, "Cdto": 0.0417, "engine": "piston", "g": 32.2, "mu": 0.04,  "P": 310, "rho": 0.002378,  "efficiency": 0.5,  "T": 7000}
param4 = {"name":" None ", "engine": "piston",  "W": 21500, "Sw": 311.6, "Clmax": 1.65, "Clto": 0.9, "Cdto": 0.045, "engine": "jet", "g": 32.2, "mu": 0.02,  "P": 0, "rho": 0.002378,  "efficiency": 0, "T": 7000, "lamdba": 0, "wing type": "Trapezed"}
param5 = {"name":" None ", "engine": "piston",  "W": 13488.54,  "Sw": 269.1,  "Clmax": 1.69,  "Clto": 1.2, "Cdto": 0.03, "g": 32.2, "mu": 0.03, "P": 310, "rho": 0.002378,  "T": 3372.135, "n": 35,  "A1": 1.158e-2,  "A2": -5.277e-05,  "A3": 9.273e-08,  "A4": -6.21e-11 }








