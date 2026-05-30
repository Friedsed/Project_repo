

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





#===================================================================================================================
#Le samedi 30 mai, après simulation, j’ai trouvé une distance de décollage avec le data1 pour le model1 de 327 ft, et pour le model4 de 1365.251 ft, ainsi que pour le model2 de 445.07118 ft.

# Avec le site de simulation, je trouve une distance de decollage de prèes de 784ft avec le elemnt du data1                <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj

#Le model1 donne une valeur proche de celle de l’Exo, qui est de 327 ft dans le livre B1, Example 3.10.2, page 350.
#======================================================================================================================================
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


#===================================================================================================================
#Le samedi 30 mai, après simulation, j’ai trouvé une distance de décollage avec le data2 de ..... ft pour le model1, de ...... ft pour le model4, et de ..... ft pour le model2.

# Avec le site de simulation, je trouve une distance de decollage de prèes de ..... ft avec le elemnt du data2               <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj


#Le model1 donne une valeur proche de celle de l’exercice, qui est de 600 ft dans le livre B1, page 349, example 3.10.1.
#======================================================================================================================================
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


#===================================================================================================================
#Le samedi 30 mai, après simulation, j’ai trouvé une distance de décollage avec le data3 de 1100.62 ft pour le model1, de 1111.04 ft pour le model4, et de 875.71 ft pour le model2.

# Avec le site de simulation, je trouve une distance de decollage de prèes de 1177ft avec le elemnt du data3                <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj

#Le model2 donne une valeur proche de celle de l’exercice, qui est de 877 ft dans le livre B2, page 798, example 18-6.
#======================================================================================================================================
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


#===================================================================================================================
#Le samedi 30 mai, après simulation, j’ai trouvé une distance de décollage avec le data4        de 2244.0467 ft pour le model1,           de 149.251 ft pour le model4,            et de 2214.05 ft pour le model2.

# Avec le site de simulation, je trouve une distance de decollage de prèes de ..... ft avec le elemnt du data3               <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj


#Le model1 donne une valeur proche de celle de l’exercice, qui est      de 2434.4 ft dans le livre B1, page 359, example 3.11.1.
#======================================================================================================================================
data4 = {
    "Unit": "US",    "name": "Executive business jet ",    "engine": "jet",
    "W": 20000,    "Sw": 320,    "bw": 54,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.6,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.74,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 6500,    "T0": 6500,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}








#===================================================================================================================
#Le samedi 30 mai, après simulation, j’ai trouvé une distance de décollage avec le data3 de 1100.62 ft pour le model1, de 1111.04 ft pour le model4, et de 875.71 ft pour le model2.

#Le model2 donne une valeur proche de celle de l’exercice, qui est de 877 ft dans le livre B2, page 798, example 18-6.
#======================================================================================================================================
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





