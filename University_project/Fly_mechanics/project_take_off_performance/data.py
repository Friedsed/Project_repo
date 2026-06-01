

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
    "W": 20000,    "Sw": 320,    "bw": 54,    "hw": 6,    
    "Clmax": 1.6,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.74,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 6500,    "T0": 6500,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}



#===================================================================================================================
#Le samedi 30 mai, après simulation, j’ai trouvé une distance de décollage avec le data4        de .........ft pour le model1,           de ...... ft pour le model4,            et de ..... ft pour le model2.

# Avec le site de simulation, je trouve une distance de decollage de prèes de ..... ft avec le elemnt du data5               <<<<<<<<<<<<<    hook.eu2.make.com/jhr88m7bybktg3p1zrsm4ju0sdlor2hj


#Le model1 donne une valeur proche de celle de l’exercice, qui est      de .......ft  Beluga XL           https://www.airbus.com/en/newsroom/press-releases/2015-09-beluga-xl-programme-achieves-design-freeze   or         https://aircraft.fandom.com/wiki/Airbus_BelugaXL
#======================================================================================================================================
data5 = {
    "Unit": "SI",    "name": "Belugar XL ",    "engine": "jet",
    "W": 227000*9.81,    "Sw": 361.6,    "bw": 60.3,    "hw": 5.1,    
    "Clmax": 1.6,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.82,
    "g": 9.81,    "mu": 0.04,    "rho": 1.225,
    "P":316000*98.6 ,    "T": 316000 ,    "T0": 316000,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}




#===================================================================================================================
#   AVION CESSNA ; TYPE: Avion léger,; MOTEUR : Lycoming IO-360-L2A

# 
# the model4 give a ground run of 920ft close to the right one 

#   According to the website, the takeoff distance is around 960 ft             https://airfleetcapital.com/blog/cessna-172-specs-dimensions/
#======================================================================================================================================
data6 = {
    "Unit": "US",    "name": "Cessna 172 SP",    "engine": "jet",
    "W": 2550,    "Sw": 174,    "bw": 36,    "hw": 7,    
    "Clmax": 1.5,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.8,
    "g": 32.2,    "mu": 0.02,    "rho": 0.0023769,
    "P": 180,    "T": 450,    "T0": 450,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}












#===================================================================================================================
#   Avion Lockheed Martin C-130J super Hercules;    Military aicraft ;      American 4 engine turboprop 
# 
#
#       NOTICED :   1shp = 1 bhp
#       https://www.airandspaceforces.com/weapons/c-130j/           and wikipedia 
#======================================================================================================================================
data7 = {
    "Unit": "US",    "name": " Lockheed Martin C-130J  ",    "engine": "jet",
    "W": 155000,    "Sw": 1745,    "bw": 132,    "hw": 16,    
    "Clmax": 2.1,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.033,    "Cdol": 0.0,    "e": 0.8,
    "g": 32.2,    "mu": 0.02,    "rho": 0.0023769,
    "P": 4700*4,    "T": 52000,    "T0": 52000,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}





#===================================================================================================================
#   DASSAULT RAFALE 
# 
#
#           https://aviationsmilitaires.net/v3/kb/aircraft/show/9472/dassault-rafale-m#tab:tab-details      
#======================================================================================================================================
data8 = {
    "Unit": "US",    "name": " DASSAULT RAFALE ",    "engine": "jet",
    "W": 54013,    "Sw": 491.911,    "bw": 35.63,    "hw": 4,    
    "Clmax": 1.8 ,    "Cl": 0.4,    "Cd": 0.03,    "Cdo": 0.022,    "Cdol": 0.0,    "e": 0.8,
    "g": 32.2,    "mu": 0.02,    "rho": 0.0023769,
    "P": 14000,    "T": 16855*2,    "T0": 16855*2,    "T1": 0,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 35
}

data9 = {
    "Unit": "US",
    "name": "DASSAULT RAFALE",
    "engine": "jet",

    # Geometry
    "W": 54013,                 # MTOW [lb]
    "W_empty": 22478,           # Empty weight [lb]
    "Sw": 491.911,              # Wing area [ft²]
    "bw": 35.63,                # Wingspan [ft]
    "hw": 17.52,                # Height [ft]
    "length": 50.10,            # Length [ft]

    # Aerodynamics (estimates)
    "Clmax": 1.8,
    "Cl": 0.4,
    "Cd": 0.03,
    "Cdo": 0.022,
    "Cdol": 0.0,
    "e": 0.8,

    # Environment
    "g": 32.174,                # ft/s²
    "mu": 0.02,
    "rho": 0.0023769,           # slug/ft³ (sea level ISA)

    # Propulsion
    "P": 14000,                 # fuel weight [lb] ≈ 10362 lb + reserve
    "T": 33710,                 # total thrust with afterburner [lbf]
    "T0": 33710,                # static thrust [lbf]
    "T1": 0,
    "T2": 0,
    "efficiency": 0.5,

    # Wind
    "Vhw": 0,                   # headwind [ft/s]
    "Vi": 0,

    # Runway
    "alpha": 0,
    "alpha_0": 0,

    # Atmospheric model
    "A1": 1.158e-2,
    "A2": -5.277e-05,
    "A3": 9.273e-08,
    "A4": -6.21e-11,

    # Numerical integration
    "n": 35,

    # Additional aircraft data
    "MTOW": 54013,              # lb
    "MLW": 33069,               # lb
    "Fuel_weight": 10362,       # lb

    "Takeoff_distance": 1312,   # ft
    "Landing_distance": 1476,   # ft

    "Cruise_speed": 1064,       # mph
    "Vmax_SL": 863,             # mph
    "Vmax_HA": 1188,            # mph

    "Range": 808,               # miles
    "Ferry_range": 1265,        # miles

    "Ceiling": 50000,           # ft
    "ROC": 1000,                # ft/s

    "Load_factor_max": 9.0,
    "Load_factor_min": -3.2,

    "Roll_rate": 270,           # deg/s

    "Mach_max_SL": 1.1,
    "Mach_max_HA": 1.8,

    "Wing_loading_MTOW": 109.8, # lb/ft²
    "Wing_loading_empty": 45.7, # lb/ft²

    "TW_ratio_MTOW_AB": 0.624,
    "TW_ratio_empty_AB": 1.50,

    "Crew": 1,
    "Ejection_seat": "Martin-Baker F16F"
}





























#convert_units(data5, "US" , "SI")


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





"""
SOME FORMULAS 

Ra : the aspect ratio = bw^2 / Sw ;                             so wing span square divided by the wing areas



"""