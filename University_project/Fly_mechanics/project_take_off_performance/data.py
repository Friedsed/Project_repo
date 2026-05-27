

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

"""


# For model 1 simulation in :  meter    N      N

dict0 = {"Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T0":120640 , "T1": 0, "T2": 0}
dict1 = {"Sw": 127, "bw": 35.92, "hw": 12.29, "W": 82190*9.81,"Ra": 10.16, "e": 0.723, "T":  119200}
dict2 = {"Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T": 0}
dict3 = {"Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T": 0}
dict4 = {"Sw": 122.4, "bw": 35.8, "hw": 11.76, "W": 79000*9.81,"Ra": 10.47, "e": 0.717, "T": 0}

params = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,
          "T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }




#data1 = modify_dict(dict0, params )

#print(data1)