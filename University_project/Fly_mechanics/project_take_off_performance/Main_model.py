
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 04 juin 2026

B2: Book used Mechanic of flight by Warren

--------------------------------
Goal of Main_model              | Compile all the model in on code 
--------------------------------



"""



import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
from data import *
#from experince import *
import pandas as pd 





# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Management of the running data                                                                                                                             |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////
#fichier = csv.DictReader(open("excel_file/data.csv"))
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
#passagers = list(fichier)
#print (passagers[1])

#print(passagers[1].keys())
# =====================================================
#Conversion of unit
# =====================================================
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
#data = convert_units(data, "SI" , "US")
#print("Checking the unity : ", data)


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

print (" Choose the model to compile i mean the parameter you would like : ")
print("Choose between ; A for data ;     B for  data1;    C for  data2;      D for  data3;    E for  data4;      F for Beluga XL;      G for the Cessna ")
print("Choose between     E for  data4;      F for Beluga XL;      G for the Cessna ;   H for the Lockeed martin C-130J Military aicraft;   I for Dassault rafale ")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
model= input(" Enter your letter then ")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

if model == "A":0 2026
    dict=data
elif model == "B" :
    dict = data1
elif model == "C" :
    dict = data2
elif model == "D" :
    dict = data3
elif model == "E" :
    dict = data4
elif model == "F" :
    dict = data5
elif model == "G" :
    dict = data6
elif model == "H" :
    dict = data7

elif model == "I" :
    dict = data9

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

#df = pd.DataFrame(list(dict.items()), columns=["Paramètre", "Valeur"])
#print(" Aicraft  ",dict["name"], " caratetriqtique is ", df)
print("Choose the Model you want to run ")
i=int (input("Choose the Model you wanna run "))
 
# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model1.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////
if i==1 :
    from Model1 import *
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " Ground run estimation using numerical Integration method")
    print(" B1: Book used Mechanic of flight by Warren")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    param = get_model_params("Model1", dict)
    plane = AircraftTakeoff1(param) # here the parameters are in US unit
    plane.summary()

# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model2.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////
elif i== 2 :
    from Model2 import *
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " General ground run estimation using Average Acceleration ; page 796 ")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    param = get_model_params("Model2", dict)
    model = AircraftTakeoff2(param)
    model.summary()

# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model3.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////
elif i== 3 :
    from Model3 import *
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " General ground run estimation using Average Acceleration for tricycle propeller aicraft ONLY ;       page 797 ")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    param = get_model_params("Model2", dict)
    model = AircraftTakeoff3(param)
    model.summary()

# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model4.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////
elif i== 4 :
    from Model4 import *
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " Ground run estimation using numerical Integration method")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    param = get_model_params("Model4", dict)
    model = AircraftTakeoff4(param)
    model.summary()





























"""





# Geometry and aircraft data
Sw = None        # Wing surface area
bw = None        # Wing span
hw = None        # Height of the wing above the ground
W = None         # Weight
Ra = None        # Aspect ratio coefficient
lambda_ = None   # Aspect ratio parameter

# Aerodynamic coefficients
Cdo = None       # Zero-lift drag coefficient
Cdol = None      # Additional drag coefficient term
Cd = None        # Drag coefficient
Cl = None        # Lift coefficient
Clmax = None     # Maximum lift coefficient
e = None         # Oswald efficiency coefficient

# Ground and motion parameters
mu = None        # Ground friction coefficient
Vlo = None       # Lift-off speed
Vhw = None       # Reference speed
Vi = None        # Initial speed for integration
Vip1 = None      # Final speed for integration
g = None         # Gravity
rho = None       # Air density

# Propulsion parameters
P = None         # Piston engine power in BHP
T = None         # Jet engine thrust
efficiency = None  # Propeller efficiency
T0 = None        # Initial thrust or experimental thrust coefficient
T1 = None        # Experimental thrust coefficient
T2 = None        # Experimental thrust coefficient
A1 = None        # Thrust formula coefficient
A2 = None        # Thrust formula coefficient
A3 = None        # Thrust formula coefficient
A4 = None        # Thrust formula coefficient

# Numerical parameter
n = None         # Number of iterations







"""