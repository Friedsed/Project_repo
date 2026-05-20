
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 20 May 2026

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

"""




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *

import pandas as pd

df = pd.read_excel("excel_file/data_1.ods")

#print ("some part of the dictionnary are :",df["Model"] )
i=4


 
# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model1.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////

if i==1 :
    from Model1 import *

    print( " Ground run estimation using numerical Integration method")
    print(" B1: Book used Mechanic of flight by Warren")

    # ============================================================
    # definition of the parameter
    # ============================================================

    params = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
    params1 = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 0,"g": 32.2, "Vi": 0 }

    # ============================================================
    # Conversion 
    # ============================================================

    param = convert_units(params1, "SI", "US")
    print("Checking the unity : ", param)


    # ============================================================
    # definition of the parameter
    # ============================================================


    plane = AircraftTakeoff(param) # here the parameters are in US unit


    # ============================================================
    # Printing and plotting 
    # ============================================================

    plane.summary()



# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model2.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////



elif i== 2 :
    from Model2 import *

    print( " General ground run estimation using Average Acceleration ; page 796 ")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")

    # =====================================================
    # Parameters
    # =====================================================


    param = { "W": 3400, "Sw": 144.9, "Clmax": 1.69, "Clto": 0.5, "Cdto": 0.0417, "engine": "piston", "g": 32.2, "mu": 0.04,  "P": 310, "rho": 0.002378,  "efficiency": 0.5, "T": 7000, "lamdba": 0, "wing type": "Trapezed"}
    param1 = { "W": 21500, "Sw": 311.6, "Clmax": 1.65, "Clto": 0.9, "Cdto": 0.045, "engine": "jet", "g": 32.2, "mu": 0.02,  "P": 0, "rho": 0.002378,  "efficiency": 0, "T": 7000, "lamdba": 0, "wing type": "Trapezed"}


    # =====================================================
    #Conversion of unit
    # =====================================================

    param = convert_units(param, "SI" , "US")
    print("Checking the unity : ", param)


    # =====================================================
    #Definition of the model 
    # =====================================================

    model = AircraftTakeoff(param)

    # =====================================================
    #plotting and Computing
    # =====================================================
    distance = model.ground_run()
    model.summary()

    print("distance est", distance)



# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model3.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////



elif i== 3 :
    from Model3 import *

    print( " General ground run estimation using Average Acceleration for tricycle propeller aicraft ONLY ;       page 797 ")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")

    # =====================================================
    # Parameters
    # =====================================================


    param = { "W": 3400, "Sw": 144.9, "Clmax": 1.69, "Clto": 0.5, "Cdto": 0.0417, "engine": "piston", "g": 32.2, "mu": 0.04,  "P": 310, "rho": 0.002378,  "efficiency": 0.5, "T": 7000, "lamdba": 0, "wing type": "Trapezed"}
    param1 = { "W": 21500, "Sw": 311.6, "Clmax": 1.65, "Clto": 0.9, "Cdto": 0.045, "engine": "jet", "g": 32.2, "mu": 0.02,  "P": 0, "rho": 0.002378,  "efficiency": 0, "T": 7000, "lamdba": 0, "wing type": "Trapezed"}


    # =====================================================
    #Conversion of unit
    # =====================================================

    param = convert_units(param, "SI" , "US")
    print("Checking the unity : ", param)


    # =====================================================
    #Definition of the model 
    # =====================================================

    model = AircraftTakeoff(param)

    # =====================================================
    #plotting and Computing
    # =====================================================
    distance = model.ground_run()
    model.summary()

    print("distance est", distance)




# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model4.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////


    
elif i== 4 :
    from Model4 import *
    print( " Ground run estimation using numerical Integration method")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")

    # =====================================================
    # Parameters
    # =====================================================

    param = {  "W": 3400,  "Sw": 144.9,  "Clmax": 1.69,  "Clto": 0.5, "Cdto": 0.0417, "g": 32.2, "mu": 0.04, "P": 310, "rho": 0.002378,  "T": 1169,
                    "lamdba": 0,  "n": 35,  "A1": 1.158e-2,  "A2": -5.277e-05,  "A3": 9.273e-08,  "A4": -6.21e-11 }

    # =====================================================
    #Conversion of unit
    # =====================================================

    param = convert_units(param, "SI" , "US")
    print("Checking the unity : ", param)


    # =====================================================
    #Definition of the model 
    # =====================================================
    model = AircraftTakeoff(param)

    # =====================================================
    #plotting and Computing
    # =====================================================

    distance = model.ground_run()

    model.plot_forces()
    model.plot_speed()
    model.plot_acceleration()

    for i in range(len(distance)):
        print(list(distance.keys())[i], "est :", distance[list(distance.keys())[i]])
        print("")

            












