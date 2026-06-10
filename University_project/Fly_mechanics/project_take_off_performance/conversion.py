
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 04 juin 2026


--------------------------------
Goal |                          To convert from US unity to SI unity
--------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad



    # Conversion factors: US -> SI
"""
    lenght 1ft ==> 0.3048 m ; force 1 Ibf ==> 4.44822 N ; Power 1hp ==> 745.7 W
"""
CONV = {
        "Sw": 0.3048**2, "bw": 0.3048, "hw": 0.3048, "W": 4.44822,  "T0": 4.44822, 
        "T1": 4.44822 / 0.3048,  "T2": 4.44822 / (0.3048**2),  "rho": 515.3788,  "Vhw": 0.3048, 
        "Vi": 0.3048,  "g": 0.3048 , "P": 745.7, "hoc": 0.3048 , "T": 4.44822
        }

# =====================================================
# Unit conversion
# =====================================================

def convert_units(p, new_unit , last_unit):
    """
    The goal here is to convert from US Unit into SI unit or SI to US.
    The new_unit is what are you looking for.
    The last_unit is what you don't want anymore.
    """

    if last_unit == new_unit:
        print(f"Already in {new_unit}")
        return

    if last_unit == "US" and new_unit == "SI":
        for key, value in CONV.items():
            if key in p:
                p[key] *= value

    elif last_unit == "SI" and new_unit == "US":
        for key, value in CONV.items():
            if key in p:
                p[key] /= value

    return p

def get_model_params(model, data):
    """
    The aicraft is represent by 35 parameters. Not all of this 
    is use for each model that why we just select the one importante
    Basically the goal of this function is to allows each model to have the require 
    input for him.
    """
    # ---------------- MODELE 1 ----------------
    if model == "Model1":
        keys = [ 
            "Unit", "name" , "engine", "Sw", "bw", "hw", "W", "Ra", "Cdo",
                "Cdol", "e", "Clmax", "Cl", "mu", "T0", "T1", "T2", "rho", "alpha", 
                "alpha_0", "Vhw", "g", "Vi" , "tr", "hoc" 
                ]
    # ---------------- MODELE 2,3 ----------------
    elif model == "Model2":
        keys = [ 
                "Unit", "name" ,  "engine", "W", "Sw", "Clmax", "Cl", "Cd",  "g",
                "mu", "P", "rho", "efficiency", "T"    
                 ]
# ---------------- MODELE 4 ----------------
    elif model == "Model4":
        keys = [ 
            "Unit", "name" , "engine", "W", "Sw", "Clmax", "Cl", "Cd", "g", "mu",
                 "P", "rho", "T", "n", "A1", "A2", "A3", "A4" , "hoc" , "efficiency"
                ]
# ---------------- MODELE 5 ----------------
    elif model == "Model5":
        keys = [ 
            "Unit", "name" , "engine", "Sw", "bw", "hw", "W", "Ra", "Cdo",
                "Cdol", "e", "Clmax", "Cl", "mu", "T0", "T1", "T2", "rho", "alpha", 
                "alpha_0", "Vhw", "g", "Vi" , "tr", "hoc" ,"k"
                ]
    else:
        return "Modele introuvable"
    # Création du dictionnaire final
    result = {}
    for key in keys:
        if key in data:
            result[key] = data[key]
    return result