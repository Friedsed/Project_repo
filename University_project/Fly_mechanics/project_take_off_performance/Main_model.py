
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 27 May 2026

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

Seats
Cabin width
Cabin height
Cabin length
Tail height
Fuselage diameter
Baggage volume
Gross weight
Maximum takeoff weight
Maximum landing weight
Fuel capacity
Maximum payload
Maximum speed
Cruise speed
Approach speed
Range
Fuel burn
Ceiling
Rate of climb
Takeoff distance
Landing distance
Thrust


"""



import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
from data import *
import pandas as pd 




# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Management of the running data                                                                                                                             |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////
#fichier = csv.DictReader(open("excel_file/data.csv"))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
#passagers = list(fichier)
#print (passagers[2])
#print(passagers[1].keys())
# =====================================================
#Conversion of unit
# =====================================================
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
#data = convert_units(data, "SI" , "US")
#print("Checking the unity : ", data)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

dict = data1
df = pd.DataFrame(list(dict.items()), columns=["Paramètre", "Valeur"])
print(" Aicraft  ",dict["name"], " caratetriqtique is ", df)

print("Choose the Model you want to compile ")
i=int (input("Choose the Model you wanna Run "))



















 
# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Code of Model1.py                                                                                                                                   |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////

if i==1 :
    from Model1 import *
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " Ground run estimation using numerical Integration method")
    print(" B1: Book used Mechanic of flight by Warren")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

    # ============================================================
    # definition of the parameter
    # ============================================================

    param = get_model_params("Model1", dict)

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
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " General ground run estimation using Average Acceleration ; page 796 ")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    # =====================================================
    # Parameters
    # =====================================================

    param = get_model_params("Model2", dict)

    # =====================================================
    #Definition of the model 
    # =====================================================

    model = AircraftTakeoff(param)

    # =====================================================
    #plotting and Computing
    # =====================================================

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

    # =====================================================
    # Parameters
    # =====================================================

    param = get_model_params("Model2", dict)


    # =====================================================
    #Definition of the model 
    # =====================================================

    model = AircraftTakeoff(param)

    # =====================================================
    #plotting and Computing
    # =====================================================

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

    # =====================================================
    # Parameters
    # =====================================================

    param = get_model_params("Model4", dict)

    # =====================================================
    #Definition of the model 
    # =====================================================
    model = AircraftTakeoff(param)

    # =====================================================
    #plotting and Computing
    # =====================================================

    model.summary()



elif i==5 :

    from Model5 import *

            
    mon_avion = Avion(
        nom="Avion teste",
        masse_kg=9071.85,
        poussee_statique_n=28900,
        surface_alaire_m2=29.74,
        Cdo=0.023,
        Clmax=1.4,
        Ra=9.1125,
        k=0.5,
    )

    #Tout les calcule ce sont ici
    env_sol = Environnement(altitude_m=0, mu=0.014, phi=0.6)
    sim = SimulateurTakeOff(mon_avion, env_sol)
    dist_modele1 = sim.calculer_distance(modele=1)
    dist_modele2 = sim.calculer_distance(modele=2)
    time_modele2 = sim.calculer_temps(modele=2)

    print(f"Distance de décollage modèle 1 pour {mon_avion.nom} à 0 m : {dist_modele1:.2f} m")
    print(f"Distance de décollage modèle 2 pour {mon_avion.nom} à 0 m : {dist_modele2:.2f} m")
    print(f"Temps d'accélération modèle 2 jusqu'à VLO : {time_modele2:.2f} s")

    env_haut = Environnement(altitude_m=1524, mu=0.014, pente=0.01, phi=0.6)
    sim_haut = SimulateurTakeOff(mon_avion, env_haut)
    dist_haut_modele2 = sim_haut.calculer_distance(modele=2)
    approx_slo = sim.approx_distance_v_moyenne()
    approx_slo_hp = sim.approx_distance_haute_performance()

    print(f"Distance de décollage modèle 2 à 1524 m avec pente 1% : {dist_haut_modele2:.2f} m")
    print(f"Approximation vitesse moyenne 0.7 V_LO : {approx_slo:.2f} m")
    print(f"Approximation haute performance : {approx_slo_hp:.2f} m")

    # Tracer les courbes de trajectoire pour modèle 1 et modèle 2 à 0 m
    n_points = 200
    distances1, vitesses1 = sim.tracer_trajectoire(modele=1, n_points=n_points)
    distances2, vitesses2 = sim.tracer_trajectoire(modele=2, n_points=n_points)

    plt.figure(figsize=(12, 6))
    plt.plot(distances1, vitesses1, color='blue', linewidth=2, label='Modèle 1 : poussée constante')
    plt.plot(distances2, vitesses2, color='red', linewidth=2, label='Modèle 2 : poussée T0 - k V^2 (phi=0.6)')
    plt.title('Velocity vs. Distance during Takeoff Roll')
    plt.xlabel('Distance (m)')
    plt.ylabel('Vitesse (m/s)')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('takeoff_velocity_distance_modeles.png', dpi=150)
    plt.show()

    # Tracer les effets de pente et les approximations
    slopes = [0.0, 0.005, 0.01, 0.02]
    model1_distances = sim.distance_par_pentes(slopes, modele=1)
    model2_distances = sim.distance_par_pentes(slopes, modele=2)
    approx_slo_distances = []
    approx_hp_distances = []
    for pente in slopes:
        env = Environnement(env_sol.altitude, mu=env_sol.mu, pente=pente, phi=env_sol.phi)
        sim_pente = SimulateurTakeOff(mon_avion, env)
        approx_slo_distances.append(sim_pente.approx_distance_v_moyenne())
        approx_hp_distances.append(sim_pente.approx_distance_haute_performance())

    # Tracer les distances de décollage en fonction de la pente pour les deux modèles et les approximations avec des points marqués
    plt.figure(figsize=(12, 6))
    plt.plot([p * 100 for p in slopes], model1_distances, marker='o', label='Modèle 1')
    plt.plot([p * 100 for p in slopes], model2_distances, marker='o', label='Modèle 2')
    plt.plot([p * 100 for p in slopes], approx_slo_distances, marker='x', linestyle='--', label='Approx. 0.7 V_LO')
    plt.plot([p * 100 for p in slopes], approx_hp_distances, marker='x', linestyle='--', label='Approx. haute perf.')
    plt.title('Distance de décollage en fonction de la pente')
    plt.xlabel('Pente (%)')
    plt.ylabel('Distance de décollage (m)')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('takeoff_distance_vs_slope.png', dpi=150)
    plt.show()

    # Comparaison des distances de décollage à pente 0% pour les deux modèles et les approximations avec un graphique à barres
    labels = ['Modèle 1', 'Modèle 2', 'Approx. 0.7 V_LO', 'Approx. haute perf.']
    values = [dist_modele1, dist_modele2, approx_slo, approx_slo_hp]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color=['blue', 'red', 'orange', 'green'])
    plt.title('Comparaison des distances de décollage à pente 0%')
    plt.ylabel('Distance (m)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('takeoff_approximation_comparison.png', dpi=150)
    plt.show()
                






























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