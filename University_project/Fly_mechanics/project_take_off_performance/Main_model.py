
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




# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Management of the running data                                                                                                                             |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////

fichier = csv.DictReader(open("excel_file/data.csv"))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

passagers = list(fichier)
#print (passagers[2])
#print(passagers[1].keys())



data = { "name": "None", "W": 13488.54,  "Sw": 269.1, "Clmax": 1.69,  "Clto": 1.2,  "Cdto": 0.03,   "g": 32.2,  "mu": 0.03,  "P": 310, "rho": 0.002378,  "T": 3372.135,  "n": 35,  "A1": 1.158e-2,
    "A2": -5.277e-05,  "A3": 9.273e-08,   "A4": -6.21e-11,   "engine": "piston",  "efficiency": 0.5,    "bw": 33,   "hw": 6,   "Ra": 6.05,   "Cdo": 0.036,   "Cdol": 0.0,
    "e": 0.82,   "Cl": 0.3485,   "T0": 1200,   "T1": -4,   "T2": 0,   "alpha": 0,   "alpha_0": 0,   "Vhw": 29.33,   "Vi": 29.33 }

modify_dict(data, param1)
# =====================================================
#Conversion of unit
# =====================================================

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
#data = convert_units(data, "SI" , "US")
#print("Checking the unity : ", data)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")








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

    param = get_model_params("Model1", data)

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

    param = get_model_params("Model2", data)

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
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " General ground run estimation using Average Acceleration for tricycle propeller aicraft ONLY ;       page 797 ")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

    # =====================================================
    # Parameters
    # =====================================================

    param = get_model_params("Model2", data)

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
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")
    print( " Ground run estimation using numerical Integration method")
    print(" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           ")

    # =====================================================
    # Parameters
    # =====================================================

    param = get_model_params("Model4", data)

    # =====================================================
    #Definition of the model 
    # =====================================================
    model = AircraftTakeoff(param)

    # =====================================================
    #plotting and Computing
    # =====================================================

    model.summary()
    model.plot_forces()
    model.plot_speed()
    model.plot_acceleration()



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
                












