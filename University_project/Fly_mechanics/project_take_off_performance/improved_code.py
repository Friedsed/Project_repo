"""
import math

class Aircraft:
    def __init__(self, data):
        self.W = data['weight']          # Poids (lbf)
        self.S = data['wing_area']      # Surface alaire (ft2)
        self.b = data['wingspan']       # Envergure (ft)
        self.hw = data['wing_height']    # Hauteur de l'aile (ft)
        self.Cd0 = data['cd0']          # Traînée parasite
        self.e = data['oswald_factor']   # Facteur d'Oswald
        self.Cl_max = data['cl_max']    # Cl max (config décollage)
        self.Cl_accel = data['cl_accel'] # Cl pendant le roulage
        self.mu_r = data['friction_mu']  # Coeff friction
        self.tr = data.get('t_rotation', 1.0) # Temps de rotation (s)
        self.h_oc = data.get('h_obstacle', 35.0) # Hauteur obstacle (ft)
        # Poussée T = T0 + T1*V + T2*V^2
        self.T0 = data['thrust_static']
        self.T1 = data['thrust_v']
        self.T2 = data['thrust_v2']

class TakeoffPerformance:
    def __init__(self, aircraft, rho=0.0023769, g=32.2):
        self.ac = aircraft
        self.rho = rho
        self.g = g

    def get_thrust(self, v):
        return self.ac.T0 + self.ac.T1 * v + self.ac.T2 * (v**2)

    def get_drag(self, v, cl, in_ground_effect=True):
        q = 0.5 * self.rho * (v**2) * self.ac.S
        ra = (self.ac.b**2) / self.ac.S
        phi = 1.0
        if in_ground_effect:
            h_ratio = 16 * self.ac.hw / self.ac.b
            phi = (h_ratio**2) / (1 + h_ratio**2)
        cd = self.ac.Cd0 + (phi * (cl**2)) / (math.pi * self.ac.e * ra)
        return cd * q

    def calculate_distances(self):
        # 1. Vitesses de référence [6, 7]
        v_stall = math.sqrt((2 * self.ac.W) / (self.ac.S * self.ac.Cl_max * self.rho))
        v_lo = 1.1 * v_stall  # Vitesse de décollage (Liftoff)
        v_oc = 1.2 * v_stall  # Vitesse au franchissement d'obstacle

        # 2. Distance d'accélération Sa (Eq. 3.10.25) [2]
        k0 = (self.ac.T0 / self.ac.W) - self.ac.mu_r
        k1 = self.ac.T1 / self.ac.W
        
        ra = (self.ac.b**2) / self.ac.S
        h_ratio = 16 * self.ac.hw / self.ac.b
        phi = (h_ratio**2) / (1 + h_ratio**2)
        cd_accel = self.ac.Cd0 + (phi * (self.ac.Cl_accel**2)) / (math.pi * self.ac.e * ra)
        
        k2 = (self.ac.T2 / self.ac.W) + (self.rho / (2 * self.ac.W / self.ac.S)) * (self.ac.Cl_accel * self.ac.mu_r - cd_accel)
        
        f_s = k0
        f_lo = k0 + k1 * v_lo + k2 * (v_lo**2)
        kr = 4 * k0 * k2 - k1**2
        
        if kr < 0:
            sq_kr = math.sqrt(-kr)
            kw = (1 / sq_kr) * math.log((( (k1 + 2*k2*v_lo) - sq_kr) * (k1 + sq_kr)) / 
                                        (( (k1 + 2*k2*v_lo) + sq_kr) * (k1 - sq_kr)))
            kt = (1 / (2 * k2)) * math.log(f_lo / f_s) - (k1 * kw) / (2 * k2)
        else:
            sq_kr = math.sqrt(kr)
            kw = (2 / sq_kr) * (math.atan((k1 + 2*k2*v_lo) / sq_kr) - math.atan(k1 / sq_kr))
            kt = (1 / (2 * k2)) * math.log(f_lo / f_s) - (k1 * kw) / (2 * k2)
        
        sa = kt / self.g

        # 3. Distance de rotation Sr (Eq. 3.10.36) [3]
        sr = v_lo * self.ac.tr

        # 4. Distance de montée Sc (Eq. 3.11.13) [8]
        # Forces à V_LO
        t_lo = self.get_thrust(v_lo)
        d_lo = self.get_drag(v_lo, cl=self.ac.W / (0.5 * self.rho * v_lo**2 * self.ac.S))
        
        # Forces à V_OC (hors effet de sol)
        t_oc = self.get_thrust(v_oc)
        cl_oc = self.ac.W / (0.5 * self.rho * v_oc**2 * self.ac.S)
        d_oc = self.get_drag(v_oc, cl=cl_oc, in_ground_effect=False)
        
        gamma_oc = math.asin((t_oc - d_oc) / self.ac.W) # Angle de montée [9]
        
        f_c_avg = ((t_lo - d_lo) + (t_oc - d_oc) / math.cos(gamma_oc)) / 2 # Force nette effective [5]
        sc = (self.ac.W / f_c_avg) * (self.ac.h_oc + (v_oc**2 - v_lo**2) / (2 * self.g))

        return sa, sr, sc

# --- VALEURS DE TEST (Exemple 3.11.1 du document) [8] ---
data_jet = {
    'weight': 20000.0, 'wing_area': 320.0, 'wingspan': 54.0, 'wing_height': 6.0,
    'cd0': 0.033, 'oswald_factor': 0.74, 'cl_max': 1.6, 'cl_accel': 0.4,
    'friction_mu': 0.04, 't_rotation': 1.0, 'h_obstacle': 35.0,
    'thrust_static': 6500.0, 'thrust_v': 0.0, 'thrust_v2': 0.0
}
params_beluga = {
    'weight': 1520550.0,    # N (Mass ~ 155t au décollage)
    'wing_area': 260.0,     # m2 [1]
    'wingspan': 44.8,       # m (Allongement lambda ~ 7.7 [7])
    'wing_height': 4.5,     # m (Estimation hauteur aile basse)
    'cd0': 0.0175,          # [1]
    'oswald_factor': 0.97,  # [1]
    'cl_max': 2.5,          # Double flap wing [2]
    'cl_accel': 0.45,
    'friction_mu': 0.02,    # Piste ciment [5]
    't_rotation': 3.0,      # Transport lourd [8]
    'h_obstacle': 10.7,     # 35 ft en mètres [6]
    'thrust_static': 520000.0, # N (2x GE CF6-80C2)
    'thrust_v': -15.0,      # Chute légère de poussée avec V
    'thrust_v2': 0.0
}

params_a320 = {
    'weight': 765000.0,     # N (Mass ~ 78t)
    'wing_area': 122.6,     # m2
    'wingspan': 34.1,       # m
    'wing_height': 3.5,     # m
    'cd0': 0.035,           # Jet aircraft [2]
    'oswald_factor': 0.6,   # Low wing aircraft [2]
    'cl_max': 2.4,          # Volets déployés
    'cl_accel': 0.4,
    'friction_mu': 0.025,   # Macadam [5]
    't_rotation': 2.0,      # Moyen courrier
    'h_obstacle': 10.7,     # 35 ft
    'thrust_static': 220000.0, # N (2x CFM56-5B)
    'thrust_v': 0.0,
    'thrust_v2': 0.0
}

calc = TakeoffPerformance(Aircraft( params_beluga ))
sa, sr, sc = calc.calculate_distances()

print(f"Sa (Accélération): {sa:.1f} ft")
print(f"Sr (Rotation): {sr:.1f} ft")
print(f"Sc (Montée): {sc:.1f} ft")
print(f"St (Total): {sa + sr + sc:.1f} ft")



"""





##########################################################################"
#distance de rotation notebook lm
########################################################################""






"""








import math

class AircraftSpecs:
    #Stocke les caractéristiques techniques de l'avion.
    def __init__(self, data):
        self.W = data['weight']          # Poids (lbf)
        self.S = data['wing_area']      # Surface alaire (ft2)
        self.Cl_max = data['cl_max']    # Cl max pour décollage
        self.Cd_min = data['cd_min']    # Traînée parasite min
        self.k = data['k_factor']       # Facteur k de la polaire (1/pi*lambda*e)
        self.T = data['thrust_tr']      # Poussée à la vitesse V_TR (lbf)
        self.h_obst = data['h_obstacle']# Hauteur de l'obstacle (ft)

class TransitionClimbCalculator:
    #Calcule les segments TR et Climb selon la méthode de la photo.
    def __init__(self, aircraft, rho=0.002378, g=32.2):
        self.ac = aircraft
        self.rho = rho
        self.g = g

    def run_calculation(self):
        # 1. Vitesse de décrochage et de transition (p. 805)
        v_stall = math.sqrt((2 * self.ac.W) / (self.rho * self.ac.S * self.ac.Cl_max))
        v_tr = 1.15 * v_stall
        
        # 2. Rayon de transition (Basé sur n = 1.1903, p. 804)
        n = 1.1903
        r = (v_tr**2) / (self.g * (n - 1))
        
        # 3. Angle de montée (gamma) à V_TR
        q_tr = 0.5 * self.rho * (v_tr**2)
        cl_tr = self.ac.W / (q_tr * self.ac.S)
        cd_tr = self.ac.Cd_min + self.ac.k * (cl_tr**2)
        ld_ratio = cl_tr / cd_tr # Rapport Finesse L/D
        
        # sin(gamma) = T/W - 1/(L/D)
        sin_gamma = (self.ac.T / self.ac.W) - (1 / ld_ratio)
        gamma = math.asin(sin_gamma)
        
        # 4. Hauteur gagnée pendant la transition (h_TR)
        h_tr = r * (1 - math.cos(gamma))
        
        # 5. Calcul des distances horizontales
        if h_tr >= self.ac.h_obst:
            # Cas 5b : Obstacle franchi pendant la transition
            s_total = math.sqrt(r**2 - (r - self.ac.h_obst)**2)
            phase = "Transition uniquement (Obstacle franchi tôt)"
        else:
            # Cas standard : Transition + Montée rectiligne
            s_tr = r * math.sin(gamma)
            s_c = (self.ac.h_obst - h_tr) / math.tan(gamma)
            s_total = s_tr + s_c
            phase = "Transition + Montée initiale"

        return {
            "V_stall (ft/s)": round(v_stall, 1),
            "V_TR (ft/s)": round(v_tr, 1),
            "Rayon R (ft)": round(r, 0),
            "Angle gamma (deg)": round(math.degrees(gamma), 2),
            "h_TR (ft)": round(h_tr, 1),
            "Distance Horizontale Totale (ft)": round(s_total, 1),
            "Phase": phase
        }

# --- VALEURS POUR TESTER LE CODE (Exemple 18-8, p. 805) ---
data_cirrus = {
    'weight': 3400.0,       # lbf
    'wing_area': 144.9,     # ft2
    'cl_max': 1.69,
    'cd_min': 0.0350,
    'k_factor': 0.04207,
    'thrust_tr': 825.0,     # Poussée à 73.6 knots
    'h_obstacle': 50.0      # ft
}

# Lancement
avion = AircraftSpecs(data_cirrus)
calc = TransitionClimbCalculator(avion)
resultats = calc.run_calculation()

for cle, val in resultats.items():
    print(f"{cle} : {val}")
"""


"""


import math

class AircraftData:
    def __init__(self, params):
        self.W = params['weight']          # Poids (lbf ou N)
        self.S = params['wing_area']      # Surface alaire (ft2 ou m2)
        self.cl_max = params['cl_max']    # Cl max configuration décollage
        self.cl_to = params['cl_takeoff'] # Cl constant pendant le roulage
        self.cd0 = params['cd0']          # Traînée parasite
        self.k = params['k_factor']       # Facteur d'Oswald (1 / (pi * AR * e))
        self.mu = params['friction_mu']   # Coefficient de friction (piste)
        # Fonctions de poussée ou constantes
        self.T_static = params['thrust_static']
        self.T_func = params.get('thrust_func') # Fonction optionnelle T(v)

class TakeoffCalculator:
    def __init__(self, aircraft, rho=0.002378, g=32.17):
        self.ac = aircraft
        self.rho = rho
        self.g = g
        # Calcul de la vitesse de décollage (V_LOF = 1.1 * Vs) [6]
        self.v_stall = math.sqrt((2 * self.ac.W) / (self.rho * self.ac.S * self.ac.cl_max))
        self.v_lof = 1.1 * self.v_stall

    def get_forces(self, v):
        #Calcule la poussée, traînée et portance à une vitesse donnée [4, 7].
        q = 0.5 * self.rho * (v**2)
        L = q * self.ac.S * self.ac.cl_to
        D = q * self.ac.S * (self.ac.cd0 + self.ac.k * (self.ac.cl_to**2))
        T = self.ac.T_func(v) if self.ac.T_func else self.ac.T_static
        return T, D, L

    def method_average_acceleration(self):
        #Méthode 1 : Accélération évaluée à V_LOF / sqrt(2) [1, 8].
        v_eval = self.v_lof / math.sqrt(2)
        T, D, L = self.get_forces(v_eval)
        accel = (self.g / self.ac.W) * (T - D - self.ac.mu * (self.ac.W - L))
        return (self.v_lof**2) / (2 * accel)

    def method_analytical_classic(self):
        #Méthode 2 : Modèle classique s = -1/2B * ln(1 - B/A * V^2) [9].
        # On définit A (accel à V=0) et B (terme de traînée/poussée variable)
        T0, _, _ = self.get_forces(0)
        A = (self.g / self.ac.W) * (T0 - self.ac.mu * self.ac.W)
        
        # Évaluation à V_LOF pour trouver B (simplifié)
        T_lo, D_lo, L_lo = self.get_forces(self.v_lof)
        accel_lo = (self.g / self.ac.W) * (T_lo - D_lo - self.ac.mu * (self.ac.W - L_lo))
        B = (A - accel_lo) / (self.v_lof**2)
        
        return -(1 / (2 * B)) * math.log(1 - (B / A) * (self.v_lof**2))

    def method_numerical_integration(self, dt=0.1):
        #Méthode 3 : Intégration pas à pas (Euler) [4, 5].
        v, s, t = 0.0, 0.0, 0.0
        while v < self.v_lof:
            T, D, L = self.get_forces(v)
            accel = (self.g / self.ac.W) * (T - D - self.ac.mu * (self.ac.W - L))
            s += v * dt + 0.5 * accel * (dt**2)
            v += accel * dt
            t += dt
        return s

# --- VALEURS DE TEST (Exemple 18-5 : Learjet 45) [8] ---
params_learjet = {
    'weight': 21500.0,       # lbf
    'wing_area': 311.6,      # ft2
    'cl_max': 1.65,
    'cl_takeoff': 0.90,      # Cl pendant le roulage
    'cd0': 0.045,
    'k_factor': 0.040,       # Estimé
    'friction_mu': 0.02,
    'thrust_static': 7000.0, # lbf (moyenne sur le roulage)
}

# Simulation
avion = AircraftData(params_learjet)
calc = TakeoffCalculator(avion)

print(f"--- RÉSULTATS POUR {params_learjet['weight']} lbs ---")
print(f"Vitesse de décollage cible : {calc.v_lof:.1f} ft/s")
print(f"Distance (Accélération Moyenne) : {calc.method_average_acceleration():.1f} ft")
print(f"Distance (Analytique Classique) : {calc.method_analytical_classic():.1f} ft")
print(f"Distance (Intégration Numérique): {calc.method_numerical_integration():.1f} ft")



"""


"""

import math

class AircraftCirrus:
    def __init__(self, data):
        self.W = data['weight']          # lb
        self.S = data['wing_area']      # ft2
        self.P_BHP = data['power']       # hp
        self.Cl_max = data['cl_max']    # Cl max
        self.Cl_TO = data['cl_takeoff']  # Cl pendant le roulage
        self.Cd0 = data['cd0']          # Traînée parasite
        self.mu = data['friction_mu']    # Coefficient de friction
        self.eta_p = data['eta_p']       # Rendement à VR/sqrt(2)

class Page798Calculator:
    def __init__(self, aircraft, rho=0.002378, g=32.174):
        self.ac = aircraft
        self.rho = rho # Masse volumique air (slug/ft3)
        self.g = g

    def calculate_takeoff(self):
        # 1. Vitesse de rotation VR (1.1 * Vs)
        v_r = 1.1 * math.sqrt((2 * self.ac.W) / (self.rho * self.ac.S * self.ac.Cl_max))
        
        # 2. Vitesse d'évaluation pour l'accélération moyenne
        v_eval = v_r / math.sqrt(2)
        
        # 3. Pression dynamique à v_eval
        q = 0.5 * self.rho * (v_eval**2)
        
        # 4. Forces calculées à v_eval (comme à la page 798)
        thrust = (self.ac.eta_p * 550 * self.ac.P_BHP) / v_eval
        lift = q * self.ac.S * self.ac.Cl_TO
        drag = q * self.ac.S * self.ac.Cd0
        
        # 5. Accélération moyenne (Eq. 18-4)
        accel = (self.g / self.ac.W) * (thrust - drag - self.ac.mu * (self.ac.W - lift))
        
        # 6. Distance de roulement (SG) et Temps (t)
        s_g = (v_r**2) / (2 * accel)
        time = v_r / accel
        
        return {
            "Vitesse Rotation VR (ft/s)": round(v_r, 1),
            "Poussée à v_eval (lb)": round(thrust, 1),
            "Portance à v_eval (lb)": round(lift, 1),
            "Trainée à v_eval (lb)": round(drag, 1),
            "Accélération Moyenne (ft/s2)": round(accel, 2),
            "Distance de roulement SG (ft)": round(s_g, 0),
            "Temps de décollage (s)": round(time, 1)
        }

# --- Simulation avec les données exactes de la page 798 ---
cirrus_specs = {
    'weight': 3400.0,
    'wing_area': 144.9,
    'power': 310.0,
    'cl_max': 1.69,
    'cl_takeoff': 0.500,
    'cd0': 0.0417,
    'friction_mu': 0.04,
    'eta_p': 0.50
}

calc = Page798Calculator(AircraftCirrus(cirrus_specs))
res = calc.calculate_takeoff()

print("Vérification des résultats de la page 798 :")
for k, v in res.items():
    print(f"{k} : {v}")




"""

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres Cirrus SR22 (Pages 798-799) ---
W = 3400.0        # Poids (lb)
S = 144.9         # Surface alaire (ft2)
Cl_TO = 0.500     # Coeff de portance (roulage)
Cd_TO = 0.0417    # Coeff de traînée (roulage)
mu = 0.04         # Coeff de friction au sol
rho = 0.002378    # Masse volumique air (slug/ft3)
g = 32.174        # Accélération pesanteur (ft/s2)
V_target = 118.9  # Vitesse de rotation cible (ft/s)
dt = 0.5          # Pas de temps (secondes)

# Modèle de poussée T(V) approximé par spline (Table 18-5 p. 799)
# T = -0.0116*V^2 - 1.18*V + 1169
def get_thrust(v):
    return -0.0116 * (v**2) - 1.18 * v + 1169

# --- Boucle d'intégration numérique ---
t, v, s = 0.0, 0.0, 0.0
res = {'t': [], 'v': [], 's': [], 'a': [], 'T': [], 'D': [], 'Ff': []}

while v < V_target:
    q = 0.5 * rho * v**2
    L = q * S * Cl_TO
    D = q * S * Cd_TO
    T = get_thrust(v)
    Ff = mu * (W - L) # Force de friction
    
    # Équation (18-15) : calcul de l'accélération
    accel = (g / W) * (T - D - Ff)
    
    # Stockage des données
    res['t'].append(t); res['v'].append(v); res['s'].append(s)
    res['a'].append(accel); res['T'].append(T); res['D'].append(D); res['Ff'].append(Ff)
    
    # Mise à jour (Eq. 18-16 & 18-17)
    s += v * dt + 0.5 * accel * dt**2
    v += accel * dt
    t += dt

# --- Tracé des courbes (similaires aux données de la Table 18-5) ---
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(res['s'], res['v'], 'b-', label='Vitesse (ft/s)')
plt.ylabel('Vitesse [ft/s]')
plt.title('Performance de Décollage - Simulation Numérique (Cirrus SR22)')
plt.grid(True); plt.legend()

plt.subplot(2, 1, 2)
plt.plot(res['s'], res['T'], 'g-', label='Poussée (T)')
plt.plot(res['s'], res['D'], 'r-', label='Traînée (D)')
plt.plot(res['s'], res['Ff'], 'k--', label='Friction ($\mu N$)')
plt.xlabel('Distance parcourue [ft]')
plt.ylabel('Forces [lb]')
plt.grid(True); plt.legend()
plt.show()
