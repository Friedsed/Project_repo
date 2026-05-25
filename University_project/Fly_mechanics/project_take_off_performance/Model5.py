import math
import matplotlib.pyplot as plt

class Avion:
    def __init__(self, nom,masse_kg,poussee_statique_n,surface_alaire_m2,Cdo,Clmax,Ra,k=0.0):  
        self.nom =nom
        self.m =masse_kg
        self.W =masse_kg * 9.80665
        self.T0 =poussee_statique_n
        self.Sw =surface_alaire_m2
        self.Cdo =Cdo
        self.Clmax =Clmax
        self.Ra =Ra
        self.k = k  # coefficient de perte de poussée quadratique (N/(m/s)^2)

class Environnement:
    def __init__(self, altitude_m, mu=0.04, pente=0.0, phi=1.0):
        self.altitude = altitude_m
        self.mu = mu
        self.pente = pente
        self.phi = phi
        self.g = 9.80665

        rho0 = 1.225
        self.rho = rho0 * (1 - 2.25577e-5 * altitude_m) ** 5.2559

class SimulateurTakeOff:
    def __init__(self, avion, env):
        self.avion = avion
        self.env = env

    def _calculer_constantes(self, modele=1):
        rho0 = 1.225
        Vi = math.sqrt((2 * self.avion.W) / (self.env.rho * self.avion.Sw * self.avion.Clmax))
        Vlo = 1.1 * Vi
        poussee_actuelle = self.avion.T0 * (self.env.rho / rho0)
        A = self.env.g * ((poussee_actuelle / self.avion.W) - self.env.mu - self.env.pente)

        
        Cl_takeoff = 0.75 * self.avion.Clmax
        Cd = self.avion.Cdo + self.env.phi * (Cl_takeoff**2 / (math.pi * self.avion.Ra * 0.82))

        if modele == 1:
            B = (self.env.rho * self.avion.Sw * (Cd - self.env.mu * Cl_takeoff)) / (2 * self.avion.m)
        elif modele == 2:
            B = (self.env.g / self.avion.W) * (
                0.5 * self.env.rho * self.avion.Sw * (Cd - self.env.mu * Cl_takeoff) + self.avion.k
            )
        else:
            raise ValueError("Modèle inconnu : utilisez 1 ou 2.")

        return Vlo, A, B

    def calculer_distance(self, modele=1):
        Vlo, A, B = self._calculer_constantes(modele)
        if A > B * Vlo**2:
            s_accel = -(1 / (2 * B)) * math.log(1 - (B / A) * Vlo**2)
        else:
            return "Erreur : la poussée est insuffisante pour décoller."

        t_rotation = 1.0
        s_rot = Vlo * t_rotation
        return s_accel + s_rot

    def calculer_temps(self, modele=2):
        Vlo, A, B = self._calculer_constantes(modele)
        if A > B * Vlo**2:
            return (1 / math.sqrt(A * B)) * math.atanh(math.sqrt(B / A) * Vlo)
        return None

    def tracer_trajectoire(self, modele=1, n_points=200):
        Vlo, A, B = self._calculer_constantes(modele)
        if A <= 0 or B <= 0:
            raise ValueError("Impossible de tracer la trajectoire : constantes invalides.")

        vitesses = [i * Vlo / (n_points - 1) for i in range(n_points)]
        distances = [-(1 / (2 * B)) * math.log(1 - (B / A) * v**2) if v > 0 else 0.0 for v in vitesses]
        return distances, vitesses

    def distance_par_pentes(self, slopes, modele=1):
        distances = []
        for pente in slopes:
            env = Environnement(self.env.altitude, mu=self.env.mu, pente=pente, phi=self.env.phi)
            sim = SimulateurTakeOff(self.avion, env)
            distances.append(sim.calculer_distance(modele))
        return distances

    def approx_distance_v_moyenne(self, mu_rel=None):
        rho = self.env.rho
        g = self.env.g
        W = self.avion.W
        Vlo, _, _ = self._calculer_constantes(modele=1)
        V_avg = 0.7 * Vlo
        Cl = 0.75 * self.avion.Clmax
        Cd = self.avion.Cdo + self.env.phi * (Cl**2 / (math.pi * self.avion.Ra * 0.82))
        D = 0.5 * rho * self.avion.Sw * Cd * V_avg**2
        L = 0.5 * rho * self.avion.Sw * Cl * V_avg**2
        mu = self.env.mu if mu_rel is None else mu_rel
        T = self.avion.T0 * (rho / 1.225)
        denom = g * rho * self.avion.Sw * self.avion.Clmax * (T - (D + mu * (W - L)))
        if denom <= 0:
            return None
        return 1.44 * W**2 / denom

    def approx_distance_haute_performance(self):
        rho = self.env.rho
        g = self.env.g
        W = self.avion.W
        T = self.avion.T0 * (rho / 1.225)
        denom = g * rho * self.avion.Sw * self.avion.Clmax * T
        if denom <= 0:
            return None
        return 1.44 * W**2 / denom

# et les informations de l'aion se font ici
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