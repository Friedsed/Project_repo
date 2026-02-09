############################################################################### VALIDATE THE 02/01/2026 BY ALEXANDRE FRIEDLY
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Données physiques -----------------------


D   = 0.003          # diamètre intérieur (m)
rho = 1000           # masse volumique eau (kg/L ~ 1000 kg/m3, ici unité TP)
nu  = 1.01e-6        # viscosité cinématique (m2/s)
g   = 9.8            # gravité (m/s2)
L   = 0.524          # longueur entre prises de pression (m)

rhom = 13600         # masse volumique relative mercure (kg/L)

# ----------------------- Données expérimentales -----------------------

# Eau
deau   = [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40]      # Δh eau (cm)
dtseau = [275, 151, 97, 75, 59, 45, 41, 39, 35, 33, 30]  # temps (s)

# Mercure (Δh=0 supprimé)
dmercure  = [7.2, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1]        # Δh mercure (cm)
dtsmecure = [20, 22, 24, 27, 32, 37, 49]               # temps (s)

# ----------------------- Traitement EAU -----------------------

# 0.2 L collectés à chaque fois
Q_eau = [0.2*1e-3 / t for t in dtseau]                     # débit volumique (L/s),  on a choisi 0.2L pour chaque mesure

#  vitesse moyenne
#mass_flow_eau = [rho * Q for Q in Q_eau]
U_eau = [Q / (np.pi * (D*D / 4)) for Q in Q_eau]

Re_eau = [D * U / nu for U in U_eau]

# ΔP eau en Pascal convertir éventuellement si besoin
deltaP_eau = [rho * g * h*0.01 for h in deau]

# coefficient de perte mesuré
lambda_eau = [ (D * dP) / (0.5 * L * rho * U**2) for dP, U in zip(deltaP_eau, U_eau) ]

# coefficient théorique laminaire
lambda_th_eau = [64.0 / Re for Re in Re_eau]

# ----------------------- Traitement MERCURE -----------------------

Q_mer = [0.2*1e-3 / t for t in dtsmecure]                  # L/s


U_mer = [Q / ( np.pi * (D*D / 4)) for Q in Q_mer]

Re_mer = [D * U / nu for U in U_mer]

deltaP_mer = [rhom * g * h*0.01 for h in dmercure]

lambda_mer = [ (D * dP) / (0.5 * L * rhom * U**2) for dP, U in zip(deltaP_mer, U_mer)]

lambda_th_mer = [64.0 / Re for Re in Re_mer]

# ----------------------- Tracé EAU -----------------------

plt.figure(figsize=(7, 5))
plt.loglog(Re_eau, lambda_th_eau,
           marker="+", color="red", linewidth=2,
           label="Théorie laminaire : $\\lambda = 64/Re$")
plt.loglog(Re_eau, lambda_eau,
           "o", color="black", label="Données expérimentales (eau)")

plt.xlabel("$Re_{eau}$", fontsize=12)
plt.ylabel("$\\lambda$", fontsize=12)
plt.title("Coefficient de perte de charge régulière (eau)\n"
          "Comparaison théorie Poiseuille / mesures")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------- Tracé MERCURE -----------------------

plt.figure(figsize=(7, 5))
plt.loglog(Re_mer, lambda_th_mer,
           marker="+", color="green", linewidth=2,
           label="Théorie laminaire : $\\lambda = 64/Re$")
plt.loglog(Re_mer, lambda_mer,
           "o", color="black", label="Données expérimentales (mercure)")

plt.xlabel("$Re_{mercure}$", fontsize=12)
plt.ylabel("$\\lambda$", fontsize=12)
plt.title("Coefficient de perte de charge régulière (mercure)\n"
          "Comparaison théorie Poiseuille / mesures")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
