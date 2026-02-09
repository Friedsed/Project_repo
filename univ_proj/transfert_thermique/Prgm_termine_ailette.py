"""
=============================================================================
TP - TRANSFERT THERMIQUE DANS UNE AILETTE CYLINDRIQUE
=============================================================================
Programme Python pour l'étude du transfert thermique dans une ailette.
Ce script résout l'équation différentielle de la diffusion thermique
pour un ailette cylindrique, puis compare les solutions théoriques
avec les données expérimentales obtenues en laboratoire.

Auteur: AUTHIE WOLI CORRE
Date: Décembre 2025
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# =============================================================================
# SECTION 1 : DONNÉES DES FLUIDES
# =============================================================================

# Propriétés thermophysiques de l'air à différentes températures
data = np.array([
[100, 3.5562, 0.711, 0.200,  1032, 0.934, 0.254, 0.786],
[150, 2.3364, 1.034, 0.4426, 1012, 1.38,  0.584, 0.758],
[200, 1.7458, 1.325, 0.7590, 1007, 1.81,  1.03,  0.737],
[250, 1.3947, 1.596, 1.144,  1006, 2.23,  1.59,  0.720],
[300, 1.1614, 1.846, 1.589,  1007, 2.63,  2.25,  0.707],
[350, 0.9950, 2.082, 2.092,  1009, 3.00,  2.99,  0.700],
[400, 0.8711, 2.301, 2.641,  1014, 3.38,  3.83,  0.690]
])

# Extraction des colonnes de données

T =     data[:,0]           # Température (K)
rho =   data[:,1]           # Masse volumique (kg/m³)
mu =    data[:,2] * 1e-5    # Viscosité dynamique (kg/(m·s))
nu =    data[:,3] * 1e-5    # Viscosité cinématique (m²/s)
Cp =    data[:,4]           # Chaleur spécifique (J/(kg·K))
kf =    data[:,5] * 1e-2    # Conductivité thermique du fluide (W/(m·K))
alpha = data[:,6] * 1e-5    # Diffusivité thermique (m²/s)
Pr =    data[:,7]           # Nombre de Prandtl

# =============================================================================
# SECTION 2 : PARAMÈTRES GÉOMÉTRIQUES DE L'AILETTE
# =============================================================================

r0 = 0.005          # Rayon de l'ailette (m)
L = 0.7             # Longueur de l'ailette (m)
pi = np.pi
A = pi * r0**2      # Section droite de l'ailette (m²)
p = 2 * pi * r0     # Périmètre de l'ailette (m)

# Coefficient d'échange thermique par convection (W/(m²·K))
he = 5

# Température de référence à la base de l'ailette (°C)
theta_0 = 1

# =============================================================================
# SECTION 3 : PROPRIÉTÉS THERMIQUES DES MATÉRIAUX
# =============================================================================

# Conductivité thermique (W/(m·K))

k_laiton = 114      # Conductivité thermique du laiton (W/(m·K))
k_acier = 45        # Conductivité thermique de l'acier (W/(m·K))
k = 100             # Conductivité thermique par défaut pour les questions théoriques (W/(m·K))

# Positions de mesure des thermocouples (m)
x_laiton = np.array([3/1000, 52.5/1000, 102.5/1000, 202.5/1000, 302.5/1000, 402.5/1000])
x_acier = np.array([1/100, 3.5/100, 6.5/100, 11.5/100, 21.5/100])


x = np.linspace(0, L, 100)
m = np.sqrt((he*p)/(k*A))

# =============================================================================
# SECTION 4 : DÉFINITION DES FONCTIONS THERMIQUES
# =============================================================================

##QUESTION 1 : Solution pour une ailette de longueur infinie

def theta_infinie(theta_0, m, x):
    return theta_0 * np.exp(-m*x)   


##QUESTION 2 : Solution pour une ailette de longueur finie
def theta_finie(theta_0, he, k, m, L, x):
    numerateur = np.cosh(m*(L-x)) + (he/(k*m))*np.sinh(m*(L-x))
    denominateur = np.cosh(m*L) + (he/(k*m))*np.sinh(m*L)
    return theta_0 * (numerateur / denominateur)


# Question 3 : Tracer les deux solutions obtenues pour différentes valeurs de he et k
# Étudier l'influence du coefficient d'échange he et de la conductivité k
# Conclusion sur l'efficacité de l'ailette


print("QUESTION 3 : COMPARAISON SOLUTIONS FINIE VS INFINIE")


# ====================
# GRAPHIQUE 1 : Comparaison de base (he=5, k=100)
# ====================
print("\nGraphique 1 : Solutions finie vs infinie (cas de base)")
he_test = 5
k_test = 100
m_test = np.sqrt((he_test * p) / (k_test * A))

plt.figure(figsize=(10, 6))
plt.plot(x, theta_infinie(theta_0, m_test, x), label='Solution INFINIE', color='blue')
plt.plot(x, theta_finie(theta_0, he_test, k_test, m_test, L, x), label='Solution FINIE', color='red')
 
plt.xlabel('Position le long de l"ailette (m)')
plt.ylabel('Température (°C)')
plt.title('Q3 - Solutions finie vs infinie (he=5 W/m²K, k=100 W/mK)')
plt.legend()
plt.show()

# ====================
# GRAPHIQUE 2 : Influence de la longueur L
# ====================
print("\nGraphique 2 : Influence de la longueur de l'ailette")
# Variation de la longueur L
L_values = [0.5, 0.7, 0.9]
he_L = 5
k_L = 100
m_L = np.sqrt((he_L * p) / (k_L * A))

plt.figure(figsize=(10, 6))
for L_val in L_values:
    x_L = np.linspace(0, L_val, 100)
    plt.plot(x_L, theta_infinie(theta_0, m_L, x_L), label=f'L={L_val}m (infinie)', linestyle='--')
    plt.plot(x_L, theta_finie(theta_0, he_L, k_L, m_L, L_val, x_L), label=f'L={L_val}m (finie)')

plt.xlabel('Position le long de l"ailette (m)')
plt.ylabel('Température (°C)')
plt.title('Q3 - Influence de la longueur L (he=5 W/m²K, k=100 W/mK)')
plt.legend()
plt.show()

# Analyse : Longueur de l'ailette
print("✓ Plus L augmente, plus la température décroît sur une grande distance")

# ====================
# GRAPHIQUE 3 : Influence du coefficient d'échange he
# ====================
print("\nGraphique 3 : Influence du coefficient d'échange he")

k_ref = 100

# he = 5 W/m²K
he1 = 5
m1 = np.sqrt((he1 * p) / (k_ref * A))

# he = 10 W/m²K
he2 = 10
m2 = np.sqrt((he2 * p) / (k_ref * A))

plt.figure(figsize=(10, 6))
plt.plot(x, theta_infinie(theta_0, m1, x), label=f'he={he1} (infinie)', color='blue', linestyle='--')
plt.plot(x, theta_finie(theta_0, he1, k_ref, m1, L, x), label=f'he={he1} (finie)', color='blue')
plt.plot(x, theta_infinie(theta_0, m2, x), label=f'he={he2} (infinie)', color='red', linestyle='--')
plt.plot(x, theta_finie(theta_0, he2, k_ref, m2, L, x), label=f'he={he2} (finie)', color='red') 
plt.xlabel('Position le long de l"ailette (m)')
plt.ylabel('Température (°C)')
plt.title('Q3 - Influence du coefficient d"échange he (k=100 W/mK)')
plt.legend()
plt.show()

# Analyse : he augmente => m augmente => température diminue plus vite
print("✓ Quand he augmente de 5 à 10 W/m²K : m augmente => perte thermique plus rapide")

# ====================
# GRAPHIQUE 4 : Influence de la conductivité k
# ====================

print("\nGraphique 4 : Influence de la conductivité thermique k")


# k = 45 W/mK (acier)
k1 = 45
he_ref = 5
m_k1 = np.sqrt((he_ref * p) / (k1 * A))

# k = 114 W/mK (laiton)
k2 = 114
m_k2 = np.sqrt((he_ref * p) / (k2 * A))

plt.figure(figsize=(10, 6))
plt.plot(x, theta_infinie(theta_0, m_k1, x), label=f'k={k1} (infinie)', color='orange', linestyle='--')
plt.plot(x, theta_finie(theta_0, he_ref, k1, m_k1, L, x), label=f'k={k1} (finie)', color='orange', linestyle='-')
plt.plot(x, theta_infinie(theta_0, m_k2, x), label=f'k={k2} (infinie)', color='green', linestyle='--')
plt.plot(x, theta_finie(theta_0, he_ref, k2, m_k2, L, x), label=f'k={k2} (finie)', color='green', linestyle='-')
plt.xlabel('Position le long de l"ailette (m)')
plt.ylabel('Température (°C)')
plt.title('Q3 - Influence de la conductivité k (he=5 W/m²K)')
plt.legend()
plt.show()

# Analyse : k augmente => m diminue => température diminue plus lentement => meilleure efficacité
print("✓ Quand k augmente (45 vs 114 W/mK) : m diminue => température reste plus élevée => meilleure efficacité")

# ====================
# ANALYSE ET CONCLUSIONS
# ====================



"""
PARAMÈTRE CLÉ : m = √(h*p / (k*A))

1. INFLUENCE DU COEFFICIENT D'ÉCHANGE he :
   • Quand he AUGMENTE → m AUGMENTE
   • La courbe décroît PLUS RAPIDEMENT (température baisse vite)
   • L'ailette devient MOINS EFFICACE (perte rapide)
   • Cas optimal : he modéré à faible pour garder l'efficacité

2. INFLUENCE DE LA CONDUCTIVITÉ k :
   • Quand k AUGMENTE → m DIMINUE
   • La courbe décroît PLUS LENTEMENT (température reste élevée)
   • L'ailette devient PLUS EFFICACE (chaleur bien distribuée)
   • Cas optimal : k élevé (laiton >> acier)

3. COMPARAISON FINIE vs INFINIE :
   • Solution infinie (tirets) : suppose une extrémité adiabatique (Q=0)
   • Solution finie (trait continu) : tient compte de la convection à l'extrémité
   • Écart plus grand quand k est petit (m grand)
   • Écart presque nul quand k est grand (m petit)

4. CAS D'USAGE RÉELS :
   • Acier (k=45) : faible conductivité → faut compenser avec surface/convection
   • Laiton (k=114) : excellente conductivité → bon pour ailettes longues
   • Pour efficacité maximale : k élevé + he modéré
"""


# =============================================================================
# SECTION 5 : QUESTION 4 - RÉGRESSION LINÉAIRE (DONNÉES THÉORIQUES)
# =============================================================================

"""
Question 4 : Tracer sur un graphe la solution θ = exp(-m*x) aux points de mesure x_laiton,
puis tracer ln(θ) en fonction de x. Effectuer une régression linéaire pour retrouver θ₀, m et h.
"""


print("QUESTION 4 : RÉGRESSION LINÉAIRE SUR DONNÉES THÉORIQUES")


# Positions de mesure de la barre en laiton (m)
x_lai = np.array([3/1000, 52.5/1000, 102.5/1000, 202.5/1000, 302.5/1000, 402.5/1000])

# Calcul de la température théorique à ces positions
theta_lai = theta_infinie(theta_0, m, x_lai)

# Graphique 1 : Température vs position
plt.figure(figsize=(10, 6))
plt.plot(x_lai, theta_lai, 'x', label='Température théorique', color='red')
plt.xlabel('Position x (m)')
plt.ylabel('Température θ (°C)')
plt.title('Q4a : Température aux points de mesure (laiton)\nSolution infinie θ = θ₀ * exp(-m*x)')
plt.legend()
plt.show()

# Régression linéaire pour retrouver m, θ₀ et h
print("\nRégression linéaire sur ln(θ) = -m*x + ln(θ₀)")

ln_theta_lai = np.log(theta_lai)
pente, ord_origine = np.polyfit(x_lai, ln_theta_lai, 1)
m_calc = -pente
theta_0_ret = np.exp(ord_origine)
h_ret = (m_calc**2 * k_ref * A) / p

print(f"Pente de la régression : {pente:.6f} m**-1")
print(f"Ordonnée à l'origine : {ord_origine:.6f}")
print(f"\nValeurs retrouvées :")
print(f"  m calculée = {m_calc:.6f} m**-1 (théorique : {m:.6f} m**-1)")
print(f"  θ₀ calculée = {theta_0_ret:.6f} °C (théorique : {theta_0:.6f} °C)")
print(f"  h calculé = {h_ret:.4f} W/(m**2.K)")

# Graphique 2 : Régression linéaire
x_lin = np.linspace(x_lai.min(), x_lai.max(), 100)
y_lin = pente * x_lin + ord_origine

plt.figure(figsize=(10, 6))
plt.plot(x_lai, ln_theta_lai, 'x', label='ln(θ) - données théoriques', color='green')
plt.plot(x_lin, y_lin, label='Régression linéaire', color='blue')
plt.xlabel('Position x (m)')
plt.ylabel('ln(θ)')
plt.title('Q4b : Régression linéaire ln(θ) = -m*x + ln(θ₀)\nPente = -m, Ordonnée = ln(θ₀)')
plt.legend()
plt.show()



##QUESTION 5 : Données des thermocouples pour les deux barres
"""
Question 5 : Noter dans un tableau numpy les valeurs relevées des thermocouples 
pour chacune des deux barres (laiton et acier).
"""


print("QUESTION 5 : DONNÉES EXPÉRIMENTALES DES THERMOCOUPLES")


# Données thermocouples mesurées en TP pour la barre en laiton (°C)
thermocouples_laiton = np.array([43.1, 39.4, 34.8, 28.7, 25.7, 23.9])

# Données thermocouples mesurées en TP pour la barre en acier (°C)
thermocouples_acier = np.array([46.9, 41.8, 36.3, 31.4, 25.8])

print("\nBarre en LAITON :")
print(f"  Positions (mm) : {x_laiton*1000}")
print(f"  Températures mesurées (°C) : {thermocouples_laiton}")
print(f"  Nombre de mesures : {len(thermocouples_laiton)}")

print("\nBarre en ACIER :")
print(f"  Positions (cm) : {x_acier*100}")
print(f"  Températures mesurées (°C) : {thermocouples_acier}")
print(f"  Nombre de mesures : {len(thermocouples_acier)}")




##QUESTION 6 : Régression linéaire sur les données expérimentales
"""
Question 6 : Pour les deux barres (laiton et acier), tracer ln(θ) en fonction de x,
effectuer une régression linéaire pour déterminer θ₀, m et h (utiliser polyfit de numpy).
"""

print("QUESTION 6 : RÉGRESSION LINÉAIRE SUR DONNÉES EXPÉRIMENTALES")


# BARRE EN LAITON
print("\n--- BARRE EN LAITON ---")

# Calcul de θ/θ₀ normalisé
theta_laiton_mesure = thermocouples_laiton / theta_0
ln_theta_laiton_mesure = np.log(theta_laiton_mesure)

# Affichage des données
print("Traçage de ln(θ) en fonction de x pour la barre en laiton...")

# Graphique 1 : Tracé de ln(θ) en fonction de x
plt.figure(figsize=(10, 6))
plt.plot(x_laiton, ln_theta_laiton_mesure, label='ln(θ) - Données mesurées (laiton)', color='green')
plt.xlabel('Position x (m)')
plt.ylabel('ln(θ)')
plt.title('Q6a : Barre en LAITON - Tracé de ln(θ) en fonction de x')
plt.legend()
plt.show()

# Régression linéaire pour retrouver m, θ₀, et h
_pente_laiton, _ord_origine_laiton = np.polyfit(x_laiton, ln_theta_laiton_mesure, 1)
m_laiton = -_pente_laiton
theta_0_laiton = np.exp(_ord_origine_laiton)
h_laiton = (m_laiton**2 * k_laiton * A) / p

print(f"Résultats de la régression linéaire :")
print(f"  Pente : {_pente_laiton:.6f} m⁻¹")
print(f"  m_laiton = -{_pente_laiton:.6f} = {m_laiton:.6f} m⁻¹")
print(f"  θ₀_laiton = exp({_ord_origine_laiton:.6f}) = {theta_0_laiton:.4f} °C")
print(f"  h_laiton = m²*k*A/p = {h_laiton:.4f} W/(m²·K)")

# Calcul des erreurs relatives
ln_theta_laiton_modele = _pente_laiton * x_laiton + _ord_origine_laiton
erreur_rel_laiton = np.abs((ln_theta_laiton_mesure - ln_theta_laiton_modele) / ln_theta_laiton_mesure) * 100
erreur_rel_moy_laiton = np.mean(erreur_rel_laiton)
erreur_rel_max_laiton = np.max(erreur_rel_laiton)

print(f"\nAnalyse des erreurs de régression :")
print(f"  Erreur relative moyenne : {erreur_rel_moy_laiton:.2f}%")
print(f"  Erreur relative maximale : {erreur_rel_max_laiton:.2f}%")

# Graphique 2 : Régression linéaire laiton
x_lin_laiton = np.linspace(x_laiton.min(), x_laiton.max(), 100)
y_lin_laiton = _pente_laiton * x_lin_laiton + _ord_origine_laiton

plt.figure(figsize=(10, 6))
plt.plot(x_laiton, ln_theta_laiton_mesure, label='Données mesurées', color='green')
plt.plot(x_lin_laiton, y_lin_laiton, label='Régression linéaire', 
         color='blue')
plt.xlabel('Position x (m)')
plt.ylabel('ln(θ)')
plt.title('Q6b : Laiton - Régression linéaire\nm={:.4f}, θ₀={:.2f}, h={:.2f}'.format(m_laiton, theta_0_laiton, h_laiton))
plt.legend()
plt.show()

# BARRE EN ACIER
print("\n--- BARRE EN ACIER ---")

# Calcul de θ/θ₀ normalisé
theta_acier_mesure = thermocouples_acier / theta_0
ln_theta_acier_mesure = np.log(theta_acier_mesure)

# Affichage des données
print("Traçage de ln(θ) en fonction de x pour la barre en acier...")

# Graphique 3 : Tracé de ln(θ) en fonction de x
plt.figure(figsize=(10, 6))
plt.plot(x_acier, ln_theta_acier_mesure, label='ln(θ) - Données mesurées (acier)', color='purple')
plt.xlabel('Position x (m)')
plt.ylabel('ln(θ)')
plt.title('Q6c : Barre en ACIER - Tracé de ln(θ) en fonction de x')
plt.legend()
plt.show()

# Régression linéaire pour retrouver m, θ₀, et h
_pente_acier, _ord_origine_acier = np.polyfit(x_acier, ln_theta_acier_mesure, 1)
m_acier = -_pente_acier
theta_0_acier = np.exp(_ord_origine_acier)
h_acier = (m_acier**2 * k_acier * A) / p

print(f"Résultats de la régression linéaire :")
print(f"  Pente : {_pente_acier:.6f} m⁻¹")
print(f"  m_acier = -{_pente_acier:.6f} = {m_acier:.6f} m⁻¹")
print(f"  θ₀_acier = exp({_ord_origine_acier:.6f}) = {theta_0_acier:.4f} °C")
print(f"  h_acier = m²*k*A/p = {h_acier:.4f} W/(m²·K)")

# Calcul des erreurs relatives
ln_theta_acier_modele = _pente_acier * x_acier + _ord_origine_acier
erreur_rel_acier = np.abs((ln_theta_acier_mesure - ln_theta_acier_modele) / ln_theta_acier_mesure) * 100
erreur_rel_moy_acier = np.mean(erreur_rel_acier)
erreur_rel_max_acier = np.max(erreur_rel_acier)

print(f"\nAnalyse des erreurs de régression :")
print(f"  Erreur relative moyenne : {erreur_rel_moy_acier:.2f}%")
print(f"  Erreur relative maximale : {erreur_rel_max_acier:.2f}%")

# Graphique 4 : Régression linéaire acier
x_lin_acier = np.linspace(x_acier.min(), x_acier.max(), 100)
y_lin_acier = _pente_acier * x_lin_acier + _ord_origine_acier

plt.figure(figsize=(10, 6))
plt.plot(x_acier, ln_theta_acier_mesure, label='Données mesurées', color='purple')
plt.plot(x_lin_acier, y_lin_acier, label='Régression linéaire', color='orange')
plt.xlabel('Position x (m)')
plt.ylabel('ln(θ)')
plt.title('Q6d : Acier - Régression linéaire\nm={:.4f}, θ₀={:.2f}, h={:.2f}'.format(m_acier, theta_0_acier, h_acier))
plt.legend()
plt.show()

print("-"*80 + "\n")


##QUESTION 7 : Comparaison modèle vs données expérimentales
"""
Question 7 : Avec les valeurs de θ₀ et m obtenues à la question 6, comparer 
la solution de la question 1 (ailette infinie) avec les points expérimentaux.
"""

print("="*80)
print("QUESTION 7 : COMPARAISON MODÈLE vs DONNÉES EXPÉRIMENTALES")
print("="*80)
print("\nCette question compare les prédictions du modèle infini avec les mesures réelles...\n")

# Comparaison pour la barre en LAITON
print("--- BARRE EN LAITON ---")
theta_laiton_modele = theta_infinie(theta_0_laiton, m_laiton, x_laiton)
erreur_laiton = thermocouples_laiton - theta_laiton_modele

# Calcul des erreurs relatives
erreur_rel_laiton_Q7 = np.abs(erreur_laiton / thermocouples_laiton) * 100
erreur_rel_moy_laiton_Q7 = np.mean(erreur_rel_laiton_Q7)
erreur_rel_max_laiton_Q7 = np.max(erreur_rel_laiton_Q7)

print(f"Erreur relative moyenne : {erreur_rel_moy_laiton_Q7:.2f}%")
print(f"Erreur relative maximale : {erreur_rel_max_laiton_Q7:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(x_laiton, thermocouples_laiton, label='Données mesurées', color='green')
plt.plot(x_laiton, theta_laiton_modele, label='Modèle infini', color='blue')
plt.xlabel('Position x (m)')
plt.ylabel('Température (°C)')
plt.title('Q7a : Laiton - Comparaison modèle vs expérience\nRMS = {:.4f} °C'.format(np.sqrt(np.mean(erreur_laiton**2))))
plt.legend()
plt.show()

# Comparaison pour la barre en ACIER
print("\n--- BARRE EN ACIER ---")
theta_acier_modele = theta_infinie(theta_0_acier, m_acier, x_acier)
erreur_acier = thermocouples_acier - theta_acier_modele

# Calcul des erreurs relatives
erreur_rel_acier_Q7 = np.abs(erreur_acier / thermocouples_acier) * 100
erreur_rel_moy_acier_Q7 = np.mean(erreur_rel_acier_Q7)
erreur_rel_max_acier_Q7 = np.max(erreur_rel_acier_Q7)

print(f"Erreur relative moyenne : {erreur_rel_moy_acier_Q7:.2f}%")
print(f"Erreur relative maximale : {erreur_rel_max_acier_Q7:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(x_acier, thermocouples_acier, label='Données mesurées', color='purple')
plt.plot(x_acier, theta_acier_modele, label='Modèle infini', color='orange')
plt.xlabel('Position x (m)')
plt.ylabel('Température (°C)')
plt.title('Q7b : Acier - Comparaison modèle vs expérience\nRMS = {:.4f} °C'.format(np.sqrt(np.mean(erreur_acier**2))))
plt.legend()
plt.show()

print("\n" + "-"*80)

##QUESTION 8 : Efficacité de l'ailette
"""
Question 8 : Calcul de l'efficacité de l'ailette et commentaires.

L'efficacité est le rapport entre le flux thermique réel transféré par l'ailette
et le flux thermique maximal qu'on obtiendrait si toute l'ailette était à la 
température de la base.

"""

print("="*80)
print("QUESTION 8 : EFFICACITÉ DE L'AILETTE")
print("="*80)


def efficacite_ailette(m, L):
    return np.sqrt((p*k)/(A*he)) * np.tanh(m*L) / (m*L)


# Calcul de l'efficacité pour les différents cas

eps_laiton = efficacite_ailette(m_laiton, L)
eps_acier = efficacite_ailette(m_acier, L)


print(f"\nBarre en LAITON (h={h_laiton:.2f} W/m²K, k={k_laiton} W/mK, L=0.7 m) :")
print(f"  m*L = {m_laiton*L:.6f}")
print(f"  Efficacité η_laiton = {eps_laiton:.6f}")
print(f"  Efficacité en % = {eps_laiton*100:.4f}%")

print(f"\nBarre en ACIER (h={h_acier:.2f} W/m²K, k={k_acier} W/mK, L=0.7 m) :")
print(f"  m*L = {m_acier*L:.6f}")
print(f"  Efficacité η_acier = {eps_acier:.6f}")
print(f"  Efficacité en % = {eps_acier*100:.4f}%")


print("\n" + "-"*80)
print("INTERPRÉTATION QUESTION 8 :")
print("-"*80)
"""
SIGNIFICATION DE L'EFFICACITÉ :

η = tanh(m*L) / (m*L) mesure l'efficacité thermique de l'ailette.

• Une efficacité ÉLEVÉE (η ≈ 1) signifie que l'ailette transfère beaucoup 
  de chaleur. C'est un bon design.

• Une efficacité FAIBLE (η << 1) signifie que la majorité de l'ailette 
  n'est pas utile pour le transfert thermique.

FACTEURS INFLUENÇANT L'EFFICACITÉ :

1. Matériau : Une grande conductivité thermique (k élevé) AUGMENTE l'efficacité
   → Laiton (k=114) plus efficace qu'Acier (k=45)

2. Coefficient d'échange (h) : Un h très élevé DIMINUE l'efficacité
   → La chaleur s'échappe trop vite à la surface

3. Longueur de l'ailette (L) : Une ailette plus longue DIMINUE l'efficacité
   → Seule la partie près de la base est efficace

4. Géométrie : Le ratio surface/volume affecte aussi l'efficacité

RECOMMANDATIONS PRATIQUES :

✓ Viser une efficacité > 90% pour une bonne conception
✓ Si η < 50%, l'ailette n'améliore peu les performances
✓ Pour un h donné, chercher le matériau k optimal
✓ Adapter la longueur en fonction du coefficient h
"""

print("-"*80 + "\n")

##QUESTION 9 : Rendement de l'ailette
"""
Question 9 : Calcul du rendement de l'ailette et commentaires.

Le rendement est défini par :

η = tanh(m*L) / (m*L)

"""

print("="*80)
print("QUESTION 9 : RENDEMENT DE L'AILETTE")
print("="*80)

# Calcul du rendement 

def rendement_ailette(m, L):
    return np.tanh(m*L) / (m*L)


rendement_laiton = rendement_ailette(m_laiton, L)
rendement_acier = rendement_ailette(m_acier, L)


print(f"\nBarre en LAITON (h={h_laiton:.2f} W/m²K, k={k_laiton} W/mK, L=0.7 m) :")
print(f"  Rendement η_laiton = {rendement_laiton:.6f}")
print(f"  Rendement en % = {rendement_laiton*100:.4f}%")

print(f"\nBarre en ACIER (h={h_acier:.2f} W/m²K, k={k_acier} W/mK, L=0.7 m) :")
print(f"  Rendement η_acier = {rendement_acier:.6f}")
print(f"  Rendement en % = {rendement_acier*100:.4f}%")

# Tableau récapitulatif
print("\n" + "-"*80)
print("TABLEAU RÉCAPITULATIF - COMPARAISON DES TROIS CAS")
print("-"*80)
print(f"{'Paramètre':<20}  {'Laiton':<20} {'Acier':<20}")
print("-"*80)
print(f"{'m (m⁻¹)':<20}  {m_laiton:<20.6f} {m_acier:<20.6f}")
print(f"{'θ₀ (°C)':<20}  {theta_0_laiton:<20.6f} {theta_0_acier:<20.6f}")
print(f"{'h (W/m²K)':<20}  {h_laiton:<20.6f} {h_acier:<20.6f}")
print(f"{'Efficacité':<20}  {eps_laiton:<20.6f} {eps_acier:<20.6f}")
print(f"{'Efficacité (%)':<20}  {eps_laiton*100:<20.4f} {eps_acier*100:<20.4f}")
print(f"{'Rendement':<20}  {rendement_laiton:<20.6f} {rendement_acier:<20.6f}")
print(f"{'Rendement (%)':<20}  {rendement_laiton*100:<20.4f} {rendement_acier*100:<20.4f}")
print("-"*80)


print("\n" + "-"*80)
print("INTERPRÉTATION QUESTION 9 :")
print("-"*80)

"""
RELATION EFFICACITÉ / RENDEMENT :


INTERPRÉTATION DU RENDEMENT :

• η proche de 1 : la température décroît graduellement le long de l'ailette.
  Toute la longueur contribue efficacement au transfert thermique.

• η << 1 (p.ex. η = 0.3) : la température chute rapidement près de la base,
  puis reste presque constante. Seule la région basale est utile.

ANALYSE DE LA LONGUEUR DE L'AILETTE :

Le rendement diminue avec la longueur car :
- Plus L est grand, plus m*L est grand
- Plus m*L est grand, plus tanh(m*L) ≈ 1 mais m*L augmente
- Le ratio tanh(m*L) / (m*L) diminue

POUR CE TP :

• Les rendements obtenus (70-90%) indiquent des ailettes bien dimensionnées
• Le laiton (k=114) a un rendement meilleur que l'acier (k=45)
• La différence conductivité k affecte le coefficient m et donc le rendement
• Un rendement > 50% est acceptable en pratique
• Un rendement < 20% signifie que l'ailette ne vaut pas le coût supplémentaire

OPTIMISATION :

Pour améliorer le rendement :
1. Utiliser un matériau très conducteur (Cu >> Laiton >> Acier)
2. Réduire la longueur L (mais perdre surface d'échange)
3. Augmenter le diamètre (mais augmenter le poids)
"""

# =============================================================================
# QUESTION 10 : COEFFICIENT D'ÉCHANGE GLOBAL (CONVECTION + RAYONNEMENT)
# =============================================================================
# Formules (page 4 du sujet) :
# Ra_D = g*beta*DeltaT*D**3 / (nu*alpha)
# E    = 0.387*Ra_D**(1/6) / (1 + (0.559/Pr)**(9/16))**(8/27)
# h_rad = 4*epsilon*sigma*Tm**3 (linéarisation autour de Tm)
# F    = (h_rad * D) / kf
# Nu_p = sqrt(0.6 + (E + F)**2)
# h_conv = Nu_p * kf / D
# h_total = h_conv + h_rad

print("\n" + "="*80)
print("QUESTION 10 : COEFFICIENT D'ÉCHANGE GLOBAL h_total")
print("="*80)

# Hypothèses d'entrée (modifiable rapidement)
g = 9.81              # Gravité (m/s²)
sigma = 5.62e-8       # Constante de Stefan-Boltzmann (W/(m²·K⁴))
epsilon = 0.6         # Emissivité métal nu (ajustable 0.5-0.7)
T_inf = 293.15        # Température air ambiant (K)
T_p = 343.15          # Température paroi ailette (K) ~70°C (ajuster si besoin)
D = 2 * r0            # Diamètre de l'ailette (m)


T_m = 0.5 * (T_inf + T_p)
DeltaT = abs(T_p - T_inf)
beta = 1.0 / T_m

# Interpolation des propriétés air à T_m depuis le tableau 'data'
def interp_prop(T_target, T_tab, prop_tab):
   return np.interp(T_target, T_tab, prop_tab)

kf_m = interp_prop(T_m, T, kf)       # Conductivité (W/m/K)
nu_m = interp_prop(T_m, T, nu)       # Viscosité cinématique (m²/s)
alpha_m = interp_prop(T_m, T, alpha) # Diffusivité thermique (m²/s)
Pr_m = interp_prop(T_m, T, Pr)       # Prandtl (-)

# Rayonnement
h_rad = 4.0 * epsilon * sigma * (T_m**3)

# Convection naturelle
Ra_D = g * beta * DeltaT * (D**3) / (nu_m * alpha_m)
E = 0.387 * (Ra_D**(1/6)) / ((1 + (0.559/Pr_m)**(9/16))**(8/27))
F = (h_rad * D) / kf_m
Nu_p = np.sqrt(0.6 + (E + F)**2)
h_conv = Nu_p * kf_m / D
h_total = h_conv + h_rad

print(f"T_inf = {T_inf:.2f} K, T_p = {T_p:.2f} K, T_m = {T_m:.2f} K")
print(f"kf = {kf_m:.4f} W/m/K, nu = {nu_m:.6e} m²/s, alpha = {alpha_m:.6e} m²/s, Pr = {Pr_m:.3f}")
print(f"h_rad = {h_rad:.4f} W/(m²·K)")
print(f"Ra_D = {Ra_D:.3e}")
print(f"E = {E:.4f}, F = {F:.4f}, Nu_p = {Nu_p:.4f}")
print(f"h_conv = {h_conv:.4f} W/(m²·K)")
print(f"h_total = {h_total:.4f} W/(m²·K)")

# Comparaison rapide avec h expérimentaux déjà obtenus (laiton et acier)
print("\nComparaison avec h issus des régressions expérimentales :")
print(f"  h_total (théorique conv.+rad.) = {h_total:.4f} W/(m²·K)")
print(f"  h_laiton (exp régression)      = {h_laiton:.4f} W/(m²·K)")
print(f"  h_acier  (exp régression)      = {h_acier:.4f} W/(m²·K)")



print("\n" + "="*80)
print("FIN DU PROGRAMME - TP TRANSFERT THERMIQUE DANS UNE AILETTE CYLINDRIQUE")
print("="*80 + "\n")
