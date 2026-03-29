import numpy as np
import matplotlib.pyplot as plt

'''
Reponse Q1 (objectif et adequation aux donnees)
Les mesures fournies (tirants en plusieurs positions pour trois pentes et plusieurs Qs) couvrent bien le cas d'ecoulement uniforme demande:
tirant quasi constant le long du canal et variations faibles sauf aux Qs les plus hauts. 
Ces donnees permettent de tester les lois de frottement Chezy, Manning et Darcy-Weisbach comme demande dans le TP.

Reponse Q2 (fonctionnement du dispositif)
- Pompe qui alimente la cuve amont; Q impose par la frequence pompe et la vanne de sortie.
- Sortie dans la cuve de recuperation; niveau dans le canal ajuste par plaques plexiglass en sortie.
- Mesure du tirant par limnimetre/pointe ou camera; Q lu sur le controleur.

Reponse Q3 (mise en route, reglages pente et Q)
- Reglage pente: mettre en eau, plaque en sortie, arreter pompe, mesurer tirant pour verifier la pente (ici on traite 0.50%, 0.73%, 0.95%).
- Mise en route: pompe a ~60% puis ajuster frequence et vanne pour atteindre les Qs cibles (ici 2.3-13.7 m3/h selon la pente). Les tableaux ci-dessous reprennent ces Qs convertis en m3/s.
'''


# ---------------------------------------------------------------------
# Constantes physiques et geometrie du canal (Question 4)
"""
Organisation demandee :
1) Donnees experimentales
2) Question 5.1 (uniformite H)
3) Question 5.2 (tau_moy)
4) Question 5.3 (vitesse Qante et Re)
5) Question 5.4 (lois de frottement + graphes)
"""
g = 9.81            # m/s2
rho_eau = 1000.0    # kg/m3
B = 0.099  # m (9.9 cm)
nu_eau = 1.01e-6    # m2/s


# ---------------------------------------------------------------------
# Donnees experimentales (tirants et Qs pour chaque pente)
# Profondeur totale du canal: 185 mm (mesure depuis le haut de la bordure jusqu'au fond)
# Les hauteurs mesurees sont depuis le haut de la bordure, donc tirant reel = 185 - hauteur mesuree
# ---------
pente_pct_238 = 0.238
pente_pct_050 = 0.50
pente_pct_073 = 0.73
pente_pct_095 = 0.95

profondeur_canal_mm = 185.0  # Profondeur totale du canal en mm

Q_m3h_073 = [2.5, 5.0, 7.5, 10.0, 12.5, 13.6]
# Mesures brutes depuis le haut de la bordure (en mm)
mesures_brutes_073 = np.array([
    [140.0, 142.0, 149.0, 155.0, 162.0, 169.5],
    [137.5, 140.0, 148.0, 154.5, 162.0, 169.5],
    [149.0, 143.0, 149.0, 154.5, 161.0, 169.5],
], dtype=float)
# Conversion en tirants d'eau reels (en mm)
tirant_mm_073 = profondeur_canal_mm - mesures_brutes_073

Q_m3h_050 = [2.3, 5.3, 7.6, 10.1, 12.6, 13.7]
mesures_brutes_050 = np.array([
    [133.0, 136.0, 143.0, 150.0, 156.0, 167.0],
    [131.0, 135.0, 140.0, 149.0, 156.0, 167.0],
    [135.5, 139.0, 149.5, 151.0, 156.0, 167.0],
], dtype=float)
tirant_mm_050 = profondeur_canal_mm - mesures_brutes_050

Q_m3h_095 = [2.5, 5.1, 7.4, 10.0, 12.4, 13.6]
mesures_brutes_095 = np.array([
    [142.5, 145.5, 150.0, 156.5, 163.0, 170.0],
    [141.5, 143.5, 149.0, 156.0, 162.0, 170.0],
    [141.0, 144.0, 150.0, 156.0, 163.0, 170.0],
], dtype=float)
tirant_mm_095 = profondeur_canal_mm - mesures_brutes_095

Q_m3h_238 = [2.5, 5.0, 7.3, 10.2, 12.5, 13.8]
mesures_brutes_238 = np.array([
    [164.0, 154.0, 146.0, 134.0, 128.0, 128.0],
    [164.0, 154.0, 146.0, 139.0, 133.0, 128.0],
    [163.5, 153.5, 147.0, 138.0, 133.0, 129.5],
], dtype=float)
tirant_mm_238 = profondeur_canal_mm - mesures_brutes_238


# ---------------------------------------------------------------------
# Conversions et grandeurs derivees (une fois pour toutes)
# ---------------------------------------------------------------------
pente_050 = pente_pct_050 / 100.0
pente_073 = pente_pct_073 / 100.0
pente_095 = pente_pct_095 / 100.0
pente_238 = pente_pct_238 / 100.0


Q_073 = np.array(Q_m3h_073, dtype=float) / 3600.0
tirant_m_073 = tirant_mm_073 / 1000.0
tirant_moy_073 = tirant_m_073.mean(axis=0)
variation_h_073 = np.ptp(tirant_m_073, axis=0)
non_uniformite_073 = variation_h_073 / tirant_moy_073
Rh_073 = B * tirant_moy_073 / (B + 2.0 * tirant_moy_073)
U_deb_073 = Q_073 / (B * tirant_moy_073)
tau_moy_073 = rho_eau * g * pente_073 * B * tirant_moy_073 / (B + 2.0 * tirant_moy_073)
Re_073 = U_deb_073 * (4.0 * Rh_073) / nu_eau
c_chezy_073 = U_deb_073 / np.sqrt(Rh_073 * pente_073)
n_manning_073 = Rh_073 ** (2.0 / 3.0) * np.sqrt(pente_073) / U_deb_073
f_darcy_073 = 8.0 * tau_moy_073 / (rho_eau * U_deb_073 ** 2)

Q_050 = np.array(Q_m3h_050, dtype=float) / 3600.0
tirant_m_050 = tirant_mm_050 / 1000.0
tirant_moy_050 = tirant_m_050.mean(axis=0)
variation_h_050 = np.ptp(tirant_m_050, axis=0)
non_uniformite_050 = variation_h_050 / tirant_moy_050
Rh_050 = B * tirant_moy_050 / (B + 2.0 * tirant_moy_050)
U_deb_050 = Q_050 / (B * tirant_moy_050)
tau_moy_050 = rho_eau * g * pente_050 * B * tirant_moy_050 / (B + 2.0 * tirant_moy_050)
Re_050 = U_deb_050 * (4.0 * Rh_050) / nu_eau
c_chezy_050 = U_deb_050 / np.sqrt(Rh_050 * pente_050)
n_manning_050 = Rh_050 ** (2.0 / 3.0) * np.sqrt(pente_050) / U_deb_050
f_darcy_050 = 8.0 * tau_moy_050 / (rho_eau * U_deb_050 ** 2)

Q_095 = np.array(Q_m3h_095, dtype=float) / 3600.0
tirant_m_095 = tirant_mm_095 / 1000.0
tirant_moy_095 = tirant_m_095.mean(axis=0)
variation_h_095 = np.ptp(tirant_m_095, axis=0)
non_uniformite_095 = variation_h_095 / tirant_moy_095
Rh_095 = B * tirant_moy_095 / (B + 2.0 * tirant_moy_095)
U_deb_095 = Q_095 / (B * tirant_moy_095)
tau_moy_095 = rho_eau * g * pente_095 * B * tirant_moy_095 / (B + 2.0 * tirant_moy_095)
Re_095 = U_deb_095 * (4.0 * Rh_095) / nu_eau
c_chezy_095 = U_deb_095 / np.sqrt(Rh_095 * pente_095)
n_manning_095 = Rh_095 ** (2.0 / 3.0) * np.sqrt(pente_095) / U_deb_095
f_darcy_095 = 8.0 * tau_moy_095 / (rho_eau * U_deb_095 ** 2)

Q_238 = np.array(Q_m3h_238, dtype=float) / 3600.0
tirant_m_238 = tirant_mm_238 / 1000.0
tirant_moy_238 = tirant_m_238.mean(axis=0)
variation_h_238 = np.ptp(tirant_m_238, axis=0)
non_uniformite_238 = variation_h_238 / tirant_moy_238
Rh_238 = B * tirant_moy_238 / (B + 2.0 * tirant_moy_238)
U_deb_238 = Q_238 / (B * tirant_moy_238)
tau_moy_238 = rho_eau * g * pente_238 * B * tirant_moy_238 / (B + 2.0 * tirant_moy_238)
Re_238 = U_deb_238 * (4.0 * Rh_238) / nu_eau
c_chezy_238 = U_deb_238 / np.sqrt(Rh_238 * pente_238)
n_manning_238 = Rh_238 ** (2.0 / 3.0) * np.sqrt(pente_238) / U_deb_238
f_darcy_238 = 8.0 * tau_moy_238 / (rho_eau * U_deb_238 ** 2)

# ---------------------------------------------------------------------
# Question 5.1 : uniformite du tirant H
# ---------------------------------------------------------------------
print("=== Question 5.1 : Uniformite ===")
print("Pente 0.238% :", np.round(non_uniformite_238, 3))
print("Pente 0.50% :", np.round(non_uniformite_050, 3))
print("Pente 0.73% :", np.round(non_uniformite_073, 3))
print("Pente 0.95% :", np.round(non_uniformite_095, 3))
print()

# ---------------------------------------------------------------------
# Question 5.2 : contrainte pari-etale moyenne tau_moy
# ---------------------------------------------------------------------
print("=== Question 5.2 : Tau moyenne (Pa) ===")
print("Pente 0.238% :", np.round(tau_moy_238, 2))
print("Pente 0.50% :", np.round(tau_moy_050, 2))
print("Pente 0.73% :", np.round(tau_moy_073, 2))
print("Pente 0.95% :", np.round(tau_moy_095, 2))
print()

# ---------------------------------------------------------------------
# Question 5.3 : vitesse Qante et Re
# ---------------------------------------------------------------------
print("=== Question 5.3 : Vitesse Qante (m/s) et Re ===")
print("Pente 0.238% : Vitesse", np.round(U_deb_238, 3), "Re", np.round(Re_238, 0))
print("Pente 0.50% : Vitesse", np.round(U_deb_050, 3), "Re", np.round(Re_050, 0))
print("Pente 0.73% : Vitesse", np.round(U_deb_073, 3), "Re", np.round(Re_073, 0))
print("Pente 0.95% : Vitesse", np.round(U_deb_095, 3), "Re", np.round(Re_095, 0))
print()

# ---------------------------------------------------------------------
# Question 5.4 : lois de frottement (Chezy, Manning, Darcy)
# ---------------------------------------------------------------------
print("=== Question 5.4 : Chezy / Manning / Darcy ===")
print("Pente 0.238% : C", np.round(c_chezy_238, 2), "n", np.round(n_manning_238, 4), "f_D", np.round(f_darcy_238, 4))
print("Pente 0.50% : C", np.round(c_chezy_050, 2), "n", np.round(n_manning_050, 4), "f_D", np.round(f_darcy_050, 4))
print("Pente 0.73% : C", np.round(c_chezy_073, 2), "n", np.round(n_manning_073, 4), "f_D", np.round(f_darcy_073, 4))
print("Pente 0.95% : C", np.round(c_chezy_095, 2), "n", np.round(n_manning_095, 4), "f_D", np.round(f_darcy_095, 4))
print()

plt.figure(figsize=(10, 8))

# Chezy
plt.subplot(3, 1, 1)
plt.plot(U_deb_238, c_chezy_238, marker="o", label="S = 0.238%")
plt.plot(U_deb_050, c_chezy_050, marker="o", label="S = 0.50%")
plt.plot(U_deb_073, c_chezy_073, marker="o", label="S = 0.73%")
plt.plot(U_deb_095, c_chezy_095, marker="o", label="S = 0.95%")
plt.ylabel("C (Chezy)")
plt.grid(True)
plt.legend()
plt.title("Lois de frottement vs vitesse Qante")

# Manning
plt.subplot(3, 1, 2)
plt.plot(U_deb_238, n_manning_238, marker="o", label="S = 0.238%")
plt.plot(U_deb_050, n_manning_050, marker="o", label="S = 0.50%")
plt.plot(U_deb_073, n_manning_073, marker="o", label="S = 0.73%")
plt.plot(U_deb_095, n_manning_095, marker="o", label="S = 0.95%")
plt.ylabel("n (Manning)")
plt.grid(True)

# Darcy-Moody
plt.subplot(3, 1, 3)
plt.loglog(Re_238, f_darcy_238, marker="o", label="S = 0.238%")
plt.loglog(Re_050, f_darcy_050, marker="o", label="S = 0.50%")
plt.loglog(Re_073, f_darcy_073, marker="o", label="S = 0.73%")
plt.loglog(Re_095, f_darcy_095, marker="o", label="S = 0.95%")
plt.xlabel("Re")
plt.ylabel("f_D (Darcy-Weisbach)")
plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Conclusion sur les lois de frottement
# ---------------------------------------------------------------------
# - Chézy : très sensible au Re dans nos mesures (Re 7k-35k). La constance
#   attendue en turbulence rugueuse n'est pas atteinte ; variation de C cohérente
#   avec un régime de transition (C augmente avec Re).
# - Manning : les valeurs restent bien au-dessus des 0.009-0.011 d'un canal lisse,
#   ce qui confirme que l'écoulement n'est pas dans le régime turbulent pleinement
#   développé ; n décroît avec Re, signature du régime transitoire.
# - Darcy-Weisbach : f_D décroit fortement avec Re, ce qui est normal en phase
#   laminaire → transition → turbulent. Les valeurs restent au-dessus de 0.025
#   pour les plus petits Re, puis se rapprochent des plages classiques en montant
#   le débit.
# En résumé : les écarts aux lois « constantes » proviennent du régime de transition
# (Re encore trop faible pour une turbulence établie). Les tendances observées sont
# physiquement cohérentes avec la plage de Re mesurée.