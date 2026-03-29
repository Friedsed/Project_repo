import numpy as np
import matplotlib.pyplot as plt


# ======================================================================================
# Données géométriques
# ======================================================================================
A_t = 0.157  # m²
D_ext_f = 3.05e-2  # m
D_int_f = 2.69e-2  # m
D_c = 2.37e-2      # m

Surf = np.pi * (D_int_f**2 - D_c**2) / 4      # surface annulaire eau froide
Pm_f = np.pi * (D_ext_f + D_c)                # périmètre mouillé
Dh = 4 * Surf / Pm_f                          # diamètre hydraulique


# ======================================================================================
# Données fluides (eau) – en fonction de T (°C)
# ======================================================================================
T_tab   = np.array([0, 10, 20, 40, 60])
Cp_tab  = np.array([4218, 4192, 4184, 4184, 4184])        # J/kg/K
nu_tab  = np.array([1.79e-06, 1.30e-06, 1.01e-06, 6.60e-07, 4.77e-07])  # m²/s
k_tab   = np.array([0.552, 0.586, 0.597, 0.628, 0.651])   # W/m/K
Pr_tab  = np.array([13.5, 9.3, 7.0, 4.34, 3.0])


# ======================================================================================
# Données expérimentales (°C, l/h)
# ======================================================================================
mf_lph = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])  # fluide froid
mc_lph = np.array([500] * 9)                                      # fluide chaud (constant)

Tf_out_C = np.array([22.1, 21.3, 21.6, 21.9, 22.7, 23.4, 24.4, 25.6, 29.0])
Tf_in_C  = np.array([18.4, 17.2, 17.0, 16.9, 16.9, 17.0, 17.1, 17.1, 17.2])

Tc_out_C = np.array([33.2, 33.1, 33.3, 33.5, 33.9, 34.3, 35.2, 36.0, 37.0])
Tc_in_C  = np.array([40.0, 39.8, 39.9, 39.9, 39.8, 39.7, 39.9, 39.8, 39.9])


# ======================================================================================
# Calculs principaux (Questions 1 à 5)
# ======================================================================================

# Conversion l/h -> kg/s avec ρ=1 kg/L
mf_kg_s = mf_lph / 3600
mc_kg_s = mc_lph / 3600

# Températures moyennes en °C
Tf_moy_C = 0.5 * (Tf_out_C + Tf_in_C)

# Propriétés interpolées (eau)
Cp_int = np.interp(Tf_moy_C, T_tab, Cp_tab)
nu = np.interp(Tf_moy_C, T_tab, nu_tab)
k = np.interp(Tf_moy_C, T_tab, k_tab)
Pr = np.interp(Tf_moy_C, T_tab, Pr_tab)

# Question 1 : flux reçu par le fluide froid
Q = mf_kg_s * Cp_int * (Tf_out_C - Tf_in_C)

# Question 2 : ΔT (en °C)
delta_Tin = Tc_in_C - Tf_in_C
delta_Tout = Tc_out_C - Tf_out_C
delta_Tln = (delta_Tin - delta_Tout) / np.log(delta_Tin / delta_Tout)
delta_Tmoy = 0.5 * (delta_Tin + delta_Tout)

# Question 3 : hg global
hg = Q / (A_t * delta_Tln)

# Question 4 : efficacité
Mf_point = mf_kg_s * Cp_int
Mc_point = mc_kg_s * Cp_int
Mmin_point = np.minimum(Mf_point, Mc_point)
Mmax_point = np.maximum(Mf_point, Mc_point)

eff_exp = Q / (Mmin_point * (Tc_in_C - Tf_in_C))
NTU = hg * A_t / Mmin_point
C_star = Mmin_point / Mmax_point
eff_th = (1 - np.exp(-NTU * (1 - C_star))) / (1 - C_star * np.exp(-NTU * (1 - C_star)))

# Question 5 : Nu/Re + corrélation
rho = 1000  # kg/m³ (densité de l'eau)
U = mf_kg_s / (rho * Surf)  # vitesse en m/s
Re = U * Dh / nu
Nu_Re = hg * Dh / k
log_Nu_Pr = np.log(Nu_Re / (Pr ** (1 / 3)))

# Ajustement linéaire : ln(Nu/Pr^(1/3)) = a * ln(Re) + b
coeffs = np.polyfit(np.log(Re), log_Nu_Pr, 1)
log_Re = np.log(Re)
pred = np.polyval(coeffs, log_Re)
ss_res = np.sum((log_Nu_Pr - pred) ** 2)
ss_tot = np.sum((log_Nu_Pr - log_Nu_Pr.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
log_Re_fit = np.linspace(log_Re.min(), log_Re.max(), 200)
log_Nu_Pr_fit = np.polyval(coeffs, log_Re_fit)


# Question 6 : Calculs du coefficeient d'écahnge convectif h_global

# on calcule efficatité théorique en fonction de h_global

# ======================================================================================
# Tracés
# ======================================================================================
mf_axis = mf_lph

plt.figure(1, figsize=(10, 6))
plt.plot(mf_axis, Q, marker="o", color='#2E86AB', linewidth=2, markersize=8)
plt.xlabel("Débit fluide froid (l/h)", fontsize=12, fontweight='bold')
plt.ylabel("Flux de chaleur Q (W)", fontsize=12, fontweight='bold')
plt.title("Flux de chaleur reçu par le fluide froid", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.figure(2, figsize=(10, 6))
plt.plot(mf_axis, delta_Tin, "o-", label="ΔTin", linewidth=2, markersize=7)
plt.plot(mf_axis, delta_Tout, "s-", label="ΔTout", linewidth=2, markersize=7)
plt.plot(mf_axis, delta_Tln, "^-", label="ΔTln (LMTD)", linewidth=2.5, markersize=7)
plt.plot(mf_axis, delta_Tmoy, "d-", label="ΔTmoy", linewidth=2, markersize=7)
plt.xlabel("Débit fluide froid (l/h)", fontsize=12, fontweight='bold')
plt.ylabel("Différence de température (°C)", fontsize=12, fontweight='bold')
plt.title("Différences de température en fonction du débit", fontsize=14, fontweight='bold')
plt.legend(fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.figure(3, figsize=(10, 6))
plt.plot(mf_axis, hg, marker="o", color='#A23B72', linewidth=2, markersize=8)
plt.xlabel("Débit fluide froid (l/h)", fontsize=12, fontweight='bold')
plt.ylabel("Coefficient global hg (W/m²·K)", fontsize=12, fontweight='bold')
plt.title("Coefficient d'échange global", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.figure(4, figsize=(10, 6))
plt.plot(NTU, eff_exp, "o-", color="#18A558", linewidth=2.5, markersize=8, label="ε expérimentale")
plt.plot(NTU, eff_th, "s--", color="#F77F00", linewidth=2.5, markersize=8, label="ε théorique (contre-courant)")
plt.xlabel("NTU", fontsize=12, fontweight='bold')
plt.ylabel("Efficacité ε", fontsize=12, fontweight='bold')
plt.title("Efficacité de l'échangeur : comparaison mesures/théorie", fontsize=14, fontweight='bold')
plt.legend(fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.figure(5, figsize=(10, 6))
plt.plot(Re, Nu_Re, "o-", color='#D62828', linewidth=2, markersize=8, label="Nu mesuré")
plt.xlabel("Nombre de Reynolds (Re)", fontsize=12, fontweight='bold')
plt.ylabel("Nombre de Nusselt (Nu)", fontsize=12, fontweight='bold')
plt.title("Nombre de Nusselt en fonction de Reynolds", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Nu vs Re avec superposition de la corrélation
a, b = coeffs
Nu_fit = np.exp(b) * (Re ** a) * (Pr ** (1 / 3))
plt.figure(6, figsize=(10, 6))
plt.plot(Re, Nu_Re, "o", color='#003049', markersize=10, label="Nu mesuré")
plt.plot(Re, Nu_fit, "--", color='#F77F00', linewidth=2.5, label="Nu corrélé")
plt.xlabel("Nombre de Reynolds (Re)", fontsize=12, fontweight='bold')
plt.ylabel("Nombre de Nusselt (Nu)", fontsize=12, fontweight='bold')
plt.title("Corrélation Nu = f(Re, Pr)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, framealpha=0.9)
plt.tight_layout()

plt.figure(7, figsize=(10, 6))
plt.plot(np.log(Re), log_Nu_Pr, "o", color='#6A4C93', markersize=10, label="Données expérimentales")
plt.plot(log_Re_fit, log_Nu_Pr_fit, "-", color='#F72585', linewidth=2.5, label=f"Ajustement : a={a:.3f}, b={b:.3f}")
plt.xlabel("ln(Re)", fontsize=12, fontweight='bold')
plt.ylabel("ln(Nu/Pr^(1/3))", fontsize=12, fontweight='bold')
plt.title("Détermination des coefficients a et b (régression log-log)", fontsize=14, fontweight='bold')
plt.legend(fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()


# ======================================================================================
# Affichage des résultats
# ======================================================================================
print(f"Q (bilan froid) [W] : {Q}")
print(f"Correlation ajustée : Nu/Pr^(1/3) = exp({b:.3f}) * Re^{a:.3f} (R2={r2:.3f})")
print(f"a = {a:.4f}")
print(f"b = {b:.4f}")

plt.show()