import numpy as np
import math  # Pour la fonction log (ln)
import matplotlib.pyplot as plt # Pour le tracé (Tâche 5)

# Caracteristiques étudiants 
A_t = 0.157 # m**2
D_ext_f = 3.05e-2 # m
D_int_f = 2.69e-2
D_c = 2.37e-2 
"""Sigma = 
Pm = 
Dh = 
"""
import numpy as np

# Données fluides  (T_tab en °C)
T_tab = np.array([0, 10, 20, 40, 60])
Cp_tab = np.array([4218, 4192, 4184, 4184, 4184])    
nu_tab = np.array([1.79e-06, 1.30e-06, 1.01e-06, 6.60e-07, 4.77e-07])
k_tab  = np.array([0.552, 0.586, 0.597, 0.628, 0.651])
Pr_tab = np.array([13.5, 9.3, 7, 4.34, 3])

mf_point = np.array([100,200,300,400,500,600,700,800,900])*(0.001/3600)

Tf_out = np.array([22.1, 21.3, 21.6, 21.9, 22.7, 23.4, 24.4, 25.6, 29.0]) + 273.15
Tf_in  = np.array([18.4, 17.2, 17.0, 16.9, 16.9, 17.0, 17.1, 17.1, 17.2]) + 273.15

Tc_out = np.array([33.2, 33.1, 33.3, 33.5, 33.9, 34.3, 35.2, 36.0, 37.0]) + 273.15
Tc_in  = np.array([40.0, 39.8, 39.9, 39.9, 39.8, 39.7, 39.9, 39.8, 39.9]) + 273.15

mc_point = np.array([500,500,500,500,500,500,500,500,500])*(0.001/3600)  # même débit chaud partout

# Température moyenne du fluide froid (K)
Tf_moy = (Tf_out + Tf_in) / 2

# CORRECTION : interpolation avec Tf_moy en °C
Cp_int = np.interp(Tf_moy - 273.15, T_tab, Cp_tab)  # J/kg/K

# Débits « capacitifs » Mf, Mc (ici encore en (l/h)*J/kg/K, tu convertiras ensuite en W/K si besoin)
Mf_point = mf_point * Cp_int
Mc_point = mc_point * Cp_int

# CORRECTION : C_min point à point
Mmin_point = np.minimum(Mf_point, Mc_point)




print("Mf_point",Mf_point)
print("Mc_point",Mc_point)
print("Mmin_point",Mmin_point)






















###################################################################################################################
############################################################################################################

"""
=============================================================================
TP - TRANSFERT THERMIQUE DANS UNE AILETTE CYLINDRIQUE
=============================================================================
Programme Python pour l'étude du transfert thermique portant sur l'échange de chaleur.


Auteur: AUTHIE WOLI CORRE
Date: Décembre 2025
=============================================================================
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate

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
T_tab   = np.array([0,   10,    20,    40,    60])
Cp_tab  = np.array([4218,4192, 4184,  4184,  4184])        # J/kg/K
nu_tab  = np.array([1.79e-06,1.30e-06,1.01e-06,6.60e-07,4.77e-07])  # m²/s
k_tab   = np.array([0.552,0.586,0.597,0.628,0.651])        # W/m/K
Pr_tab  = np.array([13.5, 9.3, 7.0, 4.34, 3.0])

# ======================================================================================
# Données expérimentales
# ======================================================================================

# Débit massique fluide froid (l/h -> kg/s en supposant ρ ≈ 1000 kg/m³)
#mf      = np.array([900,800,700,600,500,400,300,200,100])     # l/h (ne sert plus pour les calculs)
mf_point = np.array([100,200,300,400,500,600,700,800,900]) / 3600  # kg/s

# Températures (°C -> K)
Tf_out = np.array([22.1, 21.3, 21.6, 21.9, 22.7, 23.4, 24.4, 25.6, 29.0]) + 273.15
Tf_in  = np.array([18.4, 17.2, 17.0, 16.9, 16.9, 17.0, 17.1, 17.1, 17.2]) + 273.15

Tc_out = np.array([33.2, 33.1, 33.3, 33.5, 33.9, 34.3, 35.2, 36.0, 37.0]) + 273.15
Tc_in  = np.array([40.0, 39.8, 39.9, 39.9, 39.8, 39.7, 39.9, 39.8, 39.9]) + 273.15

# Débit massique fluide chaud (500 l/h -> kg/s, constant)
mc_point = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500]) /3600

# ======================================================================================
# Propriétés interpolées au fluide froid (en fonction de Tf_moy)
# ======================================================================================
Tf_moy = (Tf_out + Tf_in) / 2  # K

Cp_int = np.interp(Tf_moy - 273.15, T_tab, Cp_tab)       # J/kg/K
nu     = np.interp(Tf_moy - 273.15, T_tab, nu_tab)       # m²/s
k      = np.interp(Tf_moy - 273.15, T_tab, k_tab)        # W/m/K
Pr     = np.interp(Tf_moy - 273.15, T_tab, Pr_tab)

# Vitesse moyenne dans l’anneau
U = mf_point / Surf   # kg/s / m² ≈ m/s si ρ ≈ 1000 kg/m³

# Débits capacitifs (W/K)
Mf_point = mf_point * Cp_int        # fluide froid
Mc_point = mc_point * Cp_int        # fluide chaud (≈ Cp eau)

Mmin_point = np.minimum(Mf_point, Mc_point)
Mmax_point = np.maximum(Mf_point, Mc_point)

# ======================================================================================
# Question 1 – Q reçu par le fluide froid (méthode bilan froid)
# Q = m_f * Cp_f * (T_f_out - T_f_in)
# ======================================================================================
Q = mf_point * Cp_int * (Tf_out - Tf_in)   # W

print("Q (méthode bilan froid) [W] : ", Q)

plt.figure(0)
plt.plot(mf_point, Q, marker='o')  # repasse sur l/h pour l’axe
plt.xlabel("mf (kg/s)")
plt.ylabel("Q (W)")
plt.title("Flux de chaleur Q en fonction du débit massique fluide froid mf")
plt.grid(True)

# ======================================================================================
# Question 2 – ΔTin, ΔTout, ΔTln, ΔTmoy
# ======================================================================================
delta_Tin   = Tc_in  - Tf_in
delta_Tout  = Tc_out - Tf_out
delta_Tln   = (delta_Tin - delta_Tout) / np.log(delta_Tin / delta_Tout)
delta_Tmoy  = (delta_Tin + delta_Tout) / 2

plt.figure(1)
plt.plot(mf_point , delta_Tin,  'o-', label='ΔTin')
plt.plot(mf_point, delta_Tout, 'o-', label='ΔTout')
plt.plot(mf_point, delta_Tln,  'o-', label='ΔTln')
plt.plot(mf_point, delta_Tmoy, 'o-', label='ΔTmoy')
plt.xlabel("mf (kg/s)")
plt.ylabel("ΔT (K)")
plt.title("Différences de température en fonction de mf")
plt.legend()
plt.grid(True)

# ======================================================================================
# Question 3 – Coefficient d’échange global hg par LMTD
# hg = Q / (A_t * ΔTln)
# ======================================================================================
hg = Q / (A_t * delta_Tln)   # W/m²/K

plt.figure(2)
plt.plot(mf_point, hg, marker='o')
plt.xlabel("mf (kg/s)")
plt.ylabel("hg (W/m²K)")
plt.title("Coefficient d'échange global hg en fonction de mf")
plt.grid(True)

# ======================================================================================
# Question 4 – Efficacité ε (formule 13) vs théorie (formule 17, contre-courant)
# ε = Q / (Mmin * (T_c_in - T_f_in))
# NTU = hg * A_t / Mmin
# ε_théorie (échangeur tubulaire contre-courant) :
# ε = [1 - exp(-NTU * (1 - C*))] / [1 - C* exp(-NTU * (1 - C*))]
# avec C* = Mmin / Mmax
# ======================================================================================
# Efficacité par bilans (relation 13)
eff13=Mf_point*(Tf_out-Tf_in)/ ( (Tc_in-Tf_in)*Mmin_point ) # adimensionnel

# NTU
NTu = hg * A_t / Mmin_point
M_point = Mmin_point / Mmax_point            # C*

# Efficacité théorique (relation 17, co-courant)
eff17= ( 1- np.exp(-(1 + M_point)*NTu) ) / (1+M_point)

plt.figure(4)
plt.plot(NTu, eff13, 'o-', color='green', label='ε (bilan, rel.13)')
plt.plot(NTu, eff17, 's-', color='orange', label='ε_th (théorie, rel.17)')
plt.xlabel("NTU")
plt.ylabel("Efficacité ε")
plt.title("Efficacité de l'échangeur : bilan vs théorie")
plt.legend()
plt.grid(True)

# ======================================================================================
# Question 5 – Nusselt vs Reynolds, et log(Nu / Pr^(1/3))
# NU = h_i * Dh / k, ici approximé avec h_g côté froid
# ======================================================================================
Re = U * Dh / nu
Nu_Re = hg * Dh / k     # attention : Nu basé sur h_g, c’est un Nu « global »
log_Nu_Pr = np.log(Nu_Re / (Pr**(1/3)))

plt.figure(5)
plt.plot(Re, Nu_Re, 'o-', label='Nu(Re)')
plt.xlabel("Re")
plt.ylabel("Nu")
plt.title("Nu en fonction de Re")
plt.grid(True)
plt.legend()

plt.figure(6)
plt.plot(Re, log_Nu_Pr, 'o-', label='log(Nu / Pr^(1/3))')
plt.xlabel("Re")
plt.ylabel("log(Nu / Pr^(1/3))")
plt.title("log(Nu / Pr^(1/3)) en fonction de Re")
plt.grid(True)
plt.legend()

plt.show()

# ======================================================================================

X = np.log(Re)
Y = np.log(Nu_Re / Pr**(1/3))

coeffs = np.polyfit(X, Y, 1)  # degré 1
b = coeffs[0]
A = coeffs[1]
a = np.exp(A)

print("Pente de la droite log(Nu/Pr^(1/3)) en fonction de log(Re) ",b)
print("La a de la formule de Nusselt  :",a)
