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
# Données fluides 
T_tab=np.array([  0,        10,       20,       40,       60])
Cp_tab=np.array([ 4218,     4192,     4184,     4184,     4184])	
nu_tab=np.array([ 1.79e-06, 1.30e-06, 1.01e-06, 6.60e-07, 4.77e-07])
k_tab=np.array([  0.552,    0.586,    0.597,    0.628,    0.651])
Pr_tab=np.array([ 13.5,     9.3,      7,        4.34,     3])



mc_point = 500 # litre/heure 


mf=np.array([900,800,700,600,500,400,300,200,100])  # Débit massique fluide caloporteur (eau) "en l/h
'''
Tf_outd=np.array([22.1, 21.3, 21.6, 21.9, 22.7, 23.4, 24.4 ])
Tf_ind=np.array([ 18.4, 17.2, 17,  16.9, 16.9, 17, 17.1 ])


Tc_outd=np.array([ 33.2, 33.1, 33.3, 33.5, 33.9, 34.3, 35.2])
Tc_ind=np.array([   40,  39.8,  39.9, 39.9, 39.8, 39.7, 39.9 ])
'''


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
Mmax_point = np.max(Mf_point, Mc_point)



delta_out=Tc_out-Tf_out
delta_in=Tc_in-Tf_in
delta_Tln=(delta_in - delta_out)/np.log(delta_in/delta_out)

Cp_int=np.interp(Tf_moy,T_tab,Cp_tab) # voir doc interp numpy

Q=mf*Cp_int*(delta_out) /3600  # conversion l/h en kg/s
# Tracé
print("dum is: ",Cp_int)
plt.figure(0)

plt.plot(mf, Q, marker='o')        # courbe + points
plt.xlabel("mf")
plt.ylabel("Q")
plt.title("Q en fonction de mf")
plt.grid(True)
plt.show()
"""
plt.figure(1)

plt.plot(mf, delta_out, marker='o', label='delta_out')        # courbe + points*
plt.plot(mf, delta_in, marker='o', label='delta_in')        # courbe + points
plt.plot(mf, delta_Tln, marker='o', label='delta_Tln')
plt.xlabel("mf")
##plt.ylabel("delta_out")
plt.title("Delta_T en fonction de mf")
plt.legend()
plt.grid(True)

plt.show()


 """






#_________________________________________________________________________________________________________
"""1. Tracer le flux de chaleur reçu par le fluide froid en fonction du débit de fluide froid en prenant
un Cf à la température moyenne du fluide froid."""

"""  ANSWER commentaire:
Le flux de chaleur Q reçu par le fluide froid peut être calculé en utilisant la formule suivante :
Q = m_f * C_f * (T_f_out - T_f_in)  """

Q = np.zeros(len(mf))
for i in range(len(mf)):
    Tf_moy = (Tf_in[i] + Tf_out[i]) / 2
    Cf = np.interp(Tf_moy, T_tab, Cp_tab)  # Interpolation pour obtenir Cf à la température moyenne
    Q[i] = mf[i] * Cf * (Tf_out[i] - Tf_in[i]) / 3600  # Conversion de l/h en kg/s
print(f"Débit massique fluide froid: {mf} l/h, Flux de chaleur Q: {Q} W")

figure = plt.figure(0)
plt.plot(mf, Q, marker='o')        # courbe + points
plt.xlabel("mf (l/h)")
plt.ylabel("Q (W)")
plt.title("Flux de chaleur Q en fonction du débit massique fluide froid mf")


plt.show()


#_________________________________________________________________________________________________________

"""2. Tracer les ∆Tin , ∆Tout , ∆Tln et ∆Tmoy en fonction du débit de fluide froid. ∆Tmoy = (∆Tin +
∆Tout )/2."""

""" ANSWER commentaire:
Les différences de température ∆Tin, ∆Tout, ∆Tln et ∆Tmoy peuvent être calculées comme suit :
- ∆Tin = T_c_in - T_f_in - différence de température à l'entrée
- ∆Tout = T_c_out - T_f_out - différence de température à la sortie et - ∆Tln = (∆Tin - ∆Tout) / ln(∆Tin / ∆Tout) - différence de température logarithmique
- ∆Tmoy = (∆Tin + ∆Tout) / 2 - différence de température moyenne """

delta_Tin = Tc_in - Tf_in
delta_Tout = Tc_out - Tf_out
delta_Tln = (delta_Tin - delta_Tout) / np.log(delta_Tin / delta_Tout)
delta_Tmoy = (delta_Tin + delta_Tout) / 2
figure = plt.figure(1)
plt.plot(mf, delta_Tin, marker='o', label='∆Tin')        # courbe + points
plt.plot(mf, delta_Tout, marker='o', label='∆Tout')        # courbe + points
plt.plot(mf, delta_Tln, marker='o', label='∆Tln')     # courbe + points
plt.plot(mf, delta_Tmoy, marker='o', label='∆Tmoy')     # courbe + points
plt.xlabel("mf (l/h)")  
plt.title("Différences de température en fonction du débit massique fluide froid mf")
plt.legend()
plt.show()

#_____________________________________________________________________________________________________
"""3. Tracer le coefficient d échange global par la méthode de la température moyenne logarithmique
en fonction du débit de fluide froid."""

""" ANSWER commentaire:
Le coefficient d'échange global hg peut être calculé en utilisant la formule suivante :
hg = Q / (A_t * ∆Tln)
où A_t est la surface d'échange thermique et ∆Tln est la différence de température logarithmique.

"""
hg = Q / (A_t * delta_Tln)
figure = plt.figure(2)  
plt.plot(mf, hg, marker='o')        # courbe + points
plt.xlabel("mf (l/h)")
plt.ylabel("hg (W/m²K)")
plt.title("Coefficient d'échange global hg en fonction du débit massique fluide froid mf")
plt.legend()
plt.show()
#_________________________________________________________________________________________________________

"""4. Tracer l efficacité de l échangeur à partir des bilans (relation 13) en fonction de NTU. Puis
comparer cette efficacité à celle obtenue par la théorie (relation 17)."""

""" ANSWER commentaire:
L'efficacité ε de l'échangeur peut être calculée en utilisant la formule suivante :
 ε = Q / (Q_max) où Q_max = M_min * (T_c_in - T_f_in)
 NTU = hg * A_t / M_min
"""


B=(1/Mf_point ) + 1/Mc_point
eff13=Mf_point*(Tf_out-Tf_in)/(Tc_in-Tf_in)*Mmin_point
NTu=hg*A_t/Mmin_point
M_point=Mmin_point/Mmax_point
eff17=  1- np.exp(-(1 + M_point)*NTu)/(1+M_point)


plt.plot(mf_point,eff13, marker='o', color='green')        # courbe + points
plt.plot(NTu,eff17, marker='o', color='yellow')    
plt.xlabel("mf_point")
plt.ylabel("efficacite (W/m²K)")
plt.title("Comparaison de l'efficacite calculer avec different formule la 13 en vert et la 17 en jaune ")
plt.legend()
plt.show()


#_____________________________________________________________________________________________________

"""(a) Tracer N u = f (Re) puis log(N u/P r1/3 ).
(b) Trouver une loi de la forme N u = aReb P r1/3 en déterminant a et b et le domaine de validité."""


#_________________________________________________________________________________________________________


"""6. Les températures de sortie pour les débits d eau froide 250, 550 et 750 /h et d eau chaude de
500 /h en utilisant la correlation N u = aReb P r1/3 pour déterminer la valeur du coefficient
d échange global pour une température deau chaude et froide à l entrée de 60◦ C et 20◦ C avec
la méthode ϵ-NTU."""
#_____________________________________________________________________________________________________

#_________________________________________________________________________________________________________


plt.show()
# Bilan fluide froid
#Q_f =

# Delta_T 
#D_T_in =
#D_T_out = 
#D_T_ln = 
#D_T_moy =

# Coef d'echange
#h_g = 
# Nusselt

#Efficacite
