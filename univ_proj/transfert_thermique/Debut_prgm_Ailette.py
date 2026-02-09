import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Données Fluide 
data = np.array([
[100, 3.5562, 0.711, 0.200,  1032, 0.934, 0.254, 0.786],
[150, 2.3364, 1.034, 0.4426, 1012, 1.38,  0.584, 0.758],
[200, 1.7458, 1.325, 0.7590, 1007, 1.81,  1.03,  0.737],
[250, 1.3947, 1.596, 1.144,  1006, 2.23,  1.59,  0.720],
[300, 1.1614, 1.846, 1.589,  1007, 2.63,  2.25,  0.707],
[350, 0.9950, 2.082, 2.092,  1009, 3.00,  2.99,  0.700],
[400, 0.8711, 2.301, 2.641,  1014, 3.38,  3.83,  0.690]
])
T =     data[:,0]           # K
rho =   data[:,1]           # Kg/m3
mu =    data[:,2] * 1e-5    # Kg/ms
nu =    data[:,3] * 1e-5    # m2/S
Cp =    data[:,4]           # J/KgK
kf =    data[:,5] * 1e-2    # W/mK
alpha = data[:,6] * 1e-5    # m2/s
Pr =    data[:,7]

# Données Ailette
r0 = 0.005 # rayon ailette m
L = 0.7        # Longueur Ailette m
pi = np.pi
A = pi * r0**2   # section droite
p = 2 * pi * r0    # perimetre
he= 5     # Coefficient d'echange W/m2K
theta_0= 1  # Température à la base de l'ailette °C


#k_laiton = 114 # 121    #W/mK
x_laiton= np.array([3/1000, 52.5/1000, 102.5/1000, 202.5/1000,302.5/1000, 402.5/1000] ) # longueur de l'ailette en laiton en mm

#k_acier = 45
x_acier=np.array([1/100,3.5/100,6.5/100,11.5/100,21.5/100])

k= 100 # W/mK

x = np.linspace(0, L, 100)
m = np.sqrt((he*p)/(k*A))

##___________________________________________________________________________________


##Question 1:

def theta_infinie(theta_0, m, x):
    return theta_0*np.exp(-m*x)   

##Question 2:

def theta_finie(theta_0, he, k, m, L, x):
    return theta_0*( np.cosh(m*(L-x)) + (he/k*m)*np.sinh(m*(L-x)) ) / ( np.cosh(m*L) + (he/k*m)*np.sinh(m*L) )

##Question 3: Tracer les deux solutions obtenues pour un coefficient d’échange de h=he variant de 5 à
# 10 W/m2K et une conductivité k de variant de 40 à 120 W/mK et θ0 = 1, Faites aussi varier la
# longueur de l’ailette. Conclure.

plt.figure()
plt.plot(x, theta_infinie(theta_0, m, x), label='Solution INFINIE', color='blue')
plt.plot(x, theta_finie(theta_0, he, k, m, L, x), label='Solution FINIE', color='orange')
 
plt.xlabel('Position le long de l\'ailette (m)')
plt.ylabel('Température (°C)')
plt.title('Solutions finie vs infinie pour h=5 W/m2K et k=100 W/mK question 3')
plt.legend()
plt.grid()


# Variation de la longueur L
L_values = [0.5, 0.7, 0.9]
he = 5
kk = 100
m = np.sqrt((he * p) / (kk * A))

plt.figure(figsize=(10, 6))

for L_val in L_values:
    x_L = np.linspace(0, L_val, 100)
    plt.plot(x_L, theta_infinie(theta_0, m, x_L), label=f'Solution INFINIE L={L_val}m', linestyle='--')
    plt.plot(x_L, theta_finie(theta_0, he, kk, m, L_val, x_L), label=f'Solution FINIE L={L_val}m', linestyle='-')

plt.xlabel('Position le long de l\'ailette (m)')
plt.ylabel('Température (°C)')
plt.title('Influence de la longueur de l\'ailette L pour h=5 W/m²K et k=100 W/mK')
plt.legend()
plt.grid()



'''Conclusion :
La solution finie est plus réaliste car elle tient compte de la perte de chaleur à l'extrémité de l'ailette, 
contrairement à la solution infinie qui suppose une perte nulle à l'extrémité. 
La solution finie montre une diminution plus progressive de la température le long de l'ailette, ce qui est plus cohérent avec les conditions réelles d'échange thermique.
'''

##Question 4:  tracer sur un autre graphe, la solution
# θlai = exp(−mxlai)où xlai sont les points de mesure de la barre en laiton, puis ln(θlai).
# Faites une régression linéaire afin de retrouver theta_0 et m, puis h.



x_lai= np.array([3/1000, 52.5/1000, 102.5/1000, 202.5/1000,302.5/1000, 402.5/1000] )

theta_lai = theta_infinie(theta_0, m, x_lai)

plt.figure()
plt.plot(x_lai, theta_lai, label='T aux points de mesure x_lai', marker='x',linestyle='None', color='red')
plt.xlabel('Position le long de l\'ailette (m)')
plt.ylabel('Température (°C)')
plt.title('T aux differents points de mesure le long de la barre en laiton')
plt.legend()
plt.grid()



# Régression linéaire pour trouver theta_0, m, puis h
ln_theta_lai = np.log(theta_lai)
pente, ord_origine = np.polyfit(x_lai, ln_theta_lai, 1)
m_ret = -pente
theta_0_ret = np.exp(ord_origine)
h_ret = (m_ret**2 * k * A) / p

print(f"m ={m_ret:.4f}")
print(f"theta_0 = {theta_0_ret:.4f}")
print(f"h = {h_ret:.4f}")

plt.figure()
plt.plot(x_lai, ln_theta_lai, 'x', label='ln(theta) aux points de mesure')
plt.plot(x_lai, pente * x_lai + ord_origine, label='Régression linéaire')
plt.xlabel('Position x (m)')
plt.ylabel('ln(theta)')
plt.title('Régression linéaire pour retrouver m et theta_0')
plt.legend()
plt.grid()


##Question 5: Notez dans un tableau numpy les valeurs relevées des thermocouples pour chacune des deux barres


# Données thermocouples pour la barre en laiton

thermocouples_laiton = np.array([43.1,39.4,34.8,28.7,25.7,23.9])  # Températures mesurées en °C

# Données thermocouples pour la barre en acier

thermocouples_acier = np.array([46.9,41.8,36.3,31.4,25.8])  # Températures mesurées en °C


##Question 6: Pour les deux barres, tracer sur un graphe ln(θ) en fonction de x. 

# Barre en laiton

theta_laiton_mesure = thermocouples_laiton / theta_0
ln_theta_laiton_mesure = np.log(theta_laiton_mesure)
plt.figure()
plt.plot(x_lai, ln_theta_laiton_mesure, 'o', label='Données mesurées laiton', color='green')
plt.legend()
plt.xlabel('Position x (m)')
plt.ylabel('ln(theta)')
plt.title('ln(theta) en fonction de x pour la barre en laiton')
plt.grid()


# Barre en acier

theta_acier_mesure = thermocouples_acier / theta_0
ln_theta_acier_mesure = np.log(theta_acier_mesure)
plt.figure()
plt.plot(x_acier, ln_theta_acier_mesure, 'o', label='Données mesurées acier', color='purple')
plt.legend()
plt.xlabel('Position x (m)')
plt.ylabel('ln(theta)')
plt.title('ln(theta) en fonction de x pour la barre en acier')
plt.grid()




#régression linéaire afin de determiner θ0 et m, puis h (prendre olyfit de la bibliothèque numpy).

# Barre en laiton
_pente_laiton, _ord_origine_laiton = np.polyfit(x_lai, ln_theta_laiton_mesure, 1)
m_laiton = -_pente_laiton
theta_0_laiton = np.exp(_ord_origine_laiton)
h_laiton = (m_laiton**2 * k * A) / p

print(f"Barre en laiton : m_laiton = {m_laiton:.4f}, theta_0_laiton = {theta_0_laiton:.4f}, h_laiton = {h_laiton:.4f}")

# Barre en acier
_pente_acier, _ord_origine_acier = np.polyfit(x_acier, ln_theta_acier_mesure, 1)
m_acier = -_pente_acier
theta_0_acier = np.exp(_ord_origine_acier)
h_acier = (m_acier**2 * k * A) / p

print(f"Barre en acier : m_acier = {m_acier:.4f}, theta_0_acier = {theta_0_acier:.4f}, h_acier = {h_acier:.4f}")

# Tracer les régressions linéaires pour les deux barres
plt.figure()
plt.plot(x_lai, ln_theta_laiton_mesure, 'o', label='Données mesurées laiton', color='green')
plt.plot(x_lai, _pente_laiton * x_lai + _ord_origine_laiton, label='Régression linéaire laiton', color='blue')
plt.xlabel('Position x (m)')
plt.ylabel('ln(theta)')
plt.title('Régression linéaire pour la barre en laiton')
plt.legend()
plt.grid()


plt.figure()
plt.plot(x_acier, ln_theta_acier_mesure, 'o', label='Données mesurées acier', color='purple')
plt.plot(x_acier, _pente_acier * x_acier + _ord_origine_acier, label='Régression linéaire acier', color='orange')
plt.xlabel('Position x (m)')
plt.ylabel('ln(theta)')
plt.title('Régression linéaire pour la barre en acier')
plt.legend()
plt.grid()



##Question 7: Avec les valeurs de theta_0 et m obtenues a la question 6 comparer la solution de la question 1 avec les points expérimentaux

# Comparaison pour la barre en laiton
theta_laiton_modele = theta_infinie(theta_0_laiton, m_laiton, x_lai)
plt.figure()
plt.plot(x_lai, thermocouples_laiton, 'o', label='Données mesurées laiton', color='green')
plt.plot(x_lai, theta_laiton_modele, label='Modèle laiton', color='blue')
plt.xlabel('Position x (m)')
plt.ylabel('Température (°C)')
plt.title('Comparaison modèle vs données mesurées pour la barre en laiton')
plt.legend()
plt.grid()


# Comparaison pour la barre en acier
theta_acier_modele = theta_infinie(theta_0_acier, m_acier, x_acier)
plt.figure()
plt.plot(x_acier, thermocouples_acier, 'o', label='Données mesurées acier', color='purple')
plt.plot(x_acier, theta_acier_modele, label='Modèle acier', color='orange')
plt.xlabel('Position x (m)')
plt.ylabel('Température (°C)')
plt.title('Comparaison modèle vs données mesurées pour la barre en acier')
plt.legend()
plt.grid()
plt.show()



##Question 8: Calcul de l'efficacite de l'ailette et commentaires : 

def efficacite_ailette(m, L, he, A): 
    eps = np.sqrt(p*k /(he*A)) * np.tanh(m*L)
    return eps

eta = efficacite_ailette(m, L, he, A)
print(f"Efficacité de l'ailette : {eta:.4f}")

'''Commentaire :
L'efficacité de l'ailette mesure la capacité de l'ailette à transférer la chaleur par rapport à une situation sans ailette. 
On trouve une efficacité d'environ 89, ce qui montre que l'ailette améliore significativement le transfert de chaleur par rapport à une situation sans ailette.
'''

##Question 9: Calcul du rendement de l'ailette et commentaires :

def rendement_ailette(m, L): 
    rendement = np.tanh(m*L) / (m*L) 
    return rendement

rendement = rendement_ailette(m, L)
print(f"Rendement de l'ailette : {rendement:.4f}")

'''Commentaire :
On trouve un rendement d'environ 32%, donc le rendement est assez faible ce qui suggère que l'ailette est trop grande, et qu'il y a des pertes thermiques à considérer.
'''




