# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:39:21 2026

@author: aberg
"""


import numpy as np
import matplotlib.pyplot as plt
import time  # <-- AJOUT

# Terme source 
# -----------------------------------------------------------------------------

def f_func(x, y, t, omega, Lx, Ly):
    """Terme source avec fréquence omega et domaine [0,Lx]x[0,Ly]."""
    return np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) * np.cos(omega*t)

#  Valeur de Dirichlet sur les bords
# -----------------------------------------------------------------------------

def Td(x, y):
    """Condition de Dirichlet au bord."""
    return 0.0

# Solution exacte 
# -----------------------------------------------------------------------------

def T_ex(x, y, t, D, omega, Lx, Ly):
    lambda_ = (np.pi/Lx)**2 + (np.pi/Ly)**2
    return (
        np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) *
        (D * lambda_ * np.cos(omega * t) + omega * np.sin(omega * t)) /
        ((D * lambda_)**2 + omega**2) )

# -----------------------------------------------------------------------------
# Caclul de T^(n+1) de maniere explicite
# -----------------------------------------------------------------------------

def Tnp1_explicite(Tn, Dx, Dy, dt, D, f_func, t, x, y, omega, Lx, Ly):
    """Calcule T^{n+1} à partir de T^n par un schéma explicite."""
    N, M = Tn.shape
    Tnp1 = Tn.copy()

    Dx_coeff = D * dt / Dx**2
    Dy_coeff = D * dt / Dy**2

    for i in range (1, N-1):
        for j in range(1, M-1):
            Tnp1[i,j]=Dx_coeff * Tn[i-1,j] +Dy_coeff*Tn[i,j-1]+ (1-2*(Dx_coeff+Dy_coeff))*Tn[i,j]+ Dx_coeff*Tn[i+1,j]+Dy_coeff*Tn[i,j+1]+dt*f_func(x[i], y[j], t, omega, Lx, Ly)   

# Completer et écrire le calcul de Tnp1 à partir de Tn
# et de la la fonction qui clacule le terme source.



    return Tnp1

# -----------------------------------------------------------------------------
# Paramètres physique et maillage
# -----------------------------------------------------------------------------

Lx, Ly = 1.0, 1.0
Nx, Ny = 3, 3
Dx = Lx / Nx
Dy = Ly / Ny
dt = 0.000080
D = 1.0
Tmax = 10
omega = 2*np.pi

Dmx = D*dt/Dx**2
Dmy = D*dt/Dx**2

print(f"Delta t = {dt:.6e}")
print(f"Delta x = {Dx:.6e}")
print(f"Delta y = {Dy:.6e}")
print(f"Nombre de diffusion de maille Dmx = {Dmx:.6e}")
print(f"Nombre de diffusion de maille Dmx = {Dmy:.6e}")
print(f"Critère de stabilité  Dmy + Dm < 0.5")


# Maillage
# -----------------------------------------------------------------------------

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

i_center = np.argmin(np.abs(x - Lx/2))
j_center = np.argmin(np.abs(y - Ly/2))

# Solution initiale
# -----------------------------------------------------------------------------

Tn = T_ex(X, Y, 0.0, D, omega, Lx, Ly)

# -----------------------------------------------------------------------------
# Boucle temporelle
# -----------------------------------------------------------------------------

temps = []
erreur_L2_list = []
erreur_max_list = []
T_center_time = []
t = 0.0
erreur_max_globale = 0.0 


# Debut du chronométrage 
# -----------------------------------------------------------------------------
t_start = time.time()

dict_temp={}

while t < Tmax:
    
    # Solution numerique à t n+1   
    Tn = Tnp1_explicite(Tn, Dx, Dy, dt, D, f_func, t, x, y, omega, Lx, Ly)
    t += dt
    dict_temp["T^"+str(t)]=Tn
    
    # Solution exacte à t n+1 
    Tex = T_ex(X, Y, t, D, omega, Lx, Ly)

    # Calcul des erreurs 
    erreur = Tn - Tex
    erreur_L2 = np.sqrt(np.sum(erreur**2) * Dx * Dy)
    erreur_max = np.max(np.abs(erreur))
    erreur_max_globale = max(erreur_max_globale, erreur_max)

    # Collecte des valeurs des erreurs, des instants et T au milieu
    temps.append(t)
    erreur_L2_list.append(erreur_L2)
    erreur_max_list.append(erreur_max)
    T_center_time.append(Tn[i_center, j_center])


# Fin du chronometrage et calcul du temps de calcul
# -----------------------------------------------------------------------------

t_end = time.time()
temps_calcul = t_end - t_start


# -----------------------------------------------------------------------------
# Affichage console temps de calcul et erreur max
# -----------------------------------------------------------------------------

print(f"Erreur maximale globale sur tous les pas de temps : {erreur_max_globale:.6e}")
print(f"Temps total de calcul : {temps_calcul:.3f} secondes")

# -----------------------------------------------------------------------------
# Graphiques
# -----------------------------------------------------------------------------
# Se familiariser avec la représentation des isovaleurs

plt.figure()
plt.plot(temps, erreur_L2_list, label="Erreur L2")
plt.plot(temps, erreur_max_list, label="Erreur max")
plt.xlabel("Temps")
plt.ylabel("Erreur")
plt.title("Évolution de l'erreur dans le temps")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
cf = plt.contourf(X, Y, Tn, 50)
plt.colorbar(cf, label="Température")
cs = plt.contour(X, Y, Tn, 10, colors='k', linewidths=0.7)
plt.clabel(cs, inline=True, fontsize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solution numérique au temps final")
plt.show()

plt.figure()
cf = plt.contourf(X, Y, np.abs(Tn - Tex), 50)
plt.colorbar(cf, label="Erreur |T - Tex|")
cs = plt.contour(X, Y, np.abs(Tn - Tex), 10, colors='k', linewidths=0.7)
plt.clabel(cs, inline=True, fontsize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Erreur spatiale au temps final")
plt.show()


plt.figure()
plt.plot(temps, T_center_time, 'b-', lw=2, label='Solution numérique')
T_center_exact = [T_ex(Lx/2, Ly/2, t, D, omega, Lx, Ly) for t in temps]
plt.plot(temps, T_center_exact, 'r--', lw=2, label='Solution exacte')
plt.xlabel("Temps")
plt.ylabel("T(Lx/2, Ly/2, t)")
plt.title(f"Solution numérique vs exacte au centre, omega={omega}")
plt.legend()
plt.grid(True)
plt.show()
