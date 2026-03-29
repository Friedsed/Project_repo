# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:56:38 2026
@author: aberg
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# -----------------------------------------------------------------------------
# Terme source
# -----------------------------------------------------------------------------

def f_func(x, y, t, omega, Lx, Ly):
    return np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) * np.cos(omega*t)


# -----------------------------------------------------------------------------
# Dirichlet
# -----------------------------------------------------------------------------

def Td(x, y):
    return 0.0


# -----------------------------------------------------------------------------
# Solution exacte
# -----------------------------------------------------------------------------

def T_ex(x, y, t, D, omega, Lx, Ly):

    lambda_ = (np.pi/Lx)**2 + (np.pi/Ly)**2

    return (
        np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) *
        (D * lambda_ * np.cos(omega * t) + omega * np.sin(omega * t)) /
        ((D * lambda_)**2 + omega**2)
    )


# -----------------------------------------------------------------------------
# Gauss-Seidel


#  HERE WE DON'T KNOW THE VALUE OF rhs FOR THE FUNCTION f(x,y,t_{n+1}) , SO WE JUST HAVE A FUNCTION DEPENDING OF X,Y AND T
# -----------------------------------------------------------------------------

def solve_GS(T_init, rhs, Dx, Dy, tol=1e-8, max_iter=5000):

    T = T_init.copy()
    N, M = T.shape
    a = 1 + 2*(Dx + Dy)
    itergs = 0
    for it in range(max_iter):
        T_old = T.copy()
        for i in range(1, N-1):
            for j in range(1, M-1):
                T[i, j] = (1/a) * (  T[i,j]  + Dx * T[i-1,j] + Dy* T[i,j-1] + Dy* T_old[i,j+1] +  Dx *T_old[i+1,j] + dt * rhs[i,j] )      
        if np.linalg.norm(T - T_old) < tol:
            break
        itergs += 1
    return T, itergs

""" OUR LAST GAUSS-SEIDEL CODE  TO COMPARE WITH MY CODE IN SEANCE1_ETUDIANT.PY


    def gauss_seidel(A, b, tol, maxit):
        x_gs = np.zeros_like(b)
        x_gs[0], x_gs[-1] = b[0], b[-1]
        # Question 5 : Compléter la routine en vous inspirant de Jacobi
        for k in range(maxit):
            x_old = x_gs.copy()
            for i in range(1, len(b)-1):
                x_gs[i] = (b[i] - A[i,i-1]*x_gs[i-1] - A[i,i+1]*x_old[i+1]) / A[i,i]
            if np.linalg.norm(x_gs - x_old, ord=2) < tol:
                break
        return x_gs, k+1
"""


# -----------------------------------------------------------------------------
# Calcul de T^(n+1) implicite

# HERE WE CALCULATE THE VALUE OF rhs(x,y) SO  WE HAVE SOME DIGIT REAL NUMBER AND THEN WE CALCULE THE VALUE OF T^(n+1) USING THE GAUSS-SEIDEL FUNCTION TO HAVE SOME NUMBERS 
# -----------------------------------------------------------------------------

def Tnp1_implicite(Tn, Dx, Dy, dt, D, f_func, tnp1, x, y, omega, Lx, Ly):
    N, M = Tn.shape
    rhs = Tn.copy()

    for i in range(1, N-1):
        for j in range(1, M-1):
            rhs[i, j] = Tn[i, j] + dt * f_func(x[i], y[j], tnp1, omega, Lx, Ly)

    Dx_coeff = D * dt / Dx**2
    Dy_coeff = D * dt / Dy**2
    Tnp1, itergs = solve_GS(Tn, rhs, Dx_coeff, Dy_coeff)   #        NO NEED OF tol ANS max_inter BECAUSE HE ALREADY DEFINE THEY VALUE IN THE FUNCTION solve_GS

    return Tnp1, itergs


# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------

Lx, Ly = 1.0, 1.0
Nx, Ny = 20, 20

Dx = Lx / Nx
Dy = Ly / Ny

dt = 0.00060
D = 1.0
Tmax = 10

omega = 2*np.pi

Dmx = D*dt/Dx**2
Dmy = D*dt/Dy**2

print(f"Delta t = {dt:.6e}")
print(f"Delta x = {Dx:.6e}")
print(f"Delta y = {Dy:.6e}")
print(f"Nombre de diffusion Dmx = {Dmx:.6e}")
print(f"Nombre de diffusion Dmy = {Dmy:.6e}")


# -----------------------------------------------------------------------------
# Maillage
# -----------------------------------------------------------------------------

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)

X, Y = np.meshgrid(x, y, indexing='ij')

j_mid = np.argmin(np.abs(y - 0.5))
i_center = np.argmin(np.abs(x - Lx/2))
j_center = np.argmin(np.abs(y - Ly/2))


# -----------------------------------------------------------------------------
# Condition initiale
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


# Chronométrage
t_start = time.time()


while t < Tmax:

    t += dt

    Tn, itergs = Tnp1_implicite(Tn, Dx, Dy, dt, D, f_func, t, x, y, omega, Lx, Ly)

    print(f"t = {t:.4f}, itérations GS = {itergs}")

    Tex = T_ex(X, Y, t, D, omega, Lx, Ly)

    erreur = Tn - Tex

    erreur_L2 = np.sqrt(np.sum(erreur**2) * Dx * Dy)
    erreur_max = np.max(np.abs(erreur))

    erreur_max_globale = max(erreur_max_globale, erreur_max)

    temps.append(t)
    erreur_L2_list.append(erreur_L2)
    erreur_max_list.append(erreur_max)

    T_center_time.append(Tn[i_center, j_center])


# Temps de calcul
t_end = time.time()
temps_calcul = t_end - t_start


print(f"Erreur maximale globale : {erreur_max_globale:.6e}")
print(f"Temps total de calcul : {temps_calcul:.3f} secondes")


# -----------------------------------------------------------------------------
# Graphiques
# -----------------------------------------------------------------------------

plt.figure()
plt.plot(temps, erreur_L2_list, label="Erreur L2")
plt.plot(temps, erreur_max_list, label="Erreur max")
plt.xlabel("Temps")
plt.ylabel("Erreur")
plt.legend()
plt.grid()
plt.show()


plt.figure()
cf = plt.contourf(X, Y, Tn, 50)
plt.colorbar(cf)
plt.title("Solution numérique finale")
plt.show()


plt.figure()
cf = plt.contourf(X, Y, np.abs(Tn - Tex), 50)
plt.colorbar(cf)
plt.title("Erreur finale")
plt.show()


plt.figure()
plt.plot(temps, T_center_time, label="numérique")

T_center_exact = [T_ex(Lx/2, Ly/2, t, D, omega, Lx, Ly) for t in temps]

plt.plot(temps, T_center_exact, '--', label="exacte")
plt.legend()
plt.grid()
plt.show()