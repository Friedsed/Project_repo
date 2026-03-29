# -*- coding: utf-8 -*-
"""
TP3 – Neumann + normalisation (Questions 1 à 4)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----------------------
# Paramètres physiques
# ----------------------
L = 2.0 * np.pi
lambda_ = 1.0

# ----------------------
# Choix de f(x) et solution exacte
#   -> pour Q1/Q3/Q4 : cos(x)
#   -> pour Q2/Q4 : mettre sin(x)
# ----------------------
def T_analytique(x):
    return np.cos(x)      # mettre np.sin(x) pour la version "sin"

def RHS_analytique(x):
    return np.cos(x)      # mettre np.sin(x) pour la version "sin"


# ----------------------
# Normalisation (moyenne nulle, trapèzes)
# ----------------------
def sol_T(T, N):
    """
    Approximation de l'intégrale / moyenne par la méthode des trapèzes
    1/2 T0 + sum(Ti) + 1/2 TN
    puis division par N (proportionnelle à l'intégrale / L)
    """
    t = 0.0
    for j in range(1, N):
        t += T[j]
    t += 0.5 * (T[0] + T[N])
    t /= N
    return t


# ----------------------
# Méthode de Jacobi (avec normalisation)
# ----------------------
def jacobi(A, b, tol, maxit, N):
    x = np.zeros_like(b)
    x[0], x[-1] = b[0], b[-1]   # impose les lignes de bord

    for k in range(maxit):
        x_new = x.copy()
        for i in range(1, N):
            x_new[i] = (b[i]
                        - A[i, i-1] * x[i-1]
                        - A[i, i+1] * x[i+1]) / A[i, i]

        if np.linalg.norm(x_new - x, ord=2) < tol:
            x = x_new
            break
        x = x_new

    # Normalisation (moyenne nulle)
    t = sol_T(x, N)
    for j in range(N + 1):
        x[j] -= t

    return x, k + 1


# ----------------------
# Méthode de Gauss-Seidel (avec normalisation)
# ----------------------
def gauss_seidel(A, b, tol, maxit, N):
    x = np.zeros_like(b)
    x[0], x[-1] = b[0], b[-1]

    for k in range(maxit):
        x_old = x.copy()
        for i in range(1, N):
            x[i] = (b[i]
                    - A[i, i-1] * x[i-1]
                    - A[i, i+1] * x_old[i+1]) / A[i, i]

        if np.linalg.norm(x - x_old, ord=2) < tol:
            brWeak

    # Normalisation (moyenne nulle)
    t = sol_T(x, N)
    for j in range(N + 1):
        x[j] -= t

    return x, k + 1


# ----------------------
# Boucle principale sur N (pour étude convergence)
# ----------------------
N_values = list(range(10, 82, 2))
err_numpy, err_jacobi, err_gs = [], [], []
iter_jacobi, iter_gs = [], []

for N in N_values:
    dx = L / N
    x = np.linspace(0, L, N + 1)

    f = RHS_analytique(x)
    dg = 0.0   # dérivée T'(0) = dg (à adapter selon le problème si besoin)
    dd = 0.0   # dérivée T'(L) = dd

    # ----------------------
    # Construction du système Neumann (points fantômes)
    # ----------------------
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # Lignes de bord (Neumann) – formulation point fantôme
    # i = 0
    A[0, 0] = 2.0 / dx**2
    A[0, 1] = 2.0 / dx**2
    b[0]    = f[0] + 2.0 * dg / dx

    # i = N
    A[N, N]   = -2.0 * lambda_ / dx**2
    A[N, N-1] =  2.0 * lambda_ / dx**2
    b[N]      = f[N] + 2.0 * lambda_ * dd / dx

    # Points intérieurs : schéma centré classique
    for i in range(1, N):
        A[i, i-1] = -lambda_ / dx**2
        A[i, i]   =  2.0 * lambda_ / dx**2
        A[i, i+1] = -lambda_ / dx**2
        b[i] = f[i]

    # ----------------------
    # Solutions
    # ----------------------
    tol = 1e-7
    T_numpy = np.linalg.solve(A, b)
    T_jac, it_jac = jacobi(A, b, tol, 100000, N)
    T_gs,  it_gs  = gauss_seidel(A, b, tol, 100000, N)

    T_exact = T_analytique(x)

    # Erreurs (par rapport à la solution exacte choisie)
    err_numpy.append(np.linalg.norm(T_exact - T_numpy, ord=np.inf))
    err_jacobi.append(np.linalg.norm(T_exact - T_jac, ord=np.inf))
    err_gs.append(np.linalg.norm(T_exact - T_gs, ord=np.inf))

    # Nombre d'itérations
    iter_jacobi.append(it_jac)
    iter_gs.append(it_gs)

    print(f"N={N:3d} | Jacobi it={it_jac:5d} | GS it={it_gs:5d}")

# ----------------------
# Vérification normalisation sur le dernier N
# ----------------------
print("Verification de la normalisation avec Jacobi :", sol_T(T_jac, N))
print("Verification de la normalisation avec Gauss-Seidel :", sol_T(T_gs, N))

# ----------------------
# Graphiques
# ----------------------
plt.figure(figsize=(8, 6))
plt.loglog(N_values, err_numpy, 'k-o', label='Direct (numpy)')
plt.loglog(N_values, err_jacobi, 'r-s', label='Jacobi')
plt.loglog(N_values, err_gs, 'b-^', label='Gauss-Seidel')
plt.xlabel('N')
plt.ylabel('Erreur ||T - T_exact||∞')
plt.title('Erreur en fonction de N')
plt.grid(True, which='both', ls='--')
plt.legend()

plt.figure(figsize=(8, 6))
plt.plot(N_values, iter_jacobi, 'r-s', label='Jacobi')
plt.plot(N_values, iter_gs, 'b-^', label='Gauss-Seidel')
plt.xlabel('N')
plt.ylabel('Nombre d’itérations')
plt.title('Convergence des méthodes itératives')
plt.grid(True)
plt.legend()
plt.show()

# Pentes en log-log (facultatif)
logN = np.log(N_values)
from scipy.stats import linregress
slope_np, _, _, _, _ = linregress(logN, np.log(err_numpy))
slope_jac, _, _, _, _ = linregress(logN, np.log(err_jacobi))
slope_gs, _, _, _, _ = linregress(logN, np.log(err_gs))
print(f"Pente (Directe)      : {slope_np:.2f}")
print(f"Pente (Jacobi)       : {slope_jac:.2f}")
print(f"Pente (Gauss-Seidel) : {slope_gs:.2f}")
