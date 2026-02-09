# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:58:45 2025

@author: aberg (corrigé)
"""

import numpy as np
import matplotlib.pyplot as plt

# Lagrange
# -------------------


def lagrange_interp(x_data, y_data, x_eval):
    """
    Interpolation de Lagrange (vectorisée).
    x_data, y_data: tableaux de taille n
    x_eval: tableau de points d'évaluation
    retourne P(x_eval)
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    x_eval = np.asarray(x_eval)
    n = len(x_data)
    P = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        # construit L_i(x) = prod_{j!=i} (x - x_j)/(x_i - x_j)
        L = np.ones_like(x_eval, dtype=float)
        xi = x_data[i]
        for j in range(n):
            if j == i:
                continue
            xj = x_data[j]
            L *= (x_eval - xj) / (xi - xj)
        P += y_data[i] * L
    return P

# Newton
# -------------------

def newton_coeff(x_data, y_data):
    """
    Calcul des coefficients a (différences divisées) pour la forme de Newton.
    a[0] = f[x0], a[1] = f[x0,x1], ...
    """
    n = len(x_data)
    a = np.copy(y_data).astype(float)
    for k in range(1, n):
        # mise à jour in-place des différences divisées
        a[k:n] = (a[k:n] - a[k-1:n-1]) / (x_data[k:n] - x_data[0:n-k])
    return a

def newton_eval(x_data, a, x_eval):
    """
    Évaluation du polynôme de Newton en x_eval via Horner généralisé.
    x_data: noeuds x0..x_{n-1}
    a: coefficients de Newton (taille n)
    x_eval: tableau de points
    """
    x_data = np.asarray(x_data)
    a = np.asarray(a)
    x_eval = np.asarray(x_eval)
    P = np.zeros_like(x_eval, dtype=float)
    n = len(a)
    for idx, x in enumerate(x_eval):
        val = a[-1]
        # Horner adapté : val = a_{k} + (x - x_k)*val
        for k in range(n-2, -1, -1):
            val = a[k] + (x - x_data[k]) * val
        P[idx] = val
    return P

# -----------

# Laplacien 1D
L = 2*np.pi
lambda_ = 1.0

def T_analytique(x): return np.cos(x)

def RHS_analytique(x): return np.cos(x)   # exemple : -u'' = cos(x) with solution cos(x) if BC consistent

def build_systeme(N):
    """
    Construction du système linéaire pour le 1D -u'' = f avec conditions de Dirichlet
    sur x in [0, L]. Maillage uniforme avec N cellules -> N+1 points.
    """
    dx = L / N
    x = np.linspace(0, L, N+1)
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    # conditions de Dirichlet: valeur imposée aux noeuds 0 et N
    A[0,0] = 1.0
    b[0] = T_analytique(x[0])
    A[N,N] = 1.0
    b[N] = T_analytique(x[N])
    # intérieurs : discrétisation centrale pour -u'' = f  -> A_i = 1/dx^2 * [1 -2 1]
    coef = 1.0 / (dx*dx)
    for i in range(1, N):
        A[i,i-1] = coef
        A[i,i]   = -2.0 * coef
        A[i,i+1] = coef
        b[i] = RHS_analytique(x[i])
    return A, b, x

# Jacobi
def jacobi(A, b, T0, tol=1e-6, maxit=100000):
    """
    Méthode de Jacobi. T0 contient la condition initiale (y compris valeurs aux bords).
    Retourne la solution approchée et le nombre d'itérations effectuées.
    """
    T = T0.copy().astype(float)
    # imposer Dirichlet (au cas où)
    T[0], T[-1] = b[0], b[-1]
    Np1 = len(b)
    T_new = T.copy()
    # pré-calculer diagonales pour efficacité
    diag = np.diag(A)
    for k in range(maxit):
        # mise à jour des points intérieurs
        for i in range(1, Np1-1):
            # somme des A[i,j]*T[j] pour j != i
            s = 0.0
            # j = i-1
            s += A[i,i-1] * T[i-1]
            # j = i+1
            s += A[i,i+1] * T[i+1]
            T_new[i] = (b[i] - s) / diag[i]
        # bords restent imposés
        T_new[0], T_new[-1] = b[0], b[-1]
        # critère d'arrêt : norme max des différences
        err = np.max(np.abs(T_new - T))
        T[:] = T_new
        if err < tol:
            return T, k+1
    return T, maxit

# Gauss-Seidel
def gauss_seidel(A, b, T0, tol=1e-6, maxit=10000):
    """
    Méthode de Gauss-Seidel (mise à jour in-place).
    """
    T = T0.copy().astype(float)
    T[0], T[-1] = b[0], b[-1]
    Np1 = len(b)
    diag = np.diag(A)
    for k in range(maxit):
        T_old = T.copy()
        for i in range(1, Np1-1):
            s = A[i,i-1] * T[i-1] + A[i,i+1] * T[i+1]
            T[i] = (b[i] - s) / diag[i]
        err = np.max(np.abs(T - T_old))
        if err < tol:
            return T, k+1
    return T, maxit

# Maillage grossier et fin
N_coarse = 10
N_fine = 80

# Constructions des jeux de matrices, membres de droite et 
# maillage grossier et fin.

A_c, b_c, x_c = build_systeme(N_coarse)
A_f, b_f, x_f = build_systeme(N_fine)

# Question 3, 4 et 5
# Résolution directe sur petit maillage (coarse)
T_coarse = np.linalg.solve(A_c, b_c)

# Interpolation du coarse vers fine
# Newton
a_newt = newton_coeff(x_c, T_coarse)
T_init_newt = newton_eval(x_c, a_newt, x_f)
# Lagrange
T_init_lagr = lagrange_interp(x_c, T_coarse, x_f)

# Questions 4 et 5
# Jacobi avec départ à 0 
T_jac_zero, it_jac_zero = jacobi(A_f, b_f, np.zeros_like(b_f))
# Jacobi avec depart sur solution interpolee par Newton ou par Lagrange  
T_jac_newt, it_jac_newt = jacobi(A_f, b_f, T_init_newt)
T_jac_lagr, it_jac_lagr = jacobi(A_f, b_f, T_init_lagr)

# Gauss-Seidel avec départ à 0 
T_gs_zero, it_gs_zero = gauss_seidel(A_f, b_f, np.zeros_like(b_f))
# Gauss-Seidel avec depart sur solution interpolee par Newton ou par Lagrange 
T_gs_newt, it_gs_newt = gauss_seidel(A_f, b_f, T_init_newt)
T_gs_lagr, it_gs_lagr = gauss_seidel(A_f, b_f, T_init_lagr)

# Résultats
print(f"Jacobi init 0      : {it_jac_zero} iterations")
print(f"Jacobi Newton      : {it_jac_newt} iterations")
print(f"Jacobi Lagrange    : {it_jac_lagr} iterations")
print(f"GS init 0          : {it_gs_zero} iterations")
print(f"GS Newton          : {it_gs_newt} iterations")
print(f"GS Lagrange        : {it_gs_lagr} iterations")

# Visualisation
plt.figure(figsize=(8,5))
plt.plot(x_f, T_analytique(x_f), 'k--', label='Exacte')
plt.plot(x_f, T_jac_zero, 'r-', label='Jacobi init=0')
plt.plot(x_f, T_jac_newt, 'b-', label='Jacobi Newton')
plt.plot(x_f, T_jac_lagr, 'g-', label='Jacobi Lagrange')
plt.plot(x_f, T_gs_zero, 'r--', label='GS init=0')
plt.plot(x_f, T_gs_newt, 'b--', label='GS Newton')
plt.plot(x_f, T_gs_lagr, 'g--', label='GS Lagrange')
plt.plot(x_c, T_coarse, 'm-o', label='Solution coarse')
plt.xlabel('x')
plt.ylabel('T(x)')
plt.grid(True)
plt.legend()
plt.title('Jacobi et Gauss-Seidel avec différentes initialisations')
plt.show()
