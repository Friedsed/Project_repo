# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:58:45 2025

@author: aberg
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
# Question 2 
# expliquer l'algorithme de calcul des coefficients 
#______________________________________________________________________________________________________________________________________________
# L’algorithme prend les coordonnées des points sous forme de différents tableaux :
# un tableau pour les x (x_data) et un tableau pour les y (y_data).
# Il initialise n, qui est la longueur du tableau x_data.
# Il copie y_data dans un nouveau tableau a en le convertissant en float.
# Ensuite, pour chaque k allant de 1 à n-1, il met à jour les valeurs de a
# de l’indice k jusqu’à n en utilisant la formule des différences divisées.
# Pour le numérateur, il utilise le tableau a car c’est le tableau des y,
# et pour le dénominateur, il utilise les différences des x_data.

#_______________________________________________________________________________________________________________________________________________
# de la fonction
    n = len(x_data)
    a = np.copy(y_data).astype(float)
    for k in range(1, n):
        a[k:n] = (a[k:n] - a[k-1:n-1]) / (x_data[k:n] - x_data[0:n-k]) 
    return a

def newton_eval(x_data, a, x_eval):
    P = np.zeros_like(x_eval)
    for idx, x in enumerate(x_eval):     ## elle permet de parcourir un tableau x_eval en récupérant à la fois l’indice idx et la valeur x.
# Initialise val à a[n-1]
        val = a[-1]
# Boucle commence a k=n-2 (avant dernier indice), 
# finie à -1 donc à k=0, par pas de -1
        for k in range(len(a)-2, -1, -1):                       # Question 2 : compléter le calcul de val par Horner
            val = a[k]+ (x - x_data[k]) * val
        P[idx] = val 
    return P

# -----------

# Laplacien 1D
L = 2*np.pi
lambda_ = 1.0

def T_analytique(x): return  np.cos(x)
def RHS_analytique(x): return -np.cos(x)

def build_systeme(N):
# Question 0
# Attention : A la difference du TP1 ou la construction de la 
# matrice est dans le programme principal,² elle est ici dans 
# une fonction afin de pouvoir en construire plusieurs 
    dx = L / N
    x = np.linspace(0, L, N+1)
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
# Compléter la construction de A (voir TP1)
    A[0,0], b[0] = 1, T_analytique(x[0])  
    A[N,N], b[N] = 1, T_analytique(x[-1])
    for i in range(1, N):
        A[i,i-1] =-lambda_ / dx**2
        A[i,i]   = 2 * lambda_ / dx**2
        A[i,i+1] = - lambda_ / dx**2
        b[i] = RHS_analytique(x[i])
    return A, b, x

# Jacobi
def jacobi(A, b, T0, tol=1e-6, maxit=100000):
    T = T0.copy()
    T[0], T[-1] = b[0], b[-1]  
    for k in range(maxit):
        T_last = T0.copy()
        for i in range(1, len(b)-1):
            T_last[i] = (b[i]- A[i, i-1] * T[i-1]  - A[i, i+1] * T[i+1]) / A[i, i]
        # test de convergence
        if np.linalg.norm(T_last - T, ord=2) < tol:
            break
        T = T_last
# Question 0
# Compléter la fonction à partir du TP1
# Attention : A la difference du TP1, la valeur de l'iteration 0 est 
# en argument de la fonction

    return T, k+1

# Gauss-Seidel
def gauss_seidel(A, b, T0, tol=1e-6, maxit=10000):
    T = T0.copy()
    T[0], T[-1] = b[0], b[-1]
    for k in range(maxit):
        T_last = T.copy()
        for i in range(1, len(b)-1):
            T[i] = (b[i]- A[i, i-1] * T[i-1]  - A[i, i+1] * T[i+1]) / A[i, i]
        # test de convergence
        if np.linalg.norm(T - T_last, ord=2) < tol:
            break
# Question 0
# Compléter la fonction a partir du TP1
# Attention : A la difference du TP1, la valeur de l'iteration 0 est 
# en argument de la fonction
    return T, k+1

# Maillage grossier et fin
N_coarse = 10
N_fine = 80

# Constructions des jeux de matrices, membres de droite et 
# maillage grossier et fin.

A_c, b_c, x_c = build_systeme(N_coarse)
A_f, b_f, x_f = build_systeme(N_fine)

# Question 3, 4 et 5
# Completer la resolution du systeme par methode directe 
# sur petit maillage
T_coarse =np.linalg.solve(A_c, b_c)

# Question 3, 4 et 5
# Completer le calcul de T_coarse par interpolation de  Newton 
# et de Lagrange du maillage grossier x_c vers le maillage fin x_f 
a_newt =  newton_coeff(x_c, T_coarse)
T_init_newt =  newton_eval(x_c, a_newt, x_f)
T_init_lagr = lagrange_interp(x_c, T_coarse, x_f)

# Questions 4 et 5
# Jacobi avec depart à 0 
T_jac_zero, it_jac_zero = jacobi(A_f, b_f, np.zeros_like(b_f))
# Jacobi avec depart sur solution interpolee par Newton ou par 
# Lagrange  
T_jac_newt, it_jac_newt = jacobi(A_f, b_f, T_init_newt)
T_jac_lagr, it_jac_lagr = jacobi(A_f, b_f, T_init_lagr)

# Questions 4 et 5
# Gauss-Seidel avec départ à 0 
T_gs_zero, it_gs_zero = gauss_seidel(A_f, b_f, np.zeros_like(b_f))
# Gauss-Seidel avec depart sur solution interpolee par Newton ou par 
# Lagrange 
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
plt.plot(x_c, T_coarse, 'm-o', label='Solution coarse')  # <-- ajout
plt.xlabel('x')
plt.ylabel('T(x)')
plt.grid(True)
plt.legend()
plt.title('Jacobi et Gauss-Seidel avec différentes initialisations')
plt.show()