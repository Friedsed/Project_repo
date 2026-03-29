# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:25:37 2025

@author: ABergeon
"""

# Plan general du code :


#       i==0
#  ----------------------
# Méthodes itératives

## les fonctions jacobi et gauss_seidel


# Boucle principale

## remplissage de la matrice A et du vecteur b
## appel des méthodes directes et itératives
## calcul des erreurs
## conservation des erreurs et du nombre d'itérations pour chaque N
## conservation du nombre d'itérations pour chaque N
## affichage pour chaque N le nombre de points, le nombre d'itérations pour Jacobi et Gauss-Seidel


# Graphiques

## Courbes des erreurs en fonction de N le nombre de points donc la dimension du systeme
## Courbes du nombre d'itérations en fonction de N le nombre de points donc la dimension du systeme
## Calcul des pentes log-log a bas des erreurs et des N par chaque méthode


# NB : Le code ne resout pas des equations par ces methodes car il manqueras des imformations comme: 
# comment entrer la matrice A et le vecteur b par l utilisateur ce qui est pas pris en compte ici. 
# je compte ecrire ce code pour la resolution qui demanderas la matrice A et le vecteur b à l utilisateur puis que sort les solutions dans un autre code.
# ----------------.

#       i==0



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

i=0  # Choisir 0 pour exécuter le code principal, 1 pour exécuter une autre partie si nécessaire

if (i==0):
        

    # Paramètres physiques
    L = 2.0 * np.pi
    lambda_ = 1.0


    # Solution analytique 
    def T_analytique(x):
        return np.cos(x)


    # Membre de droite
    def RHS_analytique(x):
        return np.cos(x)


    # ----------------------
    # Méthodes itératives
    # ----------------------
    def jacobi(A, b, tol, maxit):
        x_j = np.zeros_like(b)
        x_j[0], x_j[-1] = b[0], b[-1]
        for k in range(maxit):
            x_new = x_j.copy()
            for i in range(1, len(b)-1):
                # Question 5 : Compléter la ligne ci-dessous
                x_new[i] = (b[i] - A[i,i-1]*x_j[i-1] - A[i,i+1]*x_j[i+1]) / A[i,i]
            if np.linalg.norm(x_new - x_j, ord=2) < tol:
                break
            x_j = x_new
        return x_j, k+1


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


    # ----------------------
    # Boucle principale
    # ----------------------
    N_values = list(range(10, 82, 2))
    err_numpy, err_jacobi, err_gs = [], [], []
    iter_jacobi, iter_gs = [], []
   
    for N in N_values:
        # Question 1. Compléter
        dx = L / N
        x = np.linspace(0, L, N+1)     # constrution de vecteur x qui est d inconnues du systeme d equations

        f = RHS_analytique(x)
        Tg = T_analytique(x[0])  
        Td = T_analytique(x[-1])
        
        # Construction du système linéaire 
        # Question 3 : Compléter la construction de A et b
        A = np.zeros((N+1, N+1))
        b = np.zeros(N+1)
        A[0,0], b[0] = 1.0, Tg
        A[N,N], b[N] = 1.0, Td
        
        for i in range(1, N):
            A[i,i-1] = -lambda_ / dx**2
            A[i,i]   = 2 * lambda_ / dx**2
            A[i,i+1] = -lambda_ / dx**2
            b[i] = f[i]
        
        # Solutions
        tol = 1e-7
        # Question 4 : je calcule Jacobi et Gauss-Seidel en plus du nb d'iteration avec la matrice A et le vecteur b construits que je viens de construire
        T_numpy = np.linalg.solve(A, b)
        T_jac, it_jac = jacobi(A, b, tol, 100000)
        T_gs, it_gs = gauss_seidel(A, b, tol, 100000)
        
        # Solution exacte
        T_exact = T_analytique(x)
        
        # Erreurs  : je calcule les erreurs et je conserve dans un tableau pour ensuite faire des graphiques
        err_numpy.append(np.linalg.norm(T_exact - T_numpy, ord=np.inf))
        err_jacobi.append(np.linalg.norm(T_exact - T_jac, ord=np.inf))
        err_gs.append(np.linalg.norm(T_exact - T_gs, ord=np.inf))
        
        # Itérations     : je conserve dans un tableau les iterations pour faire des graphiques
        iter_jacobi.append(it_jac)
        iter_gs.append(it_gs)
        
        print(f"N={N:3d} | Jacobi it={it_jac:5d} | GS it={it_gs:5d}")

       

    # ----------------------
    # Graphiques
    # ----------------------
    # Erreur en fonction de N
    plt.figure(figsize=(8,6))
    plt.loglog(N_values, err_numpy, 'k-o', label='Direct (numpy)')
    plt.loglog(N_values, err_jacobi, 'r-s', label='Jacobi')
    plt.loglog(N_values, err_gs, 'b-^', label='Gauss-Seidel')
    plt.xlabel('N')
    plt.ylabel('Erreur ||T - T_exact||_∞')
    plt.title('Erreur en norme sup en fonction de N')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.show()

    # Nombre d'itérations en fonction de N
    plt.figure(figsize=(8,6))
    plt.plot(N_values, iter_jacobi, 'r-s', label='Jacobi')
    plt.plot(N_values, iter_gs, 'b-^', label='Gauss-Seidel')
    plt.xlabel('N')
    plt.ylabel('Nombre d \'itérations jusqu\'à convergence')
    plt.title('Convergence de Jacobi et Gauss-Seidel en fonction de N')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----------------------
    # Calcul des pentes log-log
    # ----------------------
    logN = np.log(N_values)
    slope_jac, _, _, _, _ = linregress(logN, np.log(err_jacobi))
    slope_gs, _, _, _, _ = linregress(logN, np.log(err_gs))
    slope_np, _, _, _, _ = linregress(logN, np.log(err_numpy))
    print(f"Pente (Jacobi)       : {slope_jac:.2f}")
    print(f"Pente (Gauss-Seidel) : {slope_gs:.2f}")
    print(f"Pente (Directe)      : {slope_np:.2f}")






if (i==1):
    # resolution tp 2
    print ("Résolution TP2" )
