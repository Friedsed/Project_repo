# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:25:37 2025

@author: ABergeon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----------------------
# Physical parameters
# ----------------------
L = 2.0 * np.pi
lambda_ = 1.0

# Analytical solution
def T_analytique(x):
    return np.sin(x)

# RHS
def RHS_analytique(x):
    return np.sin(x)

# ----------------------
# Iterative methods
# ----------------------_________________________________________________________________________________________________
def sol_T(T,N):
        t=0
        for j in range(1,N):
            t= T[j]+t
        t=t+ (T[0]+T[N])/2
        t=t/N
        return t

def jacobi(A, b, tol, maxit, N):
    x = np.zeros_like(b)
    x[0], x[-1] = b[0], b[-1]

    for k in range(maxit):
        x_new = x.copy()
        for i in range(1, N):
            x_new[i] = (b[i]
                        - A[i, i-1] * x[i-1]
                        - A[i, i+1] * x[i+1]) / A[i, i]

        if np.linalg.norm(x_new - x, ord=2) < tol:
            break
        x = x_new

    
    t=sol_T(x,N)
    for j in range(N):
        x[j]=x[j] - t

    return x, k + 1


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
            break

    t=sol_T(x,N)
    for j in range(N):
        x[j]=x[j] - t

    return x, k + 1




# ----------------------_____________________________________________________________________________________________________
# Main loop
# ----------------------
N_values = list(range(10, 82, 2))
err_numpy, err_jacobi, err_gs = [], [], []
iter_jacobi, iter_gs = [], []

for N in N_values:

    dx = L / N
    x = np.linspace(0, L, N + 1)

    f = RHS_analytique(x)
    dg = T_analytique(x[0])
    dd = T_analytique(x[-1])

    # ----------------------
    # Linear system
    # ----------------------
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # Boundary rows
    A[0, 0], A[0, 1] = 2/dx**2, 2/dx**2
    A[N, N], A[N, N-1] = -2*lambda_/dx**2  ,    2*lambda_/dx**2
    b[0], b[N] = f[0]+ 2*dg / dx   ,    f[N]+2*lambda_*dd/dx

    # Interior points
    for i in range(1, N):
        A[i, i-1] = -lambda_ / dx**2
        A[i, i]   =  2 * lambda_ / dx**2
        A[i, i+1] = -lambda_ / dx**2
        b[i] = f[i]

    C=A.copy()

    # ----------------------
    # Solutions
    # ----------------------
    tol = 1e-7

    T_numpy = np.linalg.solve(A, b)
    T_jac, it_jac = jacobi(A, b, tol, 100000, N)
    T_gs, it_gs = gauss_seidel(A, b, tol, 100000, N)

    T_exact = T_analytique(x)

    # Errors
    err_numpy.append(np.linalg.norm(T_exact - T_numpy, ord=np.inf))
    err_jacobi.append(np.linalg.norm(T_exact - T_jac, ord=np.inf))
    err_gs.append(np.linalg.norm(T_exact - T_gs, ord=np.inf))

    # Iterations
    iter_jacobi.append(it_jac)
    iter_gs.append(it_gs)

    print(f"N={N:3d} | Jacobi it={it_jac:5d} | GS it={it_gs:5d}")

# ----------------------
# Plots
# ----------------------
print("Len of A", len(C))
print("The Jacobi solution is  :", T_jac)
print("The Gauss-Seidel solution is  :", T_gs)  
print("Verification de la normalisation avec Jacobi :", sol_T(T_jac,N))
print("Verification de la normalisation avec Gauss-Seidel :", sol_T(T_gs,N))
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

# ----------------------
# Slopes (log-log)
# ----------------------
logN = np.log(N_values)

slope_np, _, _, _, _ = linregress(logN, np.log(err_numpy))
slope_jac, _, _, _, _ = linregress(logN, np.log(err_jacobi))
slope_gs, _, _, _, _ = linregress(logN, np.log(err_gs))

print(f"Pente (Directe)      : {slope_np:.2f}")
print(f"Pente (Jacobi)       : {slope_jac:.2f}")
print(f"Pente (Gauss-Seidel) : {slope_gs:.2f}")
