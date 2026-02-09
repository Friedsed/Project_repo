# -*- coding: utf-8 -*-
"""
TP3 – Interpolation + accélération Jacobi/GS (Questions 5 et 6)
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# ----------------------
# Newton
# ----------------------
def newton_coeff(x_data, y_data):
    """
    Calcul des coefficients de Newton par différences divisées.
    """
    n = len(x_data)
    a = np.copy(y_data).astype(float)
    for k in range(1, n):
        a[k:n] = (a[k:n] - a[k-1:n-1]) / (x_data[k:n] - x_data[0:n-k])
    return a

def newton_eval(x_data, a, x_eval):
    """
    Évaluation du polynôme de Newton par Horner.
    """
    P = np.zeros_like(x_eval, dtype=float)
    for idx, x in enumerate(x_eval):
        val = a[-1]
        for k in range(len(a)-2, -1, -1):
            val = a[k] + (x - x_data[k]) * val
        P[idx] = val
    return P

# ----------------------
# Laplacien 1D (Dirichlet ici, comme TP2)
# ----------------------
L = 2*np.pi
lambda_ = 1.0

def T_exact(x):
    return -np.cos(x)

def RHS_analytique(x):
    return -np.cos(x)

def build_systeme(N):
    dx = L / N
    x = np.linspace(0, L, N+1)
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    Tg = T_exact(x[0])
    Td = T_exact(x[-1])
    f = RHS_analytique(x)

    # Conditions de Dirichlet
    A[0,0], b[0] = 1.0, Tg
    A[N,N], b[N] = 1.0, Td

    # Points intérieurs
    for i in range(1, N):
        A[i,i-1] = -lambda_/(dx**2)
        A[i,i]   =  2*lambda_/(dx**2)
        A[i,i+1] = -lambda_/(dx**2)
        b[i]     = f[i]
    return A, b, x

# ----------------------
# Jacobi / Gauss-Seidel avec T0 en argument
# ----------------------
def jacobi(A, b, T0, tol=1e-6, maxit=100000):
    T = T0.copy()
    T[0], T[-1] = b[0], b[-1]
    for k in range(maxit):
        T_new = T.copy()
        for i in range(1, len(b)-1):
            T_new[i] = (b[i]
                        - A[i,i-1]*T[i-1]
                        - A[i,i+1]*T[i+1]) / A[i,i]
        if np.linalg.norm(T_new - T, ord=2) < tol:
            T = T_new
            break
        T = T_new
    return T, k+1

def gauss_seidel(A, b, T0, tol=1e-6, maxit=100000):
    T = T0.copy()
    T[0], T[-1] = b[0], b[-1]
    for k in range(maxit):
        T_old = T.copy()
        for i in range(1, len(b)-1):
            T[i] = (b[i]
                    - A[i,i-1]*T[i-1]
                    - A[i,i+1]*T_old[i+1]) / A[i,i]
        if np.linalg.norm(T - T_old, ord=2) < tol:
            break
    return T, k+1

# ----------------------
# Question 6 : maillages 10 -> 80, pas 10
# ----------------------
N_min = 10
N_max = 80
step  = 10
tol   = 1e-6

time_interp_newt = 0.0

time_jac_newt    = 0.0

time_gs_newt     = 0.0

T_prev = None
x_prev = None

for N in range(N_min, N_max + 1, step):
    print(f"\n=== Maillage N = {N} ===")
    A, b, x = build_systeme(N)

    if N == N_min:
        # départ champ nul
        T0 = np.zeros_like(b)

        # on calcule une solution convergée (par ex. Jacobi)
        T_jac, it_jac = jacobi(A, b, T0, tol=tol)
        print(f"Jacobi N={N} init 0 : {it_jac} itérations")

        # on garde cette solution comme "coarse" pour N=20
        T_prev = T_jac.copy()
        x_prev = x.copy()

    else:
        # interpolation du maillage précédent (x_prev, T_prev) vers x

        # --- interpolation Newton ---
        t0 = time.perf_counter()
        a_newt = newton_coeff(x_prev, T_prev)
        T_init_newt = newton_eval(x_prev, a_newt, x)
        t1 = time.perf_counter()
        time_interp_newt += (t1 - t0)

       
        # --- Jacobi avec init Newton ---
        t0 = time.perf_counter()
        T_jac_newt, it_jac_newt = jacobi(A, b, T_init_newt, tol=tol)
        t1 = time.perf_counter()
        time_jac_newt += (t1 - t0)

       

        # --- GS avec init Newton ---
        t0 = time.perf_counter()
        T_gs_newt, it_gs_newt = gauss_seidel(A, b, T_init_newt, tol=tol)
        t1 = time.perf_counter()
        time_gs_newt += (t1 - t0)


        print(f"Jacobi N={N} Newton   : {it_jac_newt} itérations")
        print(f"GS     N={N} Newton   : {it_gs_newt} itérations")


        # on peut choisir par exemple la solution GS-Newton comme nouvelle coarse
        T_prev = T_gs_newt.copy()
        x_prev = x.copy()

# ----------------------
# Bilan des temps
# ----------------------
print("\n=== Temps totaux ===")
print(f"Temps interpolation Newton   : {time_interp_newt:.6f} s")
print(f"Temps Jacobi (init Newton)   : {time_jac_newt:.6f} s")
print(f"Temps GS (init Newton)       : {time_gs_newt:.6f} s")

