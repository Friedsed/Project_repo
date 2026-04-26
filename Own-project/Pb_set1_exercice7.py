"""
AUTHOR:             FRIEDLY WOLI
Date:               MADE AT HOME SATURDAY THE 19 OF APRIL 2026 


Exam paper:         2.29 NUMERICAL FLUID MECHANICS - SPRING 2015
University:         FROM MASSACHUSSET INSTITUTE OF THECHNOLIGY 
More detail:        PROBLEM SET 1 DUE TO Monday , February 23, 2015


"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr


"""

def cgs(V):
    Vn= V.copy() # here we have the qi
   
    M, N = V.shape
    U= Q = R = np.zeros((M,N))

    for n in range (N):
        RQ=np.zeros(0)
        for i in range (n-1):
            
            RQ = RQ + Vn[ :, i ] @  V[ :, n] @ Vn [ :, i]  

        U[ :, n] = Vn[ :, n] - RQ
        Vn[ :, n] = U[ :, n] / np.sqrt( U[ :, n] @ U [ :, n ]) 


    return Vn

    
    def mgs (V):
    M, N = V.shape
    for k in range (1 , N-1):
        Vn= np.zeros((M, N))
        Uold= V[ : , k ]

        for a in range (k):
            r= Uold @ Vn[ :, a] 
            Unew= Uold - r * Vn[ :, a ]

            if a==k-1 :
                r= Unew @ Unew
                Vn[ : , k ] = Unew / r

    return Vn

"""
def cgs(V):
    Vn = V.copy()
    M, N = V.shape
    U = np.zeros((M, N), dtype=float)
    Q = np.zeros((M, N), dtype=float)
    R = np.zeros((N, N), dtype=float)

    for n in range(N):
        U[:, n] = V[:, n]

        for i in range(n):
            R[i, n] = Q[:, i] @ V[:, n]
            U[:, n] = U[:, n] - R[i, n] * Q[:, i]

        R[n, n] = np.linalg.norm(U[:, n])
        Q[:, n] = U[:, n] / R[n, n]

    return Q, R


def mgs(V):
    M, N = V.shape
    Vn = np.zeros((M, N), dtype=float)

    for k in range(N):
        Uold = V[:, k].astype(float).copy()

        for a in range(k):
            r = Uold @ Vn[:, a]
            Uold = Uold - r * Vn[:, a]

        norme = np.linalg.norm(Uold)
        if np.isclose(norme, 0.0):
            raise ValueError("Les colonnes de V sont linéairement dépendantes.")

        Vn[:, k] = Uold / norme

    return Vn


#
m= 10
np.random.seed(100)

A,_ = qr(np.random.rand(m,m))
B,_ = qr(np.random.rand(m,m))
n = np.arange(1, m+1)
S=np.diag( 2.0**(-n)) 
V= A @ S @ B.T 

print("La matrice V est ", V)

print()
print()

print("la matrice mgs V est ", mgs(V))

print()
print()

Q, R = cgs(V)
print("la matrice Q de cgs est ",  Q)

print()
print()

print("La matrice R de cgs est ",  R)


