"""
AUTHOR:             FRIEDLY WOLI
Date:               MADE AT HOME SATURDAY THE 19 OF APRIL 2026 


Exam paper:         2.29 NUMERICAL FLUID MECHANICS - SPRING 2015
University:         FROM MASSACHUSSET INSTITUTE OF THECHNOLIGY 
More detail:        PROBLEM SET 1 DUE TO Monday , February 23, 2015


"""
"""

import numpy as np
import matplotlib.pyplot as plt 


#----------------------------------------------------------
# Termes sources 
#-------------------------------------------------------------------


#----------------------------------------------------------------------
# n from 1 to 20
#----------------------------------------------------------------------


def T_ex (N,T1,x,y,a,b):
    T=0
    for n in range (N):

        T = T + (  np.sin(2*n-1)*  (np.pi*x/a) * np.sinh(2*n-1)*(np.pi*y/a) ) /  ( (2*n-1)*np.sinh((2*n-1 ) * np.pi*b/a) )

    return ( 4*T1/ np.pi )*T

def sol(N, Nx, Ny, T1, a, b):

    T = [[0 for i in range(Nx)] for j in range(Ny)]

    for i in range(Nx):
        for j in range(Ny):

            if j == Ny - 1:
                T[j][i] = 80

            elif i == 0 or j == 0 or i == Nx - 1:
                T[j][i] = 0

            else:
                x = i * a / (Nx - 1)
                y = j * b / (Ny - 1)
                T[j][i] = T_ex(N, T1, x, y, a, b)

    return T






#----------------------------------------------------------------------
# n from 20 to 1
#----------------------------------------------------------------------

def T_ex2 (N,T1,x,y,a,b):
    T=0
    n=N
    while n >=  1:

        T = T + (  np.sin(2*n-1)*  (np.pi*x/a) * np.sinh(2*n-1)*(np.pi*y/a) ) /  ( (2*n-1)*np.sinh((2*n-1 ) * np.pi*b/a) )
        n = n-1

    return ( 4*T1/ np.pi )*T



def sol2(N, Nx, Ny, T1, a, b):

    T = [[0 for i in range(Nx)] for j in range(Ny)]

    for i in range(Nx):
        for j in range(Ny):

            if j == Ny - 1:
                T[j][i] = 80

            elif i == 0 or j == 0 or i == Nx - 1:
                T[j][i] = 0

            else:
                x = i * a / (Nx - 1)
                y = j * b / (Ny - 1)
                T[j][i] = T_ex2(N, T1, x, y, a, b)

    return T


def erro(T1, T2, Nx, Ny ):
    err= [[ 0 for i in range (Nx)] for j in range (Ny)]
    for i in range (Nx):
        for j in range (Ny):
            err[j][i]= np.abs ( T1[j][i] - T2[j][i] )
    
    return err


#-----------------------------------------------------------------------
# Parametre physique 
#---------------------------------------------------------------------
a = 5
b= 4
Nx=40
Ny=32
N=20
T1=80


x = np.linspace(0, a, Nx)
y = np.linspace(0, b, Ny)

T = sol(N, Nx, Ny, T1, a, b)
T2 = sol2(N, Nx, Ny, T1, a, b)
er = erro(T, T2, Nx, Ny )


X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots()
pc = ax.pcolormesh(X,Y,T )
fig.colorbar(pc)
plt.xlabel("x")
plt.ylabel("y")


fig2, ax2 = plt.subplots()
pc2 = ax2.pcolormesh(X,Y,T2 )
fig2.colorbar(pc2)
plt.xlabel("x")
plt.ylabel("y")

fig3, ax3 = plt.subplots()
pc3 = ax3.pcolormesh(X,Y, er )
fig3.colorbar(pc3)
plt.xlabel("x")
plt.ylabel("y")


plt.show()

"""





""" 
ANSWERS 


"""


#  GOOD WAY OF PROGRAMMING




import numpy as np
import matplotlib.pyplot as plt

# =========================
# Solution exacte (vectorisée)
# =========================
def T_ex(N, T1, x, y, a, b):
    n = np.arange(1, N+1)  # n = 1..N
    k = 2*n - 1

    term = (
        np.sin(k * np.pi * x / a) *
        np.sinh(k * np.pi * y / a) /
        (k * np.sinh(k * np.pi * b / a))
    )

    return (4 * T1 / np.pi) * np.sum(term)


# =========================
# Grille + solution
# =========================
def sol(N, Nx, Ny, T1, a, b):
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)

    T = np.zeros((Ny, Nx))

    # Conditions limites
    T[-1, :] = T1   # haut
    T[:, 0] = 0     # gauche
    T[:, -1] = 0    # droite
    T[0, :] = 0     # bas

    # intérieur
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            T[j, i] = T_ex(N, T1, X[j, i], Y[j, i], a, b)

    return X, Y, T


# =========================
# Version inversée (somme descendante)
# =========================
def T_ex2(N, T1, x, y, a, b):
    n = np.arange(N, 0, -1)
    k = 2*n - 1

    term = (
        np.sin(k * np.pi * x / a) *
        np.sinh(k * np.pi * y / a) /
        (k * np.sinh(k * np.pi * b / a))
    )

    return (4 * T1 / np.pi) * np.sum(term)


def sol2(N, Nx, Ny, T1, a, b):
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)

    T = np.zeros((Ny, Nx))

    T[-1, :] = T1
    T[:, 0] = 0
    T[:, -1] = 0
    T[0, :] = 0

    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            T[j, i] = T_ex2(N, T1, X[j, i], Y[j, i], a, b)

    return X, Y, T


# =========================
# Erreur (vectorisée)
# =========================
def erro(T1, T2):
    return np.abs(T1 - T2)


# =========================
# Paramètres
# =========================
a, b = 5, 4
Nx, Ny = 40, 32
N = 20
T1 = 80

# calcul
X, Y, T = sol(N, Nx, Ny, T1, a, b)
X, Y, T2 = sol2(N, Nx, Ny, T1, a, b)
er = erro(T, T2)


# =========================
# Plots
# =========================
fig, ax = plt.subplots()
pc = ax.pcolormesh(X, Y, T, shading='auto')
fig.colorbar(pc)
ax.set_title("Solution (somme croissante)")

fig2, ax2 = plt.subplots()
pc2 = ax2.pcolormesh(X, Y, T2, shading='auto')
fig2.colorbar(pc2)
ax2.set_title("Solution (somme décroissante)")

fig3, ax3 = plt.subplots()
pc3 = ax3.pcolormesh(X, Y, er, shading='auto')
fig3.colorbar(pc3)
ax3.set_title("Erreur")

plt.show()


