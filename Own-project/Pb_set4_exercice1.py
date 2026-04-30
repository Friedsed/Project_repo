"""
AUTHOR:             FRIEDLY WOLI
Date:               MADE AT HOME SATURDAY THE 30 OF APRIL 2026 


Exam paper:         2.29 NUMERICAL FLUID MECHANICS - SPRING 2015
University:         FROM MASSACHUSSET INSTITUTE OF THECHNOLIGY 
More detail:        PROBLEM SET 1 DUE TO Monday , February 23, 2015


"""


#  code not end need to be revise 
# last modifacatiion 29 of April 2026 at 19h59 min







import numpy as np
import matplotlib.pyplot as plt


def Sol_num(D1, D2, D3, eps, ksi, Cold, Nx):

    d1 = D1 - D2
    d2 = 1 - 2 * D1 + D2 - D3
    d3 = D1

    C = np.zeros(Nx)

    # Inlet boundary (i = 0)
    C[0] = (Cold[2] * d1 + Cold[1] * d2 + Cold[0] * d3 + eps) * ksi

    # Interior points
    for i in range(1, Nx - 1):
        C[i] = Cold[i+1] * d1 + Cold[i] * d2 + Cold[i-1] * d3

    # Outlet boundary
    C[Nx-1] = C[Nx-2]

    return C


def Gen_sol(D1, D2, D3, eps, ksi, Nx, Nt):

    C = np.zeros((Nt, Nx))

    # Initial condition already zero → OK

    for i in range(1, Nt):
        Cold = C[i-1, :]
        C[i, :] = Sol_num(D1, D2, D3, eps, ksi, Cold, Nx)

    return C   #  VERY IMPORTANT


# Parameters
LTmax = 6000
Nt = 2000         # increase for stability
Nx = 50           # increase spatial resolution

L = 10
kapa = 1.7
U = 0.02
k = 0.0002        # corrected (your value was wrong)

dt = LTmax / Nt
dx = L / (Nx - 1)

D1 = kapa * dt / dx**2
D2 = U * dt / dx
D3 = k * dt

ksi = 1 / ((U * dx / kapa) + 1)
eps = U * dx*100 / kapa

x = np.linspace(0, L, Nx)
t = np.linspace(0, LTmax, Nt)
X, T = np.meshgrid(x, t)

C = Gen_sol(D1, D2, D3, eps, ksi, Nx, Nt)

plt.figure()
cf = plt.contourf(X, T, C, 50)
plt.colorbar(cf, label="Concentration")
plt.xlabel("x (m)")
plt.ylabel("t (s)")
plt.title("Advection-Diffusion-Reaction")
plt.show()