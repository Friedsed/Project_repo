
"""
AUTHOR:             FRIEDLY WOLI
Date:               MADE AT HOME SATURDAY THE 25 OF APRIL 2026 


Exam paper:         2.29 NUMERICAL FLUID MECHANICS - SPRING 2015
University:         FROM MASSACHUSSET INSTITUTE OF THECHNOLIGY 
More detail:        PROBLEM SET 1 DUE TO Monday , February 23, 2015


"""



"""
import numpy as np
import matplotlib.pyplot as plt



def Sol_num (D1, D2, D3, eps, ksi, Cold, Nx):

    d1=D1-D2
    d2= 1 - 2 * D1 + D2 - D3
    d3= D1

    C = np.arange ((Nx))
    C[0] = ( Cold[2] * d1 + Cold[1] * d2 + Cold[0] * d3 + eps ) * ksi  

    for i in range (1 , Nx - 2):
        C[i] = Cold[i+1] * d1 + Cold[i] * d2 + Cold[i-1] * d3

    C[Nx-1] = C[Nx-2]

    return C



def Gen_sol ( D1, D2, D3, eps, ksi, Nx, Nt ):

    C= np.zeros (( Nt, Nx )     )             # here the ligne is the time and the rows are the sapce cordinate x 
                           # boundary condition at t=0     here the ligne represent the time en the rows the x space cordinate 
    


    for i in range (1, Nt):

        Cold = C[ i-1 , : ]
        C[i ,:  ] = Sol_num (D1, D2, D3, eps, ksi, Cold, Nx)




# Parameters 

LTmax = 6000                # LTmax mean the lenght on the time like the maximum time
Nt = 60                        # Nt the numbres of nodes for the time
Nx = 10
L = 10                      # L for the lenght of the x 
kapa = 1.7 
U = 0.02
k = 0.0025

dt = LTmax / Nt
dx = L / Nx

D1 = kapa *dt / dx**2
D2 = U * dt / dx
D3 = k * dt

ksi= 1 / ( ( U * dx / kapa) + 1 )
eps= U* dx / kapa


x = np.linspace ( 0, L, Nx )
t= np.linspace ( 0, LTmax, Nt)
X, T = np.meshgrid ( x, t )

C = Gen_sol ( D1, D2, D3, eps, ksi, Nx, Nt )


plt.figure()

cf= plt.contour(X, T, C, 50)

plt.colorbar(cf, label ="the concentration ")

plt.show ()










"""




import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Numerical solver (stable form)
# -----------------------------
def Sol_num(D1, D2, D3, Cold, Nx, dx, kapa, U):

    C = np.zeros(Nx)
    cin = 100  # inlet concentration

    # Inlet boundary (stable formulation)
    C[0] = (U * cin + (kapa/dx) * Cold[1]) / (U + kapa/dx)

    # Interior points
    for i in range(1, Nx - 1):
        diffusion = D1 * (Cold[i+1] - 2*Cold[i] + Cold[i-1])
        advection = -D2 * (Cold[i] - Cold[i-1])
        reaction = -D3 * Cold[i]

        C[i] = Cold[i] + diffusion + advection + reaction

    # Outlet boundary (Neumann)
    C[-1] = C[-2]

    return C


# -----------------------------
# Time integration
# -----------------------------
def Gen_sol(D1, D2, D3, Nx, Nt, dx, kapa, U):

    C = np.zeros((Nt, Nx))  # time × space

    for n in range(1, Nt):
        Cold = C[n-1, :]
        C[n, :] = Sol_num(D1, D2, D3, Cold, Nx, dx, kapa, U)

        # Safety check (detect blow-up early)
        if np.max(np.abs(C[n, :])) > 1e6:
            print(f"❌ Blow-up at time step {n}")
            return C

    return C


# -----------------------------
# Parameters
# -----------------------------
LTmax = 6000

Nx = 80          # space points
Nt = 300000      # time steps (important for stability)

L = 10
kapa = 1.7
U = 0.02
k = 0.0002

dx = L / (Nx - 1)
dt = LTmax / Nt

# Dimensionless numbers
D1 = kapa * dt / dx**2
D2 = U * dt / dx
D3 = k * dt

print("D1 (diffusion) =", D1)
print("D2 (advection) =", D2)

# -----------------------------
# Grid
# -----------------------------
x = np.linspace(0, L, Nx)
t = np.linspace(0, LTmax, Nt)

# -----------------------------
# Solve
# -----------------------------
C = Gen_sol(D1, D2, D3, Nx, Nt, dx, kapa, U)

# -----------------------------
# Check before plotting
# -----------------------------
if np.isnan(C).any() or np.isinf(C).any():
    print("❌ Solution contains NaN or Inf → unstable")
else:
    # Reduce data for plotting (otherwise too heavy)
    skip = 1000
    X, T = np.meshgrid(x, t[::skip])
    C_plot = C[::skip, :]

    plt.figure(figsize=(8, 5))
    cf = plt.contourf(X, T, C_plot, 50)
    plt.colorbar(cf, label="Concentration")
    plt.xlabel("x (m)")
    plt.ylabel("t (s)")
    plt.title("Advection-Diffusion-Reaction (Stable Solution)")
    plt.show()