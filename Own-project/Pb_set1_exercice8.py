"""
AUTHOR:             FRIEDLY WOLI
Date:               MADE AT HOME SATURDAY THE 19 OF APRIL 2026 


Exam paper:         2.29 NUMERICAL FLUID MECHANICS - SPRING 2015
University:         FROM MASSACHUSSET INSTITUTE OF THECHNOLIGY 
More detail:        PROBLEM SET 1 DUE TO Monday , February 23, 2015


"""


import numpy as np
import matplotlib.pyplot as plt














def f(x, r, K):
    return r * x * (1 - x / K)


def df(x, r, K):
    return r * (1 - 2 * x / K)


def newton_raphson(r, K, x0, tol=1e-12, maxit=50):
    """
    Newton method to find an equilibrium of f(x,r,K)=0.
    For the logistic equation, equilibria are x=0 and x=K.
    """
    x = np.zeros(maxit, dtype=float)
    x[0] = x0

    for i in range(1, maxit):
        dfx = df(x[i - 1], r, K)
        if np.isclose(dfx, 0.0):
            raise ValueError("Derivative too small in Newton-Raphson.")

        x[i] = x[i - 1] - f(x[i - 1], r, K) / dfx

        if abs(x[i] - x[i - 1]) < tol:
            return x[:i + 1]

    return x


def central_diff(r, K, maxit, x0, dt):
    """
    Two-step central-difference style scheme:
    x[n+1] = x[n-1] + 2*dt*f(x[n], r, K)
    """
    x = np.zeros(maxit, dtype=float)
    x[0] = x0

    if maxit > 1:
        x[1] = x0 + dt * f(x0, r, K)

    for i in range(1, maxit - 1):
        x[i + 1] = x[i - 1] + 2.0 * dt * f(x[i], r, K)

    return x


def forward_diff(r, K, maxit, x0, dt):
    """
    Forward Euler method:
    x[n+1] = x[n] + dt*f(x[n], r, K)
    """
    x = np.zeros(maxit, dtype=float)
    x[0] = x0

    for i in range(maxit - 1):
        x[i + 1] = x[i] + dt * f(x[i], r, K)

    return x


def exact_sol(t, r, K, x0):
    """
    Exact solution of the logistic IVP:
    x(t) = x0*K*exp(rt) / ((K-x0) + x0*exp(rt))
    """
    ert = np.exp(r * t)
    return (x0 * K * ert) / ((K - x0) + x0 * ert)


# Parameters
maxit = 120
t0 = 0.0
tf = 10.0
t = np.linspace(t0, tf, maxit)
dt = t[1] - t[0]

x0 = 0.5
r = 1.0
K = 2.0
tol = 1e-12

# Computations
c_diff = central_diff(r, K, maxit, x0, dt)
n_iter = newton_raphson(r, K, x0, tol, 30)
f_diff = forward_diff(r, K, maxit, x0, dt)
x_exact = exact_sol(t, r, K, x0)

# Plot
plt.figure(figsize=(9, 5))
plt.plot(t, x_exact, label="Exact solution", linewidth=2)
plt.plot(t, c_diff, "--", label="Central difference")
plt.plot(t, f_diff, "-.", label="Forward Euler")

# Newton gives iterations, not time evolution
plt.axhline(n_iter[-1], color="gray", linestyle=":", label=f"Newton equilibrium ≈ {n_iter[-1]:.6f}")

plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Logistic equation: exact and numerical solutions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Newton iterations:", n_iter)
print("Final Newton value:", n_iter[-1])





















"""


def f(x,r,K):
    return r*x*( 1 - x/K )


def df(x, r, k):
    return r*(1 - 2*x / k )


def NewtonRaphson (r,K,df,x0,TOL, Maxit):
    x= np.arange(120)
    x[0] = x0
    for i in range (1, 120):
        x[i]= x[i-1] - f(x[i-1],r,K) / df(x[i-1],r,K)

    return x


def central_diff ( r, K , Maxit, x0, dt ):

    x=np.zeros(120)
    x[0]=x0
    x[1]=x0
    for i in range ( 1 , 118 ):
        x[i+1]= dt * f(x[i],r,K) + x[i-1]

    return x








def foward_diff ( r, K , Maxit, x0, dt ):

    x=np.zeros(120)
    x[0]=x0
    for i in range (1, 120 ):
        x[i]= dt * f(x[i-1], r, K) + x[i-1]

    return x

def exact_sol(r, T , K ):
    return (2* np.exp(r*T)/ (1+ 2* (np.exp(r*T) -1 )/K ) )

# Parameters -----------------------------------------
Maxit=30
T=np.linspace(0,120, Maxit)
dt= 120/ Maxit
x0=5
r=1
K=2
TOL=10e-12


C_diff= central_diff ( r, K , Maxit, x0, dt )
B_diff= NewtonRaphson (r,K,df,x0,TOL, Maxit)
F_diff= foward_diff ( r, K , Maxit, x0, dt )


plt.figure()

plt.plot(T,C_diff, label="C_diff" )
plt.plot(T,B_diff, label="B_diff" )
plt.plot(T,F_diff, label="F_diff" )

plt.xlabel('time')
plt.ylabel('Population')

plt.show()
"""