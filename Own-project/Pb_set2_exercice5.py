import numpy as np
import matplotlib.pyplot as plt


# ===== Solution analytique =====
def T_ex(X, h, Tn, Ta, To, Lx):

    B = ((Tn - Ta) - (To - Ta)*np.exp(Lx*np.sqrt(h))) / (2 * np.sinh(Lx*np.sqrt(h)))

    return 2*np.sinh(X*np.sqrt(h))*B + (To - Ta)*np.exp(X*np.sqrt(h)) + Ta


# ===== Gauss-Seidel =====
def solve_GS(A, b, tol, maxit, N):
    x = np.zeros_like(b)
    x[0], x[-1] = b[0], b[-1]

    for k in range(maxit):
        x_old = x.copy()

        for i in range(1, N-1):
            x[i] = (b[i]
                    - A[i, i-1]*x[i-1]
                    - A[i, i+1]*x_old[i+1]) / A[i, i]

        if np.linalg.norm(x - x_old, ord=2) < tol:
            break

    return x, k+1


# ===== Schéma différences centrées =====
def T_cent(h, dx, Ta, To, N, Tn, tol, maxit):

    d = -(2 + h*dx**2)
    dc = -dx**2 * h * Ta

    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N-1):
        A[i, i-1] = 1
        A[i, i]   = d
        A[i, i+1] = 1

    b[0] = To
    b[-1] = Tn

    for i in range(1, N-1):
        b[i] = dc

    T = solve_GS(A, b, tol, maxit, N)[0]

    return T


# ===== Étude pour plusieurs h =====
def T_h(h_list, dx, Ta, To, N, Tn, tol, maxit, Lx, X):

    Th = np.zeros((len(h_list), N))
    Th_ex = np.zeros((len(h_list), N))

    for i in range(len(h_list)):
        Th[i] = T_cent(h_list[i], dx, Ta, To, N, Tn, tol, maxit)
        Th_ex[i] = T_ex(X, h_list[i], Tn, Ta, To, Lx)

    return Th, Th_ex


# ===== Paramètres =====
Lx = 20
Nx = 6
dx = Lx / Nx

To = 25
Ta = 5
Tn = 100

h = [0.4, 0.04, 0.02, 0.00001]

maxit = 5000
tol = 1e-8

X = np.linspace(0, Lx, Nx)


# ===== Calcul =====
T_num, T_exact = T_h (h, dx, Ta, To, Nx, Tn, tol, maxit, Lx, X)


# ===== Plot solution numérique =====
fig, axs = plt.subplots(2, 2)

axs[0,0].plot(X, T_num[0], "b.", label="h=0.4")
axs[0,1].plot(X, T_num[1], "r--",  label="h=0.04")
axs[1,0].plot(X, T_num[2], "r--", label="h=0.02")
axs[1,1].plot(X, T_num[3], "g.", label="h=1e-5")

for ax in axs.flat:
    ax.set_xlabel("position")
    ax.set_ylabel("Temperature")
    ax.legend()

plt.tight_layout()



# ===== Plot solution exacte =====
fig, axs = plt.subplots(2, 2)

axs[0,0].plot(X, T_exact[0], "b.", label="h=0.4")
axs[0,1].plot(X, T_exact[1], "k", label="h=0.04")
axs[1,0].plot(X, T_exact[2], "r--", label="h=0.02")
axs[1,1].plot(X, T_exact[3], "g.", label="h=1e-5")

for ax in axs.flat:
    ax.set_xlabel("position")
    ax.set_ylabel("Temperature")
    ax.legend()

plt.tight_layout()
plt.show()



print (X)