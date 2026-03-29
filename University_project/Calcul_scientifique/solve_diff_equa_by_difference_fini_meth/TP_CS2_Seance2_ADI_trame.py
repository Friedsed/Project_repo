# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:19:33 2026

@author: aberg
"""

# -*- coding: utf-8 -*-
"""
Méthode ADI (Peaceman–Rachford) pour l'équation de la chaleur 2D
Validation par solution analytique
"""
# -*- coding: utf-8 -*-
"""
Méthode ADI (Peaceman–Rachford) pour l'équation de la chaleur 2D
Version pédagogique proche du code implicite fourni
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------
D = 1.0
Lx, Ly = 1.0, 1.0
Nx, Ny = 40, 40
dt = 0.00001
Tmax = 5
omega = 2*np.pi
Nt = int(Tmax / dt)


# -----------------------------------------------------------------------------
# Maillage
# -----------------------------------------------------------------------------
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing='ij')

# -----------------------------------------------------------------------------
# Solution exacte et source (identiques à la version implicite)
# -----------------------------------------------------------------------------
def T_ex(x, y, t, D, omega, Lx, Ly):
    lambda_ = (np.pi/Lx)**2 + (np.pi/Ly)**2
    return (
        np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) *
        (D * lambda_ * np.cos(omega * t) + omega * np.sin(omega * t)) /
        ((D * lambda_)**2 + omega**2)
    )

def f_source(x, y, t, D, omega, Lx, Ly):
    return np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) * np.cos(omega*t)

# -----------------------------------------------------------------------------
# Coefficients ADI
# -----------------------------------------------------------------------------
dx2 = dx**2
dy2 = dy**2
d_x = D * dt / (2 * dx2)
d_y = D * dt / (2 * dy2)

# -----------------------------------------------------------------------------
# Algorithme de Thomas (résolution tridiagonale)
# -----------------------------------------------------------------------------

def thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom

    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x

# -----------------------------------------------------------------------------
# Condition initiale
# -----------------------------------------------------------------------------
Tn = np.zeros((Nx+1, Ny+1))
for i in range(Nx+1):
    for j in range(Ny+1):
        Tn[i, j] = T_ex(x[i], y[j], 0.0, D, omega, Lx, Ly)

# -----------------------------------------------------------------------------
# Boucle en temps ADI
# -----------------------------------------------------------------------------
temps = np.linspace(dt, Tmax, Nt)
erreur_L2_list = []
erreur_max_list = []
T_center_time = []

erreur_max_globale = 0.0
start_time = time.time()

for n in range(Nt):
    t = temps[n]

    # -------------------------------------------------------------------------
    # Demi-pas 1 : implicite en x, explicite en y
    # -------------------------------------------------------------------------
    T_half = Tn.copy()

    for j in range(1, Ny):
        rhs = np.zeros(Nx+1)
        for i in range(1, Nx):
            rhs[i] = (
                d_y * Tn[i, j+1]
                + (1 - 2*d_y) * Tn[i, j]
                + d_y * Tn[i, j-1]
                + dt/2 * f_source(x[i], y[j], t - dt/2, D, omega, Lx, Ly)
            )

        rhs[0] = 0.0
        rhs[Nx] = 0.0

        a = -d_x * np.ones(Nx+1)
        b = (1 + 2*d_x) * np.ones(Nx+1)
        c = -d_x * np.ones(Nx+1)

        b[0] = 1.0; c[0] = 0.0
        b[Nx] = 1.0; a[Nx] = 0.0

        T_half[:, j] = thomas(a, b, c, rhs)

    # -------------------------------------------------------------------------
    # Demi-pas 2 : implicite en y, explicite en x
    # -------------------------------------------------------------------------
    Tnp1 = T_half.copy()

# Completer cette partie en vous inspirant de la précédente

    for i in range(1, Nx):
        rhs = np.zeros(Ny+1)
        for j in range(1, Ny):
            rhs[j] = (
                d_x * T_half[i+1, j]
                + (1 - 2*d_x) * T_half[i, j]
                + d_x * T_half[i-1, j]
                + dt/2 * f_source(x[i], y[j], t, D, omega, Lx, Ly)
            )

        rhs[0] = 0.0
        rhs[Ny] = 0.0

        a = -d_y * np.ones(Ny+1)
        b = (1 + 2*d_y) * np.ones(Ny+1)
        c = -d_y * np.ones(Ny+1)

        b[0] = 1.0; c[0] = 0.0
        b[Ny] = 1.0; a[Ny] = 0.0

        Tnp1[i, :] = thomas(a, b, c, rhs)

    Tn = Tnp1.copy()

    # -------------------------------------------------------------------------
    # Solution exacte et erreurs
    # -------------------------------------------------------------------------
    Tex = np.zeros_like(Tn)
    for i in range(Nx+1):
        for j in range(Ny+1):
            Tex[i, j] = T_ex(x[i], y[j], t, D, omega, Lx, Ly)

    erreur = Tn - Tex
    erreur_L2 = np.sqrt(np.mean(erreur**2))
    erreur_max = np.max(np.abs(erreur))

    erreur_L2_list.append(erreur_L2)
    erreur_max_list.append(erreur_max)
    erreur_max_globale = max(erreur_max_globale, erreur_max)

    ic = Nx // 2
    jc = Ny // 2
    T_center_time.append(Tn[ic, jc])

end_time = time.time()
temps_calcul = end_time - start_time

# -----------------------------------------------------------------------------
# Graphiques
# -----------------------------------------------------------------------------
plt.figure()
plt.plot(temps, erreur_L2_list, label="Erreur L2")
plt.plot(temps, erreur_max_list, label="Erreur max")
plt.xlabel("Temps")
plt.ylabel("Erreur")
plt.title("Évolution de l'erreur dans le temps – Méthode ADI")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
cf = plt.contourf(X, Y, Tn, 50)
plt.colorbar(cf, label="Température")
cs = plt.contour(X, Y, Tn, 10, colors='k', linewidths=0.7)
plt.clabel(cs, inline=True, fontsize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solution numérique au temps final – Méthode ADI")
plt.show()

plt.figure()
cf = plt.contourf(X, Y, np.abs(Tn - Tex), 50)
plt.colorbar(cf, label="Erreur |T - Tex|")
cs = plt.contour(X, Y, np.abs(Tn - Tex), 10, colors='k', linewidths=0.7)
plt.clabel(cs, inline=True, fontsize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Erreur spatiale au temps final – Méthode ADI")
plt.show()

plt.figure()
plt.plot(temps, T_center_time, 'b-', lw=2, label='Solution numérique')
T_center_exact = [T_ex(Lx/2, Ly/2, t, D, omega, Lx, Ly) for t in temps]
plt.plot(temps, T_center_exact, 'r--', lw=2, label='Solution exacte')
plt.xlabel("Temps")
plt.ylabel("T(Lx/2, Ly/2, t)")
plt.title(f"Solution numérique vs exacte au centre, omega={omega}")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# Affichage final : erreur max globale et temps de calcul
# -----------------------------------------------------------------------------
print(f"Erreur maximale globale sur tous les pas de temps : {erreur_max_globale:.6e}")
print(f"Temps total de calcul : {temps_calcul:.3f} secondes")
