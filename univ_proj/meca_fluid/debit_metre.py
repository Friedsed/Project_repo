import numpy as np
import matplotlib.pyplot as plt

# Constantes
dB = 16
g = 9.81
dA = 26

# Données des colonnes (corrigées pour alignement)
c0 = [160, 140, 120, 100, 80, 60, 40, 20]
c1 = [323, 343, 354, 361, 364, 365, 366, 366]
c2 = [182, 233, 271, 299, 320, 346, 349, 357]
c3 = [298, 323, 338, 348, 354, 358, 361, 363]
c4 = [305, 328, 342, 352, 357, 360, 362, 364]
c6 = [312, 333, 347, 355, 359, 361, 363, 364]
c7 = [148, 209, 252, 284, 310, 330, 345, 355]
c8 = [178, 230, 268, 297, 318, 334, 348, 355]
c9 = [145, 204, 249, 281, 308, 328, 344, 353]
c10 = [129, 195, 140, 176, 204, 225, 241, 252]

# Calculs corrigés (paires alignées par index)
x = np.sqrt(np.array(c1) - np.array(c2))  # √(c1[i] - c2[i]) pour chaque i

# Facteur constant α
alpha = (np.pi * dB**2 * np.sqrt(g)) / (np.sqrt(8 * (1 - (dB/dA)**4)) * x)

# Débits Q = α * h (où h = c0 ?)
Q = alpha * x

# Graphique corrigé
plt.figure(figsize=(10, 6))
plt.plot(x,Q, marker="o", label="S = 0.95%", linewidth=2, markersize=6)
plt.xlabel("Q (débit)")
plt.ylabel("√(ha + hb)")  # ou √(c1 - c2)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Débit Q en fonction de √(ha + hb)")
plt.tight_layout()
plt.show()
