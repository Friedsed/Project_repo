
import numpy as np
import matplotlib.pyplot as plt

# Constantes
dB = 16
g = 9.81
dA = 26

# Données des colonnes (corrigées pour alignement)
Vol=[4,   3.8,  3.6,  3.4,  3.2,   3,   2.8,  2.6,   2.4, 2.2, 2 ,   1.8,  1.6,  1.4,   1.2,   1, 0.8, 0.6,0.4]

P=  [9.7, 10.2, 10.7,11.2, 11.9, 12.6, 13.3, 14.1, 15.15, 16, 17.1, 18.3 , 19.6, 19.8, 20.1, 19.8, 20, 20 ,20.2]
P= [i+1 for i in P]
P1=  [9.8, 10.2, 10.7,11.3, 11.9, 12.6, 13.4, 14.2, 15.1, 16.2, 17.3, 18.5 , 19.9, 20.6, 20.6, 20.7, 20.8, 20.9,20.1]
P1= [i+1 for i in P1]
P2 =[9.7, 10.2, 10.7,11.3, 11.9, 12.6, 13.3, 14.1, 15, 16.1, 17.2, 18.4 , 19.5, 20.1, 20.2, 20.3, 20.3, 20.4,20.8]
P2= [i+1 for i in P2]
P3 = [9.7, 10.1, 10.7,11.2, 11.8, 12.5, 13.3, 14.1, 15, 16, 17.1, 18.3 , 19.6, 19.9, 20, 20, 20.1, 20.2,20.4]
P3= [i+1 for i in P3]

# Graphique corrigé
plt.figure(figsize=(10, 6))
plt.plot(Vol,P, marker="o", label="T20 ", linewidth=2, markersize=6)
plt.plot(Vol,P1, marker="o", label="T25 ", linewidth=2, markersize=6)
plt.plot(Vol,P2, marker="o", label="T35 ", linewidth=2, markersize=6)
plt.plot(Vol,P3, marker="o", label="T40 ", linewidth=2, markersize=6)
plt.xlabel("Volume en cm^3")
plt.ylabel("La pression en bar ")  
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("La pression en foinction du volume")
plt.tight_layout()
plt.show()
