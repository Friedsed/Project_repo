import numpy as np
import matplotlib.pyplot as plt 
from airplane_model1 import Aircraft

import sympy as sp

# Variables
V = sp.symbols('V')
a, b = sp.symbols('a b')
d0, d1, d2 = sp.symbols('d0 d1 d2')

# Fonction
D = d0 + d1*V + d2*V**2
f = (-b + a*V)/D

# Intégrale symbolique
F = sp.integrate(f, V)

print("Primitive :")
sp.pprint(F)

# Intégrale définie
V1, V2 = sp.symbols('V1 V2')
I = sp.integrate(f, (V, V1, V2))

print("\nIntégrale définie :")
sp.pprint(I)



