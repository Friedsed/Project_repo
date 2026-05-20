import numpy as np
import matplotlib.pyplot as plt 
from airplane_model1 import Aircraft

import sympy as sp

# Variables
V = sp.symbols('V')
a, b = sp.symbols('a b')
A, B, C = sp.symbols('A B C')

# Fonction
D = A + B*V + C*V**2
f = D/ (-b + V)

# Intégrale symbolique
F = sp.integrate(f, V)

print("Primitive :")
sp.pprint(F)

# Intégrale définie
V1, V2 = sp.symbols('V1 V2')
I = sp.integrate(f, (V, V1, V2))

#print("\nIntégrale définie :")
#sp.pprint(I)



