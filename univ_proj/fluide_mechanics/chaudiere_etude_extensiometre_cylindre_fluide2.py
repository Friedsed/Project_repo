import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

p1=np.array([ 0,    5,  10, 15,     20,     25,     30,     35]) #  pression en bar

epsiP01=np.array([0, 92,187, 274, 365, 456, 550, 645]) # les deformations en m   et epsiP01 au resltats de la premiere mesure et P pour piston
epsiP02=np.array([ 0, -15, -57, -86, -115, -143, -174, -202  ])
epsiP03=np.array([0, 5, 5, 4, 4, 6, 6, 7])
epsiP04=np.array([0, 35,70,98,132,165,198,230])
epsiP05=np.array([0, 62,126,184,245,307,371,432])
epsiP06=np.array([0, 92,191,280,375,469,569,862])




epsiP11=np.array([0, 35,117,205,298,387,481,574]) # les deformations en m   et epsi11 au resltats de la premiere mesure
epsiP12=np.array([ 0, -14,-49,-80,-110,-141,-174,-205  ])
epsiP13=np.array([0, -1,-7,-8,-7,-9,-7,-9])
epsiP14=np.array([0, 10,37,65,97,127,160,191])
epsiP15=np.array([0, 20,74,132,193,253,316,378])
epsiP16=np.array([0, 35,122,212,309,401,500,596])


epsiC01=np.array([0,81,173,261,354,446,540,637]) # les deformations en m   et epsi11 au resltats de la deuxieme mesure
epsiC02=np.array([ 0, -23,-57,-90,-117,-149,-181,-216  ])
epsiC03=np.array([0, 1,0,0,0,0,1,-2])
epsiC04=np.array([0, 27, 57, 87, 12, 150, 181, 214])
epsiC05=np.array([0, 53, 115, 173, 234, 296, 358, 423])
epsiC06=np.array([0, 82, 178, 268, 365, 459, 557, 657])


epsiC11=np.array([0,80,172,260,252,446,538,638]) # les deformations en m   et epsi11 au resltats de la deuxieme mesure
epsiC12=np.array([ 0, -25,-56,-86,-118,-149,-181,-213  ])
epsiC13=np.array([0, 1,1,1,0,0,0,1])
epsiC14=np.array([0, 28,59,89,121,151,184,217])
epsiC15=np.array([0, 53,114,174,234,297,358,424])
epsiC16=np.array([0, 84,177,208,364,460,557,657])


fig,ax= plt.subplots()
 


ax.plot(p1, epsiC01, label='epsiC01=f(p1)')
ax.plot(p1, epsiC02, label='epsiC02=f(p1)')
ax.plot(p1, epsiC03, label='epsiC03=f(p1)')
ax.plot(p1, epsiC04, label='epsiC04=f(p1)')
ax.plot(p1, epsiC05, label='epsiC05=f(p1)')
ax.plot(p1, epsiC06, label='epsiC06=f(p1)')

ax.set_xlabel('pression en bar')
ax.set_ylabel('deformation en m')
ax.set_title('deformation en fonction de la pression pour le piston')
ax.legend()
plt.show()







#--------------------------------------------------------------------------------------------------------------------------------------
# TRAVAIL DEMAMDER

"""



x, y, z, r= sp.symbols('x y z r ')


theta, nu, la = sp.symbols('theta nu la')
A, B, a = sp.symbols('A B a')
P1, Po, t = sp.symbols('P1 Po t')

epsilon= sp.Matrix([0.5*A -B/r**2  ,0  , 0]  , [0, 0.5*A + B/r**2, 0]  ,   [0, 0, a]  )   # matrice de deformation 

trace= epsilon[0,0] + epsilon[1,1] + epsilon[2,2]

sigma= sp.Matrix([ 2*nu*epsilon[0,0]+ la*trace  ,        0   ,        0]  ,   [0,    2*nu*epsilon[1,1] + la*trace,      0]  ,   [   0,       0,      2*nu*epsilon[2,2] + la*trace]  )   # matrice de contrainte



equations=[
     2*nu* (0.5*A -B/r**2) + la*(A+a) + P1,
     2*nu* (0.5*A -B/r**2) + la*(A+a) +Po,
     2*nu*a + la*(A+a) -t ,
    
]

resulat= sp.solve(equations, (A, B, a))

print('le resultat est : ', resulat)


 """


import sympy as sp

# Variables
x, y, z, r = sp.symbols('x y z r')
theta, nu, la = sp.symbols('theta nu la')
A, B, a = sp.symbols('A B a')
P1, Po, t = sp.symbols('P1 Po t')

# Matrice de déformation
epsilon = sp.Matrix([
    [0.5*A - B/r**2, 0, 0],
    [0, 0.5*A + B/r**2, 0],
    [0, 0, a]
])  

# Trace
trace = epsilon[0,0] + epsilon[1,1] + epsilon[2,2]

# Matrice de contrainte 
sigma = sp.Matrix([
    [2*nu*epsilon[0,0] + la*trace, 0, 0],
    [0, 2*nu*epsilon[1,1] + la*trace, 0],
    [0, 0, 2*nu*epsilon[2,2] + la*trace]
])

# Équations  (= 0)
equations = [
    2*nu*(0.5*A - B/r**2) + la*(A + a) + P1,    # = 0 implicite
    2*nu*(0.5*A - B/r**2) + la*(A + a) + Po,    # = 0 implicite  
    2*nu*a + la*(A + a) - t                     # = 0 implicite
]

# Résolution
resultat = sp.solve(equations, (A, B, a))  # Parentheses au lieu de crochets

print('Le résultat est : ', resultat)  

