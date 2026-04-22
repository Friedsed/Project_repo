import numpy as np
import matplotlib.pyplot as plt

R=0.05   # 0.05 because in the exam paper they say that the raduis is 5 cm
G=1
Io=1


traction1=np.array([5,      10,     15,    20,      30,     40,     50,     60,         70])

ptraction1=np.array(   [5.65 ,  5.8,    5.7  ,  5.7  ,  5.8   , 5.8  ,  5.85   ,5.9  ,      5.9  ])

ptraction2=np.array(   [15.8,   15.85,  15.9  ,16  ,    16.15   ,16.29  ,16.5   ,16.6  ,    16.61  ])

ptraction3=np.array(   [25.9,   26.05,  26.15,  26.3,   26.6,   26.8,   27.1,   27.28,      27.3  ])

ptraction4=np.array(   [36.1,   36.25,  36.4,   36.6,   37.05,  37.3,   37.7,   37.9,   37.9 ])

ptraction5=np.array(   [46.1,   46.35,   46.6,  46.8,   47,   47.5,  47.95 ])


torsion1=np.array(      [10,      20,      30,    40,      50,     60,     70])                 # The loard applied that generate the torque so in the unit is N 

ptorsion1= np.array(    [36,      54,      54,     54,     54,     54,     54])                 # Les valeurs du deplacement au point 1 pour z = 0.5m 

ptorsion2= np.array(    [28.5,      44,      44,     44,     44,     44,     44])               # Les valeurs du deplacement au point 1 pour z = 0.4m 

ptorsion3= np.array(    [21,      23,      23,     23,     23,     23,     23])                 # Les valeurs du deplacement au point 1 pour z = 0.3m

ptorsion4= np.array(    [14.5,      22.5,      22.5,     22.5,     22.5,     22.5,     22.5])   # Les valeurs du deplacement au point 1 pour z = 0.2m

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Answer to the question 4 in other to find the constante k
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

z=np.array(                 [0.5,       0.4,    0.30,    0.2])

delta_10N=np.array(         [ 36 ,      28.5 ,   21,    14.5 ])
delta_20N=np.array(         [ 54 ,      44 ,   23,    22.5])
delta_30N=np.array(         [ 54 ,      44 ,   23,    22.5 ])
delta_40N=np.array(         [ 54 ,      44 ,   23,    22.5 ])
delta_50N=np.array(         [ 54 ,      44 ,   23,    22.5 ])
delta_60N=np.array(         [ 54 ,      44 ,   23,    22.5 ])
delta_70N=np.array(         [ 54 ,      44 ,   23,    22.5 ])



Moment= torsion1*R     
ks=R/G*Io
fig,ax= plt.subplots()
 


ax.plot(delta_10N,z, label='z(x)')
ax.plot(delta_20N,z, label='z(x)')
ax.plot(delta_30N,z, label='z(x)')
ax.plot(delta_40N,z, label='z(x)')
ax.plot(delta_50N,z, label='z(x)')
ax.plot(delta_60N,z, label='z(x)')
ax.plot(delta_70N,z, label='z(x)')

plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Answer to the question 4 in other to find the constante k

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------





