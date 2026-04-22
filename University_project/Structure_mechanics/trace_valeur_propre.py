

import numpy as np
import matplotlib.pyplot as plt

# constantes
E = 2.3e+9 #Pa
nu = 0.37



lamb = E*nu/((1-2*nu)*(1+nu))
mu = E/(2*(1+nu))


print('lambda=',lamb)
print('mu=',mu)


a=0.025
nb_points = 100
r = np.linspace(0,a,nb_points)


sigma_r =lamb*(4*r**2-2*a**2)+2*mu*(3*r**2-a**2)
sigma_theta =lamb*(4*r**2-2*a**2)+2*mu*(r**2-a**2)
sigma_z = lamb*(4*r**2-2*a**2)
    

valeurs_propres = np.array([sigma_r, sigma_theta, sigma_z])
Tmax = np.amax(valeurs_propres, axis=0)
Tmin = np.amin(valeurs_propres, axis=0)
tau_max= (Tmax - Tmin)/2


plt.plot(r, sigma_r, label="sigma_r")
plt.plot(r, sigma_theta, label="sigma_theta")
plt.plot(r, sigma_z, label="sigma_z")
plt.plot(r, tau_max, label="tau_max")
plt.xlabel('r', fontweight='bold')
plt.legend(loc='best')
plt.show()

list_tau_max=list(tau_max)
max_value=max(list_tau_max)
indice = list_tau_max.index(max(list_tau_max))
print("Cisaillement max en r=",r[indice])
print("tau_max_max_max=", tau_max[indice])

