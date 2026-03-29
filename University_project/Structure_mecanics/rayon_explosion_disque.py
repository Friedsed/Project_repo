

import numpy as np
import matplotlib.pyplot as plt

# constantes
E = 2.3e+9 #Pa
nu = 0.37
R = 0.06 # 6 cm
nb_points = 100
pas=R/nb_points 
rho=1200 #kg.m-3
tau_c=55e+6 #Pa

lamb = E*nu/((1-2*nu)*(1+nu))
mu = E/(2*(1+nu))

k1_norm=rho/(8*(lamb+2*mu))
k2_norm=(2*lamb+3*mu)*k1_norm*R**2/(lamb+mu)

print(lamb)
print(mu)

r = np.linspace(0,R,nb_points)


sigma_r =2*k2_norm*(lamb+mu)-(4*lamb+6*mu)*k1_norm*r**2
sigma_theta =2*k2_norm*(lamb+mu)-(4*lamb+2*mu)*k1_norm*r**2
sigma_z = 2*lamb*(k2_norm-2*k1_norm*r**2)
    



valeurs_propres = np.array([sigma_r, sigma_theta, sigma_z])

Tmax = np.amax(valeurs_propres, axis=0)
Tmin = np.amin(valeurs_propres, axis=0)

tau_max= (Tmax - Tmin)/2


plt.plot(r, sigma_r)
plt.plot(r, sigma_theta)
plt.plot(r, sigma_z)
plt.plot(r, tau_max)
plt.show()

max_value=max(tau_max)
r_rupture = np.where(tau_max == max_value)
print(r_rupture)

omega_c=np.sqrt(tau_c/max_value)
print(omega_c)

u=omega_c**2*(k2_norm*r-k1_norm*r**3)

plt.plot(r, u)
plt.show()

print(omega_c**2*(k2_norm*R-k1_norm*R**3))
