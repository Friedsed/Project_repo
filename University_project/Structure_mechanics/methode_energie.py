from sympy import *
import numpy as np
import matplotlib.pyplot as plt


Pat=0
c1, c2, c3,c4, ksi = symbols('c1,c2,c3,c4,ksi')
rho, G, m, a =symbols('rho, G,m, a')

r, theta, phi = symbols('r,theta,phi')
lam, mu = symbols('lam,mu')

u = c1*r+c2*r**2+c3*r**3+c4*r**4

td = Matrix([[u.diff(r), 0, 0], [0, u/r, 0], [0, 0, u/r]])

tc=lam*trace(td)*eye(3)+2*mu*td

Eint = integrate((1/2)*r**2*sin(theta)*trace(td@tc), (r, 0, a),
                 (theta, 0, pi), (phi, 0, 2*pi))

Evext = -integrate(-rho*G*m*r**3*sin(theta)*u/a**3,
                   (r, 0, a), (theta, 0, pi), (phi, 0 , 2*pi))

Evexts = -integrate(-Pat *u.subs(r,a)*a**2*sin(theta), (theta, 0, pi), (phi, 0 , 2*pi))

# print(Evext)

Ep = Eint+Evext+Evexts+ksi*(tc[0,0].subs(r,a)+Pat)

#print('Ep=',Ep)

E = 2.3*10**9 #Pa
nu = 0.37

lamn = E*nu/((1-2*nu)*(1+nu))
mun = E/(2*(1+nu))
an = 64*10**5 # 6400 km
rhon=1200 #kg.m-3
mn=6*10**24 #kg
Gn=6.67*10**(-11)


Epp = Ep.subs([(lam, lamn), (mu, mun), (a, an), (G, Gn), (rho, rhon), (m, mn)])

res=solve([Epp.diff(c1), Epp.diff(c2), Epp.diff(c3),
      Epp.diff(c4),Epp.diff(ksi)], [c1, c2, c3, c4, ksi], dict=True)


print('c1=', res[0][c1])
print('c2=', res[0][c2])
print('c3=', res[0][c3])
print('c4=', res[0][c4])
print('ksi=', res[0][ksi])

print('u_energie(a)=',res[0][c1]*an+res[0][c2]*an**2+res[0][c3]*an**3+res[0][c4]*an**4)
