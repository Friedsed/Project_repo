# -*- coding: utf-8 -*-
"""

"""

from sympy import *
import numpy as np
import matplotlib.pyplot as plt





#definir les variables formelles

a,b,c= symbols ('a,b,c')
r,theta, z= symbols ('r,theta,z')
lambda_, mu_= symbols('lambda_,mu_')
R,e=symbols('R,e')
rho,omega=symbols('rho,omega')
ksi=symbols('ksi')
Pat=symbols('Pat')


#definir le champ de déplacement admissible (vérifie les CL)
u=a*r+b*r**2+c*r**3

#Calcul du tenseur des deformations Sym(grad(u))
td=Matrix([[u.diff(r),0,0],[0,u/r,0],[0,0,0]])

#Calcul du tenseur des contraintes (loi de Hooke)
tc=lambda_*trace(td)*eye(3)+2*mu_*td

#print(simplify(tc))

#Calcul energie volumique de déformation 1/2trace(td*tc)

Eint_v=1/2*trace(td*tc)

#Calcul energie de deformation E_int

E_int=integrate(Eint_v*r,(r,0,R),(theta,0,2*pi),(z,-e/2,e/2))

#Calcul Ev_ext pour la force volumique
f_v=rho*omega**2*r
Ev_ext=-integrate(f_v*u*r,(r,0,R),(theta,0,2*pi),(z,-e/2,e/2))

#Calcul Ev_ext2 pour la pression atmospherique (remarque c'est 0 si on la néglige)

Ev_ext2=-integrate(-Pat*u.subs(r,R)*R,(theta,0,2*pi),(z,-e/2,e/2))

#Definition Energie Potentielle totale a minimiser sous la contrainte que P(r=R)=Pat traitée avec multiplicateur ksi

Ep=E_int+Ev_ext+Ev_ext2+ksi*(tc[0,0].subs(r,R)+Pat)

#Resolution du systeme lineaire

res1=solve([Ep.diff(a),Ep.diff(b),Ep.diff(c),Ep.diff(ksi)],[a,b,c,ksi])



#Deplacement en r

print('u(r) =', simplify(expand(res1[a]*r+res1[b]*r**2+res1[c]*r**3)))


#Remettre les valeurs numeriques
lambda_n=2.39e+9
mu_n=0.84e+9
en=1.2e-3
omegan=14778
rhon=1200
Rn=0.06
Patn=10**5

Epn=Ep.subs([(lambda_,lambda_n),(mu_,mu_n),(R,Rn),(e,en),(omega,omegan),(rho,rhon),(Pat,Patn)])

#Resolution du systeme lineaire numerique si resolution formelle trop lourde

res=solve([Epn.diff(a),Epn.diff(b),Epn.diff(c),Epn.diff(ksi)],[a,b,c,ksi])
print('a=',res[a])
print('b=',res[b])
print('c=',res[c])
print('ksi=',res[ksi])

#Deplacement en r=R

print('Deplacement en R=', res[a]*Rn+res[b]*Rn**2+res[c]*Rn**3)
            
            
            
            
