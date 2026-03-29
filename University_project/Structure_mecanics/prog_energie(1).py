# -*- coding: utf-8 -*-


from sympy import *
import numpy as np
import matplotlib.pyplot as plt





#definir les variables formelles

k1,k2,k3= symbols ('k1,k2,k3')
r,theta, z= symbols ('r,theta,z')
lambda_, mu_= symbols('lambda_,mu_')
a,h=symbols('a,h')
rho,omega=symbols('rho,omega')
ksi=symbols('ksi')
P0=symbols('P0')


#definir le champ de déplacement non admissible (ne vérifie pas les CL)
u=k1*r+k2*r**2+k3*r**3


#Calcul du tenseur des deformations Sym(grad(u))
td=Matrix([[u.diff(r),0,0],[0,u/r,0],[0,0,0]])


#Calcul du tenseur des contraintes (loi de Hooke)
tc=lambda_*trace(td)*eye(3)+2*mu_*td


#Calcul energie volumique de déformation 1/2trace(td*tc)

Eint_v=1/2*trace(td*tc)

#Calcul energie de deformation E_int

E_int=integrate(Eint_v*r,(r,0,a),(theta,0,2*pi),(z,-h/2,h/2))

#Calcul Ev_ext pour la force volumique
f_v=rho*omega**2*r
Ev_ext=-integrate(f_v*u*r,(r,0,a),(theta,0,2*pi),(z,-h/2,h/2))

#Calcul Ev_ext2 pour la pression P radiale (le travail de la force F et P est nul sur les faces inferieures et superieures)

E_ext2=-integrate(-P0*u.subs(r,a)*a,(theta,0,2*pi),(z,-h/2,h/2))+0

#Definition Energie Potentielle totale a minimiser sous la contrainte que u(a)=0 (déplacement admissible) traitée avec multiplicateur ksi

Ep=E_int+Ev_ext+E_ext2+ksi*(u.subs(r,a))

#Resolution du systeme lineaire

res1=solve([Ep.diff(k1),Ep.diff(k2),Ep.diff(k3),Ep.diff(ksi)],[k1,k2,k3,ksi])

# Les 3 constantes k1, k2 et k3

print('k1=',res1[k1])
print('k2=',res1[k2])
print('k3=',res1[k3])

#Deplacement en r

print('u(r) =', simplify(expand(res1[k1]*r+res1[k2]*r**2+res1[k3]*r**3)))



#verification du Deplacement en r=a

print('Deplacement en a', res1[k1]*a+res1[k2]*a**2+res1[k3]*a**3)

#Pressurisation en fonction de omega
            
test=-tc[0,0].subs(r,a)  
test1=test.subs(k1,res1[k1])  
test2=test1.subs(k2,res1[k2])  
test3=test2.subs(k3,res1[k3])       
print('P0(omega)=',simplify(test3))
            
