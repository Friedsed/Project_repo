# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 08:03:58 2023

#Le code fonctionne bien
# derniere modification mardi 2024-06-10 à 17h54

"""

from sympy import *
import numpy as np
import matplotlib.pyplot as plt




# A ne pas modifier APM

#definir les variables formelles

c1,c2,c3,c4= symbols ('c1,c2,c3,c4')
r,theta, phi= symbols ('r,theta,phi')
lambda_, mu_= symbols('lambda_,mu_')
R,e=symbols('R,e')
rho,omega=symbols('rho,omega')
ksi=symbols('ksi')
Pat=0
G,m=symbols('G,m')
a=symbols('a')
young=symbols('young')

#definir le champ de déplacement admissible (vérifie les CL)
u=c1*r+c2*r**2+c3*r**3+c4*r**4

#Calcul du tenseur des deformations Sym(grad(u))
td=Matrix([[u.diff(r),0,0],[0,u/r,0],[0,0,u/r]])

#Calcul du tenseur des contraintes (loi de Hooke)
tc=lambda_*trace(td)*eye(3)+2*mu_*td                                                                    # APM

#print(simplify(tc))

#Calcul energie volumique de déformation 1/2trace(td*tc)

Eint_v=1/2*trace(td*tc)                                                                                 # APM

#---------------------------------- ATTENTION ON EST EN SPHERIQUE DONC INTEGRATION AVEC r**2*sin(theta) ----------------------------------

#Calcul energie de deformation E_int

E_int=integrate(Eint_v*r**2*sin(theta),     (r,0,a),     (theta,0,pi) ,    (phi  ,  0,  2*pi))                                                      # APM

#Calcul Ev_ext pour la force volumique
f_v=G*m*r*rho/a**3
Ev_ext=-integrate(  f_v* u *sin(theta)**2 *r**2,   (r,0,a),    (theta,0,pi),   (phi,0,2*pi)    )                     # bon

#Calcul Ev_ext2 pour la pression atmospherique (remarque c'est 0 si on la néglige)

Ev_ext2=-integrate(-Pat*u.subs(r,a)*a,(theta,0,pi),(phi,0,2*pi))                                        # = 0 CAR pression atmospherique =0

#Definition Energie Potentielle totale a minimiser sous la contrainte que P(r=R)=Pat traitée avec multiplicateur ksi

Ep =     E_int -  Ev_ext -   Ev_ext2 +     ksi*(tc[0,0].subs(r,a)+Pat)

#Resolution du systeme lineaire

res1    =   solve(     [Ep.diff(c1),Ep.diff(c2),Ep.diff(c3),Ep.diff(c4),Ep.diff(ksi)]   ,   [c1,c2,c3,c4,ksi]   )



#Deplacement en r

print('u(r) =', simplify(   expand(res1[c1]*r+res1[c2]*r**2+res1[c3]*r**3+res1[c4]*r**4)    )  )


#Remettre les valeurs numeriques

mu_n=0.37
E=2.3e9
en=1.2e-3
rhon=1200
Patn=10**5
GG=6.67e-11
mm=6e24
lambda_n= mu_n*E/((1+mu_n)*(1-2*mu_n))
an=64e5


Epn=Ep.subs([(lambda_,lambda_n),(mu_,mu_n),(e,en),(rho,rhon),(G,GG),(m,mm),(a,an)])

#Resolution du systeme lineaire numerique si resolution formelle trop lourde

res=solve([Epn.diff(c1),Epn.diff(c2),Epn.diff(c3),Epn.diff(c4),Epn.diff(ksi)],[c1,c2,c3,c4,ksi])
print('c1=',res[c1])
print('c2=',res[c2])
print('c3=',res[c3])
print('c4=',res[c4  ])
print('ksi=',res[ksi])

#Deplacement en r=R

print('Deplacement en R=', res[c1]*an+res[c2]*an**2+res[c3]*an**3+res[c4]*an**4)
            
            
            
            