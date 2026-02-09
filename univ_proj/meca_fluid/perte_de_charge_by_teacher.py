#%% Calcul de la puissance d'une pompe pour alimenter un circuit solaire
#% (exercice de cours, chapitre 6)
import numpy as np

#%% Donnees

d = 0.008            #% diametre (m)
L = 60               #% longueur du circuit
k = 1.5e-6           #% rugosité (m)
rho = 980           #% masse volumique (kg/m^3)
nu = 2e-6            #% viscosité (m^2/s)
Q = 250*1e-3/3600    #% debit de volume (m^3/s)

#%% Parametres de l'ecoulement
S = np.pi/4*d**2  #% surface
U = Q/S       #% vitesse
Re = U*d/nu   #% Reynolds
epsilon = k/d #% rugosité relative
print("Parametres : Re = "+str(Re)+" ; epsilon = "+str(epsilon))


#%% estimation du coef de perte de charge linéaire
print("Calcul iteratif du coefficient lambda : ")
lambda1 = 4e-2 #% estimation avec le diagramme de moody
#% on fait en suite 3 iterations avec la formule de colbroook
lambda1 = 1/(-2*np.log10(2.55/(Re*np.sqrt(lambda1)))+epsilon/3.71)**2
print("iteration 1 : "+str(lambda1))
lambda1 = 1/(-2*np.log10(2.55/(Re*np.sqrt(lambda1)))+epsilon/3.71)**2
print("iteration 2 : "+str(lambda1))
lambda1 = 1/(-2*np.log10(2.55/(Re*np.sqrt(lambda1)))+epsilon/3.71)**2
print("iteration 3 : "+str(lambda1))
lambda1 = 1/(-2*np.log10(2.55/(Re*np.sqrt(lambda1)))+epsilon/3.71)**2
print("iteration 4 : "+str(lambda1))


#%% calcul de la perte de charge linéaire
Deltalin = .5*rho*U**2*L/d*lambda1
print("Perte de charge linéaire   [Pa] : "+str(Deltalin))
#%% estimation de la perte de charge singulière 
K = 20*0.75+2*6.4
Deltasing = .5*rho*U**2*K
print("Perte de charge singulière [Pa] : "+str(Deltasing))

#%% perte de charge totale 
Delta = Deltalin+Deltasing
print("Perte de charge totale     [Pa] : "+str(Delta))

#%% Puissance de la pompe
Puiss = Delta*S*U
print("Puissance de la pompe      [W]  : "+str(Puiss))
