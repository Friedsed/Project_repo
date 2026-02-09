################################################################# MODEL 1{##################################################}
################################################################# MODEL 1{##################################################}
################################################################# MODEL 1{##################################################}


# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

@author: C. Airiau
@date: 30/10/2023

Partie 1: apprentissage sur le modèle analytique, Model_1

"""
import numpy as np  # module de math
import matplotlib.pyplot as plt  # module graphique
from scipy.constants import g    # constante en m/s^2.
#from .import colored_messages as cm
#from .import constantes as cs

class Model_1(object):
    """ Class of the analytical model"""
    def __init__(self, h=10, v_0=10, alpha=30):
        """ Le constructeur de classe est lancé dès la création de la classe"""
        self.h = h
        self.v_0 = v_0
        self.alpha = np.deg2rad(alpha)  # on met alpha en radians directement
        #self.initial_message()
        self.impact_values = None
        self.time, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.t_1, self.h_1 = None, None


    #def initial_message():
     #   set_title("Création d'une instance du modèle analytique (class initiation)")

    # SETTERS

    def update_parameters(self, params_dict):
        """ set new values for parameters from a dictionary"""
        if params_dict is None:
            params_dict = dict(v_0=10, h=20, alpha=30)
        else:
            for key in params_dict.keys():
                if key == "v_0":
                    self.v_0 = params_dict[key]
                elif key == "h":
                    self.h = params_dict[key]
                elif key == "alpha":
                    self.alpha = np.deg2rad(params_dict[key])
        self.get_parameters()

    def set_velocity(self, t):
        v_z = self.v_0 * np.sin(self.alpha) - g * t
        v_x = self.v_0 * np.cos(self.alpha)
        v = np.sqrt(v_x ** 2 + v_z ** 2)
        return v_x, v_z, v

    def set_impact_values(self):
        """
        retourne les valeurs temps, portée, vitesse et angle à l'impact
        """
        var = self.v_0 * np.sin(self.alpha)
        t_i = (var + np.sqrt(var ** 2 + 2 * g * self.h)) / g
        v_x, v_z, v = self.set_velocity(t_i)
        x_i = v_x * t_i  # ici des variables locales, pas de self devant
        theta_i = np.rad2deg(np.arctan(v_z / v_x))
        self.impact_values = {"t_i": t_i, "p": x_i, "angle": theta_i, "v": [v_x, v_z, v]}
        return self.impact_values

##############################################################JE COMPLETE ICI########################################################### 

#__________________________________________--------------------- SET-TRAJECTORY-------------------- 9.2.1 
    def set_trajectory (self,t_end,npt ):    # t_end is a nombers and npt is number
        self.time=np.linspace(0,t_end,npt)
        self.x = self.v_0*np.cos(self.alpha)*self.time
        self.z= -(g*0.5*(self.time)**2) + self.v_0 * self.time*np.sin(self.alpha) + self.h
        self.v_x= self.v_0*np.cos(self.alpha)
        self.v_z= self.v_0 * np.sin(self.alpha) - g*self.time
        self.v= np.sqrt(self.v_x**2 + self.v_z**2)

#________________________________________________------------SET TRAJECTORIES--------------------- 10 a


    def set_trajectories(self,alphas):
        list_x , list_z , list_v_x , list_v_z , list_v = [], [], [], [], []

        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            impact_values = self.set_impact_values()
            self.set_trajectory(impact_values["t_i"], 50)
            list_x.append(self.x)
            list_z.append(self.z)
            list_v_x.append(self.v_x)
            list_v_z.append(self.v_z )
            list_v.append(self.v)

        return [list_x , list_z, list_v_x, list_v_z, list_v]
    
##############################################################    FIN    ########################################################### 
  
  
  
  
  
  
  
  
    # GETTERS################################################################################################################################
    ########################################################################################################################################


    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        set_info("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])

#_________________________________________________________________________________________________


    def get_parameters(self):
        """
        Affichage formatté des paramètres
        """
        set_info("Parameters:")
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))



##############################################################JE COMPLETE ICI###########################################################    





# PLOTTING METHODS  #####################################################################################################################
########################################################################################################################################

#______________________________________________________-------------------PLOT TRAJECTORY ---------------- 9.2.2 

    def plot_trajectory(self):
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.title("PLOT TRAJECTORY")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()
#__________________________________________________------------------ PLOTS COMPONENTS -------------------- 9.2.3

    def plot_component(self):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)        
        fig.suptitle("Composantes")
        ax1.plot(self.time, self.v_z, marker="+",color="red",linewidth=3)
        ax1.fill_between(self.time, self.v_z, alpha=0.3, color='red')
        ax1.set_ylabel("$v_z$")
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(self.time, self.v, marker="o",color="green",linewidth=3)
        ax2.fill_between(self.time,self.v, alpha=0.3, color='green')
        ax2.set_ylabel("$v$")
        ax2.set_xlabel("time (s)")
        ax2.grid(True, linestyle='--', alpha=0.5)
     
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # ajuste les marges
        plt.title("PLOT COMPONENT")
        plt.show()

#_____________________________________________------------------PLOT TRAJECTORIES --------------- 10.b

    def plot_trajectories(self, xliste, zliste,alphas):
         
        plt.figure(figsize=(10, 6)) # Crée une nouvelle figure

        for i , alpha in enumerate(alphas):
            X=xliste[i]
            Z=zliste[i]
            plt.plot(X, Z, label=f"α = {alpha}")
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Z (m)")
        plt.title("FIGURE Différentes trajectoires en fonction de l'angle de lancement")
        plt.legend(title="Angle de lancement", loc='upper right')
        plt.grid(True, linestyle='--')
        plt.show()

#___________________________________________------------------ PORTET MAXIMAL EN FONCTION DE X ----------------10.2


    def plot_maximum_distance(self, alphas):
        """Trace la portée maximale en fonction de l'angle."""
        distances = []
        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            impact_values = self.set_impact_values()
            distances.append(impact_values["p"])
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, distances)
        plt.xlabel("Angle ")
        plt.ylabel("Portée maximale (m)")
        plt.title("Portée maximale en fonction de l'angle")
        plt.grid(True, linestyle='--')
        plt.show()

#_________________________________________________________________________________________________ DONE BY ME 


    def find_optimal_angle(self, alphas):
        """Trouve l'angle optimal pour la portée maximale."""
        distances = []
        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            impact_values = self.set_impact_values()
            distances.append(impact_values["p"])
        optimal_index = np.argmax(distances)

        return alphas[optimal_index], distances[optimal_index]

#_________________________________________________________________________________________________  DONE BY ME


    def plot_maximum_height(self, alphas):
        """Trace l'altitude maximale en fonction de l'angle."""
        heights = []
        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            t_1 = (self.v_0 * np.sin(self.alpha)) / g
            h_1 = self.h + (self.v_0 * np.sin(self.alpha) * t_1) - (0.5 * g * t_1**2)
            heights.append(h_1)
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, heights)
        plt.xlabel("Angle")
        plt.ylabel("Altitude maximale (m)")
        plt.title("Altitude maximale en fonction de l'angle")
        plt.grid(True, linestyle='--')
        plt.show()



        












































































################################################################# MODEL 2{##################################################}
################################################################# MODEL 2{##################################################}
################################################################# MODEL 2{##################################################}







# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

@author: C. Airiau
@date: 30/10/2023

Partie 2: résolution ODE du problème balistique
Implémentation du modèle 1

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g    # constante en m/s^2.
from scipy.integrate import odeint


import colored_messages as cm
import constantes as cs

class Model_2(object):
    def __init__(self, params):
        """ Le constructeur de classe est lancé dès la création de la classe"""
        self.h = params["h"]
        self.v_0 = params["v_0"]
        self.alpha = np.deg2rad(params["alpha"])
        self.npt = params["npt"]
        #self.initial_message()

        self.t, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.impact_values = None

    @staticmethod
    def initial_message():
        cm.set_title("Création d'une instance du modèle  ODE (exemple d'apprentissage)")

    def ode(self, y,t):
        dy = np.zeros(4)
        dy[0] = y[2]  # dx / dt = v_x
        dy[1] = y[3]  # dz / dt = v_z
        dy[2] = 0     # dv_x / dt = 0
        dy[3] = -g   # dv_z / dt = - g

        return dy


    def solve_trajectory(self, alpha=30, t_end=1):
        """
        trajectory in a gravity field
        t_end is an approximation of the trajectory duration

        On préfère ici mettre alpha et t_end en paramètres plutôt que d'utiliser l'attribut self.alpha.
        c'est plus clair pour les études paramétriques ultérieures
        """
        self.t = np.linspace(0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)
        # initial condition
        y_init = [0, self.h, self.v_0 * np.cos(self.alpha), self.v_0 * np.sin(self.alpha)]
        y = odeint(self.ode, y_init, t=self.t)       # résolution de l'ode
        
        self.x, self.z, self.v_x, self.v_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    def plot_trajectory(self):
        """
        dessin de la trajectoire
        Ajout des étudiants
        """
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()
        

    def validation(self,t_end,npt):
        """
        on se contente de calculer l'erreur sur la position du dernier point.
        ajout des étudiants
        """
        #set_msg("Validation")
        print("analytical solution at t = %f" % self.t[-1])
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x, z                       : %f  %f" % (x_ref, z_ref))
        print("numerical solution at the same time:")
        print("x, z                       : %f  %f" % (self.x[-1], self.z[-1]))
        
        # Ajouter le calcul de l'erreur et l'affichage 
         

        ################################### COMPLETER PAR MOI     ################################################
        self.time=np.linspace(0,t_end,npt)
        x = self.v_0*np.cos(self.alpha)*self.time
        z= -(g*0.5*(self.time)**2) + self.v_0 * self.time*np.sin(self.alpha) + self.h
        v_x= self.v_0*np.cos(self.alpha)
        v_z= self.v_0 * np.sin(self.alpha) - g*self.time
        v= np.sqrt(self.v_x**2 + self.v_z**2)
        ecart=[ np.max( np.abs(x-self.x) ) , np.max( np.abs (z-self.z) ) ] # np.max( np.abs(v_x-self.v_x) ), np.max( np.abs(v_z-self.v_z) )   ]
       
        if np.max(ecart) < 1e-7:
            print(" LA VALIDATION EST BONNE ")
        else:
            print(" Pas de validation")

        print("l erreur max est ", np.max(ecart))
        plt.plot(self.x, self.z, marker="+", color="red", markersize = 12, linewidth=3)
        plt.plot(x, z, marker="+", color="green",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()

        ################################## FIN ################################################################################
      

    def set_reference_solution(self, t):
        x = self.v_0 * np.cos(self.alpha) * t
        z = - g / 2 * t ** 2 + self.v_0 * np.sin(self.alpha) * t + self.h
        return x, z

    def set_impact_values(self):
        """
        partie à modifier par les étudiants

        méthode: trouver le temps d'impact t_i tel que v_z est juste au dessus de 0.
        v_z(t_{i+1}) < 0. Puis remplir v_x, v_z, v, theta_i, x_i à cet instant d'impact.
        """
        # partie à coder
        ######################################################## COMPLETER PAR MOI     ################################################
        

        # 1) Trouver le premier indice où z devient négatif
        n = None
        for i in range(len(self.z) -1):
            if self.z[i] > 0 and self.z[i+1] <= 0:
                n = i
                break

        if n is None:
            raise ValueError("Aucun impact détecté : z ne devient jamais négatif.")

            # 2) Calcul du coefficient d’interpolation

        def interpo(a, n, u):
            # interpolation linéaire entre u[n] et u[n+1]
            return u[n] + a * (u[n + 1] - u[n])

        a = - self.z[n] / (self.z[n+1] - self.z[n])


        t_i  = interpo(a, n, self.t)
        x_i  = interpo(a, n, self.x)
        z_i  = interpo(a, n, self.z)
        vx_i = interpo(a, n, self.v_x)
        vz_i = interpo(a, n,self. v_z)
        #v_i = interpo(a, n, self.v)

        # angle d’impact
        theta_i = np.rad2deg(np.arctan(vz_i / vx_i))

        # 4) Stockage des valeurs
        self.impact_values = {"t_i": t_i, "p": x_i, "angle": np.rad2deg(theta_i), "v": [ vx_i, vz_i]}


        # ##########################################################################################################################


        # résultat à conserver:
        
        return self.impact_values

    def get_parameters(self):
        """
        Affichage formatté des paramètres
        """
        set("Parameters:")
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))

    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])





















































































################################################################# MODEL 3{##################################################}
################################################################# MODEL 3{##################################################}
################################################################# MODEL 3{##################################################}





# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
from scipy.integrate import odeint

import colored_messages as cm
import constantes as cs


class Model_3(object):
    def __init__(self, params):
        self.h = params["h"]
        self.v_0 = params["v_0"]
        self.alpha = np.deg2rad(params["alpha"])
        self.npt = params["npt"]

        self.mass = params["mass"]
        self.rho = params["rho"]
        self.Cd = params["Cd"]
        self.Cl = params["Cl"]
        self.area = params["area"]
        self.a = params["a"]

        self.t, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.Cx, self.Cz = None, None
        self.impact_values = None

        # listes pour stocker les trajectoires multiples
        self.list1 = [None, None]
        self.list2 = [None, None]
        self.list3 = [None, None]

        self.T0 = self.a * self.mass * g
        self.beta = (self.rho * self.area) / (2 * self.mass)
        self.Ct = self.T0 / self.mass

    #____________________________________________________________________________________________ TO UPDATE THE MODEL PARAMETERS

    def update_param(self, param1):
        self.h = param1["h"]
        self.v_0 = param1["v_0"]
        self.alpha = np.deg2rad(param1["alpha"])
        self.npt = param1["npt"]
        self.mass = param1["mass"]
        self.rho = param1["rho"]
        self.Cd = param1["Cd"]
        self.Cl = param1["Cl"]
        self.area = param1["area"]
        self.a = param1["a"]

        self.T0 = self.a * self.mass * g
        self.beta = (self.rho * self.area) / (2 * self.mass)
        self.Ct = self.T0 / self.mass

    #____________________________________________________________________________________________

    @staticmethod
    def initial_message():
        cm.set_title("Création d'une instance du modèle ODE (exemple d'apprentissage)")

    #____________________________________________________________________________________________ FUNCTION ODE

    def ode(self, y, t):
        dy = np.zeros(4)
        dy[0] = y[2]
        dy[1] = y[3]
        v2 = y[2]**2 + y[3]**2
        theta = np.arctan2(y[3], y[2])

        Cx = -self.Cd * np.cos(theta) - self.Cl * np.sin(theta)
        Cz = self.Cl * np.cos(theta) - self.Cd * np.sin(theta)

        dy[2] = self.beta * v2 * Cx + self.Ct * np.cos(theta)
        dy[3] = -g + self.beta * v2 * Cz + self.Ct * np.sin(theta)

        return dy

    #____________________________________________________________________________________________ TO GET THE (X  , Z  , V_X ,  V_Z )     FOR THE MODEL WE ARE WORKING ONE

    def solve_trajectory(self, alpha, t_end):
        self.t = np.linspace(0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)

        y_init = [0,
                  self.h,
                  self.v_0 * np.cos(self.alpha),
                  self.v_0 * np.sin(self.alpha)]
        y = odeint(self.ode, y_init, t=self.t)

        self.x, self.z, self.v_x, self.v_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    #____________________________________________________________________________________________ TO PLOT ONE TRAJECTORY OF THE NUMERICAL ( X, Y ) COORDINATE

    def plot_trajectory(self):
        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position X"], fontsize=12)
        plt.grid()
        plt.show()

    #____________________________________________________________________________________________  TO PLOT DIFFERENT TRAJECTORIES DEPENDING ONE THE ( CD ,  CL)
    def plot_trajectories(self, param1, param2, param3, listC, t_end):
        # listC = [Cl4, Cd4, Cl5, Cd5, Cl7, Cd7]

        # 1ère trajectoire
        self.update_param(param1)
        self.solve_trajectory(param1["alpha"], t_end)
        self.list1[0], self.list1[1] = self.x, self.z

        # 2ème trajectoire
        self.update_param(param2)
        self.solve_trajectory(param2["alpha"], t_end)
        self.list2[0], self.list2[1] = self.x, self.z

        # 3ème trajectoire
        self.update_param(param3)
        self.solve_trajectory(param3["alpha"], t_end)
        self.list3[0], self.list3[1] = self.x, self.z

        lab4 = r"$C_l, C_d$ : " + f"{listC[0]:.2f}, {listC[1]:.2f}"
        lab5 = r"$C_l, C_d$ : " + f"{listC[2]:.2f}, {listC[3]:.2f}"
        lab7 = r"$C_l, C_d$ : " + f"{listC[4]:.2f}, {listC[5]:.2f}"

        plt.plot(self.list1[0], self.list1[1], marker="+", color="blue",
                 linewidth=3, label=lab4)
        plt.plot(self.list2[0], self.list2[1], marker="+", color="red",
                 linewidth=3, label=lab5)
        plt.plot(self.list3[0], self.list3[1], marker="+", color="green",
                 linewidth=3, label=lab7)
        plt.title("tracé de 4 trajectoires")

        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()

    #____________________________________________________________________________________________      TO VALIDATE THE MODEL NUMERICAL COMPRE TO ANALYTICAL SOLUTION 

    def validation(self, t_end, npt):
        cm.set_msg("Validation")

        print("analytical solution at t = %f" % self.t[-1])
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x, z                       : %f  %f" % (x_ref, z_ref))
        print("numerical solution at the same time:")
        print("x, z                       : %f  %f" % (self.x[-1], self.z[-1]))

        # Courbe analytique
        self.time = np.linspace(0, t_end, npt)
        x = self.v_0 * np.cos(self.alpha) * self.time
        z = -(g * 0.5 * (self.time)**2) + self.v_0 * self.time * np.sin(self.alpha) + self.h

        ecart = [np.max(np.abs(x - self.x)),
                 np.max(np.abs(z - self.z))]

        if np.max(ecart) < 1e-7:
            print("La validation est vraie (erreur max < 1e-7)")
        else:
            print("Pas de validation (erreur trop grande)")

        print("L'erreur max est ", np.max(ecart))

        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3, label="Numérique" )
        plt.plot(x, z, marker="+", color="green", linewidth=3, label="Analytique" )
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Z (m)")
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.title("Comparaison Model_3 / analytique sans traînée")
        plt.show()

    #____________________________________________________________________________________________      TO GET  (X, Z ) ANALYTICAL COORDINATE

    def set_reference_solution(self, t):
        x = self.v_0 * np.cos(self.alpha) * t
        z = - g / 2 * t ** 2 + self.v_0 * np.sin(self.alpha) * t + self.h
        return x, z

    #____________________________________________________________________________________________       FIND THE IMPACT VALUES 

    def set_impact_values(self):
        def interpo(a, n, u):
            # interpolation linéaire entre u[n] et u[n+1]
            return u[n] + a * (u[n + 1] - u[n])

        # on cherche le premier passage de z de >0 à <=0
        n = 0
        for i in range(len(self.z) - 1):
            if self.z[i] > 0 and self.z[i + 1] <= 0:
                n = i
                break

        # paramètre d'interpolation a tel que z_i = 0
        a = - self.z[n] / (self.z[n + 1] - self.z[n])

        # temps et position à l'impact
        t_i = interpo(a, n, self.t)
        x_i = interpo(a, n, self.x)
        z_i = 0.0  # par définition de l’impact

        # vitesses à l'impact
        v_x = interpo(a, n, self.v_x)
        v_z = interpo(a, n, self.v_z)
        v = np.sqrt(v_x**2 + v_z**2)

        # angle de la vitesse
        theta_i = np.arctan2(v_z, v_x)

        # résultat conservé
        self.impact_values = {
            "t_i": t_i,
            "p": x_i,
            "angle": np.rad2deg(theta_i),
            "v": [v_x, v_z, v],
        }
        return self.impact_values

    #____________________________________________________________________________________________

    def get_parameters(self):
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))
        print("mass       : %.2f kg" % self.mass)
        print("rho        : %.2f kg/m^3" % self.rho)
        print("area       : %.2f m^2" % self.area)
        print("Cd         : %.2f" % self.Cd)
        print("Cl         : %.2f" % self.Cl)

    #____________________________________________________________________________________________

    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])
  #____________________________________________________________________________________________      DRAW THE CONTOURS PLOT DEPENDING ONE THE ( ALPHA , CdC) CHANGING

# Cdc et alphaC pour dire Cd pour la focntion de contour et alpha pouhr la fonction de contour
    def plot_contour (self , alphaC, CdC, param_base, t_end):

        R = np.zeros((len(alphaC), len(CdC)))

        for i, alphai in enumerate(alphaC):
            for j,Cdj in enumerate (CdC):
                param_new={}
                param_new["v_0"], param_new["h"],  param_new["npt"],param_new["a"] =  param_base["v_0"], param_base["h"],  param_base["npt"], param_base["a"]
                param_new["area"], param_new["mass"] , param_new["Cl"], param_new["rho"] =  param_base["area"], param_base["mass"], param_base["Cl"], param_base["rho"]
                param_new["alpha"], param_new["Cd"]  =  alphai, Cdj
                self.update_param(param_new)       # mettre a jour les parametre du model pour avoir un nouveau model
                self.solve_trajectory(alphai, t_end)     # resoudre pour avoir les valeurs de position en x, z les vitesse et autre 
                impact= self.set_impact_values()         # recuperer les valeurs d'impact
                R[i,j]=impact["p"]                        #  Sauvegarder les valeurs de l impact en fonction de l angle alpha et puis de Cdj

        CD, A = np.meshgrid(CdC, alphaC)
        fig, ax = plt.subplots()
        cont = ax.contourf(A, CD, R, levels=20, cmap="jet")

        cbar= fig.colorbar(cont,ax=ax)
        cbar.set_label("Portee R[m]")

        ax.set_ylabel(r"$c_d$")
        ax.set_xlabel(r"$\alpha$ [deg]")
        ax.set_title("Contours de la portée dans le plan ($\\alpha$, $c_d$)")
        plt.show()
