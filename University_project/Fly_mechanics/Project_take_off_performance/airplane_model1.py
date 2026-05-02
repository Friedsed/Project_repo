"""
Author: Friedly WOLI
Date: 27th of april 2026

Students in third year in mechanics and energetic at Toulouse university

Modified the 02/05/2026







"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g  # Accélération de la pesanteur en m/s²
import sympy as sp

T_ref = 273.15    # 0 degrés Celcius en Kelvins
g_ref = 9.80665   # Accélération de la gravité moyenne en m/s^2
R_ref = 8.31432   # constante des gaz parfaits
gamma = 1.4

Gam = u"\u03b3"
Rho = u"\u03c1"
Delt = u"\u03b4"
Sigm = u"\u03c3"
Thet = u"\u0398"



#-===========================================================================================
# Definition de la classe aircraft
#-===========================================================================================

class Aircraft :


    def __init__(self, W, hw, Sw, bw ):

        self.W = W                  # mass of the aircraft
        self.hw= hw                 # height of the wing above the ground
        self.Sw = Sw                # wing surface
        self.bw = bw                # wingspan 
        

#----------------------------------- drag forces 

    def drag ( self, rho, V, Cd ):

        return 0.5*rho*V**2 * self.Sw *Cd
    
#----------------------------------- frictionnal forces 

    def frictionnal (self, mu,  L):

        return mu*( self.W - L )
    
#----------------------------------- lift forces 

    def lift(self, rho, V, Cl):

        return 0.5*rho* V**2 * self.Sw * Cl 
    
#----------------------------------- drag coef 

    # e is the Oswald efficient coefficient or span efficiency factor         , Ra is the aspect ratio b**2/ S  page 53 of the book1

    def drag_coef(self, Cdo, Cdol, e, Ra, Cl):

        Cdoll = (16* self.hw / self.bw)**2 / (1 + ( 16*self.hw/ self.bw)**2 )

        return  Cdo +   Cdol* Cl +    Cdoll * Cl**2 / (np.pi * e* Ra)

#----------------------------------- lift coef 

    def lift_coef(self, alpha, alpha_o):

        return 2* np.pi *( alpha- alpha_o)

     


    def dictance ( self, Vi, Vip1, Toi , Ti , Tii, Vhw, g  , mu , Cl, Cd, rho) :

        K0i= (Toi / self.W ) - mu  
        K1i = Ti/ self.W
        K2i = ( Tii / self.W ) + (rho* self.Sw / 2* self.W) * (Cl*mu - Cd)

        # Variables
        V = sp.symbols('V')
        a, b = sp.symbols('a b')
        d0, d1, d2 = sp.symbols('d0 d1 d2')

        # Fonction
        D = d0 + d1*V + d2*V**2
        f = (-b + a*V)/D

        # Intégrale symbolique
        F = sp.integrate(f, V)

        print("Primitive :")
        sp.pprint(F)

        # Intégrale définie
        V1, V2 = sp.symbols('V1 V2')
        I = sp.integrate(f, (V, V1, V2))

        d = I.subs ([(V1, Vi), (V2, Vip1), (a, -1* Vhw) , (b, 1) , (d0, K0i) , (d1, K1i) , (d2, K2i) ])

        return d/g