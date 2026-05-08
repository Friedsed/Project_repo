"""
Author: Friedly WOLI
Date: 27th of april 2026

Students in third year in mechanics and energetic at Toulouse university

Modified the 05/05/2026





"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


"""

T_ref = 273.15    # 0 degrés Celcius en Kelvins
g_ref = 9.80665   # Accélération de la gravité moyenne en m/s^2
R_ref = 8.31432   # constante des gaz parfaits
gamma = 1.4



#-===========================================================================================
# Definition de la classe aircraft
#-===========================================================================================



hw, bw, Ra, e, mu,  g  = sp.symbols( "hw, bw, Ra, e, mu, g " )
rho, V, Vhw, Sw, W  , D, T, L   = sp.symbols(" rho, V, Vhw,  Sw, W , D, T, L" )
C_D0, C_D0l, C_L, C_D= sp.symbols(" C_D0, C_D0l, C_L, C_D " )
T0, T1, T2 = sp.symbols("T0, T1, T2 ")
V1, V2 = sp.symbols(" V1, V2 ")
C_L_alpha, C_L_alpha_sect, alpha, alpha_0 = sp.symbols("C_L_alpha, C_L_alpha_sect, alpha, alpha_0")






C_L_alpha_sect= 2*np.pi * (alpha - alpha_0 )
C_L_alpha = C_L_alpha_sect / (1 + C_L_alpha_sect /(np.pi * Ra))
C_L = C_L_alpha *(alpha - alpha_0)
C_D = C_D0 + C_D0l * C_L + (16*hw/bw)**2 * C_L**2 / (1 + (16*hw/bw))*np.pi * e *Ra

L = 0.5 * V **2* Sw * C_L 
D = 0.5 * V **2* Sw * C_D
T= T0 + T1 * V + T2 * V**2
Fr= mu * ( W - L )

pre_dist= (W/g) * (V - Vhw ) / (T - D - Fr)
dis = sp.integrate(pre_dist, (V, V1, V2))


print(" la distance est ", dis)

"""





# Définition des symboles
hw, bw, Ra, e, mu, g = sp.symbols("hw bw Ra e mu g")
rho, V, Vhw, Sw, W, D, T, L = sp.symbols("rho V Vhw Sw W D T L")
C_D0, C_D0l, C_L, C_D = sp.symbols("C_D0 C_D0l C_L C_D")
T0, T1, T2 = sp.symbols("T0 T1 T2")
V1, V2 = sp.symbols("V1 V2")
C_L_alpha, C_L_alpha_sect, alpha, alpha_0 = sp.symbols("C_L_alpha C_L_alpha_sect alpha alpha_0")



# Expressions aérodynamiques
C_L_alpha_sect = 2 * sp.pi * (alpha - alpha_0)
C_L_alpha = C_L_alpha_sect / (1 + C_L_alpha_sect / (sp.pi * Ra))
C_L = C_L_alpha * (alpha - alpha_0)

C_D = C_D0 + C_D0l * C_L + ((16 * hw / bw)**2 * C_L**2) / (1 + (16 * hw / bw)) * sp.pi * e * Ra

# Forces
L = 0.5 * V**2 * Sw * C_L
D = 0.5 * V**2 * Sw * C_D
T = T0 + T1 * V + T2 * V**2
Fr = mu * (W - L)

# Intégrale
pre_dist = (W / g) * (V - Vhw) / (T - D - Fr)
dis = sp.integrate(pre_dist, (V, V1, V2))

print("La distance est :", dis)


def Kti(Vip1, Vi, koi, k2i, k1i , kwi):

    fi = koi + k1i * Vi + k2i * Vi**2
    fip1 = koi + k1i * Vip1 + k2i * Vip1**2
    dfi = k1i + 2* k2i * Vi
    dfip1 = k1i + 2*k2i * Vip1
    kri = 4* koi * k2i - k1i **2

    if k2i == 0 and k1i == 0 :

        return (Vip1**2 - Vi**2)/ 2* koi
    
    elif k2i == 0 and k1i != 0 :

        return ( koi/ k1i**2 ) * np.ln(fi/fip1) + (Vip1 - Vi)/ k1i
    
    else:

        return (1/ 2* k2i )* np.ln(fip1/fi) - (k1i * kwi ) / 2*k2i


def kwi(Vip1, Vi, koi, k2i, k1i , kwi):


    fi = koi + k1i * Vi + k2i * Vi**2
    fip1 = koi + k1i * Vip1 + k2i * Vip1**2
    dfi = k1i + 2* k2i * Vi
    dfip1 = k1i + 2*k2i * Vip1
    kri = 4* koi * k2i - k1i **2

    if k2i==0 and k1i==0 : 

        return (Vip1 - Vi ) / koi 
    
    elif k1i !=0 and k2i == 0 :

        return (1/ k1i )






