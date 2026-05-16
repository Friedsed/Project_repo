"""
Author: Friedly WOLI
Date: 27th of april 2026

Students in third year in mechanics and energetic at Toulouse university

Modified the 13 mai 2026 à 15h39

la fonction calcule de distance fonctionne avec les exemples des pages 348 et aussi 350 du livre flight of mechanics by Warren




"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
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




# on au total 36 variables

# Définition des symboles

"""
hw, bw, Ra, e, mu, g = sp.symbols("hw bw Ra e mu g")
rho, V, Vhw, Sw, W, D, T, L, Fr = sp.symbols("rho V Vhw Sw W D T L Fr")
C_D0, C_D0l, C_L, C_D = sp.symbols("C_D0 C_D0l C_L C_D")
T0, T1, T2 = sp.symbols("T0 T1 T2")
V1, V2 = sp.symbols("V1 V2")
C_L_alpha, C_L_alpha_sect, alpha, alpha_0 = sp.symbols("C_L_alpha C_L_alpha_sect alpha alpha_0")

Vip1, Vi, koi, k2i, k1i , kwi = sp.symbols("Vip1, Vi, koi, k2i, k1i , kwi")
dfi, dfip1, kri = sp.symbols("dfi, dfip1, kri ")

"""

#===============================================================
def distance_integral_direct (Vhw, koi, k1i, k2i,Vi, Vip1, g ):

    f = lambda V : (-Vhw + V) / ( koi + k1i*V + k2i*V**2 )
    I , error = quad(f, Vi, Vip1)

    return I/g

# ============================================================
# Plot distance versus speed
# ============================================================

def plot_dist_speed(Vhw, koi, k1i, k2i, Vi, Vip1, g):

    speed = np.linspace(Vi, Vip1, 100)

    S = np.array([
        distance_integral_direct(Vhw, koi, k1i, k2i, Vi, v, g)
        for v in speed
    ])

    plt.figure(figsize=(8, 5))
    plt.plot(speed, S)
    plt.xlabel("Velocity (ft/s)")
    plt.ylabel("Distance (ft)")
    plt.title("Distance vs Velocity")
    plt.grid(True)
    plt.show()

def plot_acc_speed ( koi, k1i, k2i, Vi, Vip1, g):

    v=np.linspace( Vi, Vip1, 100)
    a= ((koi + k1i *v + k2i *v**2 ) *g )


    plt.figure()
    plt.plot(v,a)
    plt.xlabel("la vitesse ")
    plt.ylabel("l'acceleration ")
    plt.show ( )


#=============================================================



# Expressions aérodynamiques
def C_L_alpha_sect(alpha, alpha_0):
    
    return  2 * sp.pi * (alpha - alpha_0)


def C_L_alpha(C_L_alpha_sect, Ra ):
    
    return  C_L_alpha_sect / (1 + C_L_alpha_sect / (sp.pi * Ra))


def C_L_function (C_L_alpha, alpha, alpha_0 ):
    
    return C_L_alpha * (alpha - alpha_0)



def C_D_function (C_D0, C_D0l, C_L, hw, bw, e, Ra ):
    
    return C_D0 + C_D0l * C_L + ((16 * hw / bw)**2 * C_L**2) / ((1 + (16 * hw / bw)**2) * np.pi * e * Ra)

# Forces
def lift ( V, Sw, C_L ):
    return  0.5 * V**2 * Sw * C_L


def drag(V, Sw, C_D):  
    return  0.5 * V**2 * Sw * C_D


def trust(T0, T1, V, T2):  
    return  T0 + T1 * V + T2 * V**2


def friction(W, L, mu ):
    return  mu * (W - L)

def Koi (Toi, W, mu):
    return Toi / W - mu


def K1i(T1, W):
    return T1/W


def K2i(T2, W, rho, Cl, mu, Cd, Sw ):
    return T2/W + rho/ (2*W/Sw) * (Cl*mu - Cd)


def Kri(Koi, K1i, K2i):
    return 4*Koi *K2i - K1i **2


def fi_function( koi, k1i, k2i, Vi):
    return koi + k1i * Vi + k2i * Vi**2
    # fip1 = koi + k1i * Vip1 + k2i * Vip1**2


def dfi_function( k1i, k2i, Vi): 
    return k1i + 2* k2i * Vi
    #  dfip1 = k1i + 2*k2i * Vip1    ; donc pour avoir dfip1 il faut remplacer Vi par Vip1


def Vlo_function(clmax, W, Sw, rho):

    return 1.1* np.sqrt(2/clmax)* np.sqrt(W/(Sw*rho))

def distance( Kti, Vhw, Kwi, g):
    return (Kti - Vhw*Kwi) / g


def Kti_function(Vip1, Vi, koi, k1i, k2i , kwi, fi, fip1):

    if k2i == 0 and k1i == 0 :

        return (Vip1**2 - Vi**2)/ 2* koi
    
    elif k2i == 0 and k1i != 0 :

        return ( koi/ k1i**2 ) * np.log(fi/fip1) + (Vip1 - Vi)/ k1i
    
    else:

        return (1/ (2* k2i) ) * np.log(fip1 / fi) - (k1i * kwi ) / (2*k2i)


def kwi_function(Vip1, Vi, koi, k2i, k1i , fi , fip1, dfi, dfip1, kri):


    if k2i==0 and k1i==0 : 

        return (Vip1 - Vi ) / koi 
    
    elif k1i !=0 and k2i == 0 :

        return (1/ k1i )*np.log(fip1/fi)
    
    elif kri <0 :

        return (1/np.sqrt(-kri)) *np.log ( (dfip1 - np.sqrt(-kri) ) * (dfi  + np.sqrt(-kri) ) / ((dfip1 + np.sqrt(-kri))*(dfi - np.sqrt(-kri)) ))


    elif kri ==0 :

        return (2/dfi) - 2/dfip1
    
    else :

        return (2/np.sqrt(kri) ) * (  1/ np.tan(dfip1 / np.sqrt(kri)) - 1/ np.tan(dfi / np.sqrt( kri)) )




# l= [S_w , b_w , h_w, W, Ra, Cdo , Cdol, e, Clmax, Cl, mu, T0, T1, T2]


def calcule_distance (alpha, alpha_0, Sw , bw , hw,rho, W, Ra, Cdo , Cdol, e, Clmax, Cl, mu, T0, T1, T2 ,  Vhw, g, Vi):

    Ko= Koi (T0, W, mu)
    K1 = K1i(T1, W)                         # in sec/ft
    Cd = C_D_function (Cdo, Cdol, Cl, hw, bw, e, Ra )
    K2 = K2i(T2, W, rho, Cl, mu, Cd, Sw )           # sec^2/ft^2
    Kr= Kri(Ko, K1, K2)
    Vip1 = Vlo_function(Clmax, W, Sw, rho)
    fi= fi_function( Ko, K1, K2, Vi)
    fip1= fi_function( Ko, K1, K2, Vip1)
    dfi= dfi_function( K1, K2, Vi)
    dfip1= dfi_function( K1, K2, Vip1)
    Kw= kwi_function(Vip1, Vi, Ko, K2, K1 , fi , fip1, dfi, dfip1, Kr)          # in ft/sec
    Kt = Kti_function(Vip1, Vi, Ko, K1, K2 , Kw, fi, fip1)   # in ft^2 /sec^2
    dist = distance( Kt, Vhw, Kw, g)
    dist_integ = distance_integral_direct (Vhw, Ko, K1, K2,Vi, Vip1, g )

   

    print (" La valeur de Ko est ", Ko )
    print ("")
  
    print(" La valeurs de K1 est ", K1)
    print("")
    
    print(" La valeurs de K2 est ", K2)
    print("")
    
    print("La valeur de Kr est :", Kr)
    print("")

    print("La valeur de Vip1 est :", Vip1)
    print("")
   
    print("La valeur de fi est : ", fi, " et celui de dfi est :",dfi)
    print("")
   
    print("La valeur de fip1 est : ", fip1, " et celui de dfip1 est :",dfip1)
    print("")

    print("La valeur de Kw est :", Kw)
    print("")

    print("La valeur de Kt est :", Kt)
    print("")

    print(" La valeur de la distance est : ", dist)
    print ("")

    print(" La valeur de la distance de l'integrale est : ", dist_integ)
    print ("")
    print ("")
    print ("")

    plot_acc_speed ( Ko, K1, K2,  Vi, Vip1, g)
    plot_dist_speed(Vhw, Ko, K1, K2, Vi, Vip1, g)




# 









# Let's solve a problem 

Sw , bw , hw, W, Ra= 180, 33, 6, 2700 , 6.05

Cdo , Cdol, e, Clmax, Cl =  0.036, 0.0, 0.82, 1.4, 0.3485


mu= 0.04

T0, T1, T2 = 1200, -4 , 0
rho = 0.0023769
alpha = 0
alpha_0 = 0
Vhw =0
g= 32.2     # in ft/sec
Vi=0

    

    
print ("the drag coefficient is ", C_D_function (Cdo, Cdol, Cl, hw, bw, e, Ra ) ) # le calcule fontionne et donne de bon resultat ; vérifié avec l'exo de la page 347



calcule_distance (alpha, alpha_0, Sw , bw , hw,rho, W, Ra, Cdo , Cdol, e, Clmax, Cl, mu, T0, T1, T2, Vhw, g, Vi)









calcule_distance(alpha=0 ,alpha_0=0 ,Sw=180, bw=33, hw=6, rho=0.0023769 , W=2700, Ra=6.05 , Cdo= 0.036 , Cdol=0, e=0.82, Clmax= 1.4, Cl= 0.3485 ,  mu= 0.04, T0= 1200 , T1= -4 , T2=0 , Vhw=29.33 , g=32.2, Vi=29.33  )

#dist= (Kti, Vhw, Kwi, g)
#print("la distance de decollage est ", dist)