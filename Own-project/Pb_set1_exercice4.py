"""
AUTHOR:             FRIEDLY WOLI
Date:               MADE AT HOME SATURDAY THE 19 OF APRIL 2026 


Exam paper:         2.29 NUMERICAL FLUID MECHANICS - SPRING 2015
University:         FROM MASSACHUSSET INSTITUTE OF THECHNOLIGY 
More detail:        PROBLEM SET 1 DUE TO Monday , February 23, 2015


"""









import numpy as np
import matplotlib.pyplot as plt

# fonction
def f_ex(x):
    return np.sin(x)

# dérivée exacte
def df_ex(x):
    return np.cos(x)
#------------------------------------------------------------------------------------

# dérivée centrée
def f_central_diff(x, h):
    return (f_ex(x + h) - f_ex(x - h)) / (2 * h)

# dérivée forward
def f_forward_diff(x, h):
    return (f_ex(x + h) - f_ex(x)) / h

#------------------------------------------------------------------------------------

# erreurs
def absolute_error(f_exact, f_app):
    return np.abs(f_exact - f_app)

def relative_error(f_exact, f_app):
    return np.abs((f_exact - f_app) / f_exact)


# Paramètres
h = 10**np.arange(-20, 0, 0.5)
x = np.pi - 0.1

# vraie valeur de la dérivée
f_exact = df_ex(x)

# approximations
f_central = f_central_diff(x, h)
f_forward = f_forward_diff(x, h)

# erreurs
err_central_diff = absolute_error(f_exact, f_central)
err_forward_diff = absolute_error(f_exact, f_forward)

# plot
plt.subplot(211)

plt.loglog(h, err_central_diff, label="Central diff error")
plt.loglog(h, err_forward_diff, label="Forward diff error ")
plt.xlabel("h")
plt.ylabel("error")

plt.subplot(211)
plt.plot(h, f_central, label="Central diff")
plt.plot(h, f_forward, label= " forward diff ")




#plt.show()







# noisy data function

#------------------------------------------------------------------------------------

def noisy_f_ex( w, x ):
    return (1+.01*w ) * f_ex(x)

def noisy_df_ex(w, x):
    return ( 1+ 0.01 *w ) * np.cos(x)

def noisy_f_central_diff(x,h,w):
    return (noisy_f_ex(w, x + h) - noisy_f_ex(w, x - h)) / (2 * h)

def noisy_f_forward_diff(x,h,w):
    return (noisy_f_ex(w, x + h) - noisy_f_ex(w, x)) / h
 




 # Parameter 

h=0.01
x= np.arange(0, 2*np.pi, h)
w=0.1





# vraie valeur de la dérivée
noisy_f_exact = noisy_df_ex(w, x)

# approximations
noisy_f_central = noisy_f_central_diff(x, h,w)
noisy_f_forward = noisy_f_forward_diff(x, h,w)

# erreurs
err_central_diff = absolute_error(noisy_f_exact, noisy_f_central)
err_forward_diff = absolute_error(noisy_f_exact, noisy_f_forward)

# plots
plt.subplot(212)

plt.plot(x,noisy_f_exact, "bo",label="exact function")
plt.loglog(x, noisy_f_central, "k", label="Central diff error")
plt.loglog(x, noisy_f_forward, "r--",label="Forward diff error ")
plt.xlabel("x")
plt.ylabel("f,fcentral, fforward")




plt.show()

