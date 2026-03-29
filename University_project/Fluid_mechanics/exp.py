############################################################################### VALIDATE THE 02/01/2026 BY ALEXANDRE FRIEDLY
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


t=3
if t==0:

    t=np.linspace(0,2*np.pi, 1000)
    x=5*np.sin(t)
    y=(13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))*100




    plt.plot(x,y ,
            marker="+", color="hotpink", linewidth=2,
            label="Sophie's Love")


    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Rayon", fontsize=12)
    plt.title("Sophie VS Friedly")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()



elif t==1:
    x=np.linspace(-1.5,1.5,200)
    y=np.linspace(-1.5,1.5,200)
    z=np.linspace(-1.5,1.5,200)
    X, Y, Z = np.meshgrid(x,y,z)
    F=(X**2 + (9/4)*Y**2 +Z**2 -1)**3 -X**2*Z**3 - (9/80)*Y**2*Z**3

    mask =np.abs(F)<0.01
    Xc=X[mask]
    Yc=Y[mask]
    Zc=Z[mask]

    fig = plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    ax.scatter(Xc, Yc,Zc, s=1)

    ax.set_title("Coeur en 3D ")
    ax.set_box_aspect([1,1,1])
    plt.show()


elif t==2:
       
    x=np.linspace(-2,2,400)
    y=np.linspace(-2,2,400)
    X, Y= np.meshgrid(x,y)

    F=(X**2 + Y**2 -1)**3 -X**2 * Y**3

    plt.contour(X, Y , F, levels=[-1e3,0], colors=["hotpink"])
    plt.axis("equal")
    plt.title("Coeur for SierraOscaPapaHotelIndiaEcho")
    plt.show()


elif t==3:


    def prime(n):
        for i in range(2,n-1):
            if n%i== 0 :
                return False
            
            else :
                return True
            

    print(prime(109))
