import matplotlib as mp
import math as math
import numpy as np
import matplotlib.pyplot as plt


rho=1000 # masse volumique de l'eau en kg/m3
g=9.81 # acceleration de la gravite en m/s2
Spvc = 0.25*math.pi*0.019**2 # section Surf_pvc_opaque en m2 de diametre 19 mm
Splexitra= 0.25*math.pi*0.0212**2 # section Surf_plexi transparent en m2 de diametre 21.2 mm
Splexi= 0.25*math.pi*0.0425**2 # section Surf_plexi grand rayon en m2 de diametre 42.5 mm

Hdebit1=0.01*np.array(   [87.3, 74.3, 70.7, 65, 67, 57, 40.4, 32, 22.1 ])  # expriment les hauteurs pour le debit1 au hauteur [7, 8, 9 , 10, 11, 12, 13, 14, 15] en mS

Hdebit2=0.01*np.array(   [80.3, 68.1, 64.7, 59.5, 61, 51.7, 37.1, 28.1, 19  ])

Hdebit3=0.01*np.array(   [73.1, 61.7, 58.6, 53.7, 55.2, 46, 32.7, 24.2, 15.4  ])

Hdebit4=0.01*np.array(   [ 67.8, 57.5, 54.2, 49.4, 51, 42, 29.4, 20.9, 12.9 ])

Hdebit5=0.01*np.array(   [60.8, 50.7, 47.9, 43.6, 44.7, 37, 25.3, 17.6, 10  ])

Hdebit6=0.01*np.array(   [ 53.2, 44.4, 41.6, 37.6, 38.8, 31.4, 20.6, 13.6, 6.4 ])

duree=np.array(   [25.11, 26.01,  26.92,  27.92,  28.46,  30.32]) # la durre en seconde pour 10L deau 

debitmeme= 0.001/duree # debit2 et debit meme sont les meme pour verifier que le code est bon EN M3/s


print('les valeur du debit sont ', debitmeme)

print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p1-p2', penteP1)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p2-p3', penteP2)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p3-p4', penteP3)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p4-p5', penteP4)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour 

# calcule des perte de charges pour les 6 debit

deltaP1= np.array([rho*g*(Hdebit1[i] - Hdebit1[i+1]) for i in range(len(Hdebit1)-1)])  # perte de charge pour le debit 1

deltaP2= np.array([rho*g*(Hdebit2[i] - Hdebit2[i+1]) for i in range(len(Hdebit2)-1)])  # perte de charge pour le debit 2

deltaP3= np.array([rho*g*(Hdebit3[i] - Hdebit3[i+1]) for i in range(len(Hdebit3)-1)])  # perte de charge pour le debit 3

deltaP4= np.array([rho*g*(Hdebit4[i] - Hdebit4[i+1]) for i in range(len(Hdebit4)-1)])  # perte de charge pour le debit 4

deltaP5= np.array([rho*g*(Hdebit5[i] - Hdebit5[i+1]) for i in range(len(Hdebit5)-1)])  # perte de charge pour le debit 5

deltaP6= np.array([rho*g*(Hdebit6[i] - Hdebit6[i+1]) for i in range(len(Hdebit6)-1)])  # perte de charge pour le debit 6

#   chgh
deltaH1_2=np.array( [deltaP1[0], deltaP2[0], deltaP3[0], deltaP4[0], deltaP5[0], deltaP6[0]]) # perte de charge p1-p2 quelle que soit le debit

deltaH2_3=np.array( [deltaP1[1], deltaP2[1], deltaP3[1], deltaP4[1], deltaP5[1], deltaP6[1]]) # perte de charge p2-p3

deltaH3_4=np.array( [deltaP1[2], deltaP2[2], deltaP3[2], deltaP4[2], deltaP5[2], deltaP6[2]]) # perte de charge p3-p4

deltaH4_5=np.array( [deltaP1[3], deltaP2[3], deltaP3[3], deltaP4[3], deltaP5[3], deltaP6[3]]) # perte de charge p4-p5

deltaH5_6=np.array( [deltaP1[4], deltaP2[4], deltaP3[4], deltaP4[4], deltaP5[4], deltaP6[4]]) # perte de charge p5-p6

deltaH6_7=np.array( [deltaP1[5], deltaP2[5], deltaP3[5], deltaP4[5], deltaP5[5], deltaP6[5]]) # perte de charge p6-p7

deltaH7_8=np.array( [deltaP1[6], deltaP2[6], deltaP3[6], deltaP4[6], deltaP5[6], deltaP6[6]]) # perte de charge p7-p8

deltaH8_9=np.array( [deltaP1[7], deltaP2[7], deltaP3[7], deltaP4[7], deltaP5[7], deltaP6[7]]) # perte de charge p8-p9 quelle que soit le debit

Coef_perte= (1/Spvc)**2*rho* debitmeme**2 # coefficient de perte de charges est celuis qui multiplie epsilon dans la formule de perte de charge epsi*Coef_perte*L

print('les valeur de perte de charge', deltaP1)
print('les valeur de perte de charge  ', deltaP2)
print('les valeur de perte de charge  ', deltaP3)
print('les valeur de perte de charge  ', deltaP4)
print('les valeur de perte de charge ', deltaP5)
print('les valeur de perte de charge ', deltaP6)

# tracer les courbes de perte de charge en fonction du debit
plt.figure(figsize=(10, 6))
plt.plot(Coef_perte, deltaH1_2, label='Perte de charge pour debit 1')
plt.plot(Coef_perte, deltaH2_3, label='Perte de charge pour debit 2')
plt.plot(Coef_perte, deltaH3_4, label='Perte de charge pour debit 3')
plt.plot(Coef_perte, deltaH4_5, label='Perte de charge pour debit 4')
plt.plot(Coef_perte, deltaH5_6, label='Perte de charge pour debit 5')
plt.plot(Coef_perte, deltaH6_7, label='Perte de charge pour debit 6')
plt.xlabel('Débit (M³/s)')
plt.ylabel('Perte de charge (Pa)')
plt.title('Perte de charge en fonction du débit')
plt.legend()
plt.grid()
plt.show()
# REGRESSION LINEAIRE POUR TROUVER LA PENTE DE LA COURBE DE PERTE DE CHARGE EN FONCTION DU DEBIT
penteP1, _ord_origine_P1 = np.polyfit(Coef_perte, deltaH1_2, 1)
penteP2, _ord_origine_P2 = np.polyfit(Coef_perte, deltaH2_3, 1)
penteP3, _ord_origine_P3 = np.polyfit(Coef_perte, deltaH3_4, 1)
penteP4, _ord_origine_P4 = np.polyfit(Coef_perte, deltaH4_5, 1)
penteP5, _ord_origine_P5 = np.polyfit(Coef_perte, deltaH5_6, 1)
penteP6, _ord_origine_P6 = np.polyfit(Coef_perte, deltaH6_7, 1)
penteP7, _ord_origine_P7 = np.polyfit(Coef_perte, deltaH7_8, 1)
penteP8, _ord_origine_P8 = np.polyfit(Coef_perte, deltaH8_9, 1)


print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p1-p2', penteP1)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p2-p3', penteP2)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p3-p4', penteP3)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p4-p5', penteP4)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p5-p6', penteP5)
print('les valeur de pente de   la courbe de perte de charge en fonction du debit pour p6-p7', penteP6)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p7-p8', penteP7)
print('les valeur de pente de la courbe de perte de charge en fonction du debit pour p8-p9', penteP8)


print ('Coefficient de perte de charge pvc et plexi transparent ', 1-(Spvc/Splexi)**2)
print ('Coefficient de perte de charge Plexi transparent grand rayon et plexi transparent', 1-(Splexitra/Splexi)**2)
print ('Coefficient de perte de charge pvc et plexi transparent grand rayon', 1-(Spvc/Splexi)**2)
