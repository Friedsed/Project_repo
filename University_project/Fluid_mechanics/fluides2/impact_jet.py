import matplotlib as mp
import math as math
import numpy as np
import matplotlib.pyplot as plt




temps25=np.array([26.76, 47.30, 49.62, 53.86, 59.18, 80.35, 84.19, 91.94, 112.83 , 229.34])   # en ms

pos_25 = np.array([0.13, 0.112, .092, .087, 0.0725, .0435, 0.0415, 0.0255, 0.0145, 0.005  ])

debit=0.025/temps25

plt.figure(figsize=(10, 6))

plt.plot(pos_25, debit, 'o-', label='debit en fonction de la position')
plt.xlabel('Position')
plt.ylabel('debi')
plt.title('position ')
plt.legend()
plt.grid()





temps25=np.array([44.9 , 47.21, 56.10, 60.04, 60.09, 72.87, 74.90 , 90.03, 100.63, 128.79 ])   # en ms

pos_25 = np.array([0.205, .1785, .151, .116 , 0.095,0.074, 0.061, 0.052, 0.04, 0.028  ])

debit=0.025/temps25

plt.figure(figsize=(10, 6))

plt.plot(pos_25, debit, 'o-', label='debit en fonction de la position')
plt.xlabel('Position')
plt.ylabel('debi')
plt.title('position ')
plt.legend()
plt.grid()
plt.show()

















