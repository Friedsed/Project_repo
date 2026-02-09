import numpy as np                      # module de math
import matplotlib.pyplot as plt         # module graphique
from scipy.constants import g           # constante en m/s^2.   
from model_3 import Model_3


task = 1


if task == 0:         # test of AnalyticalModel
    t_end, alpha_ref = 3, 20
    npt=1001
    model_2 = Model_3({"v_0": 20, "h": 20, "alpha": 36, "npt": 1001,"area": 0.01, "rho": 1.3, "mass": 100,"Cl": 0.18, "Cd": 0.01, "a": 0.3})

    model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)
    print("solution:", model_2.x, model_2.z)

    plt.figure(0)
    # Validation par rapport à l’erreur
    model_2.validation(t_end, npt)
    model_2.plot_trajectory()

elif task == 1:       # test of AnalyticalModel
    t_end = 20
    alpha_ref = 36

    
    listC = [0, 0,     0 , 0.01 ,     0.01    , 0.18]
 
    param1 = {"v_0": 20, "h": 20, "alpha": 36, "npt": 101,"area": 0.01, "rho": 1.3, "mass":0.100,"Cl": 0, "Cd": 0, "a": 0.3}
    param2 = {"v_0": 20, "h": 20, "alpha": 36, "npt": 101,"area": 0.01, "rho": 1.3, "mass": 0.100,"Cl": 0, "Cd": 0.01, "a": 0.3}
    param3 = {"v_0": 20, "h": 20, "alpha": 36, "npt": 101, "area": 0.01, "rho": 1.3, "mass":0.100, "Cl": 0.18, "Cd": 0.01, "a": 0.3 }

    # on utilise une instance pour tracer les 3 trajectoires
    model_plot = Model_3(param1)
    model_plot.plot_trajectories(param1, param2, param3, listC, t_end)


    alphaC = np.linspace(20, 60, 11)
    CdC = np.linspace(0, 0.1, 11)
    model_plot.plot_contour(alphaC, CdC, param1,t_end)
