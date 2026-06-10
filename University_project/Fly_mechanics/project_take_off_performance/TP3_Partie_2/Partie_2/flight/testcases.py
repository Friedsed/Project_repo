# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

Testcases module :
    get_aircraft
    get_flight_data
    get_polar_data
    get_engine

@author: Christophe Airiau
"""
import numpy as np
from flight.general import ft2m


def get_aircraft(case=0):
    if case == 0:
        return dict(name="avion_1", m=20000, s=61, lambda_=0, b=27.05, e=0.88, rho=1, v=np.array([0, 0]))
    elif case == 1:
        return dict(name="ATR 72-500", m=22000, s=61, lambda_=0, b=27.05, e=0.85, rho=1, v=np.array([0, 0]))
    elif case == 2:
        return dict(name="TBM-700", m=2900, s=18, lambda_=0, b=12.78, e=0.85, rho=1, v=np.array([0, 0]))
    elif case == 3:
        return dict(name="B747-400", m=300000, s=525, lambda_=0, b=62.3, e=0.85, rho=1, v=np.array([0, 0]))


def get_flight_data(case=0):
    if case == 0:
        h1 = np.linspace(0, 5000, 5)
        h2 = np.linspace(5000, 5500, 5)
        h = np.hstack([h1, h2])
        return dict(h=h, mach=np.linspace(0.02, 0.45, 51))
    elif case == 1:
        # ATR-72-500
        h1 = np.linspace(0, 6000, 5)
        h2 = np.linspace(6500, 8100, 6)
        h = np.hstack([h1, h2])
        return dict(h=h, mach=np.linspace(0.02, 0.5, 51))
    elif case == 2:
        h1 = np.linspace(0, 4300, 5)
        h2 = np.linspace(4400, 4800, 5)
        h = np.hstack([h1, h2])
        return dict(h=h, mach=np.linspace(0.02, 0.35, 51))
    elif case == 3:
        h1 = np.linspace(0, 6000, 5)
        h2 = np.linspace(6100, 6700, 6)
        h = np.hstack([h1, h2])
        return dict(h=h, mach=np.linspace(0.02, 0.5, 51))
    elif case == 4:
        # B747-400
        h1 = np.array([0, 5000, 10000, 20000, 25000, 28000])
        h2 = np.linspace(30000, 36000, 7)
        h = ft2m(np.hstack([h1, h2]))
        return dict(h=h, mach=np.linspace(0.1, 1.2, 51))
    elif case == 5:
        # TBM 700
        h1 = np.linspace(0, 24000, 7)
        h2 = np.linspace(25000, 31000, 7)
        h = ft2m(np.hstack([h1, h2]))
        return dict(h=h, mach=np.linspace(0.02, 550/340/3.6, 21))

    elif case == 10:
        return dict(h=[0, 4500], mach=np.array([0.02, 0.3]))


def get_polar_data(case=0):
    """
    return the CD_0, and CL_0
    k is calculated elsewhere
    """
    if case == 0:
        return [0.028, 0.1]
    elif case == 1:
        # ATR 72-500
        return [0.029, 0.1]
    elif case == 3:
        # TBM 700
        return [0.023, 0.1]
    elif case == 4:
        # B747-400
        return [0.025, 0.0]


def get_engine(case=0):
    if case == 0:
        return dict(type="piston", model="simple", pa_sl=1.5e6, thrust_sl=1, correction=2)
    elif case == 1:
        return dict(type="piston", model="turbo", hc=ft2m(18000), boost=0.,  pa_sl=1.5e6, thrust_sl=1, correction=2)
    elif case == 2:
        return dict(type="piston", model="GF",  pa_sl=1.5e6, thrust_sl=1, correction=2)
    elif case == 3:
        # ATR 72-500
        return dict(type="turboprop",  thrust_sl=90000, tr=1.072)
    elif case == 4:
        return dict(type="turbojet",  thrust_sl=60000, tr=1.072, mil=False)
    elif case == 5:
        # B747-400
        return dict(type="turbofan",  thrust_sl=282e3*4, tr=1.11, mil=False, high_bpr=True)
    elif case == 6:
        # TBM 700
        return dict(type="turboprop", thrust_sl=13500, tr=1.04)


