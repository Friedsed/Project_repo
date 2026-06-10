# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

aerodynamic fields :
    AeroFields : a simple containers for v, q, C_L, C_D, f, Pr, D, L

@author: Christophe Airiau
"""

import numpy as np
from flight.aircraft import Aircraft


class AeroFields(object):
    """
    Container of the aerodynamic fields for plot purposes
    s, v, q, cl, cd, pr, f,
    indexes of optimal aerodynamics efficiency and required power
    \(P_r = \sqrt{\dfrac{2 W^3}{rho S}} \dfrac{C_D}{C_L^{3/2}} \)
    """
    def __init__(self, model, aircraft, cd):
        """
        SI unit only
        """
        if aircraft is None:
            aircraft = Aircraft()
        self.model = model
        self.v = aircraft.v
        self.q = aircraft.q
        self.cl, self.cd = aircraft.cl, cd
        self.drag = self.q * aircraft.s * cd
        self.lift = self.q * aircraft.s * self.cl
        self.f = self.cl / self.cd
        self.pr = self.drag * self.v
        self.pr_coef = 1 / (self.f * np.sqrt(self.cl))    # C_D / C_L^{3/2}
        self.index_f = np.argmax(self.f)
        self.index_pr = np.argmin(self.pr)

    def __repr__(self):
        """
        used when printing the class instance
        optimal values for min(Pr)
        """
        # it could be modified
        n = 60
        print()
        print("=" * n)
        print("# Model : ", self.model)
        print("=" * n)
        print(" v_opt [km/h]     q_opt [Pa]     Pr_min [kW]")
        print("-" * n)
        form = " %6.1f           %6.1f         %6.2f"
        print(form % (self.v[self.index_pr] * 3.6, self.q[self.index_pr], self.pr[self.index_pr] / 1000))
        print("=" * n)
        return " "

