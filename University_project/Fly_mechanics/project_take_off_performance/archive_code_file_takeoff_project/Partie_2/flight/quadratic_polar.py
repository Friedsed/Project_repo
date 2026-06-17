# -*- coding: utf-8 -*-
"""
Flight Mechanics project
March 2023

Quadratic polar class


Polar:
    cd = a_0 cl^2 + a_1 cl + a_2
    cd = cd_0 + k (cl-cl_0)^2
    k = 1 / (2 pi lambda)

    p_2 = [a_0, a_1, a_2]
    a_0 = k
    a_1 = - 2 k cl_0
    a_2 = k  cl_0^2 + cd_0

Required power:
    Pr = a_1 V^3 + a_2 / V + a_3 V

+--------------+---------------+------------------------------+
| Python index |  doc index    |      optimal quantity        |
+==============+===============+==============================+
|      0       |     1, 3      |    aerodynamic efficiency    |
|      1       |     2, 4      |        required power        |
+--------------+---------------+------------------------------+

@author: Christophe Airiau
"""
import numpy as np
from flight.atmosphere_std import RHOZERO, SimpleAtmosphere
from flight.aircraft import Aircraft, __G__


class QuadraticPolar(object):
    """
    Aircraft quadratic polar
    Attributes :
    k, cd_0, cl_0, e
    p_2 : polynomial of the polar
    ar_x_e : ar x e
    cl_ref, f_ref, pr_ref   : reference values when C_{L_0} = 0
    epsilon, epsilon2
    cl,  cd, f : lift and drag coefficients, f = C_L / C_D
    q : dynamic pressure
    v, pr : velocity and required power for optimal aerodynamic efficiency and required power
    w, s, rho, sigma : usual meaning
    w_s, ar : idem
    a_1, a_2, a_3 : coefficients to evaluate the required power
    """
    def __init__(self, d=None):
        """
        """
        if d is None:
            d = {"k": 0.02, "cd_0": 0.02, "cl_0": 0.01}
        if "p_2" in d.keys():
            self.p_2 = d["p_2"]
            self.k = self.p_2[0]
            self.cl_0 = -self.p_2[1] / (2 * self.k)
            self.cd_0 = self.p_2[2] - self.k * self.cl_0**2
        else:
            self.k, self.cd_0, self.cl_0 = d["k"], d["cd_0"], d["cl_0"]
            self.p_2 = [self.k, - 2 * self.k * self.cl_0, self.k * self.cl_0**2 + self.cd_0]

        self.ar_x_e = 1 / (self.k * np.pi)
        self.cl_ref = self.f_ref = None
        self.epsilon_2 = self.epsilon = None
        self.cl, self.cd, self.f, self.q = [0, 0], [0, 0], [0, 0], [0, 0]
        self.v, self.pr = [0, 0], [0, 0]
        self.w, self.s, self.rho, self.sigma = None, None, None, None
        self.pr_ref = None
        self.a_1 = self.a_2 = self.a_3 = self.eta_p = None
        self.e, self.w_s, self.ar = None, None, None
        self.aircraft = None

    def __repr__(self):
        n = 80
        print("=" * n)
        print("#  Quadratic polar")
        print("=" * n)
        column_names = "  k x 100   CL_0   CD_0 x 10 000   lambda x e   a_0          a_1         a_2 "
        print(column_names)

        form = "%6.2f    %6.2f       %6.2f        %6.2f   %9.3e  %9.3e  %10.3e"
        print("-" * n)
        print(form % (self.k * 100, self.cl_0, self.cd_0 * 1e4, self.ar_x_e, self.p_2[0], self.p_2[1], self.p_2[2]))
        print("=" * n)
        if self.e and self.ar:
            print("Oswald coefficient      : %.3f " % self.e)
            print("Aspect ratio            : %.3f " % self.ar)
        return " "

    # ****************************************
    # Solvers
    # ****************************************

    def solve(self, m=10000, s=50, rho=1):
        """
        main function to solve the direct problem
        """
        self.solve_init()
        self.solve_velocities(m=m, s=s, rho=rho)
        self.set_power_coefficients()
        self.solve_power()
        self.get_data()

    def solve_init(self):
        """
        inputs: cd_0, k, epsilon
        return :
            cl, cd, f @ optimal f_max (index 0)
            f_ref, cl_ref
        """
        self.cl_ref = np.sqrt(self.cd_0/self.k)
        self.f_ref = 1 / (2 * np.sqrt(self.k * self.cd_0))
        self.epsilon = self.cl_0 / self.cl_ref
        self.cd[0] = 2 * self.cd_0 * self.drag_correction(self.epsilon)
        # self.cl[0] = self.cl_ref * self.lift_correction(self.epsilon)
        self.cl[0] = np.sqrt(self.cd_0 / self.k + self.cl_0 ** 2)
        self.f[0] = self.f_ref * self.f_correction(self.epsilon)
        print("epsilon :  %.3f, \t\t CL_opt (max f)  : %.3f " % (self.epsilon, self.cl[0]))

    def solve_velocities(self, m=10000, s=50, rho=1):
        """
        return quantities at maximum aerodynamic efficiency f_max:
        pr_ref
        v, q, Pr @ f_max
        """
        self.w, self.s, self.rho = m * __G__, s, rho
        c = np.sqrt(2 * self.w / (self.rho * self.s))
        self.pr_ref = self.w * c * pow(self.k**3*self.cd_0, 1/4)
        self.q[0] = self.w / (self.s * self.cl[0])
        self.v[0] = np.sqrt(2 * self.q[0] / self.rho)
        self.pr[0] = self.w / self.f[0] * self.v[0]
        # self.pr[0] = self.q[0] * self.s * self.cd[0] * self.v[0]  (same as previous)

    def solve_power(self):
        """
        return quantities at optimal required power (mimimum) Pr_min
        Cl_4, Cd_4, f_4, q_4, Pr_4 = Pr_min
        stored with index 1
        """
        v_4_to_4 = 1 / 3 * self.a_2 / self.a_1 + (
                    self.a_3 ** 2 - self.a_3 * np.sqrt(12 * self.a_1 * self.a_2 + self.a_3 ** 2)) \
                   / (18 * self.a_1 ** 2)
        self.v[1] = pow(v_4_to_4, 1 / 4)
        cl4_bis = 12 * self.w / (self.rho * self.s) * self.a_1 / \
                  (- self.a_3 + np.sqrt(self.a_3 ** 2 + 12 * self.a_1 * self.a_2))
        cl4 = 3 * (self.cd_0 + self.k * self.cl_0 ** 2) / \
              (self.k * self.cl_0 + 2 * np.sqrt(self.k * (3 / 4 * self.cd_0 + self.k * self.cl_0 ** 2)))
        print('test CL_4 : %.4f  == %.4f ' % (cl4, cl4_bis))
        self.cl[1] = cl4
        self.cd[1] = np.polyval(self.p_2, cl4)
        self.f[1] = self.cl[1] / self.cd[1]
        self.q[1] = self.w / (self.s * self.cl[1])
        self.pr[1] = self.w / self.f[1] * self.v[1]

    def solve_inverse(self):
        """
        return the solution of the inverse problem
        Inputs :
            Cl, f, v, Pr @ f_max, epsilon, e
        Outputs :
            q, C_D, C_L @ f_max
            C_{D_0}, C_{L_0}, p_2, k
            AR, S, b
            W, W/S,
        """
        self.w_s = self.rho * self.v[0]**2 * self.cl[0] / 2
        self.w = self.pr[0] * self.f[0] / self.v[0]
        self.s = self.w / self.w_s
        self.q[0] = 1/2 * self.rho * self.v[0]**2
        self.cd[0] = self.cl[0] / self.f[0]
        self.cd_0 = self.cd[0] / (2 * self.drag_correction(self.epsilon))
        self.k = (1 + self.epsilon**2) * self.cd_0 / self.cl[0]**2
        self.cl_0 = self.epsilon * np.sqrt(self.cd_0/self.k)
        self.p_2 = [self.k, - 2 * self.k * self.cl_0, self.k * self.cl_0 ** 2 + self.cd_0]
        self.ar_x_e = 1 / (self.k * np.pi)
        self.ar = self.ar_x_e / self.e
        self.set_power_coefficients()
        self.solve_power()
        self.get_data()
        self.aircraft = Aircraft({"name": "Plane_2", "m": self.w/__G__, "s": self.s, "AR": self.ar,
                                  "b": np.sqrt(self.ar * self.s), "e": self.e, "rho": self.rho,
                                  "v": np.linspace(100, 400)/3.6})

    # ****************************************
    #  setters
    # ****************************************

    def set_data(self, d=None):
        """
        Enter data from the inverse problem
        returns:
        f, v, Pr, C_L @ f_max
        epsilon
        with the right dimension if necessary
        """
        if d is None:
            d = {"e": 0.85, "h": 5000, "CL_1": 0.44, "f_max": 22.5, "V_1": 340, "Pr_1": 410, "epsilon": 0.1}
        self.e = d["e"]
        self.sigma, _, _ = SimpleAtmosphere(d["h"]/1000)
        self.rho = RHOZERO * self.sigma
        self.f[0] = d["f_max"]
        self.v[0] = d["V_1"] / 3.6
        self.pr[0] = d["Pr_1"] * 1e3
        self.cl[0] = d["CL_1"]
        self.epsilon = d["epsilon"]

    def set_flight_data(self, m=10000, s=50, rho=1):
        """
        data entered from the user
        """
        self.w, self.s, self.rho = m * __G__, s, rho

    def set_power_coefficients(self, unit_density=False):
        """
        unit_density: to set the coefficient with varying density later
        without unit density:
        Pr = a_1 V^3 + a_2 / V + a_3 V
        with unit density:
        Pr = rho a_1 V^3 + a_2 / rho / V + a_3 V
        """
        if unit_density:
            rho = 1
        else:
            rho = self.rho
        self.a_1 = rho * self.s / 2 * (self.cd_0 + self.k * self.cl_0**2)
        self.a_2 = 2 * self.k * self.w**2 / (rho * self.s)
        self.a_3 = - 2 * self.k * self.w * self.cl_0

        if unit_density:
            return self.a_1, self.a_2, self.a_3

    def set_aspect_ratio(self, e=0.85):
        """
        from the Oswald coefficient return the wing aspect ratio
        """
        self.ar = self.ar_x_e / e

    @staticmethod
    def drag_correction(x):
        """ given from epsilon @ f_max"""
        y = np.sqrt(1 + x ** 2)
        return y * (y - x)

    @staticmethod
    def lift_correction(x):
        """ given from epsilon @ f_max"""
        return np.sqrt(1 + x ** 2)

    @staticmethod
    def f_correction(x):
        """ Aerodynamic efficiency correction f from epsilon @ f_max"""
        return 1 / (np.sqrt(1 + x ** 2) - x)

    @staticmethod
    def speed_correction(x):
        """ from epsilon @f_max"""
        return pow(1 + x ** 2, 1 / 4)

    # ****************************************
    #  getters
    # ****************************************

    def get_data(self):
        """
        data displayed in a nice table
        """
        title = ("aerodynamic efficiency", "required power")
        column_title = " CL     CD x 10 000       f       V [km/h]    P [kW]    q [Pa] \t optimal criteria"
        form = " %6.3f   %6.1f      %6.2f     %6.1f      %6.2f     %6.1f \t %s"
        n = 100
        print("=" * n)
        print(column_title)
        print("-" * n)
        for i in range(2):
            print(form % (self.cl[i], self.cd[i]*1e4, self.f[i], self.v[i]*3.6, self.pr[i]/1000, self.q[i], title[i]))
        # print("Ratios state 2 / state 1")
        print("-" * n)
        form1 = " %6.4f   %6.4f      %6.4f     %6.4f      %6.4f     %6.4f \t %s "
        print(form1 % (self.cl[1] / self.cl[0], self.cd[1] / self.cd[0], self.f[1] / self.f[0],
                       self.v[1] / self.v[0], self.pr[1] / self.pr[0], self.q[1] / self.q[0],
                       "Ratios state 2 / state 1"))
        print("=" * n)
        print("Required power coefficients : ")
        print('[a_1 : %.4f , \t a_2 / 10^6 : %.4f , \t a_3 : %.4f ]' % (self.a_1, self.a_2 * 1e-6, self.a_3))
