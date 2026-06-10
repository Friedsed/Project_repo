# -*- coding: utf-8 -*-
"""
Module to perform Bezier interpolation

@Author: Omar Aflak
https://towardsdatascience.com/b%C3%A9zier-interpolation-8033e9a262c2

reformatted by C. Airiau, March 2023

"""

import numpy as np
import matplotlib.pyplot as plt
from flight.solution_p3 import Resolution

plt.style.use("seaborn-notebook")


def get_bezier_coef(points):
    """
    find the a & b points

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    """
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficients matrix
    c_mat = 4 * np.identity(n)
    np.fill_diagonal(c_mat[1:], 1)
    np.fill_diagonal(c_mat[:, 1:], 1)
    c_mat[0, 0] = 2
    c_mat[n - 1, n - 1] = 7
    c_mat[n - 1, n - 2] = 2

    # build points vector
    pt_vec = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    pt_vec[0] = points[0] + 2 * points[1]
    pt_vec[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    a_mat = np.linalg.solve(c_mat, pt_vec)
    b_mat = [0] * n
    for i in range(n - 1):
        b_mat[i] = 2 * points[i + 1] - a_mat[i + 1]
    b_mat[n - 1] = (a_mat[n - 1] + points[n]) / 2

    return a_mat, b_mat


def get_cubic(a, b, c, d):
    """
    returns the general Bezier cubic formula given 4 control points

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + \
                     3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


def get_bezier_cubic(points):
    """
    return one cubic curve for each consecutive points

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    a_mat, b_mat = get_bezier_coef(points)
    return [
        get_cubic(points[i], a_mat[i], b_mat[i], points[i + 1])
        for i in range(len(points) - 1)
    ]


def evaluate_bezier(points, n):
    """
    evalute each cubic curve on the range [0, 1] sliced in n points

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])


def bernstein_poly(t, a, b, c, d):
    """

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + \
           3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


def panel(eta_p, eta):  # set_panel_index
    """ """
    for k in range(len(eta_p) - 1):
        if eta_p[k] <= eta <= eta_p[k + 1]:
            print("found")
            return k
    raise ValueError(" %f not in  [%f, %f] in panel" % (eta, eta_p[0], eta_p[-1]))


def get_points_from_curves(points, d, var="x"):
    """ """
    xp, yp = points[:, 0], points[:, 1]
    v = ["x", "y"]
    if var not in v:
        raise ValueError("bad values of x or y")
    index = v.index(var)
    rhs = d[var]
    i = panel(points[:, index], d[var])
    a_bezier, b_bezier = get_bezier_coef(points)
    s = get_cubic(points[i], a_bezier[i], b_bezier[i], points[i + 1])
    coef = []
    for k in range(2):
        a_, b_, c_, d_ = points[i][k], a_bezier[i, k], b_bezier[i][k], points[i + 1][k]
        coef.append([-a_ + 3 * b_ - 3 * c_ + d_, 3 * (a_ - 2 * b_ + c_), 3 * (b_ - a_), a_])

    a, b, c, d = coef[index]
    d -= rhs
    print("a, b, c, d : ", a, b, c, d)
    r = Resolution(a, b, c, d, vb=False)
    # r = p3_roots([a, b, c, d])
    for v in r:
        if 0 <= v.x <= 1 and abs(v.y) <= 1e-12:
            break
    t_value = v.x
    x_value = np.polyval(coef[0], t_value)
    y_value = np.polyval(coef[1], t_value)
    return i, t_value, x_value, y_value
