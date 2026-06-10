# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:21:14 2023

@author: airiau
"""
import numpy as np
import matplotlib.pyplot as plt


def set_local_zeros_vec(x=None, y=None, vb=False):
    """
    search for the zeros of a given discrete function z = f(x)
    """
    g = y[:-1] * y[1:]
    # dz, dx = np.diff(y), np.diff(x)
    zeros = []
    counter = 0
    for k in range(len(y) - 1):
        if g[k] <= 0:
            counter += 1
            zeros.append(x[k] - y[k] * (x[k+1] - x[k]) / (y[k+1] - y[k]))
    if abs(y[-1]) <= 1e-10:
        zeros.append(x[-1])
    if vb:
        print("number of zeros found : ", counter)
    return zeros


def set_local_zeros_mat(x=None, y=None, vb=False):
    """
    search for the seros of given discrete data x: abscissa, y: matrix of column functions
    """
    zeros = []
    n, m = y.shape
    for i in range(m):
        zeros.append(set_local_zeros_vec(x=x, y=y[:, i], vb=vb))
    return zeros

def f(x, n):
    return np.sin(np.pi * (n+1) * x)
nx = 201
ny = 3
y = np.zeros((nx, ny), dtype=float)
x = np.linspace(-0.1, 6.5, nx)

for i in range(ny):
    y[:, i] = f(x, i)

x_zero = set_local_zeros_mat(x, y)
    
print(x_zero)

plt.figure()
for j in range(ny):
    plt.plot(x, y[:, j], label="n=%d" % i)
    plt.plot(x_zero[j], f(np.array(x_zero[j]), j), "ro")
plt.show()


        