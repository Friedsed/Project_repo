# -*- coding: utf-8 -*-
"""
résolution de l'équation du troisième degré par la méthode de Cardan

Trouvé sur internet.
Mise en forme en Mars 2023 par C. Airiau
"""

import math 
import numpy as np


class MyComplex(object):
    """ modélisation des nombres complexes """ 
    def __init__(self, a=0.0, b=0.0):
        """le constructeur soit en cartésiennes soit en polaires""" 
        self.x = a
        self.y = b
 
    def __str__(self): 
        """ représentation externe pour print et str """ 
        if self.y == 0:
            return str(self.x) 
        if self.x == 0 and self.y == 1:
            return "i" 
        if self.x == 0 and self.y == -1:
            return "-i" 
        if self.x == 0:
            return str(self.y) + "i"
        if self.y == 1:
            return str(self.x) + "+i"
        if self.y == -1:
            return str(self.x) + "-i"
        if self.y > 0:
            return str(self.x) + "+"+str(self.y) + "i"
        else: 
            return str(self.x) + str(self.y) + "i"
 
    def __add__(self, other):
        """somme de deux complexes""" 
        a = self.x + other.x
        b = self.y + other.y
        return MyComplex(a, b)
 
    def __sub__(self, other):
        """différence de deux complexes""" 
        a = self.x - other.x
        b = self.y - other.y
        return MyComplex(a, b)
 
    def __neg__(self): 
        """opposé d'un complexe""" 
        return MyComplex(-self.x, -self.y)
 
    def null(self): 
        """test de nullité""" 
        return self.x == 0 and self.y == 0
 
    def __mul__(self, other):
        """produit de deux complexes""" 
        a = self.x * other.x - self.y * other.y
        b = self.x * other.y + self.y * other.x
        return MyComplex(a, b)
 
    def __truediv__(self, other):
        """quotient de deux complexes""" 
        return self*(~other) 
 
    def conj(self): 
         """conjugué d'un complexe""" 
         return MyComplex(self.x, -self.y)
 
    def module(self): 
        """module d'un complexe""" 
        return math.sqrt(self.x * self.x + self.y * self.y)

    def argument(self):
        """argument d'un complexe""" 
        if self.x == 0 and self.y == 0:
            return 0 
        if self.x == 0 and self.y > 0:
            return math.pi/2 
        if self.x == 0 and self.y < 0:
            return -math.pi/2 
        if self.x > 0:
            return math.atan(self.y / self.x)
        return math.pi - math.atan(-self.y / self.x)
 
    def __invert__(self): 
        """inverse d'un complexe""" 
        if self.null(): 
            raise ZeroDivisionError 
        return MyComplex(self.x / (self.x * self.x + self.y * self.y), -self.y / (self.x * self.x + self.y * self.y))
 
    def __pow__(self, n):
        """puissances d'un complexe"""
        r = MyComplex(1, 0)
        if n <= 0 and self.null():
            raise ZeroDivisionError 
        if n >= 0:
            while n: 
                r = r * self
                n = n-1
            return r
        if n < 0:
            return (~r)**(-n)
 
    def racines(self, n):
        """calcule les n racines n-ièmes du nombre""" 
        # on utilise les racines de l'unité 
        return [MyComplex(self.module() ** (1.0 / n) * math.cos((k * 2 * math.pi + self.argument()) / n),
                          self.module() ** (1.0 / n) * math.sin((k * 2 * math.pi + self.argument()) / n))
                for k in range(0, n)]


def Cardan(a, b, c, d):
    """ a, b, c, d sont les coefficients initiaux de l'équation"""
    # on commence par mettre sous forme canonique
    b, c, d = b / a, c / a, d / a
    p = c - b * b / MyComplex(3.0, 0)
    q = d - b * c / MyComplex(3.0, 0) - (b ** 3) / MyComplex(27.0, 0) + (b ** 3) / MyComplex(9.0, 0)
    B, C = q, -p * p * p / MyComplex(27.0, 0)
    D = B * B - MyComplex(4.0, 0) * C
    R = D.racines(2)
    U = (-B + R[0]) / MyComplex(2.0, 0)
    roots = U.racines(3)
    sol1 = [u - p / (MyComplex(3.0, 0) * u) for u in roots]
    sol2 = [z - b / MyComplex(3.0, 0) for z in sol1]
    return sol2 


def print_roots_(r):
    """ display roots on screen """
    for z in r:
        if abs(z.y) <= 1e-13:
            print("\t real    : %.5f" % z.x)
        else:
            print("\t complex : %.5f + i %.5f" % (z.x, z.y))


def Resolution(a, b, c, d, vb=False):
    """Résout l'équation az^3+bz^2+c^z+d=0""" 
    # les coefficients peuvent être entiers, réels ou complexes 
    # Dans tous les cas on convertit en complexes pour commencer 
    print("résolution")
    if isinstance(a, float) or isinstance(a, int):
        a = MyComplex(float(a), 0)
    if isinstance(b, float) or isinstance(b, int):
        b = MyComplex(float(b), 0)
    if isinstance(c, float) or isinstance(c, int):
        c = MyComplex(float(c), 0)
    if isinstance(d,float) or isinstance(d, int):
        d = MyComplex(float(d), 0)
    Z = Cardan(a, b, c, d)
    print("roots from solution_p3 : ")
    print_roots_(Z)
    if vb:
        print("Error = P(Z)")
        for z in Z:
            print(a * z**3 + b * z**2 + c * z + d)
    return Z


def main():
    t = [[1, 1/2, -5/2, 1 ],
         [1, MyComplex(-3 / 2, 2), MyComplex(1 / 2, -3), MyComplex(0, 1)],
         [1, MyComplex(0, 3 / 2), 3 / 2, MyComplex(0, 1)]]
    for v in t:
        Resolution(v[0], v[1], v[2], v[3])


def poly_roots(p, vb=True):
    """
    Parameters
    ----------
    p: polynomial

    Returns
    -------
    list of roots
    """
    if vb:
        print("roots from numpy.roots : ")
    r = np.roots(p)
    real_roots = []
    complex_roots = []
    for k, z in enumerate(r):
        if abs(z.imag) <= 1e-13:
            real_roots.append(z.real)
        else:
            complex_roots.append(z)
    roots = []
    if real_roots:
        for r in sorted(real_roots):
            if vb:
                print("\t real    : %.5f" % r)
            roots.append(r)
    if complex_roots:
        for z in sorted(complex_roots):
            if vb:
                print("\t complex : %.5f + i %.5f" % (z.real, z.imag))
            roots.append(z)
    return roots


if __name__ == '__main__': 
    main() 
