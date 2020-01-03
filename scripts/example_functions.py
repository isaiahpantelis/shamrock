import math
import numpy as np


ChebPoly = {'T0': lambda x: 1,
            'T1': lambda x: x,
            'T2': lambda x: 2 * (x**2) - 1,
            'T3': lambda x: 4 * (x**3) - 3 * x,
            'T4': lambda x: 8 * (x**4) - 8 * (x**2) + 1}


def mexhat(x, alpha=0.01):
    return np.exp(-alpha * x * x) * np.cos(x)


# -- Decaying cosine
def decaying_cos(x, alpha=0.01):
    return np.exp(-alpha * x) * np.cos(x * x)
    # return np.cos(x)


# -- Runge's function
def runge(x):
    return 1 / (1 + 25.0 * x * x)


# -- Square-root cusp
def sqcusp(x):
    return math.sqrt(math.fabs(x)) ** (2.0 / 5.0)


# -- Quadratic with two small real roots
def quad(x):
    return (x - 1.0) ** 2 - 0.000001


# -- Useful for testing corner cases
def lin(x):
    return x


def expf(x):
    return np.exp(x)


def g(x):
    return np.sin(x) + 0.1 * np.sin(10*x)


def h(x):
    return 0.1 * x * np.exp(0.01 * x) * np.sin(x) + 0.1 * np.sin(10*x)


def tanh(x):
    return np.tanh(x)
