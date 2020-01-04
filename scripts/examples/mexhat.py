import os
import sys

# -- `scripts` folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shamrock as sh
import numpy as np
import matplotlib.pyplot as plt
# from pprint import pprint

a = -5.0 * np.pi
b = 5.0 * np.pi
I = (a, b)


@sh.ChebProxyDec(I=I)
def f(x):
    return np.exp(-0.01 * x * x) * np.cos(x)


nx = 1000
x = np.linspace(a, b, nx)

# -- Plot the function
y = f.eval(x, mode=b'exact')

plt.figure()
plt.plot(x, y, label='y')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()

# -- ROOT FINDING
roots = f.solve()

plt.figure()
plt.plot(x, y, label='y')
plt.plot(roots, np.zeros((len(roots),)), marker='.', linewidth=0.0, label='roots')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()

# -- OPTIMA
critical, argmin, argmax = f.optimise()
argmin = critical[np.where(argmin == 1)]
argmax = critical[np.where(argmax == 1)]
minvals = f.eval(argmin, mode=b'exact')
maxvals = f.eval(argmax, mode=b'exact')

plt.figure()
plt.plot(x, y, label='y')
plt.plot(roots, np.zeros((len(roots),)), marker='.', linewidth=0.0, label='roots')
plt.plot(argmin, minvals, marker='.', linewidth=0.0, label='minima')
plt.plot(argmax, maxvals, marker='.', linewidth=0.0, label='maxima')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()

plt.show()
