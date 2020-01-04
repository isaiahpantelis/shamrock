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
plt.show()

# # # -- DIFFERENTIATION
# # dy1 = f.eval(x, diff_order=1)
# # # # pprint(f.dcoeffs)
# # dy2 = f.eval(x, diff_order=2)
# # # # pprint(f.dcoeffs)
# # dy3 = f.eval(x, diff_order=3)
# # dy4 = f.eval(x, diff_order=4)
# # # # pprint(f.dcoeffs)
#
# # -- ROOT FINDING
# roots = f.solve()
# # print(f'roots = {roots}')
# # droots = f.solve(diff_order=1)
# # print(f'droots = {droots}')
# # ddroots = f.solve(diff_order=2)
# # # print(f'ddroots = {ddroots}')
#
# # -- OPTIMA
# critical, minima, maxima = f.optimise()
#
# # print(f'-- critical = {critical}')
# # print(f'-- minima = {minima}')
# # print(f'-- maxima = {maxima}')
#
# # print(f'[python] f.dcoeffs = {f.dcoeffs}')
# # print(f'[python] f.dcoeffs = {f.dcoeffs}')
# # print(f'[python] f.coeffs = {f.coeffs}')
# # print(f'[python] f.dcoeffs = {f.dcoeffs}')
# # print(f'[python] order = {len(f.dcoeffs)}')
# # print(f'[python] #f.dcoeffs = {len(f.dcoeffs[0])}')
#
# # -- PLOTS
# plt.figure()
# plt.plot(x, y, label='y')
# # plt.plot(x, yhat, label=f'yhat')
# # plt.plot(x, dy1, linestyle='--', linewidth=1.0, label=f'dy1')
# # plt.plot(x, dy2, linestyle='--', linewidth=1.0, label=f'dy2')
# # plt.plot(x, dy3, linestyle='--', linewidth=1.0, label=f'dy3')
# # plt.plot(x, dy4, linestyle='--', linewidth=1.0, label=f'dy4')
# # plt.plot(f.p.x, f.fvals, marker='.', linewidth=0.0, label='fvals')
# plt.plot(roots, np.zeros((len(roots),)), marker='.', linewidth=0.0, markersize=12, label='roots')
# # plt.plot(droots, np.zeros((len(droots),)), marker='.', linewidth=0.0, markersize=12, label='droots')
# # plt.plot(ddroots, np.zeros((len(ddroots),)), marker='.', linewidth=0.0, markersize=12, label='ddroots')
#
# # plt.plot(critical, np.zeros((len(critical),)), marker='.', linewidth=0.0, markersize=12, label='critical')
# minima = np.array([_[0] for _ in zip(critical, minima) if _[1] > 0])
# plt.plot(minima, f(minima), marker='.', linewidth=0.0, markersize=12, label='minima')
# maxima = np.array([_[0] for _ in zip(critical, maxima) if _[1] > 0])
# plt.plot(maxima, f(maxima), marker='.', linewidth=0.0, markersize=12, label='maxima')
#
# plt.legend()
# plt.grid()
# # plt.show()
#
# # -- ERROR
# # plt.figure()
# # plt.plot(x, y - yhat, label='error')
# # plt.legend()
# # plt.grid()
#
# # -- COEFFS
# coeffs = f.coeffs[0]
# plt.figure()
# plt.plot(list(range(len(coeffs))), coeffs, linestyle='--', marker='.', label='coeffs')
# plt.legend()
# plt.grid()
# # plt.show()
#
# # # -- Root checking
# # vals = []
# # for root in roots:
# #     vals.append(f(np.array(root, ndmin=1)))
# # plt.figure()
# # plt.plot(roots, vals, linestyle='--', linewidth=0, marker='.', label='values at roots')
# # plt.legend()
# # plt.grid()
#
# plt.show()