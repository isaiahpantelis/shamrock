"""
Contains python wrappers around cython objects or python objects built out of cython objects.

TODO: 1. Handle graciously tiny intervals (a almost equal to b) with a large number of interpolation points. In extreme
         cases, the interpolation points will be equal.
      2. Log the test results
"""

# -- Library modules
import numpy as np
import sys
import pprint
from typing import Callable
# from time import process_time

# -- Project modules
from shamrock import shamrock_cy
from shamrock.common import errmsg


# ----------------------------------------------------------------------------------------------------------------------

# -- `chebcoeffs` is more than a simple reference to shamrock_cy.chebcoeffs because some additional steps are needed
# -- before calling `shamrock_cy.chebcoeffs`: we inherit from a Cython class and pass an object that wraps f
def chebcoeffs(f, p):

    class FunctionWrapper(shamrock_cy.FunctionWrapper):
        pass

    F = FunctionWrapper()
    F.eval = f

    return shamrock_cy.chebcoeffs(F, p)


# ----------------------------------------------------------------------------------------------------------------------


# -- `chebcoefft` is more than a simple reference to shamrock_cy.chebcoeffs because some additional steps are needed
# -- before calling `shamrock_cy.chebcoeffs`: we inherit from a Cython class and pass an object that wraps f
def chebcoefft(f, p):

    class FunctionWrapper(shamrock_cy.FunctionWrapper):
        pass

    F = FunctionWrapper()
    F.eval = f

    return shamrock_cy.chebcoefft(F, p)


# ----------------------------------------------------------------------------------------------------------------------


def chebeval(coeffs, a, b, x):

    if not isinstance(x, np.ndarray):
        x = np.array(x, copy=False)

    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs, copy=False)

    return shamrock_cy.chebeval(coeffs, a, b, x)


# ======================================================================================================================
# -- ChebProxy
# ======================================================================================================================
def ChebProxy(f: Callable[[float], float], I=(-1.0, 1.0), K=None, settings=shamrock_cy.Settings_cy()):
    """
    Representation of a function by a truncated Chebyshev series. There is a decorator, below, based on this function.
    """

    class FunctionWrapper(shamrock_cy.FunctionWrapper):
        pass

    F = FunctionWrapper()
    F.eval = f.__call__

    exception = ValueError(errmsg(fileName='shamrock.py',
                                  className='ChebProxy.py',
                                  methodName='__init__',
                                  message=f'Illegal value K = {K}. The order of approximation K must be an integer greater than 0 and less than {settings.Kmax + 1}.'))

    if K is None:
        proxy = shamrock_cy.ChebProxy_cy(F, I, -1, settings)
        return proxy
    else:
        if not isinstance(K, int) or K < 1 or K > settings.Kmax:
            raise exception
        return shamrock_cy.ChebProxy_cy(F, I, K, settings)


# ======================================================================================================================
# -- ChebProxy decorator
# ======================================================================================================================
def ChebProxyDec(I=(-1.0, 1.0), K=None, settings=shamrock_cy.Settings_cy()):
    def decorator(f):
        return ChebProxy(f, I, K, settings)
    return decorator
