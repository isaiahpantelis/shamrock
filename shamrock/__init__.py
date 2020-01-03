"""
[2019-11-05]
Algorithms that need to evaluate a user-supplied function cannot be implemented entirely in C++. This is, mainly, the
reason that there are two sets of imports, below. The first set is for calculations (algorithms) that are in pure C++
and wrapped in Cython. The second set of imports is for python objects (functions, classes, etc) that wrap algorithms
implemented in Cython.
"""
# -- Library modules
import numpy as np

# -- Import objects from the Cython extension `shamrock_cy`
from .shamrock_cy import Partition_cy as Partition
from .shamrock_cy import ChebPoly_cy as ChebPoly
from .shamrock_cy import chebdf
from .shamrock_cy import chebyshev_companion_matrix_cy as chebyshev_companion_matrix
from .shamrock_cy import Settings_cy as Settings

# -- Import objects from the python module `shamrock.py`
# -- These imports make the imported objects available at the top level of the package. For example, we can write
# -- `shamrock.chebcoefft` instead of `shamrock.shamrock.chebcoefft`
from .shamrock import chebcoeffs
from .shamrock import chebcoefft
from .shamrock import chebeval
from .shamrock import ChebProxy
from .shamrock import ChebProxyDec
