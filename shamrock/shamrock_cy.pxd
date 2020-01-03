from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.map cimport map
from libcpp cimport bool

# ----------------------------------------------------------------------------------------------------------------------
# -- This cdef extern seems vacuous, but it's used to include all the code in `shamrock_cpp.cpp`.
# ----------------------------------------------------------------------------------------------------------------------
cdef extern from "shamrock_cpp.cpp":
    pass

# ----------------------------------------------------------------------------------------------------------------------
# -- Declaration of a specialisation of the C++ `partition` template function
#
# -- Returns the Chebyshev interpolation points (C++ implementation)
# -- Note that the C++ function is a template and we could cdef-extern it using Cython's support for templates. For now,
# -- the flexibility is kept in the C++ code, but the Cython function is specialised.
# ----------------------------------------------------------------------------------------------------------------------
cdef extern from 'shamrock_cpp.h' namespace 'chebyshev':
    vector[double] partition(const double, const double, const unsigned long) except +;

# ----------------------------------------------------------------------------------------------------------------------
# -- Declaration of the C++ `Partition` class
# ----------------------------------------------------------------------------------------------------------------------
cdef extern from 'shamrock_cpp.h' namespace 'chebyshev':
    cdef cppclass Partition:
        unsigned long K, N
        double a, b
        vector[unsigned long] K_history
        vector[unsigned long] N_history
        vector[double] partition
        Partition()
        Partition(const double, const double, const unsigned long)
        void refine()
        void coarsen()

# ----------------------------------------------------------------------------------------------------------------------
# -- Declaration of the C++ `ChebPoly` class
# ----------------------------------------------------------------------------------------------------------------------
cdef extern from 'shamrock_cpp.h' namespace 'chebyshev':
    cdef cppclass ChebPoly:
        map[unsigned long, vector[long]] look_up_table
        unsigned long kind
        #ChebPoly()
        ChebPoly(unsigned long) except+
        vector[long] T(const unsigned long)