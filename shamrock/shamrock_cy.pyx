# ======================================================================================================================
# -- TODO: Automatic interval sub-division
# -- TODO: Dynamic range check
# -- TODO: Newton polishing of candidate roots
# -- TODO: Consistent use of unsigned long
# ======================================================================================================================
# -- NOTES:
#    1. The use of `truncate()` is necessary only when the coefficients are used to construct a companion matrix, to
#       to avoid division by zero. If the coefficients are used for evaluation, then `truncate()` will just lead to
#       negligible computational savings.
#    2. In Cython, memory views can only be used as types of local variables and, hence, cannot be used to declare
#       class members. As a result, std::vectors, std::maps, and other C++ containers are used to declare members that
#       hold data. That leads to the need to cast to numpy arrays (or array views) before using certain variables.
# ======================================================================================================================
# -- Imports
# ======================================================================================================================
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp.string cimport string

cimport cython
from cython.operator import dereference, postincrement, postdecrement
# from cython cimport boundscheck, wraparound

# ----------------------------------------------------------------------------------------------------------------------
# -- The next line uses `shamrock_cy.pxd` to cimport the listed objects.
# ----------------------------------------------------------------------------------------------------------------------
from shamrock_cy cimport partition, Partition, ChebPoly

from shamrock.common import errmsg

import sys
import pprint
# from collections import OrderedDict
import numpy as np
cimport numpy as np
from scipy.fftpack import dct

ctypedef np.float64_t Real
ctypedef np.uint8_t uint8
ctypedef np.int64_t Int

# ======================================================================================================================
# -- Std lib functions
# ======================================================================================================================
# -- The std lib power function.
cdef extern from 'math.h':
    int pow(int, int)
    double abs(double)
    double cos(double)
    double fabs(double)

#cdef extern from 'algorithm.h':
#    int max(int, int)
#    int min(int, int)

cdef extern from 'math.h':
    double M_PI

# ======================================================================================================================
# -- Functions
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# -- FUNCTION: chebcoeffs
# ----------------------------------------------------------------------------------------------------------------------
# -- Chebyshev coefficients
# ----------------------------------------------------------------------------------------------------------------------
# -- NOTE: In most cases, chebcoefft should be used instead.
#
# -- From [Boyd, $B.2, p.411]
# -- NOTE: This function is implemented in Cython and not C++ because it uses array multiplication. However, array x
# -- vector multiplication is not strictly necessary; it simply represents a weighted sum which can be implemented in
# -- C++ without a matrix library.
# ----------------------------------------------------------------------------------------------------------------------
cpdef np.ndarray[Real, ndim=1] chebcoeffs(FunctionWrapper f, Partition_cy p):

    cdef unsigned long N = p.x.shape[0] - 1
    cdef unsigned long Np1 = p.x.shape[0]

    # -- Compute the values of f at the partition points.
    cdef unsigned long k = 0
    cdef np.ndarray[Real, ndim=1] fvals = np.empty((Np1,))

    for k in range(Np1):
        fvals[k] = f.eval(p.x[N - k])

    # -- The interpolation array [Boyd, $3.2, p.50]
    cdef np.ndarray[Real, ndim=2] I = np.empty((Np1, Np1))

    cdef unsigned long j = 0
    cdef double pj, pk

    for j in range(Np1):
        if j == 0 or j == N:
            pj = 2
        else:
            pj = 1
        for k in range(Np1):
            if k == 0 or k == N:
                pk = 2
            else:
                pk = 1
            I[j, k] = 2 * cos(k * j * M_PI / N) / N / pj / pk

    cdef np.ndarray[Real, ndim=1] coeffs = np.matmul(I, fvals)

    return coeffs

# ----------------------------------------------------------------------------------------------------------------------
# -- FUNCTION: Chebyshev coefficients via FFT
# ----------------------------------------------------------------------------------------------------------------------
#cpdef np.ndarray[Real, ndim=1] chebcoefft(FunctionWrapper f, Partition_cy p):
#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef pair[vector[Real], vector[Real]] chebcoefft(FunctionWrapper f, Partition_cy p):

    cdef unsigned long N = p.x.shape[0] - 1
    cdef unsigned long Np1 = p.x.shape[0]

    # -- Compute the values of f at the partition points.
    cdef np.ndarray[Real, ndim=1] fvals = np.empty((Np1,))

    cdef unsigned long k = 0

    for k in range(Np1):
        fvals[k] = f.eval(p.x[N - k])

    cdef np.ndarray[Real, ndim=1] coeffs = dct(fvals, type=1) / N
    coeffs[0] /= 2.0
    coeffs[N] /= 2.0

    return coeffs, fvals[::-1]


# ----------------------------------------------------------------------------------------------------------------------
# -- FUNCTION: Chebyshev coefficients of the first derivative
# ----------------------------------------------------------------------------------------------------------------------
# -- [Boyd, $B.2, p.414]
cpdef np.ndarray[Real, ndim=1] chebdf(double[:] coeffs, double a, double b):

    cdef unsigned long N = coeffs.shape[0] - 1
    cdef unsigned long Np1 = coeffs.shape[0]

    cdef np.ndarray[Real, ndim=1] out = np.empty((Np1,))

    S = 2.0 / (b - a)
    out[N] = 0.0
    out[N - 1] = 2 * N * coeffs[N]

    cdef unsigned long k = 0

    for k in range(N - 2, 0, -1):
        out[k] = out[k + 2] + 2 * (k + 1) * coeffs[k + 1]

    out[0] = out[2] / 2.0 + coeffs[1];
    out = S * out;

    return out


# ----------------------------------------------------------------------------------------------------------------------
# -- FUNCTION: Evaluate a truncated Chebyshev series.
# ----------------------------------------------------------------------------------------------------------------------
# -- [Boyd, $B.2, p.413]

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef np.ndarray[Real, ndim=1] chebeval(double[:] coeffs, double a, double b, double[:] x):

    # -- sums a Chebyshev series
    # -- at a point x in [A,B] where the coefficients
    # -- a(1), a(2), . . . ,a(N) are the Chebyshev coefficients of f(x).

    cdef unsigned long nvals = x.shape[0]
    cdef unsigned long ncoeffs = coeffs.shape[0]

    cdef np.ndarray[Real, ndim=1] out = np.empty((nvals,))
    cdef unsigned long k = 0
    cdef unsigned long j = 0
    cdef double ksi, b0, b1, b2, b3

    for k in range(nvals):

        ksi = (2 * x[k] - (a + b)) / (b - a)
        b0=0.0;
        b1=0.0;
        b2=0.0;
        b3=0.0;

        for j in range(ncoeffs):

            b0 = 2 * ksi * b1 - b2 + coeffs[ncoeffs - 1 - j];
            b3 = b2
            b2 = b1
            b1 = b0

        out[k] = 0.5 * (b0 - b3) + 0.5 * coeffs[0];

    return out


# ----------------------------------------------------------------------------------------------------------------------
# -- Evaluate a truncated Chebyshev series of an even function.
# ----------------------------------------------------------------------------------------------------------------------
# -- [Mason & Handscomb, p.49, Ch.2, Problem 7]
cpdef np.ndarray[Real, ndim=1] chebeval_even(double[:] coeffs, double a, double b, double[:] x):

    cdef unsigned long nvals = x.shape[0]
    cdef unsigned long ncoeffs = coeffs.shape[0]

    cdef np.ndarray[Real, ndim=1] out = np.empty((nvals,))
    cdef unsigned long k = 0
    cdef unsigned long j = 0
    cdef double ksi, b0, b1, b2, b3

    for k in range(nvals):

        ksi = (2 * x[k] - (a + b)) / (b - a)
        b0=0.0;
        b1=0.0;
        b2=0.0;
        b3=0.0;

        for j in range(ncoeffs):

            b0 = 2 * (2 * ksi * ksi - 1) * b1 - b2 + coeffs[ncoeffs - 1 - j];
            b3 = b2
            b2 = b1
            b1 = b0

        out[k] = 0.5 * (b0 - b3) + 0.5 * coeffs[0];

    return out

# ----------------------------------------------------------------------------------------------------------------------
# -- FUNCTION: Chebyshev companion matrix
# ----------------------------------------------------------------------------------------------------------------------
cpdef np.ndarray[Real, ndim=2] chebyshev_companion_matrix_cy(np.ndarray[Real, ndim=1] coeffs):

    cdef unsigned long Np1 = coeffs.shape[0]
    cdef unsigned long N = Np1 - 1

    #if N == 1:
    #    return np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)

    cdef np.ndarray[Real, ndim=2] CCM = np.zeros((N, N))

    #print(f'[cython][CCM] coeffs = {coeffs}')
    #print(f'[cython][CCM] CCM = {CCM}')

    CCM[0, 1] = 1.0

    cdef unsigned long j = 0

    for j in range(1, N-1):
        CCM[j, j-1] = 0.5
        CCM[j, j+1] = 0.5

    for j in range(0, N):
        CCM[N-1, j] = -coeffs[j] / (2 * coeffs[N])

    CCM[N-1, N-2] = CCM[N-1, N-2] + 0.5

    return CCM


# ======================================================================================================================
# -- Classes
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# -- Some functions or methods that would normally take a function as input are designed to accept instead an object
# -- of a class derived from the following `Function` class.
# ----------------------------------------------------------------------------------------------------------------------
# -- CLASS: FunctionWrapper
# ----------------------------------------------------------------------------------------------------------------------
cdef class FunctionWrapper:
    cpdef double eval(self, double x):
        raise NotImplementedError('Method `eval` of interface class `Function` was called')


# ----------------------------------------------------------------------------------------------------------------------
# -- CLASS: Vector wrapper(s)
# ----------------------------------------------------------------------------------------------------------------------
cdef class VectorWrapperDouble:

    cdef double* vec_data_ptr
    cdef unsigned long size

    # -- Members used by `__buffer__`
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    # -- Constructor
    def __cinit__(self, unsigned long vec_data_ptr, const unsigned long size):
        """
        vec_data_ptr: memory address passed as unsigned integer (not as a pointer to double) to avoid complications with
                      Cython ('Cannot convert double* to Python object' etc)
        """
        self.vec_data_ptr = <double *>vec_data_ptr
        self.size = size

    # -- Destructor
    # -- No destructor because vec_data_ptr points to shared data.

    # -- Buffer protocol
    def __getbuffer__(self, Py_buffer *buffer, int flags):

        cdef Py_ssize_t itemsize = sizeof(double)
        self.shape[0] = self.size
        self.strides[0] = sizeof(double)
        buffer.buf = <char *>(self.vec_data_ptr)
        buffer.format = 'd'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.size * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL


# ======================================================================================================================
# -- CLASS: Partition_cy
# ======================================================================================================================
# -- A class representing a Chebyshev partition of an interval.
# ======================================================================================================================
cdef class Partition_cy:

    # -- Partition_cy is a wrapper around the C++ class `Partition`.

    # -- `Instance_ptr`: a pointer to an instance of the "real" C++ class. The type `Partition *` is understood here
    #                    by cython because at the top of this file we did `from shamrock_cy cimport partition,
    #                    Partition`.
    # -- `array`: a reference to the to the underlying data of the std::vector that holds that data of the
    #                 `Partition` to which `Instance_ptr` points. This is for convenience.

    cdef Partition* Instance_ptr
    cdef VectorWrapperDouble array

    # -- Constructor
    def __cinit__(self, const double a = np.nan, const double b = np.nan, const unsigned long K = 0):

        # -- Combination of default values for a and b used to indicate that the default constructor will be called.
        if np.isnan(a) or np.isnan(b):
            self.default_construct()
        # -- Unbounded intervals are not supported.
        elif (a == -np.inf) or (b == np.inf):
            if a == b:
                self.default_construct()
            else:
                raise NotImplementedError('Unbounded intervals are not supported')
        else:
            self.Instance_ptr = new Partition(a, b, K)
            self.array = VectorWrapperDouble(<unsigned long>(self.Instance_ptr[0].partition.data()), self.Instance_ptr[0].partition.size())

    # -- Default constructor
    cpdef default_construct(self):
        self.Instance_ptr = new Partition()
        self.array = VectorWrapperDouble(<unsigned long>(self.Instance_ptr[0].partition.data()), self.Instance_ptr[0].partition.size())

    # -- Destructor
    def __dealloc__(self):
        if self.Instance_ptr is not NULL:
            del self.Instance_ptr

    # -- Printing
    def __repr__(self):
        repr = [('a', self.a),
                ('b', self.b),
                ('K', self.K),
                ('N', self.N),
                ('nPoints', self.x.size),
                ('x', self.x)]
        return pprint.pformat(repr)

    # -- Methods
    cpdef void refine(self):
        self.Instance_ptr[0].refine()
        self.array = VectorWrapperDouble(<unsigned long>(self.Instance_ptr[0].partition.data()), self.Instance_ptr[0].partition.size())

    cpdef void coarsen(self):
        self.Instance_ptr[0].coarsen()
        self.array = VectorWrapperDouble(<unsigned long>(self.Instance_ptr[0].partition.data()), self.Instance_ptr[0].partition.size())

    # -- Properties
    #@property
    #def partition(self):
    #    return self.Instance_ptr[0].partition

    @property
    def x(self):
        x = np.asarray(self.array)
        return x

    @property
    def K(self):
        return self.Instance_ptr[0].K

    @property
    def N(self):
        return self.Instance_ptr[0].N

    @property
    def a(self):
        return self.Instance_ptr[0].a

    @property
    def b(self):
        return self.Instance_ptr[0].b

    @property
    def K_history(self):
        return self.Instance_ptr[0].K_history

    @property
    def N_history(self):
        return self.Instance_ptr[0].N_history


# ======================================================================================================================
# -- CLASS: ChebPoly_cy
# ======================================================================================================================
# -- A class for generating Chebyshev polynomials.
# ======================================================================================================================
cdef class ChebPoly_cy:
    """
    `ChebPoly_cy` is a wrapper around the C++ class `ChebPoly`.

    `Instance_ptr`: a pointer to an instance of the "real" C++ class. The type `Partition *` is understood here
                    by cython because at the top of this file we did `from shamrock_cy cimport partition,
                    Partition`.
    """

    cdef ChebPoly* Instance_ptr

    # -- Constructor
    def __cinit__(self, const unsigned long kind = 1):
        self.Instance_ptr = new ChebPoly(kind)

    # -- Destructor
    def __dealloc__(self):
        if self.Instance_ptr is not NULL:
            del self.Instance_ptr

    # -- Method: Get the n-th Chebyshev polynomial. The kind of the polynomial (1st, 2nd, etc) is whatever is the kind
    # -- of polynomials for which the class was initialised.
    cpdef vector[long] T(self, n):
        return self.Instance_ptr[0].T(n)

    # -- Evalute Tn at x
    cpdef vector[Real] eval(self, n, x):
        return np.polyval(self.T(n), np.array(x, copy=False, ndmin=1, dtype=np.float_))


# ======================================================================================================================
# -- CLASS: Settings_cy
# ======================================================================================================================
# -- A class holding the settings for `ChebProxy`
# ======================================================================================================================
cdef class Settings_cy:

    cdef unsigned long Kmax
    cdef Real _intervalTolerance
    cdef Real _intervalToleranceMinVal
    cdef Real maxInterstitialError
    cdef bool issueWarnings
    cdef Real tau, sigma, rho, coeffTailCutOff
    cdef Real almostZero

    # ------------------------------------------------------------------------------------------------------------------
    # -- Cython-level initialiser
    # ------------------------------------------------------------------------------------------------------------------
    def __cinit__(self):
        """
        For some of the default values used below see [STE].
        """

        cdef Real eps_x_100 = 100 * np.finfo(float).eps
        #cdef Real eps_x_1000 = 1000 * np.finfo(float).eps

        self.Kmax = 11
        self.maxInterstitialError = eps_x_100
        self._intervalTolerance = eps_x_100
        self._intervalToleranceMinVal = eps_x_100
        self.issueWarnings = True

        # -- Threshold for discarding roots with imaginary part > tau
        self.tau = 1e-8

        # -- Threshold for discarding roots outside [-1, 1]
        self.sigma = 1e-6

        # -- Threshold for discarding small Chebyshev coefficients
        self.coeffTailCutOff = 1e-13

        # -- A point r on the x-axis is a root (candidate) if abs(f(r)) < rho
        self.rho = 1e-8

        # -- Almost zero
        self.almostZero = eps_x_100

    # ------------------------------------------------------------------------------------------------------------------
    # -- Properties
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def maxInterstitialError(self):
        return self.maxInterstitialError

    @maxInterstitialError.setter
    def maxInterstitialError(self, value):
        self.maxInterstitialError = value

    # ------------------------------------------------------------------------------------------------------------------

    @property
    def coeffTailCutOff(self):
        return self.coeffTailCutOff

    @coeffTailCutOff.setter
    def coeffTailCutOff(self, value):
        self.coeffTailCutOff = value

    # ------------------------------------------------------------------------------------------------------------------

    @property
    def rho(self):
        return self.rho

    @rho.setter
    def rho(self, value):
        self.rho = value

    # ------------------------------------------------------------------------------------------------------------------

    @property
    def Kmax(self):
        return self.Kmax

    @Kmax.setter
    def Kmax(self, value):
        self.Kmax = value

    # ------------------------------------------------------------------------------------------------------------------

    @property
    def intervalTolerance(self):
        return self._intervalTolerance

    @intervalTolerance.setter
    def intervalTolerance(self, value):

        if value >= self._intervalToleranceMinVal:
            self._intervalTolerance = value
        else:
            if self.issueWarnings:
                #print(errmsg(fileName='shamrock.py', className='Settings', methodName='coefficientThreshold.setter', message=f'Cannot set the coefficient threshold equal to {value}. The minimum allowable value is {self._intervalToleranceMinVal} and will be used instead.]'), file=sys.stderr, flush=True)
                sys.stderr.write(errmsg(fileName='shamrock.py', className='Settings', methodName='coefficientThreshold.setter', message=f'Cannot set the coefficient threshold equal to {value}. The minimum allowable value is {self._intervalToleranceMinVal} and will be used instead.]'))

            self._intervalTolerance = self._intervalToleranceMinVal

    # ------------------------------------------------------------------------------------------------------------------

    @property
    def issueWarnings(self):
        return self.issueWarnings

    @issueWarnings.setter
    def issueWarnings(self, value):
        self.issueWarnings = value

    # ------------------------------------------------------------------------------------------------------------------
    # -- dunder methods
    # ------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        return pprint.pformat({key: getattr(self, key) for key in dir(self) if not key.startswith('__') and not key.startswith('_')})


# ======================================================================================================================
# -- FUNCTION: truncate()
# ======================================================================================================================
# -- For further truncating the Chebyshev (finite) series; i.e., for cutting off a negligible part of the tail of the
# -- sequence of Chebyshev coefficients.
# ======================================================================================================================
cpdef np.ndarray[Real, ndim=1] truncate(np.ndarray[Real, ndim=1] coeffs, Real coeffTailCutOff):

        #print('[cython][truncate] Inside truncate')

        cdef unsigned long k = 0
        cdef unsigned long ncoeffs = <unsigned long>(coeffs.shape[0])

        # -- Initialise the tail sum to be equal to the last coefficient
        cdef Real tailSum = fabs(coeffs[ncoeffs - 1])

        # [PRINT]
        #print(f'k = {k}')
        #print(f'tailSum = {tailSum}')
        #print(f'ncoeffs = {ncoeffs}')
        #print(f'coeffs = {coeffs}')

        # -- Don't let the chopping of the tail lead to an empty vector of coefficients
        while (fabs(tailSum) < coeffTailCutOff) and k < ncoeffs:
            k += 1
            tailSum += fabs(coeffs[ncoeffs - k - 1])

        #print('[cython][truncate] Leaving truncate')

        return coeffs[:(ncoeffs - k)]


# ======================================================================================================================
# -- CLASS: ChebProxy_cy
# ======================================================================================================================
# -- The main class for representing the approximation of a function by a truncated Chebyshev series.
# ======================================================================================================================
cdef class ChebProxy_cy:
    """
    In `shamrock.py`, there is a python-level wrapper for this class.
    """

    # -- In Cython, class members are declared outside `__cinit__`.
    # -- Reference: `https://stackoverflow.com/questions/42632297/mixing-cdef-and-regular-python-attributes-in-cdef-class`
    # -- NOTE: memory views can only be local variables, hence the declaration of std::vectors here which may require
    #          castings.

    cdef FunctionWrapper F
    cdef Settings_cy settings
    cdef Partition_cy p, q

    cdef pair[Real, Real] I
    cdef Real a, b, error
    #cdef unsigned long K
    cdef vector[Real] fvals
    cdef map[unsigned long, vector[Real]] coeffs, roots

    # ------------------------------------------------------------------------------------------------------------------
    # -- METHOD: Cython-level initialiser
    # ------------------------------------------------------------------------------------------------------------------
    def __cinit__(self, FunctionWrapper F, pair[Real, Real] I, long K, settings):

        # [PRINT]
        #print('-- Inside ChebProxy_cy.__cinit__')

        if I.second- I.first < settings.intervalTolerance:
            raise ValueError(errmsg(fileName='shamrock_cy.pyx',
                                    className='ChebProxy_cy',
                                    methodName='__cinit__',
                                    message=f'The length of the interval {I} is less than the tolerance {settings.intervalTolerance}'))

        self.F = F
        self.I = I
        self.a = I.first
        self.b = I.second
        self.settings = settings

        #pprint.pprint(settings)

        self.approximate(K)

    # ------------------------------------------------------------------------------------------------------------------
    # -- METHOD: approximate()
    # ------------------------------------------------------------------------------------------------------------------
    cpdef approximate(self, long Kin):
        """
        This method is called inside `__cinit__`
        """

        # -- Successive partitions used to estimate the interstitial error (see [STE]).
        cdef Partition_cy p, q
        cdef unsigned long K, k, nx_interstitial
        cdef vector[Real] coeffs, fvals
        cdef np.ndarray[Real, ndim=1] x_interstitial, y_hat_interstitial, y_interstitial
        cdef Real error, tailSum

        # -- Determine the value of K automatically
        if Kin == -1:
            Kmin = 1
            Kmax = self.settings.Kmax + 1
        # -- User-supplied value for K
        else:
            Kmin = Kin
            Kmax = Kin + 1

        for K in range(Kmin, Kmax):

            p = Partition_cy(self.a, self.b, K)
            q = Partition_cy(self.a, self.b, K + 1)

            coeffs, fvals = chebcoefft(self.F, p)

            x_interstitial = q.x[1:2 ** (K + 1):2]
            nx_interstitial = x_interstitial.shape[0]
            y_interstitial = np.empty((nx_interstitial,))

            for k in range(nx_interstitial):
                y_interstitial[k] = self.F.eval(x_interstitial[k])

            y_hat_interstitial = chebeval(np.array(coeffs, copy=False), self.a, self.b, x_interstitial)

            error = np.max(np.abs(y_interstitial - y_hat_interstitial))

            if error < self.settings.maxInterstitialError:
                break

        # --------------------------------------------------------------------------------------------------------------
        # -- Chop the coefficient tail.
        # --------------------------------------------------------------------------------------------------------------
        #self.truncatedCoeffs = truncate(np.array(coeffs, copy=False), self.settings.coeffTailCutOff)

        # --------------------------------------------------------------------------------------------------------------
        # -- Save the results in class members.
        # --------------------------------------------------------------------------------------------------------------
        self.p = p
        self.coeffs[0] = coeffs
        self.fvals = fvals
        self.error = error

    # ------------------------------------------------------------------------------------------------------------------
    # -- METHOD: eval()
    # ------------------------------------------------------------------------------------------------------------------
    # -- Returns the values of the approximation (i.e., of the truncated Chebyshev series) at x.
    cpdef np.ndarray[Real, ndim=1] eval(self, np.ndarray[Real, ndim=1] x, int nterms=-1, string mode='approximate', unsigned long diff_order=0):

        # --------------------------------------------------------------------------------------------------------------
        # -- Declarations
        # --------------------------------------------------------------------------------------------------------------
        cdef int ncoeffs = self.coeffs[0].size()

        # [PRINT]
        #print(f'[cython][eval] mode = {mode}')
        #print(f'[cython][eval] diff_order = {diff_order}')

        # --------------------------------------------------------------------------------------------------------------
        # -- Argument checking #1
        # --------------------------------------------------------------------------------------------------------------
        # -- `mode` can take only the values `exact` or `approximate`.
        # --------------------------------------------------------------------------------------------------------------
        if mode not in ['exact', 'approximate']:
            raise ValueError(errmsg(fileName='shamrock_cy.pyx',
                                className='ChebProxy_cy',
                                methodName='eval',
                                message=f'The only legal values for the argument `mode` are `exact` and `approximate`.'))

        # --------------------------------------------------------------------------------------------------------------
        # -- Argument checking #2
        # --------------------------------------------------------------------------------------------------------------
        # -- If evaluating the finite Chebyshev series, there are constraints on the number of terms to be summed.
        # --------------------------------------------------------------------------------------------------------------
        if mode == 'approximate':

            if nterms > ncoeffs or nterms == 0:
                raise ValueError(errmsg(fileName='shamrock_cy.pyx',
                                        className='ChebProxy_cy',
                                        methodName='yhat',
                                        message=f'nterms = {nterms}, ncoeffs = {ncoeffs} -- The number of terms (`nterms`) in the truncated Chebyshev series must be between 1 and the number of coefficients (`ncoeffs`)'))

            if diff_order < 0:
                raise ValueError(errmsg(fileName='shamrock_cy.pyx',
                                                className='ChebProxy_cy',
                                                methodName='eval',
                                                message=f'The differentiation order (`diff_order`) cannot be negative'))

        # --------------------------------------------------------------------------------------------------------------
        # -- Exact evaluation (early return)
        # --------------------------------------------------------------------------------------------------------------
        if mode == 'exact':

            # -- Warning
            if diff_order != 0:
                print(errmsg(fileName='shamrock_cy.pyx',
                             className='ChebProxy_cy',
                             methodName='eval',
                             message=f'When the evaluation mode is set equal to `exact` the differentiation order (`diff_order`) is ignored'))

            # [PRINT]
            #print(f'[cython][eval] Calling `self.exact()`')

            return self.exact(x)

        # --------------------------------------------------------------------------------------------------------------
        # -- Approximate evaluation
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # -- If evaluating a derivative, make sure the coefficients are available.
        # -- NOTE: If diff_order == 0, then the coefficients are available because `__cinit__` calls self.approximate().
        # --------------------------------------------------------------------------------------------------------------
        if diff_order > 0:
            self.calcdcoeffs(diff_order)

        coeffs = np.array(self.coeffs[diff_order], copy=False)

        if nterms < 0:
            return chebeval(coeffs, self.a, self.b, x)
        else:
            return chebeval(coeffs[0:nterms], self.a, self.b, x)

    # ------------------------------------------------------------------------------------------------------------------
    # -- METHOD: exact()
    # ------------------------------------------------------------------------------------------------------------------
    # -- Returns the `exact` values of f (the function to be approximated) at x.
    cpdef np.ndarray[Real, ndim=1] exact(self, np.ndarray[Real, ndim=1] x):

        cdef unsigned long nx = x.shape[0]
        cdef unsigned long k

        cdef np.ndarray[Real, ndim=1] result = np.empty((nx,))

        for k in range(nx):
            result[k] = self.F.eval(x[k])

        return result

    # ------------------------------------------------------------------------------------------------------------------
    # -- METHOD: calcdcoeffs()
    # ------------------------------------------------------------------------------------------------------------------
    cpdef void calcdcoeffs(self, unsigned long n=1):
        """
        Calculated the Chebyshev coefficients of the derivative of order n of f.
        """

        # [PRINT]
        #print('[cython][diff] Inside `diff()`')

        # --------------------------------------------------------------------------------------------------------------
        # -- Argument checking
        # --------------------------------------------------------------------------------------------------------------
        if n < 1:
            raise ValueError(errmsg(fileName='shamrock_cy.pyx',
                                    className='ChebProxy_cy',
                                    methodName='calcdcoeffs',
                                    message=f'The order of differentiation `n` (`diff_order`) cannot be less than 1.'))

        # --------------------------------------------------------------------------------------------------------------
        # -- Declarations and definitions
        # --------------------------------------------------------------------------------------------------------------
        #      k: a counter
        #    uln: input argument `n` cast as `unsigned long`
        #   nmax: the maximum differentiation order for which coefficients are already available
        #  found: if the requested coefficients have already been cached in the map self.coeffs, then they are not
        #         calculated again. `found` is a boolean flag with self-explanatory name.
        # dcoeffs: auxiliary local variable
        # --------------------------------------------------------------------------------------------------------------

        cdef unsigned long k, nmax
        cdef bool found
        cdef np.ndarray[Real, ndim=1] dcoeffs

        # [PRINT]
        # print(f'-- nmax = {nmax}')

        # -- Casting
        #uln = <unsigned long>n

        # -- Highest derivative order available
        nmax = dereference(self.coeffs.rbegin()).first

        # -- Check if the requested coefficients have been cached after a previous calculation.
        #not_found = self.coeffs.find(n) == self.coeffs.end()

        # -- The requested coefficients have not been computed previously.
        if n > nmax:
            for k in range(nmax + 1, n + 1):
                self.coeffs[k] = chebdf(np.array(self.coeffs[k-1], copy=False), self.a, self.b)


    # ------------------------------------------------------------------------------------------------------------------
    # -- METHOD: solve()
    # ------------------------------------------------------------------------------------------------------------------
    cpdef np.ndarray[Real, ndim=1] solve(self, unsigned long diff_order=0):

        # --------------------------------------------------------------------------------------------------------------
        # -- Argument checking #1
        # --------------------------------------------------------------------------------------------------------------
        # -- `diff_order` cannot be negative.
        # --------------------------------------------------------------------------------------------------------------
        if diff_order < 0:
            raise ValueError(errmsg(fileName='shamrock_cy.pyx',
                                    className='ChebProxy_cy',
                                    methodName='solve',
                                    message=f'The differentiation order (`diff_order`) cannot be negative'))

        # --------------------------------------------------------------------------------------------------------------
        # -- Declarations
        # --------------------------------------------------------------------------------------------------------------
        # -- The return values of numpy.linalg.eig
        # -- If the eigenvalues / eigenvectors end up being real, the assignment to w and v will fail if they are typed
        # -- as in the next two lines.
        #cdef np.ndarray[complex, ndim=1] w
        #cdef np.ndarray[complex, ndim=2] v

        # -- For ease of reference in the code below
        cdef Real a = self.a
        cdef Real b = self.b

        # -- A counter
        cdef unsigned long k = 0

        # -- The Chebyshev companion matrix
        cdef np.ndarray[Real, ndim=2] CCM

        # -- A vector with candidate roots
        cdef np.ndarray[Real, ndim=1] r
        cdef np.ndarray[Real, ndim=1] truncatedCoeffs
        cdef long ntruncatedCoeffs = 0
        cdef Real rootOfLinearApproximation = 0.0

        # -- A temporary variable for storing the value of f or of a derivative at a candidate root
        cdef Real val

        # -- The maximum derivative order for which coefficients are available
        cdef unsigned long nmax = dereference(self.coeffs.rbegin()).first

        # -- Mode of evaluation: exact or approximate
        cdef string mode = 'approximate'

        # --------------------------------------------------------------------------------------------------------------
        # -- Construct the Chebyshev companion matrix (CCM).
        # --------------------------------------------------------------------------------------------------------------
        if diff_order > nmax:
            self.calcdcoeffs(diff_order)

        truncatedCoeffs = truncate(np.array(self.coeffs[diff_order], copy=False), self.settings.coeffTailCutOff)
        ntruncatedCoeffs = truncatedCoeffs.shape[0]

        # [PRINT]
        #print(f'[cython][solve] diff_order = {diff_order} -- nmax = {nmax} -- coeffs = {self.coeffs[diff_order]}')
        #print(f'[cython][solve] truncated coeffs = {truncatedCoeffs}')
        #print(f'[cython][solve] # truncated coeffs = {ntruncatedCoeffs}')

        # TODO: In `solve()` treat the special cases where the approximating polynomial is T0 or T1
        # TODO: NOTE: The coefficients, here, *do not* correspond to [-1,1] necessarily, but to general [a,b]
        # TODO: Return a success flag from solve
        # TODO: Check settings and issue warning, if allowed.

        # --------------------------------------------------------------------------------------------------------------
        # -- The CCM cannot be constructed with less than 3 coefficients.
        # --------------------------------------------------------------------------------------------------------------
        # -- T0 = 1: Therefore, if there is only one non-zero Chebyshev coefficient, the approximation is a constant
        # -- function that "intersects the x-axis at infinity".
        if ntruncatedCoeffs == 1:
            return np.array([np.inf], dtype=np.float64)

        if ntruncatedCoeffs == 2:
            rootOfLinearApproximation = -truncatedCoeffs[0] / truncatedCoeffs[1]
            rootOfLinearApproximation = rootOfLinearApproximation * 0.5 * (b -a) + 0.5 * (a + b)
            return np.array([rootOfLinearApproximation], dtype=np.float64)

        CCM = chebyshev_companion_matrix_cy(truncatedCoeffs)

        # --------------------------------------------------------------------------------------------------------------
        # -- If the CCM contains at least one NaN, return an empty vector / list.
        # --------------------------------------------------------------------------------------------------------------
        #if np.any(np.isnan(CCM)):
        #    r = np.array([], dtype=np.float64)
        #    return r

        # --------------------------------------------------------------------------------------------------------------
        # -- Find the roots
        # --------------------------------------------------------------------------------------------------------------
        w, v = np.linalg.eig(CCM)

        # -- Keep the roots with small imaginary part
        r = np.real(w[np.where(np.abs(np.imag(w)) < self.settings.tau)[0]])

        # -- Keep the roots close enough to the interval [-1, 1]
        r = r[np.where(np.abs(r) < 1 + self.settings.sigma)[0]]

        # -- Scale and shift the candidate roots back to [a, b]
        r = r * 0.5 * (b -a) + 0.5 * (a + b)

        if diff_order == 0:

            # -- Keep the roots where the value of f is close to zero
            for k in range(<unsigned long>(r.shape[0])):
                #val = self.F.eval(r[k])
                if np.abs(self.F.eval(r[k])) > self.settings.rho:
                    r[k] = np.nan

        # -- Because of argument checking when entering the function, `diff_order` cannot be negative.
        else:

            # -- Keep the roots where the value of the derivative of order `diff_order` is close to zero
            mode = 'approximate'
            for k in range(<unsigned long>(r.shape[0])):
                val = self.eval(np.array(r[k], ndmin=1), mode=mode, diff_order=<int>diff_order)[0]
                if np.abs(val) > self.settings.rho:
                    r[k] = np.nan

        r = r[np.where(~np.isnan(r))[0]]
        r = np.sort(r)

        # [PRINT]
        #print(f'[cython][solve] Returning')
        #print(f'[cython][solve] r = {r}')

        return r

    # ------------------------------------------------------------------------------------------------------------------
    # -- minimise()
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # -- minimize()
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # -- maximise()
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # -- maximize()
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # -- TODO: The implementation of this method is incomplete.
    # ------------------------------------------------------------------------------------------------------------------
    # -- METHOD: optimise()
    # ------------------------------------------------------------------------------------------------------------------
    cpdef optimise(self):

        # [PRINT]
        #print('[cython][optimise] Inside `optimise()`')

        # -- We will need the second derivative, anyway, to attempt to distinguish between minima and maxima (of
        # -- course, a second order test is not necessarilly conclusive).
        self.calcdcoeffs(n=2)

        cdef np.ndarray[Real, ndim=1] critical = self.solve(diff_order=1)
        cdef np.ndarray[uint8, ndim=1] minima = np.zeros((critical.shape[0],), dtype=np.uint8)
        cdef np.ndarray[uint8, ndim=1] maxima = np.zeros((critical.shape[0],), dtype=np.uint8)
        cdef np.ndarray[Real, ndim=1] ddy_at_root

        #minima[:] = np.nan
        #maxima[:] = np.nan

        cdef long k = 0

        for k in range(critical.shape[0]):

            if not np.isnan(critical[k]):

                ddy_at_root = self.eval(np.array([critical[k]]), diff_order=2)

                if ddy_at_root[0] > 0:
                    minima[k] = 1
                elif ddy_at_root[0] < 0:
                    maxima[k] = 1

        # [PRINT]
        #print('[cython] Leaving `optimise()`')

        return critical, minima, maxima

    # ------------------------------------------------------------------------------------------------------------------
    # -- optimize()
    # ------------------------------------------------------------------------------------------------------------------
    cpdef optimize(self):
        return self.optimise()

    # ------------------------------------------------------------------------------------------------------------------
    # -- dunder methods
    # ------------------------------------------------------------------------------------------------------------------
    # -- __call__
    # ------------------------------------------------------------------------------------------------------------------
    # -- Returns the values of the approximation (i.e., of the truncated Chebyshev series) at x.
    def __call__(self, np.ndarray[Real, ndim=1] x):
        return self.exact(x)

    # ------------------------------------------------------------------------------------------------------------------
    # -- Properties
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def p(self):
        return self.p

    @property
    def fvals(self):
        return self.fvals

    @property
    def coeffs(self):
        return self.coeffs

    @property
    def dcoeffs(self):
        return self.dcoeffs

    @property
    def truncatedCoeffs(self):
        return self.truncatedCoeffs

    #@property
    #def K(self):
    #    return self.K

    @property
    def error(self):
        return self.error

    @property
    def roots(self):
        return self.roots




# -- EOF