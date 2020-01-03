#include "shamrock_cpp.h"
#include <math.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <map>
#include <utility>
#include <algorithm>

// ---------------------------------------------------------------------------------------------------------------------
// -- MACROS
// ---------------------------------------------------------------------------------------------------------------------
#define PRINTM(MSG) std::cout << #MSG << '\n'
#define PRINTE(EXPR) std::cout << #EXPR " = " << EXPR << '\n'
#define PRINTME(MSG, EXPR) std::cout << #MSG ": " << EXPR << '\n'
#define PRINTVV(VAR, VAL) std::cout << #VAR " = " << VAL << '\n'
#define PRINTV(V) std::cout << #V " = "; for(auto& element2print : V) std::cout << element2print << "    "; std::cout << '\n'

// ---------------------------------------------------------------------------------------------------------------------
// -- Functions computing Chebyshev interpolation points in [a,b] (Boyd, p. 50)
// ---------------------------------------------------------------------------------------------------------------------
// -- OVERLOADING V1
template<typename Real, typename Int>
std::vector<Real> chebyshev::partition(const Real a, const Real b, const Int K) {

    try {

        if (K < 0) throw std::invalid_argument("The exponent K used to calculate the number of Chebyshev interpolation points cannot be negative.");

        Int N = pow(2, K);
        Real bma2 = (b - a) / 2.0;
        Real bpa2 = (b + a) / 2.0;

        std::vector<Real> ipoints(N + 1);
        std::iota(ipoints.rbegin(), ipoints.rend(), 0.0);

        for (auto& k : ipoints) k = bma2 * cos(k * M_PI / N) + bpa2;

        return ipoints;

   } catch (const std::invalid_argument& e) {

        std::cerr << "Exception raised by execution path in\n\tFILE: boyd/shamrockc.cpp\n\tFUNCTION: chebyshev::partition\nwith\n\tMESSAGE: " << e.what() << std::endl;

        return std::vector<Real>(0);

   }

}

// -- OVERLOADING V2
// -- For efficiency, the result is stored in the vector `out` which is passed as an argument.
template<typename Real, typename Int>
void chebyshev::partition(std::vector<Real>& out, const Real a, const Real b, const Int K) {

    // PRINTM([C++] void partition called);

    try {

        if (K < 0) throw std::invalid_argument("The exponent K used to calculate the number of Chebyshev interpolation points cannot be negative.");

        Int N = pow(2, K);

        if (N + 1 > out.size()) throw std::invalid_argument("The size of the output vector is smaller than the number of Chebyshev interpolation points requested.");

        Real bma2 = (b - a) / 2.0;
        Real bpa2 = (b + a) / 2.0;

        // std::vector<Real> ipoints(N + 1);
        std::iota(out.rbegin(), out.rend(), 0.0);

        for (auto& k : out) k = bma2 * cos(k * M_PI / N) + bpa2;

   } catch (const std::invalid_argument& e) {

        std::cerr << "Exception raised by execution path in\n\tFILE: boyd/shamrockc.cpp\n\tFUNCTION: chebyshev::partition\nwith\n\tMESSAGE: " << e.what() << std::endl;

   }

}

// =====================================================================================================================
// -- A class representing a Chebyshev partition of an interval.
// =====================================================================================================================

// ---------------------------------------------------------------------------------------------------------------------
// -- Default constructor
// ---------------------------------------------------------------------------------------------------------------------
// -- [2019-11-21] The default constructor is not called by __cinit__ in the .pyx file. Just keeping things simple
// -- because constructors cannot be overloaded at the Cython level. It is, of course, possible to have cases in the
// -- __cinit__ method and call the appropriate C++ constructor accordingly.
chebyshev::Partition::Partition() : K {0}, N {0}, a {0.0}, b {0.0}, bma2 {0.0}, bpa2 {0.0}, K_history {std::vector<unsigned long>()}, N_history {std::vector<unsigned long>()}, partition {std::vector<double>()} {}

// ---------------------------------------------------------------------------------------------------------------------
// -- Constructor
// ---------------------------------------------------------------------------------------------------------------------
// namespace::class::constructor
chebyshev::Partition::Partition(const double aa, const double bb, const unsigned long KK) {

    //    PRINTM([chebyshev::Partition::Partition]);
    //    PRINTE(aa);
    //    PRINTE(bb);

    a = aa;
    b = bb;
    K = KK;
    N = pow(2, K);
    N_history.push_back(N);
    K_history.push_back(K);

    bma2 = (b - a) / 2.0;
    bpa2 = (b + a) / 2.0;

    if (aa > bb) {  // -- Left end point > right end point

        partition = std::vector<double>();

    } else if (aa == bb) {  // -- Left end point == right end point

        partition = std::vector<double>{aa};

    } else {  // -- Likely sensible values have been passed for the end points of the interval

        partition = std::vector<double>(N + 1);

        // -- Can I use the stand-alone function?
        // -- The first argument is used for output.
        chebyshev::partition(this->partition, aa, bb, KK);

    }

}

// ---------------------------------------------------------------------------------------------------------------------
// -- Methods
// ---------------------------------------------------------------------------------------------------------------------
inline void chebyshev::Partition::refine() {

    if (partition.size() > 1) {

        // -- Multiply N by 2
        unsigned long N_new = (N << 1);
        unsigned long K_new = K + 1;
        // -- Save the new value of N
        N_history.push_back(N_new);
        K_history.push_back(K_new);

        // -- Calculate and add to the underlying array the new Chebyshev partition points.
        for (long j = N_new - 1, decrement = 2; j > 0; j -= decrement) {
            partition.push_back(bma2 * cos(j * M_PI / N_new) + bpa2);
        }

        std::inplace_merge(&partition[0], &partition[N + 1], &partition[N_new + 1]);

        N = N_new;
        K = K_new;

    }

}

inline void chebyshev::Partition::coarsen() {

    if (partition.size() > 2) {

        // -- Multiply N by 2
        unsigned long N_new = (N >> 1);
        unsigned long K_new = K - 1;
        // -- Save the new value of N
        N_history.push_back(N_new);
        K_history.push_back(K_new);

        // -- Index into the vector `partition`, although we iterate over `partition` using iterator.
        unsigned long k = 0;

        for (auto it = partition.begin(); it != partition.end();) {
            if (k % 2 == 0) {
                ++it;
            } else {
                it = partition.erase(it);
            }
            k += 1;
        }

        N = N_new;
        K = K_new;

    }

}

// =====================================================================================================================
// -- A class used to generate Chebyshev polynomials of the first kind. It is a class and not a function because state
// -- has to be maintained for memoization.
// =====================================================================================================================
using LookUpTable = std::map<unsigned long, std::vector<long>>;
// ---------------------------------------------------------------------------------------------------------------------
// -- Constructors
// ---------------------------------------------------------------------------------------------------------------------
// -- Default constructor
// -- There is no meaningful default constructor
// chebyshev::ChebPoly::ChebPoly() : look_up_table {std::map<unsigned long, std::vector<long>>()}, kind {} {}

// -- Accepts the kind of the Chebyshev polynomials as argument.
chebyshev::ChebPoly::ChebPoly(const unsigned long kind_arg) {

    if (kind_arg != 1) std::cerr << "Warning issued by\n\tMETHOD: chebyshev::ChebPoly::ChebPoly(const unsigned long) in\n\tFILE: shamrock/shamrockc_cy.cpp with\n\tMESSAGE: " << "Only Chebyshev polynomials of the 1st kind supported for now. The generated polynomials will be of the 1st kind." << std::endl;

    look_up_table = LookUpTable();
    kind = kind_arg;

    look_up_table.emplace(0, std::vector<long>{1});
    look_up_table.emplace(1, std::vector<long>{1, 0});

}

// ---------------------------------------------------------------------------------------------------------------------
// -- Methods
// ---------------------------------------------------------------------------------------------------------------------
std::vector<long> make_Tn(const std::vector<long>& Tnm1, const std::vector<long>& Tnm2) {

    // -- The result is a vector of degree one higher than T_{n-1}.
    std::vector<long> Tn(Tnm1.size() + 1);

    // -- T_n = 2x * T_{n-1} - T_{n-2}
    // -- Multiplication by 2x
    std::transform(Tnm1.begin(), Tnm1.end(), Tn.begin(), [](long n){return n << 1;});

    // -- Subtraction of T_{n-2}
    unsigned long k = 2;
    for (const auto& x: Tnm2) {
        Tn[k] -= x;
        ++k;
    }

    return Tn;

}

std::vector<long> T_aux(LookUpTable& look_up_table, unsigned long n) {

    // PRINTM([C++] Inside T_aux);

    // -- Works correctly because of the initialisation in the constructor.
    if (n == 0) return look_up_table[0];

    // -- Works correctly because of the initialisation in the constructor.
    if (n == 1) return look_up_table[1];

    auto iter = look_up_table.find(n);

    if (iter != look_up_table.end()) return look_up_table[n];

    auto Tn = make_Tn(T_aux(look_up_table, n-1), T_aux(look_up_table, n-2));

    look_up_table.insert(std::make_pair(n, Tn));

    return Tn;

}

inline std::vector<long> chebyshev::ChebPoly::T(unsigned long n) { return T_aux(look_up_table, n); }

//