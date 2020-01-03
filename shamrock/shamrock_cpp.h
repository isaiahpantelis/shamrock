#ifndef SHAMROCK_H
#define SHAMROCK_H

#include <vector>
#include <map>

namespace chebyshev {

    // -----------------------------------------------------------------------------------------------------------------
    // -- Compile-time look-up table of cosine values for Chebyshev partitions.
    // -----------------------------------------------------------------------------------------------------------------
    /* NOT WORKING YET
    template<unsigned long N>
    struct CosineTable
    {

        //std::map<std::pair<unsigned long, unsigned long>, double> values;
        double values[N][2]
        constexpr unsigned long NN = N;

        constexpr CosineTable() : values()
        {
            for (unsigned long j = 0; j < N; ++j)
            {
                //values[std::make_pair(j, N)] = cos(j * M_PI / N);
                values[j][0] = j;
                values[j][1] = cos(j * M_PI / N);
            }
        }

    };
    */

    // -----------------------------------------------------------------------------------------------------------------
    // -- Functions computing Chebyshev interpolation points in [a,b] (Boyd, p. 50)
    // -----------------------------------------------------------------------------------------------------------------
    // -- OVERLOADING V1
    // -- Chebyshev interpolation points in [a,b]
    template<typename Real, typename Int>
    std::vector<Real> partition(const Real, const Real, const Int);

    // -- OVERLOADING V2
    // -- Chebyshev interpolation points in [a,b]
    template<typename Real, typename Int>
    void partition(std::vector<Real>& out, const Real, const Real, const Int);

    // -----------------------------------------------------------------------------------------------------------------
    // -- A class representing a Chebyshev partition of an interval.
    // -----------------------------------------------------------------------------------------------------------------
    class Partition {

        // constexpr static CosineTable CosTbl = CosineTable<1048577>();

        public:

            // -- Members
            unsigned long K, N;
            double a, b, bma2, bpa2;
            std::vector<unsigned long> K_history;
            std::vector<unsigned long> N_history;
            std::vector<double> partition;

            // -- Constructors
            Partition();
            Partition(const double, const double, const unsigned long);

            // -- Methods
            void refine();
            void coarsen();
            std::vector<double> getx(const unsigned long) const;

    };

    // -----------------------------------------------------------------------------------------------------------------
    // -- A class for generating Chebyshev polynomials of the first kind.
    // -----------------------------------------------------------------------------------------------------------------
    class ChebPoly {

        public:

            using LookUpTable = std::map<unsigned long, std::vector<long>>;

            // -- Members
            LookUpTable look_up_table;
            unsigned long kind;

            // -- Constructors
            // ChebPoly();
            ChebPoly(const unsigned long);

            // -- Methods
            std::vector<long> T(const unsigned long);

    };

}

#endif