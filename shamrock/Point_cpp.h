#ifndef POINT_H
#define POINT_H

namespace chebyshev {

    // -----------------------------------------------------------------------------------------------------------------
    // -- A class representing a Chebyshev interpolation point
    // -----------------------------------------------------------------------------------------------------------------
    class Point {

        public:

            long num;
            long den;
            double x;

            // -- Default constructor
            Point();

            // -- Constructor
            Point(const unsigned long, const unsigned long, double);

            // -- Less than operator
            bool operator <(const Point&) const;

            // -- Setter
            void set(const unsigned long, const unsigned long, double);

            // -- Getter (redundant, but keep for now; direct access to members is provided via properties at Cython level)
            // std::pair<long, long> get() { return std::make_pair(num, den); }

    };

}

#endif