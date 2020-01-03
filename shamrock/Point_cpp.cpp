#include "Point_cpp.h"

// ---------------------------------------------------------------------------------------------------------------------
// -- A class representing Chebyshev interpolation points
// ---------------------------------------------------------------------------------------------------------------------
// -- Default constructor
chebyshev::Point::Point() : num{0}, den{0}, x{0.0} {}

// -- Constructor
chebyshev::Point::Point(const unsigned long n, const unsigned long d, double xx) {

    x = xx;

    if (n == d) {
        num = 1;
        den = 1;
    } else if (n == 0) {
        num = 0;
        den = 1;
    } else {
        // long gcd = std::gcd(n, d);
        num = n;
        den = d;
    }

}

bool chebyshev::Point::operator <(const Point& rhs) const {

    // -- The lcm of the denominators
    /*
    long lcm = std::lcm(den, rhs.den);
    return num * lcm / den < rhs.num * lcm / rhs.den;
    */
    return x < rhs.x;

}

// -- Setter
void chebyshev::Point::set(const unsigned long n, const unsigned long d, double xx) {

    x = xx;

    if (n == d) {
        num = 1;
        den = 1;
    } else if (n == 0) {
        num = 0;
        den = 1;
    } else {
        // long gcd = std::gcd(n, d);
        num = n;
        den = d;
    }

}