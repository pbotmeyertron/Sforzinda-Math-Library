#pragma once

#include <cstdint>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace sf {

/* Commonly-used macros and compiler attributes. */

#ifdef _WIN32
    #define sf_align(n) __declspec(align(n))
#else
    /* GCC and Clang */
    #define sf_align(n) __attribute__((aligned(n)))
#endif

#ifdef _WIN32
    #define sf_inline __forceinline
#else
    /* GCC and Clang */
    #define sf_inline __attribute__((always_inline))
#endif

//==============================================================================
// Mathematical Constants                                                  
//==============================================================================

/* Single-precision machine epsilon as specified in float.h */
#ifndef FLT_EPSILON
#define FLT_EPSILON     FLT_EPSILON  
#endif

/* Double-precision machine epsilon as specified in float.h */
#ifndef EPSILON
#define EPSILON         DBL_EPSILON  
#endif

/* sqrt(2) */
#ifndef SQRT_2
#define SQRT_2          1.414213562373095048801688724209698079L  
#endif


/* sqrt(3) */
#ifndef SQRT_3
#define SQRT_3          1.732050807568877293527446341505872366L  
#endif

/* sqrt(5) */
#ifndef SQRT_5
#define SQRT_5          2.236067977499789696409173668731276235L  
#endif

/* sqrt(1/2) */
#ifndef SQRT_1_DIV_2
#define SQRT_1_DIV_2    0.707106781186547524400844362104849039L  
#endif

/* sqrt(1/3) */
#ifndef SQRT_1_DIV_3
#define SQRT_1_DIV_3    0.577350269189625764509148780501957455L  
#endif

/* pi */
#ifndef PI
#define PI              3.141592653589793238462643383279502884L  
#endif

/* pi * 2 */
#ifndef TAU
#define TAU             6.283185307179586476925286766559005774L
#endif

/* pi/2 */
#ifndef PI_DIV_2
#define PI_DIV_2        1.570796326794896619231321691639751442L  
#endif

/* pi/4 */
#ifndef PI_DIV_4
#define PI_DIV_4        0.785398163397448309615660845819875721L  
#endif

/* sqrt(pi) */
#ifndef SQRT_PI
#define SQRT_PI         1.772453850905516027298167483341145183L  
#endif

/* e */
#ifndef E
#define E               2.718281828459045235360287471352662498L        
#endif

/* ln(2) */
#ifndef LN_2
#define LN_2            0.693147180559945309417232121458176568L  
#endif

/* ln(10) */
#ifndef LN_10
#define LN_10           2.302585092994045684017991454684364208L  
#endif

/* ln(pi) */
#ifndef LN_PI
#define LN_PI           1.144729885849400174143427351353058712L  
#endif

/* log_2(e) */
#ifndef LOG_BASE_2_E 
#define LOG_BASE_2_E    1.442695040888963407359924681001892137L     
#endif  
  
/* log_10(e) */
#ifndef LOG_BASE_10_E
#define LOG_BASE_10_E   0.434294481903251827651128918916605082L  
#endif

/* Euler-Mascheroni Constant */
#ifndef EULER
#define EULER           0.577215664901532860606512090082402431L  
#endif

/* Golden Ratio */
#ifndef PHI
#define PHI             1.618033988749894848204586834365638118L  
#endif

/* Apery's Constant */
#ifndef APERY
#define APERY           1.202056903159594285399738161511449991L  
#endif

//==============================================================================
// Typedefs for Primitive Types                                                  
//==============================================================================

/* Signed 8-Bit Integer */
typedef int8_t      i8;  
/* Signed 16-Bit Integer */
typedef int16_t     i16; 
/* Signed 32-Bit Integer */
typedef int32_t     i32; 
/* Signed 64-Bit Integer */
typedef int64_t     i64; 

/* Unsigned 8-Bit Integer */
typedef uint8_t     u8;  
/* Unsigned 16-Bit Integer */
typedef uint16_t    u16; 
/* Unsigned 32-Bit Integer */
typedef uint32_t    u32; 
/* Unsigned 64-Bit Integer */
typedef uint64_t    u64;

/* 32-Bit Floating-Point Number */
typedef float       f32; 
/* 64-Bit Floating-Point Number */
typedef double      f64; 
/* 128-Bit Floating-Point Number */
typedef long double f128;

//==============================================================================
// Mathematical Utilities                                                 
//==============================================================================

/* Performs equality check using machine-epsilon. */
#define equals(a,b)     (std::abs(a - b)  < EPSILON)

/* Performs non-equality check using machine-epsilon. */
#define not_equals(a,b) (std::abs(a - b) >= EPSILON)

//==============================================================================
// Mathematical Types                                                  
//==============================================================================

/*
 * The following types are defined as template classes with overloads and
 * common functions used in graphics routines:
 *
 * vec2 - 2D Vector
 * vec3 - 3D Vector
 * vec4 - 4D Vector
 * mat2 - 2x2 Matrix
 * mat3 - 3x3 Matrix
 * mat4 - 4x4 Matrix
 * quat - Quaternion
 */

//==============================================================================
// vec2                                                 
//==============================================================================

/* 2D Vector */

template<class T>
class vec2 {
public:
    union {
        struct {
            /* Coordinate notation. */
            T x, y; 
        };
        struct {
            /* Array notation. */
            T v[2]; 
        };
    };

    vec2<T>() { 
        x = 0;
        y = 0; 
    } 

    vec2<T>(T cx, T cy) { 
        x = cx; 
        y = cy; 
    }

    vec2<T>(T cx) { 
        x = cx;
        y = cx; 
    }

    vec2<T>(const vec2<T>& v) { 
        x = v.x; 
        y = v.y; 
    }

    vec2<T>(T v[2]) { 
        x = v[0]; 
        y = v[1]; 
    }

    /* Returns the {x, y} coordinates of a vec2. */
    sf_inline vec2<T> xy() const {
        return(vec2<T>(x,y));
    }

    /* Returns the length (magnitude) of a 2D vector. */
    sf_inline T length() const {
        vec2<T> a = std::sqrt(a.x * a.x + a.y * a.y);
        return a;
    }

    /* Normalizes a 2D vector. */
    sf_inline vec2<T> const normalize() const {
        T mag = std::sqrt(x * x + y * y);
        if (mag != 0.0f) {
            vec2<T> r;
            r.x = x / mag; 
            r.y = y / mag;
            return(r);
        }
        return(vec2<T>(0.0f, 0.0f));
    }

    /* Index or subscript operand. */
    sf_inline T 
    operator [] 
    (u32 i) const {
        return(v[i]);
    }

    /* Index or subscript operand. */
    sf_inline T& 
    operator[] 
    (u32 i) {
        return(v[i]);
    }
}; // vec2

/* 2D Vector Functions (non-member) */

/* Returns the length (magnitude) of a 2D vector. */
template<typename T>
sf_inline T length(const vec2<T>& a) {
    return std::sqrt(a.x * a.x + a.y * a.y);
}

/* Normalizes a 2D vector. */
template<typename T>
sf_inline vec2<T> normalize(const vec2<T>& a) {
    T mag = length(a);
    if (mag != 0.0f) {
        a.x /= mag;
        a.y /= mag;
        return(a);
    }
    return(vec2<T>(0.0f, 0.0f));
}

/* Returns the dot product of a 2D vector. */
template<typename T>
sf_inline T dot_product(const vec2<T>& a, const vec2<T>& b) {
    return(a.x * b.x + a.y * b.y);
}

/* Returns the cross product of a 2D vector. */
template<typename T>
sf_inline vec2<T> cross_product(const vec2<T>& a, const vec2<T> b) {
    vec2<T> c;
    c.x = (a.x * b.y) - (a.y * b.x);
    c.y = (a.y * b.x) - (a.x * b.y);
    return c;
}

/* Returns the angle between two 2D vectors. */
template<typename T>
sf_inline T angle_between(const vec2<T>& a, const vec2<T>& b) {
    return dot_product(a, b) / (length(a) * length(b));
}

/* Returns the normal axis between two 2D vectors. */
template<typename T>
sf_inline vec2<T> normal_axis_between(const vec2<T>& a, const vec2<T>& b) {
    return normalize(cross_product(a, b));
}

/* Returns the distance between two 2D vectors. */
template<typename T>
sf_inline T distance(const vec2<T>& a, const vec2<T>& b) {
    return std::sqrt((b.x - a.x) * (b.x - a.x) + 
                     (b.y - a.y) * (b.y - a.y));
}

/* Prints out the coordinates of a vec2. */
template<typename T>
sf_inline void print_vec(const vec2<T>& a) {
    printf("x:12f, y:12f\r\n", a.x, a.y);
}

/* 2D Vector Overloads */

/* Add two vec2s. */
template<typename T>
sf_inline vec2<T> 
operator + 
(const vec2<T> &lhs, const vec2<T> &rhs) {
    vec2<T> c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    return(c);
}

/* Add vec2 and scalar. */
template<typename T, typename U>
sf_inline vec2<T> 
operator + 
(const vec2<T> &lhs, const U &rhs) {
    vec2<T> c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    return(c);
}

/* Add scalar and vec2. */
template<typename T, typename U>
sf_inline vec2<T> 
operator + 
(const U &lhs, const vec2<T> &rhs) {
    vec2<T> c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    return(c);
}

/* Plus-equals operand with two vec2s. */
template<typename T>
sf_inline vec2<T>& 
operator += 
(vec2<T> &lhs, const vec2<T> &rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y;
    return(lhs);
}

/* Plus-equals operand with a vec2 and scalar. */
template <typename T, typename U>
sf_inline vec2<T>& 
operator += 
(vec2<T> &lhs, const U &rhs) {
    lhs.x += rhs; 
    lhs.y += rhs;
    return(lhs);
}

/* Unary minus operand. Makes vec2 negative. */
template<typename T>
sf_inline vec2<T> 
operator - 
(const vec2<T> &rhs) {
    vec2<T> c; 
    c.x =- rhs.x; 
    c.y =- rhs.y; 
    return(c);
}

/* Subtracts a vec2 from a vec2. */
template <typename T>
sf_inline vec2<T> 
operator - 
(const vec2<T> &lhs, const vec2<T> &rhs) {
    vec2<T> c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    return(c);
}

/* Subtracts a scalar from a vec2. */
template<typename T, typename U>
sf_inline vec2<T> 
operator - 
(const vec2<T> &lhs, const U &rhs) {
    vec2<T> c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    return(c);
}

/* Subtracts a vec2 from a scalar. */
template<typename T, typename U>
sf_inline vec2<T> 
operator - 
(const U &lhs, const vec2<T> &rhs) {
    vec2<T> c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    return(c);
}

/* Minus-equals operand for two vec2s. */
template<typename T>
sf_inline vec2<T>& 
operator -= 
(vec2<T> &lhs, const vec2<T> &rhs) {
    lhs.x-=rhs.x; lhs.y-=rhs.y;
    return(lhs);
}

/* Minus-equals operand for vec2 and scalar. */
template <typename T, typename U>
sf_inline vec2<T>& 
operator -= 
(vec2<T> &lhs, const U &rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs;
    return(lhs);
}

/* Multiplies two vec2s. */
template<typename T>
sf_inline vec2<T> 
operator * 
(const vec2<T> &lhs, const vec2<T> &rhs) {
    vec2<T> c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y;
    return(c);
}

/* Multiplies a vec2 and scalar. */
template<typename T, typename U>
sf_inline vec2<T> 
operator * 
(const U &lhs, const vec2<T> &rhs) {
    vec2<T> c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs;
    return(c);
}

/* Multiplies a scalar and vec2. */
template <typename T, typename U>
sf_inline vec2<T> 
operator * 
(const vec2<T> &lhs, const U &rhs) {
    vec2<T> c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    return(c);
}

/* Multiply-equals operand for vec2. */
template <typename T>
sf_inline vec2<T>& 
operator *= 
(vec2<T> &lhs, const vec2<T> &rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y;
    return(lhs);
}

/* Multiply-equals operand for vec2 and scalar. */
template<typename T, typename U>
sf_inline vec2<T>& 
operator *= 
(vec2<T> &lhs, const U &rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs;
    return(lhs);
}

/* Divides two vec2. */
template<typename T>
sf_inline vec2<T> 
operator / 
(const vec2<T> &lhs, const vec2<T> &rhs) {
    vec2<T> c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y;
    return(c);
}

/* Divides a vec2 by a scalar. */
template <typename T, typename U>
sf_inline vec2<T> 
operator / 
(const vec2<T> &lhs, const U &rhs) {
    vec2<T> c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs;
    return(c);
}

/* Divide-equals operand for two vec2s. */
template <typename T>
sf_inline vec2<T>& 
operator /= 
(vec2<T> &lhs, const vec2<T> &rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y;
    return(lhs);
}

/* Divide-equals operand for vec2 and scalar. */
template<typename T, typename U>
sf_inline vec2<T>& 
operator /= 
(vec2<T> &lhs, const U &rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs;
    return(lhs);
}

/* Add one to each element in vec2. */
template<typename T>
sf_inline vec2<T>& 
operator ++ 
(vec2<T> &rhs) {
    ++rhs.x; 
    ++rhs.y;
    return(rhs);
}

/* Add one to each element in vec2. */
template<typename T>
sf_inline vec2<T> 
operator ++ 
(vec2<T> &lhs, i32) {
    vec2<T> c = lhs;
    lhs.x++; 
    lhs.y++;
    return(c);
}

/* Subtract one from each element in vec2. */
template<typename T>
sf_inline vec2<T>& 
operator -- 
(vec2<T> &rhs) {
    --rhs.x; 
    --rhs.y;
    return(rhs);
}

/* Subtract one from each element in vec2. */
template<typename T>
sf_inline vec2<T> 
operator -- 
(vec2<T> &lhs, int) {
    vec2<T> c = lhs;
    lhs.x--; 
    lhs.y--;
    return(c);
}

/* Tests two vec2s for equality. */
template<typename T>
sf_inline bool 
operator == 
(const vec2<T> &lhs, const vec2<T> &rhs) {
    return((lhs.x == rhs.x) && (lhs.y == rhs.y));
}

/* Tests two vec2s for non-equality. */
template<typename T>
sf_inline bool 
operator != 
(const vec2<T> &lhs, const vec2<T> &rhs) {
    return((lhs.x != rhs.x) || (lhs.y != rhs.y));
}

/* Allows for printing elements of vec2 to stdout. Thanks to Ray Tracing in One
 * Weekend for this. :) */
template<typename T>
sf_inline std::ostream& 
operator << 
(std::ostream &os, const vec2<T> &rhs) {
    os << "(" << rhs.x << "," << rhs.y << ")";
    return(os);
}

//==============================================================================
// vec3                                                 
//==============================================================================

/* 3D Vector */

template<class T>
class vec3 {
public:
    union {
        struct { 
            /* Coordinate notation. */
            T x, y, z; 
        };
        struct { 
            /* Array notation. */
            T v[3]; 
        };
    };

    vec3<T>() { 
        x = 0;
        y = 0;
        z = 0; 
    }

    vec3<T>(T cx, T cy, T cz) { 
        x = cx; 
        y = cy; 
        z = cz; 
    }

    vec3<T>(T cx) { 
        x = cx;
        y = cx;
        z = cx; 
    }

    /* Initialize a vec3 with a vec2 and a scalar. */
    vec3<T>(vec2<T> v, T cz) { 
        x = v.x; 
        y = v.y; 
        z = cz; 
    }

    vec3<T>(const vec3<T> &v) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
    }

    vec3<T>(T v[3]) { 
        x = v[0]; 
        y = v[1]; 
        z = v[2]; 
    }

    /* Returns the {x, y, z} coordinates of a vec3. */
    sf_inline vec3<T> xyz() const {
        return(vec3<T>(x, y, z));
    }

    sf_inline T length() const {
        T r = std::sqrt(x * x + y * y + z * z);
        return(r);
    }

    sf_inline vec3<T> const normalize() const {
        T mag = std::sqrt(x * x + y * y + z * z);
        if (mag != 0.0f) {
            vec3<T> r;
            r.x = x / mag; 
            r.y = y / mag; 
            r.z = z / mag;
            return r;
        }
        return(vec3<T>(0.0f, 0.0f, 0.0f));
    }

    /* Index or subscript operand. */
    sf_inline T 
    operator [] 
    (u32 i) const {
        return(v[i]);
    }

    /* Index or subscript operand. */
    sf_inline T& 
    operator [] 
    (u32 i) {
        return(v[i]);
    }
}; // vec3

/* 3D Vector Functions (non-member) */

/* Returns the length (magnitude) of a 3D vector. */
template<typename T>
sf_inline T length(const vec3<T>& a) {
    return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

/* Normalizes a 3D vector. */
template<typename T>
sf_inline vec3<T> normalize(const vec3<T>& a) {
    T mag = length(a);
    if (mag != 0.0f) {
        a.x /= mag;
        a.y /= mag;
        a.z /= mag;
        return a;
    }
    return(vec3<T>(0.0f, 0.0f, 0.0f));
}

/* Returns the dot product of a 3D vector. */
template<typename T>
sf_inline T dot_product(const vec3<T>& a, const vec3<T>& b) {
    return(a.x * b.x + a.y * b.y + a.z * b.z);
}

/* Returns the cross product of a 3D vector. */
template<typename T>
sf_inline vec3<T> cross_product(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> c;
    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);
    return(c);
}

/* Returns the angle between two 3D vectors. */
template<typename T>
sf_inline T angle_between(const vec3<T>& a, const vec3<T>& b) {
    T c;
    c = dot_product(a, b) / (length(a) * length(b));
    return 2.0f * std::acos(c);
}

/* Returns the normal axis between two 3D vectors. */
template<typename T>
sf_inline vec3<T> normal_axis_between(const vec3<T>& a, const vec3<T>& b) {
    return normalize(cross_product(a, b));
}

/* Returns the distance between two 3D vectors. */
template<typename T>
sf_inline T distance(const vec3<T>& a, const vec3<T>& b) {
    return std::sqrt((b.x - a.x) * (b.x - a.x) + 
                     (b.y - a.y) * (b.y - a.y) +
                     (b.z - a.z) * (b.z - a.z));
}

/* Prints out the coordinates of a vec3. */
template<typename T>
sf_inline void print_vec(const vec3<T>& a) {
    printf("x:12f, y:12f, z:12f\r\n", a.x, a.y, a.z);
}

/* 3D Vector Overloads */

/* Add two vec3s. */
template<typename T>
sf_inline vec3<T> 
operator + 
(const vec3<T> &lhs, const vec3<T> &rhs) {
    vec3<T> c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    c.z = lhs.z + rhs.z; 
    return(c);
}

/* Add vec3 and scalar. */
template<typename T, typename U>
sf_inline vec3<T> 
operator + 
(const vec3<T> &lhs, const U &rhs) {
    vec3<T> c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    c.z = lhs.z + rhs; 
    return(c);
}

/* Add scalar and vec3. */
template<typename T, typename U>
sf_inline vec3<T> 
operator + 
(const U &lhs, const vec3<T> &rhs) {
    vec3<T> c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    c.z = lhs + rhs.z; 
    return(c);
}

/* Plus-equals operand with two vec3s. */
template<typename T>
sf_inline vec3<T>& 
operator += 
(vec3<T> &lhs, const vec3<T> &rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y; 
    lhs.z += rhs.z;
    return(lhs);
}

/* Plus-equals operand with a vec3 and scalar. */
template<typename T, typename U>
sf_inline vec3<T>& 
operator += 
(vec3<T> &lhs, const U &rhs) {
    lhs.x += rhs; 
    lhs.y += rhs; 
    lhs.z += rhs;
    return(lhs);
}

/* Unary minus operand. Makes vec3 negative. */
template<typename T>
sf_inline vec3<T> 
operator - 
(const vec3<T> &rhs) {
    vec3<T> c; 
    c.x =- rhs.x; 
    c.y =- rhs.y; 
    c.z =- rhs.z; 
    return(c);
}

/* Subtracts a vec3 from a vec3. */
template<typename T>
sf_inline vec3<T> 
operator - 
(const vec3<T> &lhs, const vec3<T> &rhs) {
    vec3<T> c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    c.z = lhs.z - rhs.z; 
    return(c);
}

/* Subtracts a scalar from a vec3. */
template<typename T, typename U>
sf_inline vec3<T> 
operator - 
(const vec3<T> &lhs, const U &rhs) {
    vec3<T> c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    c.z = lhs.z - rhs; 
    return(c);
}

/* Subtracts a vec3 from a scalar. */
template<typename T, typename U>
sf_inline vec3<T> 
operator - 
(const U &lhs, const vec3<T> &rhs) {
    vec3<T> c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    c.z = lhs - rhs.z; 
    return(c);
}

/* Minus-equals operand for two vec3s. */
template<typename T>
sf_inline vec3<T>& 
operator -= 
(vec3<T> &lhs, const vec3<T> &rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y; 
    lhs.z -= rhs.z;
    return(lhs);
}

/* Minus-equals operand for vec3 and scalar. */
template<typename T, typename U>
sf_inline vec3<T>& 
operator -= 
(vec3<T> &lhs, const U &rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs; 
    lhs.z -= rhs;
    return(lhs);
}

/* Multiplies two vec3s. */
template<typename T>
sf_inline vec3<T> 
operator * 
(const vec3<T> &lhs, const vec3<T> &rhs) {
    vec3<T> c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y; 
    c.z = rhs.z * lhs.z;
    return(c);
}

/* Multiplies a vec3 and scalar. */
template<typename T, typename U>
sf_inline vec3<T> 
operator * 
(const U &lhs, const vec3<T> &rhs) {
    vec3<T> c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs; 
    c.z = rhs.z * lhs;
    return(c);
}

/* Multiplies a scalar and vec3. */
template<typename T, typename U>
sf_inline vec3<T> 
operator * 
(const vec3<T> &lhs, const U &rhs) {
    vec3<T> c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    c.z = rhs * lhs.z;
    return(c);
}

/* Multiply-equals operand for vec3. */
template<typename T>
sf_inline vec3<T>& 
operator *= 
(vec3<T> &lhs, const vec3<T> &rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y; 
    lhs.z *= rhs.z;
    return(lhs);
}

/* Multiply-equals operand for vec3 and scalar. */
template<typename T, typename U>
sf_inline vec3<T>& 
operator *= 
(vec3<T> &lhs, const U &rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs; 
    lhs.z *= rhs;
    return(lhs);
}

/* Divides two vec3s. */
template<typename T>
sf_inline vec3<T> 
operator / 
(const vec3<T> &lhs, const vec3<T> &rhs) {
    vec3<T> c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y; 
    c.z = lhs.z / rhs.z;
    return(c);
}

/* Divides a vec3 by a scalar. */
template<typename T, typename U>
sf_inline vec3<T> 
operator / 
(const vec3<T> &lhs, const U &rhs) {
    vec3<T> c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs; 
    c.z = lhs.z / rhs;
    return(c);
}

/* Divide-equals operand for two vec3s. */
template<typename T>
sf_inline vec3<T>& 
operator /= 
(vec3<T> &lhs, const vec3<T> &rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y; 
    lhs.z /= rhs.z;
    return(lhs);
}

/* Divide-equals operand for vec3 and scalar. */
template<typename T, typename U>
sf_inline vec3<T>& 
operator /= 
(vec3<T> &lhs, const U &rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs; 
    lhs.z /= rhs;
    return(lhs);
}

/* Add one to each element in vec3. */
template<typename T>
sf_inline vec3<T>& 
operator ++ 
(vec3<T> &rhs) {
    ++rhs.x; 
    ++rhs.y; 
    ++rhs.z;
    return(rhs);
}

/* Add one to each element in vec3. */
template<typename T>
sf_inline vec3<T> 
operator ++ 
(vec3<T> &lhs, i32) {
    vec3<T> c = lhs;
    lhs.x++; 
    lhs.y++; 
    lhs.z++;
    return(c);
}

/* Subtract one from each element in vec3. */
template<typename T>
sf_inline vec3<T>& 
operator -- 
(vec3<T> &rhs) {
    --rhs.x; 
    --rhs.y; 
    --rhs.z;
    return(rhs);
}

/* Subtract one from each element in vec3. */
template<typename T>
sf_inline vec3<T> 
operator -- 
(vec3<T> &lhs, i32) {
    vec3<T> c=lhs;
    lhs.x--; 
    lhs.y--; 
    lhs.z--;
    return(c);
}

/* Tests two vec3s for equality. */
template<typename T>
sf_inline bool 
operator == 
(const vec3<T> &lhs, const vec3<T> &rhs) {
    return((lhs.x == rhs.x) && 
           (lhs.y == rhs.y) && 
           (lhs.z == rhs.z));
}

/* Tests two vec3s for non-equality. */
template<typename T>
sf_inline bool 
operator != 
(const vec3<T> &lhs, const vec3<T> &rhs) {
    return((lhs.x != rhs.x) || 
           (lhs.y != rhs.y) || 
           (lhs.z != rhs.z));
}

/* Allows for printing elements of vec3 to stdout. Thanks to Ray Tracing in One
 * Weekend for this. :) */
template<typename T>
std::ostream& 
operator << 
(std::ostream &os, const vec3<T> &rhs) {
    os << "(" << rhs.x << "," << rhs.y << "," << rhs.z << ")";
    return(os);
}

//==============================================================================
// vec4                                                 
//==============================================================================

/* 4D Vector */

template<class T>
class vec4 {
public:
    union {
        struct { 
            /* Coordinate notation. */
            T x, y, z, w; 
        };
        struct { 
            /* 32-bit color values. */
            T r, g, b, a; 
        };
        struct { 
            /* Array notation. */
            T v[4]; 
        };
    };

    vec4<T>() { 
        x = 0;
        y = 0;
        z = 0;
        w = 0; 
    }

    vec4<T>(T cx, T cy, T cz, T cw) { 
        x = cx; 
        y = cy; 
        z = cz; 
        w = cw; 
    }

    vec4<T>(T cx) { 
        x = cx;
        y = cx;
        z = cx;
        w = cx; 
    }

    /* Initialize a vec4 with a vec3 and a scalar. */
    vec4<T>(vec3<T> v, T cw) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
        w = cw; 
    }

    /* Initialize a vec4 with two vec2s. */
    vec4<T>(vec2<T> v, vec2<T> u) { 
        x = v.x; 
        y = v.y; 
        z = u.x; 
        w = u.y; 
    }   

    vec4<T>(const vec4<T> &v) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
        w = v.w; 
    }

    vec4<T>(T v[4]) { 
        x = v[0]; 
        y = v[1]; 
        z = v[2]; 
        w = v[3]; 
    }

    /* Returns the {x, y, z, w} coordinates of a vec4. */
    sf_inline vec4<T> xyzw() const {
        return(vec4<T>(x, y, z, w));
    }

    sf_inline T length() const {
        T r = std::sqrt(x * x + y * y + z * z + w * w);
        return(r);
    }

    sf_inline vec4<T> const normalize() const {
        T mag = std::sqrt(x * x + y * y + z * z + w * w);
        if (mag != 0.0f) {
            vec4<T> r;
            r.x = x / mag; 
            r.y = y / mag; 
            r.z = z / mag; 
            r.w = w / mag;
            return(r);
        }
        return(vec4<T>(0.0f, 0.0f, 0.0f, 0.0f));
    }

    /* Index or subscript operand. */
    sf_inline T 
    operator [] 
    (u32 i) const {
        return(v[i]);
    }

    /* Index or subscript operand. */
    sf_inline T& 
    operator [] 
    (u32 i) {
        return(v[i]);
    }
};

/* 4D Vector Functions (non-member) */

/* Returns the length (magnitude) of a 4D vector. */
template<typename T>
sf_inline T length(const vec4<T>& a) {
    return std::sqrt(a.x * a.x + 
                     a.y * a.y + 
                     a.z * a.z +
                     a.w * a.w);
}

/* Normalizes a 4D vector. */
template<typename T>
sf_inline vec4<T> normalize(const vec4<T>& a) {
    T mag = length(a);
    if (mag != 0.0f) {
        a.x /= mag;
        a.y /= mag;
        a.z /= mag;
        a.w /= mag;
        return(a);
    }
    return(vec4<T>(0.0f, 0.0f, 0.0f, 0.0f));
}

/* Returns the dot product of a 4D vector. */
template<typename T>
sf_inline T dot_product(const vec4<T>& a, const vec4<T>& b) {
    return(a.x * b.x + 
           a.y * b.y + 
           a.z * b.z +
           a.w * b.w);
}

/* Returns the cross product of a 4D vector. */
template<typename T>
sf_inline vec4<T> cross_product(const vec4<T>& a, const vec4<T>& b) {
    vec4<T> c;
    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);
    c.w = (a.w * b.w) - (a.w * b.w);
    return c;
}

/* Returns the distance between two 4D vectors. */
template<typename T>
sf_inline T distance(const vec4<T>& a, const vec4<T>& b) {
    return std::sqrt((b.x - a.x) * (b.x - a.x) + 
                     (b.y - a.y) * (b.y - a.y) + 
                     (b.z - a.z) * (b.z - a.z) + 
                     (b.w - a.w) * (b.w - a.w));
}

/* Prints out the coordinates of a vec4. */
template<typename T>
sf_inline void print_vec(const vec4<T>& a) {
    printf("x:12f, y:12f, z:12f, w:12f\r\n", a.x, a.y, a.z, a.w);
}

/* 4D Vector Overloads */

/* Add two vec4s. */
template<typename T>
sf_inline vec4<T> 
operator+ 
(const vec4<T> &lhs, const vec4<T> &rhs) {
    vec4<T> c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    c.z = lhs.z + rhs.z; 
    c.w = lhs.w + rhs.w; 
    return(c);
}

/* Add vec4 and scalar. */
template<typename T, typename U>
sf_inline vec4<T> 
operator + 
(const vec4<T> &lhs, const U &rhs) {
    vec4<T> c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    c.z = lhs.z + rhs; 
    c.w = lhs.w + rhs; 
    return(c);
}

/* Add scalar and vec4. */
template<typename T, typename U>
sf_inline vec4<T> 
operator + 
(const U &lhs, const vec4<T> &rhs) {
    vec4<T> c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    c.z = lhs + rhs.z; 
    c.w = lhs + rhs.w; 
    return(c);
}

/* Plus-equals operand with two vec4s. */
template<typename T>
sf_inline vec4<T>& 
operator += 
(vec4<T> &lhs, const vec4<T> &rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y; 
    lhs.z += rhs.z; 
    lhs.w += rhs.w;
    return(lhs);
}

/* Plus-equals operand with a vec4 and scalar. */
template<typename T, typename U>
sf_inline vec4<T>& 
operator += 
(vec4<T> &lhs, const U &rhs) {
    lhs.x += rhs; 
    lhs.y += rhs; 
    lhs.z += rhs; 
    lhs.w += rhs;
    return(lhs);
}

/* Unary minus operand. Makes vec4 negative. */
template<typename T>
sf_inline vec4<T> 
operator - 
(const vec4<T> &rhs) {
    vec4<T> c; 
    c.x =- rhs.x; 
    c.y =- rhs.y; 
    c.z =- rhs.z; 
    c.w =- rhs.w; 
    return(c);
}

/* Subtracts a vec4 from a vec4. */
template<typename T>
sf_inline vec4<T> 
operator - 
(const vec4<T> &lhs, const vec4<T> &rhs) {
    vec4<T> c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    c.z = lhs.z - rhs.z; 
    c.w = lhs.w - rhs.w; 
    return(c);
}

/* Subtracts a scalar from a vec4. */
template<typename T, typename U>
sf_inline vec4<T> 
operator - 
(const vec4<T> &lhs, const U &rhs) {
    vec4<T> c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    c.z = lhs.z - rhs; 
    c.w = lhs.w - rhs; 
    return(c);
}

/* Subtracts a vec4 from a scalar. */
template<typename T, typename U>
sf_inline vec4<T> 
operator - 
(const U &lhs, const vec4<T> &rhs) {
    vec4<T> c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    c.z = lhs - rhs.z; 
    c.w = lhs - rhs.w; 
    return(c);
}

/* Minus-equals operand for two vec4s. */
template<typename T>
sf_inline vec4<T>& 
operator -= 
(vec4<T> &lhs, const vec4<T> &rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y; 
    lhs.z -= rhs.z; 
    lhs.w -= rhs.w;
    return(lhs);
}

/* Minus-equals operand for vec4 and scalar. */
template<typename T, typename U>
sf_inline vec4<T>& 
operator -= 
(vec4<T> &lhs, const U &rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs; 
    lhs.z -= rhs; 
    lhs.w -= rhs;
    return(lhs);
}

/* Multiplies two vec4s. */
template<typename T>
sf_inline vec4<T> 
operator * 
(const vec4<T> &lhs, const vec4<T> &rhs) {
    vec4<T> c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y; 
    c.z = rhs.z * lhs.z; 
    c.w = rhs.w * lhs.w;
    return(c);
}

/* Multiplies a vec4 and scalar. */
template<typename T, typename U>
sf_inline vec4<T> 
operator * 
(const U &lhs, const vec4<T> &rhs) {
    vec4<T> c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs; 
    c.z = rhs.z * lhs; 
    c.w = rhs.w * lhs;
    return(c);
}

/* Multiplies a scalar and vec4. */
template<typename T, typename U>
sf_inline vec4<T> 
operator * 
(const vec4<T> &lhs, const U &rhs) {
    vec4<T> c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    c.z = rhs * lhs.z;
    c.w = rhs * lhs.w;
    return(c);
}

/* Multiply-equals operand for vec4. */
template<typename T>
sf_inline vec4<T>& 
operator *= 
(vec4<T> &lhs, const vec4<T> &rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y; 
    lhs.z *= rhs.z; 
    lhs.w *= rhs.w;
    return(lhs);
}

/* Multiply-equals operand for vec4 and scalar. */
template<typename T, typename U>
sf_inline vec4<T>& 
operator *= 
(vec4<T> &lhs, const U &rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs; 
    lhs.z *= rhs; 
    lhs.w *= rhs;
    return(lhs);
}

/* Divides two vec4s. */
template<typename T>
sf_inline vec4<T> 
operator / 
(const vec4<T> &lhs, const vec4<T> &rhs) {
    vec4<T> c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y; 
    c.z = lhs.z / rhs.z; 
    c.w = lhs.w / rhs.w;
    return(c);
}

/* Divides a vec4 by a scalar. */
template<typename T, typename U>
sf_inline vec4<T> 
operator / 
(const vec4<T> &lhs, const U &rhs) {
    vec4<T> c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs; 
    c.z = lhs.z / rhs; 
    c.w = lhs.w / rhs;
    return(c);
}

/* Divide-equals operand for two vec4s. */
template<typename T>
sf_inline vec4<T>& 
operator /= 
(vec4<T> &lhs, const vec4<T> &rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y; 
    lhs.z /= rhs.z; 
    lhs.w /= rhs.w;
    return(lhs);
}

/* Divide-equals operand for vec4 and scalar. */
template<typename T, typename U>
sf_inline vec4<T>& 
operator /= 
(vec4<T> &lhs, const U &rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs; 
    lhs.z /= rhs; 
    lhs.w /= rhs;
    return(lhs);
}

/* Add one to each element in vec4. */
template<typename T>
sf_inline vec4<T>& 
operator ++ 
(vec4<T> &rhs) {
    ++rhs.x; 
    ++rhs.y; 
    ++rhs.z; 
    ++rhs.w;
    return(rhs);
}

/* Add one to each element in vec4. */
template<typename T>
sf_inline vec4<T> 
operator ++ 
(vec4<T> &lhs, i32) {
    vec4<T> c = lhs;
    lhs.x++; 
    lhs.y++; 
    lhs.z++; 
    lhs.w++;
    return(c);
}

/* Subtract one from each element in vec4. */
template<typename T>
sf_inline vec4<T>& 
operator -- 
(vec4<T> &rhs) {
    --rhs.x; 
    --rhs.y; 
    --rhs.z; 
    --rhs.w;
    return(rhs);
}

/* Subtract one from each element in vec4. */
template<typename T>
sf_inline vec4<T> 
operator -- 
(vec4<T> &lhs, i32) {
    vec4<T> c = lhs;
    lhs.x--; 
    lhs.y--; 
    lhs.z--; 
    lhs.w--;
    return(c);
}

/* Tests two vec4s for equality. */
template<typename T>
sf_inline bool 
operator == 
(const vec4<T> &lhs, const vec4<T> &rhs) {
    return((lhs.x == rhs.x) && 
           (lhs.y == rhs.y) && 
           (lhs.z == rhs.z) && 
           (lhs.w == rhs.w));
}

/* Tests two vec4s for non-equality. */
template<typename T>
sf_inline bool 
operator != 
(const vec4<T> &lhs, const vec4<T> &rhs) {
    return((lhs.x != rhs.x) || 
           (lhs.y != rhs.y) || 
           (lhs.z != rhs.z) || 
           (lhs.w != rhs.w));
}

template<typename T>
sf_inline std::ostream& 
operator << 
(std::ostream &os, const vec4<T> &rhs) {
    os    << "(" << 
    rhs.x << "," << 
    rhs.y << "," << 
    rhs.z << "," << 
    rhs.w << ")";
    return(os);
}

//==============================================================================
// mat2                                                 
//==============================================================================
// TODO: Come back and define mat2
/* 2x2 Matrix */

//==============================================================================
// mat3                                                 
//==============================================================================

/* 3x3 Matrix */

template <class T>
class mat3 {
public:
    union {
        struct { 
            /* reference matrix [row][column] */
            T m[3][3]; 
        };
        struct { 
            T x0, x1, x2;
            T y0, y1, y2;
            T z0, z1, z2; 
        };
    };

    mat3<T>() { 
        x0 = 0; y0 = 0; z0 = 0;
        x1 = 0; y1 = 0; z1 = 0;
        x2 = 0; y2 = 0; z2 = 0;
    }

    mat3<T>(vec3<T> v1, vec3<T> v2, vec3<T> v3) { 
        x0 = v1.x; y0 = v1.y; z0 = v1.z; 
        x1 = v2.x; y1 = v2.y; z1 = v2.z; 
        x2 = v3.x; y2 = v3.y; z2 = v3.z; 
    }

    mat3<T>(const mat3<T> &v) { 
        x0 = v.x0; y0 = v.y0; z0 = v.z0; 
        x1 = v.x1; y1 = v.y1; z1 = v.z1; 
        x2 = v.x2; y2 = v.y2; z2 = v.z2; 
    }

    sf_inline mat3<T> const transpose() const {
        mat3<T> transpose;
        /* row 1 */
        transpose.m[0][0] = m[0][0]; 
        transpose.m[1][0] = m[0][1]; 
        transpose.m[2][0] = m[0][2];
        /* row 2 */
        transpose.m[0][1] = m[1][0]; 
        transpose.m[1][1] = m[1][1]; 
        transpose.m[2][1] = m[1][2];
        /* row 3 */
        transpose.m[0][2] = m[2][0]; 
        transpose.m[1][2] = m[2][1]; 
        transpose.m[2][2] = m[2][2];
        return transpose;
    }

    sf_inline T const determinant() const {
        return(m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - 
               m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) + 
               m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]));
    }

    sf_inline mat3<T> const inverse() const {
        T inverse_determinant = 
            1.0f / (m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - 
                    m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) + 
                    m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]));
        mat3<T> inv;
        inv.m[0][0] =  (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inverse_determinant;
        inv.m[1][0] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]) * inverse_determinant;
        inv.m[2][0] =  (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inverse_determinant;
        inv.m[0][1] = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]) * inverse_determinant;
        inv.m[1][1] =  (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * inverse_determinant;
        inv.m[2][1] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]) * inverse_determinant;
        inv.m[0][2] =  (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * inverse_determinant;
        inv.m[1][2] = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]) * inverse_determinant;
        inv.m[2][2] =  (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inverse_determinant;
        return(inv);
    }

    sf_inline void const print(i32 new_line = 1) const {
        printf("| %10.5f %10.5f %10.5f |\n", m[0][0], m[0][1], m[0][2]);
        printf("| %10.5f %10.5f %10.5f |\n", m[1][0], m[1][1], m[1][2]);
        printf("| %10.5f %10.5f %10.5f |\n", m[2][0], m[2][1], m[2][2]);
        if (new_line) { 
            printf("\n"); 
        }
    }
};

/* Add two mat3s. */
template<typename T>
sf_inline mat3<T> 
operator + 
(const mat3<T> &lhs, const mat3<T> &rhs) {
    mat3<T> c;
    /* row 1 */
    c.m[0][0] = lhs.m[0][0] + rhs.m[0][0]; 
    c.m[1][0] = lhs.m[1][0] + rhs.m[1][0]; 
    c.m[2][0] = lhs.m[2][0] + rhs.m[2][0];
    /* row 2 */
    c.m[0][1] = lhs.m[0][1] + rhs.m[0][1]; 
    c.m[1][1] = lhs.m[1][1] + rhs.m[1][1]; 
    c.m[2][1] = lhs.m[2][1] + rhs.m[2][1];
    /* row 3 */
    c.m[0][2] = lhs.m[0][2] + rhs.m[0][2]; 
    c.m[1][2] = lhs.m[1][2] + rhs.m[1][2]; 
    c.m[2][2] = lhs.m[2][2] + rhs.m[2][2];
    return c;
}

/* Mat3 plus-equals operand. */
template<typename T>
sf_inline mat3<T>& 
operator += 
(mat3<T> &lhs, const mat3<T> &rhs) {
    lhs = lhs + rhs;
    return lhs;
}

/* Unary minus operand. Makes mat3 negative. */
template<typename T>
sf_inline mat3<T> 
operator - 
(const mat3<T> &rhs) {
    mat3<T> c;
    /* row 1 */
    c.x0 =- rhs.x0; 
    c.y0 =- rhs.y0; 
    c.z0 =- rhs.z0;
    /* row 2 */
    c.x1 =- rhs.x1; 
    c.y1 =- rhs.y1; 
    c.z1 =- rhs.z1;
    /* row 3 */
    c.x2 =- rhs.x2; 
    c.y2 =- rhs.y2; 
    c.z2 =- rhs.z2;
    return c;
}

/* Subtract a mat3 from a mat3. */
template<typename T>
sf_inline mat3<T> 
operator - 
(const mat3<T> &lhs, const mat3<T> &rhs) {
    mat3<T> c;
    /* row 1 */
    c.m[0][0] = lhs.m[0][0] - rhs.m[0][0]; 
    c.m[1][0] = lhs.m[1][0] - rhs.m[1][0]; 
    c.m[2][0] = lhs.m[2][0] - rhs.m[2][0];
    /* row 2 */
    c.m[0][1] = lhs.m[0][1] - rhs.m[0][1]; 
    c.m[1][1] = lhs.m[1][1] - rhs.m[1][1]; 
    c.m[2][1] = lhs.m[2][1] - rhs.m[2][1];
    /* row 3 */
    c.m[0][2] = lhs.m[0][2] - rhs.m[0][2]; 
    c.m[1][2] = lhs.m[1][2] - rhs.m[1][2]; 
    c.m[2][2] = lhs.m[2][2] - rhs.m[2][2];
    return c;
}

/* Mat3 minus-equals operand. */
template<typename T>
sf_inline mat3<T>& 
operator -= 
(mat3<T> &lhs, const mat3<T> &rhs) {
    lhs = lhs - rhs;
    return lhs;
}

/* Multiply a mat3 with a vec3. */
template<typename T>
sf_inline vec3<T> 
operator * 
(const mat3<T> &lhs, const vec3<T> &rhs) {
    vec3<T> c;
    c.x = rhs.x * lhs.x0 + rhs.y * lhs.x1 + rhs.z * lhs.x2;
    c.y = rhs.x * lhs.y0 + rhs.y * lhs.y1 + rhs.z * lhs.y2;
    c.z = rhs.x * lhs.z0 + rhs.y * lhs.z1 + rhs.z * lhs.z2;
    return c;
}

/* Multiply a vec3 with a mat3. */
template<typename T>
sf_inline vec3<T> 
operator * 
(const vec3<T> &lhs, const mat3<T> &rhs) {
    vec3<T> c;
    c.x = lhs.x * rhs.x0 + lhs.y * rhs.y0 + lhs.z * rhs.z0;
    c.y = lhs.x * rhs.x1 + lhs.y * rhs.y1 + lhs.z * rhs.z1;
    c.z = lhs.x * rhs.x2 + lhs.y * rhs.y2 + lhs.z * rhs.z2;
    return c;
}

/* Multiply a mat3 with a scalar. */
template<typename T, typename U>
sf_inline mat3<T> 
operator * 
(const mat3<T> &lhs, const U &rhs) {
    mat3<T> c;
    /* row 1 */
    c.m[0][0] = lhs.m[0][0] * rhs; 
    c.m[1][0] = lhs.m[1][0] * rhs; 
    c.m[2][0] = lhs.m[2][0] * rhs;
    /* row 2 */
    c.m[0][1] = lhs.m[0][1] * rhs; 
    c.m[1][1] = lhs.m[1][1] * rhs; 
    c.m[2][1] = lhs.m[2][1] * rhs;
    /* row 3 */
    c.m[0][2] = lhs.m[0][2] * rhs; 
    c.m[1][2] = lhs.m[1][2] * rhs; 
    c.m[2][2] = lhs.m[2][2] * rhs;
    return c;
}

/* Multiply a scalar with a mat3. */
template<typename T, typename U>
sf_inline mat3<T> 
operator * 
(const U &lhs, const mat3<T> &rhs) {
    return(rhs * lhs);
}

/* Multiply two mat3s. */
template<typename T>
sf_inline mat3<T> 
operator * 
(const mat3<T> &lhs, const mat3<T> &rhs) {
    mat3<T> c;
    /* row 1 */
    c.m[0][0] = rhs.m[0][0] * lhs.m[0][0] + 
                rhs.m[1][0] * lhs.m[0][1] + 
                rhs.m[2][0] * lhs.m[0][2];
    c.m[1][0] = rhs.m[0][0] * lhs.m[1][0] + 
                rhs.m[1][0] * lhs.m[1][1] + 
                rhs.m[2][0] * lhs.m[1][2];
    c.m[2][0] = rhs.m[0][0] * lhs.m[2][0] + 
                rhs.m[1][0] * lhs.m[2][1] + 
                rhs.m[2][0] * lhs.m[2][2];
    /* row 2 */
    c.m[0][1] = rhs.m[0][1] * lhs.m[0][0] + 
                rhs.m[1][1] * lhs.m[0][1] + 
                rhs.m[2][1] * lhs.m[0][2];
    c.m[1][1] = rhs.m[0][1] * lhs.m[1][0] + 
                rhs.m[1][1] * lhs.m[1][1] + 
                rhs.m[2][1] * lhs.m[1][2];
    c.m[2][1] = rhs.m[0][1] * lhs.m[2][0] + 
                rhs.m[1][1] * lhs.m[2][1] + 
                rhs.m[2][1] * lhs.m[2][2];
    /* row 3 */
    c.m[0][2] = rhs.m[0][2] * lhs.m[0][0] + 
                rhs.m[1][2] * lhs.m[0][1] + 
                rhs.m[2][2] * lhs.m[0][2];
    c.m[1][2] = rhs.m[0][2] * lhs.m[1][0] + 
                rhs.m[1][2] * lhs.m[1][1] + 
                rhs.m[2][2] * lhs.m[1][2];
    c.m[2][2] = rhs.m[0][2] * lhs.m[2][0] + 
                rhs.m[1][2] * lhs.m[2][1] + 
                rhs.m[2][2] * lhs.m[2][2];
    return c;
}

/* Multiply-equals operand with two mat3s. */
template<typename T>
sf_inline mat3<T>& 
operator *= 
(mat3<T> &lhs, const mat3<T> &rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Multiply-equals operand with mat3 and scalar. */
template<typename T, typename U>
sf_inline mat3<T>& 
operator *= 
(mat3<T> &lhs, const U &rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Divide a mat3 by a scalar. */
template<typename T, typename U>
sf_inline mat3<T> operator / 
(const mat3<T> &lhs, const U &rhs) {
    mat3<T> c;
    /* row 1 */
    c.m[0][0] = lhs.m[0][0] / rhs; 
    c.m[1][0] = lhs.m[1][0] / rhs; 
    c.m[2][0] = lhs.m[2][0] / rhs;
    /* row 2 */
    c.m[0][1] = lhs.m[0][1] / rhs; 
    c.m[1][1] = lhs.m[1][1] / rhs; 
    c.m[2][1] = lhs.m[2][1] / rhs;
    /* row 3 */
    c.m[0][2] = lhs.m[0][2] / rhs; 
    c.m[1][2] = lhs.m[1][2] / rhs; 
    c.m[2][2] = lhs.m[2][2] / rhs;
    return c;
}

/* Mat3 division operation as multiplicative inverse. */
template<typename T>
sf_inline mat3<T> 
operator / 
(const mat3<T> &lhs, const mat3<T> &rhs) {
    return(lhs * rhs.inverse());
}

/* Divide-equals operand with two mat3s. */
template<typename T>
sf_inline mat3<T>& 
operator /= 
(mat3<T> &lhs, const mat3<T> &rhs) {
    lhs = lhs / rhs;
    return lhs;
}

/* Divide-equals operand with mat3 and scalar. */
template<typename T, typename U>
sf_inline mat3<T>& 
operator /= 
(mat3<T> &lhs, const U &rhs) {
    lhs = lhs / rhs;
    return lhs;
}

/* Tests for equality between two mat3s. */
template<typename T>
sf_inline bool 
operator == 
(const mat3<T> &lhs, const mat3<T> &rhs) {
    T M[9];
    return((lhs.M[0] == rhs.M[0]) && 
           (lhs.M[1] == rhs.M[1]) && 
           (lhs.M[2] == rhs.M[2]) &&
           (lhs.M[3] == rhs.M[3]) && 
           (lhs.M[4] == rhs.M[4]) && 
           (lhs.M[5] == rhs.M[5]) &&
           (lhs.M[6] == rhs.M[6]) && 
           (lhs.M[7] == rhs.M[7]) && 
           (lhs.M[8] == rhs.M[8]));
}

/* Tests for non-equality between two mat3s. */
template<typename T>
sf_inline bool 
operator != 
(const mat3<T> &lhs, const mat3<T> &rhs) {
    T M[9];
    return((lhs.M[0] != rhs.M[0]) || 
           (lhs.M[1] != rhs.M[1]) || 
           (lhs.M[2] != rhs.M[2]) ||
           (lhs.M[3] != rhs.M[3]) || 
           (lhs.M[4] != rhs.M[4]) || 
           (lhs.M[5] != rhs.M[5]) ||
           (lhs.M[6] != rhs.M[6]) || 
           (lhs.M[7] != rhs.M[7]) || 
           (lhs.M[8] != rhs.M[8]));
}

/* Allows for iostream output of mat3 elements. */
template<typename T>
sf_inline std::ostream& 
operator << 
(std::ostream &os, const mat3<T> &rhs) {
    std::ios_base::fmtflags f = os.flags();
    os << std::fixed;
    os << std::endl;
    os << "| " << std::setprecision(5) << std::setw(10) << rhs.x0 << " " 
               << std::setprecision(5) << std::setw(10) << rhs.x1 << " " 
               << std::setprecision(5) << std::setw(10) << rhs.x2 << " |" 
               << std::endl;
    os << "| " << std::setprecision(5) << std::setw(10) << rhs.y0 << " " 
               << std::setprecision(5) << std::setw(10) << rhs.y1 << " " 
               << std::setprecision(5) << std::setw(10) << rhs.y2 << " |" 
               << std::endl;
    os << "| " << std::setprecision(5) << std::setw(10) << rhs.z0 << " " 
               << std::setprecision(5) << std::setw(10) << rhs.z1 << " " 
               << std::setprecision(5) << std::setw(10) << rhs.z2 << " |" 
               << std::endl;
    os.flags(f);
    return(os);
}

//==============================================================================
// mat4                                                 
//==============================================================================

/* 4x4 Matrix */

template <class T>
class mat4 {
public:
    union {
        struct { 
            T m[4][4]; 
        };
        struct { 
            T M[16]; 
        };
        struct { 
            T x0, x1, x2, x3;
            T y0, y1, y2, y3; 
            T z0, z1, z2, z3; 
            T w0, w1, w2, w3; 
        };
    };

    mat4<T>() { 
        x0 = 0; y0 = 0; z0 = 0; w0 = 0;
        x1 = 0; y1 = 0; z1 = 0; w1 = 0;
        x2 = 0; y2 = 0; z2 = 0; w2 = 0;
        x3 = 0; y3 = 0; z3 = 0; w3 = 0; 
    }

    mat4<T>(vec4<T> v1, vec4<T> v2, vec4<T> v3, vec4<T> v4) { 
        x0 = v1.x; y0 = v1.y; z0 = v1.z; w0 = v1.w; 
        x1 = v2.x; y1 = v2.y; z1 = v2.z; w1 = v2.w; 
        x2 = v3.x; y2 = v3.y; z2 = v3.z; w2 = v3.w; 
        x3 = v4.x; y3 = v4.y; z3 = v4.z; w3 = v4.w; 
    }

    mat4<T>(const mat4<T> &v) { 
        x0 = v.x0; y0 = v.y0; z0 = v.z0; w0 = v.w0; 
        x1 = v.x1; y1 = v.y1; z1 = v.z1; w1 = v.w1; 
        x2 = v.x2; y2 = v.y2; z2 = v.z2; w2 = v.w2; 
        x3 = v.x3; y3 = v.y3; z3 = v.z3; w3 = v.w3; 
    }

    sf_inline mat4<T> const transpose() const {
        mat4<T> transpose;
        /* row 1 */
        transpose.m[0][0]=m[0][0]; 
        transpose.m[1][0]=m[0][1]; 
        transpose.m[2][0]=m[0][2];
        transpose.m[3][0]=m[0][3];
        /* row 2 */
        transpose.m[0][1]=m[1][0]; 
        transpose.m[1][1]=m[1][1]; 
        transpose.m[2][1]=m[1][2];
        transpose.m[3][1]=m[1][3];
        /* row 3 */
        transpose.m[0][2]=m[2][0]; 
        transpose.m[1][2]=m[2][1]; 
        transpose.m[2][2]=m[2][2];
        transpose.m[3][2]=m[2][3];
        /* row 4 */
        transpose.m[0][3]=m[3][0]; 
        transpose.m[1][3]=m[3][1]; 
        transpose.m[2][3]=m[3][2];
        transpose.m[3][3]=m[3][3];
        return transpose;
    }

    sf_inline T const determinant() const {
        T f0 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        T f1 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        T f2 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        T f3 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
        T f4 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
        T f5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
        vec4<T> dc = vec4<T>((m[1][1] * f0 - m[2][1] * f1 + m[3][1] * f2),
                            -(m[0][1] * f0 - m[2][1] * f3 + m[3][1] * f4),
                             (m[0][1] * f1 - m[1][1] * f3 + m[3][1] * f5),
                            -(m[0][1] * f2 - m[1][1] * f4 + m[2][1] * f5));
        return(m[0][0] * dc.x + 
               m[1][0] * dc.y + 
               m[2][0] * dc.z + 
               m[3][0] * dc.w);
    }

    /* Beginning of mat4 inverse() function */
    
    sf_inline void
    mat4_scale(mat4<T> mat, T s) {
        mat.m[0][0] *= s; mat.m[0][1] *= s; mat.m[0][2] *= s; mat.m[0][3] *= s;
        mat.m[1][0] *= s; mat.m[1][1] *= s; mat.m[1][2] *= s; mat.m[1][3] *= s;
        mat.m[2][0] *= s; mat.m[2][1] *= s; mat.m[2][2] *= s; mat.m[2][3] *= s;
        mat.m[3][0] *= s; mat.m[3][1] *= s; mat.m[3][2] *= s; mat.m[3][3] *= s;
    }
    
    sf_inline void
    mat4_inverse(mat4<T> mat, mat4<T> dest) {
        T t[6];
        T det;
        T     a = mat.m[0][0], b = mat.m[0][1], c = mat.m[0][2], d = mat.m[0][3],
              e = mat.m[1][0], f = mat.m[1][1], g = mat.m[1][2], h = mat.m[1][3],
              i = mat.m[2][0], j = mat.m[2][1], k = mat.m[2][2], l = mat.m[2][3],
              m = mat.m[3][0], n = mat.m[3][1], o = mat.m[3][2], p = mat.m[3][3];

        t[0] = k * p - o * l; t[1] = j * p - n * l; t[2] = j * o - n * k;
        t[3] = i * p - m * l; t[4] = i * o - m * k; t[5] = i * n - m * j;

        dest.m[0][0] =  f * t[0] - g * t[1] + h * t[2];
        dest.m[1][0] =-(e * t[0] - g * t[3] + h * t[4]);
        dest.m[2][0] =  e * t[1] - f * t[3] + h * t[5];
        dest.m[3][0] =-(e * t[2] - f * t[4] + g * t[5]);

        dest.m[0][1] =-(b * t[0] - c * t[1] + d * t[2]);
        dest.m[1][1] =  a * t[0] - c * t[3] + d * t[4];
        dest.m[2][1] =-(a * t[1] - b * t[3] + d * t[5]);
        dest.m[3][1] =  a * t[2] - b * t[4] + c * t[5];

        t[0] = g * p - o * h; t[1] = f * p - n * h; t[2] = f * o - n * g;
        t[3] = e * p - m * h; t[4] = e * o - m * g; t[5] = e * n - m * f;

        dest.m[0][2] =  b * t[0] - c * t[1] + d * t[2];
        dest.m[1][2] =-(a * t[0] - c * t[3] + d * t[4]);
        dest.m[2][2] =  a * t[1] - b * t[3] + d * t[5];
        dest.m[3][2] =-(a * t[2] - b * t[4] + c * t[5]);
    
        t[0] = g * l - k * h; t[1] = f * l - j * h; t[2] = f * k - j * g;
        t[3] = e * l - i * h; t[4] = e * k - i * g; t[5] = e * j - i * f;
    
        dest.m[0][3] =-(b * t[0] - c * t[1] + d * t[2]);
        dest.m[1][3] =  a * t[0] - c * t[3] + d * t[4];
        dest.m[2][3] =-(a * t[1] - b * t[3] + d * t[5]);
        dest.m[3][3] =  a * t[2] - b * t[4] + c * t[5];

        det = 1.0f / (a * dest.m[0][0] + b * dest.m[1][0]
                    + c * dest.m[2][0] + d * dest.m[3][0]);

        mat4_scale(dest, det);
    }  

    sf_inline mat4<T> const inverse_0() const {
        mat4<T> mat;
        mat4<T> r;
        mat4_inv(mat.m, r.m);
        return r;
    }

/* This algorithm is almost directly borrowed from GLM. */

    sf_inline mat4<T> const inverse_1() const {
        T c00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        T c02 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
        T c03 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
        T c04 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        T c06 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
        T c07 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
        T c08 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        T c10 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
        T c11 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        T c12 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
        T c14 = m[0][1] * m[3][3] - m[0][3] * m[3][1];
        T c15 = m[0][1] * m[3][2] - m[0][2] * m[3][1];
        T c16 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
        T c18 = m[0][1] * m[2][3] - m[0][3] * m[2][1];
        T c19 = m[0][1] * m[2][2] - m[0][2] * m[2][1];
        T c20 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
        T c22 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
        T c23 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        vec4<T> f0(c00, c00, c02, c03);
        vec4<T> f1(c04, c04, c06, c07);
        vec4<T> f2(c08, c08, c10, c11);
        vec4<T> f3(c12, c12, c14, c15);
        vec4<T> f4(c16, c16, c18, c19);
        vec4<T> f5(c20, c20, c22, c23);
        vec4<T> v0(m[0][1], m[0][0], m[0][0], m[0][0]);
        vec4<T> v1(m[1][1], m[1][0], m[1][0], m[1][0]);
        vec4<T> v2(m[2][1], m[2][0], m[2][0], m[2][0]);
        vec4<T> v3(m[3][1], m[3][0], m[3][0], m[3][0]);
        vec4<T> i0(v1 * f0 - v2 * f1 + v3 * f2);
        vec4<T> i1(v0 * f0 - v2 * f3 + v3 * f4);
        vec4<T> i2(v0 * f1 - v1 * f3 + v3 * f5);
        vec4<T> i3(v0 * f2 - v1 * f4 + v2 * f5);
        vec4<T> sA( 1,-1, 1,-1);
        vec4<T> sB(-1, 1,-1, 1);
        mat4<T> inv(i0 * sA, 
                    i1 * sB, 
                    i2 * sA, 
                    i3 * sB);
        vec4<T> r0(inv.m[0][0],
                   inv.m[0][1],
                   inv.m[0][2],
                   inv.m[0][3]);
        vec4<T> d0(m[0][0] * r0.x,
                   m[1][0] * r0.y,
                   m[2][0] * r0.z,
                   m[3][0] * r0.w);
        T d1 = (d0.x + d0.y) + (d0.z + d0.w);
        T invdet = static_cast<T>(1) / d1;
        return(inv * invdet);
    }


    sf_inline void const print(i32 new_line = 1) const {
        printf("| %10.5f %10.5f %10.5f %10.5f |\n",m[0][0], m[0][1], m[0][2], m[0][3]);
        printf("| %10.5f %10.5f %10.5f %10.5f |\n",m[1][0], m[1][1], m[1][2], m[1][3]);
        printf("| %10.5f %10.5f %10.5f %10.5f |\n",m[2][0], m[2][1], m[2][2], m[2][3]);
        printf("| %10.5f %10.5f %10.5f %10.5f |\n",m[3][0], m[3][1], m[3][2], m[3][3]);
        if (new_line) { 
            printf("\n"); 
        }
    }
};

template <typename T>
mat4<T> operator+ (const mat4<T> &lhs, const mat4<T> &rhs) {
    mat4<T> c;
    c.m[0][0]=lhs.m[0][0]+rhs.m[0][0]; c.m[1][0]=lhs.m[1][0]+rhs.m[1][0]; c.m[2][0]=lhs.m[2][0]+rhs.m[2][0]; c.m[3][0]=lhs.m[3][0]+rhs.m[3][0];
    c.m[0][1]=lhs.m[0][1]+rhs.m[0][1]; c.m[1][1]=lhs.m[1][1]+rhs.m[1][1]; c.m[2][1]=lhs.m[2][1]+rhs.m[2][1]; c.m[3][1]=lhs.m[3][1]+rhs.m[3][1];
    c.m[0][2]=lhs.m[0][2]+rhs.m[0][2]; c.m[1][2]=lhs.m[1][2]+rhs.m[1][2]; c.m[2][2]=lhs.m[2][2]+rhs.m[2][2]; c.m[3][2]=lhs.m[3][2]+rhs.m[3][2];
    c.m[0][3]=lhs.m[0][3]+rhs.m[0][3]; c.m[1][3]=lhs.m[1][3]+rhs.m[1][3]; c.m[2][3]=lhs.m[2][3]+rhs.m[2][3]; c.m[3][3]=lhs.m[3][3]+rhs.m[3][3];
    return c;
}

template <typename T>
mat4<T> operator- (const mat4<T> &rhs)
    {
    mat4<T> c;
    c.x0=-rhs.x0; c.y0=-rhs.y0; c.z0=-rhs.z0; c.w0=-rhs.w0;
    c.x1=-rhs.x1; c.y1=-rhs.y1; c.z1=-rhs.z1; c.w1=-rhs.w1;
    c.x2=-rhs.x2; c.y2=-rhs.y2; c.z2=-rhs.z2; c.w2=-rhs.w2;
    c.x3=-rhs.x3; c.y3=-rhs.y3; c.z3=-rhs.z3; c.w3=-rhs.w3;
    return(c);
    }

template <typename T>
mat4<T> operator- (const mat4<T> &lhs, const mat4<T> &rhs)
    {
    mat4<T> c;
    c.m[0][0]=lhs.m[0][0]-rhs.m[0][0]; c.m[1][0]=lhs.m[1][0]-rhs.m[1][0]; c.m[2][0]=lhs.m[2][0]-rhs.m[2][0]; c.m[3][0]=lhs.m[3][0]-rhs.m[3][0];
    c.m[0][1]=lhs.m[0][1]-rhs.m[0][1]; c.m[1][1]=lhs.m[1][1]-rhs.m[1][1]; c.m[2][1]=lhs.m[2][1]-rhs.m[2][1]; c.m[3][1]=lhs.m[3][1]-rhs.m[3][1];
    c.m[0][2]=lhs.m[0][2]-rhs.m[0][2]; c.m[1][2]=lhs.m[1][2]-rhs.m[1][2]; c.m[2][2]=lhs.m[2][2]-rhs.m[2][2]; c.m[3][2]=lhs.m[3][2]-rhs.m[3][2];
    c.m[0][3]=lhs.m[0][3]-rhs.m[0][3]; c.m[1][3]=lhs.m[1][3]-rhs.m[1][3]; c.m[2][3]=lhs.m[2][3]-rhs.m[2][3]; c.m[3][3]=lhs.m[3][3]-rhs.m[3][3];
    return(c);
    }

template <typename T>
vec4<T> operator* (const mat4<T> &lhs, const vec4<T> &rhs)
    {
    vec4<T> c;
    c.x=rhs.x*lhs.x0+rhs.y*lhs.x1+rhs.z*lhs.x2+rhs.w*lhs.x3;
    c.y=rhs.x*lhs.y0+rhs.y*lhs.y1+rhs.z*lhs.y2+rhs.w*lhs.y3;
    c.z=rhs.x*lhs.z0+rhs.y*lhs.z1+rhs.z*lhs.z2+rhs.w*lhs.z3;
    c.w=rhs.x*lhs.w0+rhs.y*lhs.w1+rhs.z*lhs.w2+rhs.w*lhs.w3;
    return(c);
    }

template <typename T>
vec4<T> operator* (const vec4<T> &lhs, const mat4<T> &rhs)
    {
    vec4<T> c;
    c.x=lhs.x*rhs.x0+lhs.y*rhs.y0+lhs.z*rhs.z0+lhs.w*rhs.w0;
    c.y=lhs.x*rhs.x1+lhs.y*rhs.y1+lhs.z*rhs.z1+lhs.w*rhs.w1;
    c.z=lhs.x*rhs.x2+lhs.y*rhs.y2+lhs.z*rhs.z2+lhs.w*rhs.w2;
    c.w=lhs.x*rhs.x3+lhs.y*rhs.y3+lhs.z*rhs.z3+lhs.w*rhs.w3;
    return(c);
    }

template <typename T, typename U>
mat4<T> operator* (const mat4<T> &lhs, const U &rhs)
    {
    mat4<T> c;
    c.m[0][0]=lhs.m[0][0]*rhs; c.m[1][0]=lhs.m[1][0]*rhs; c.m[2][0]=lhs.m[2][0]*rhs; c.m[3][0]=lhs.m[3][0]*rhs;
    c.m[0][1]=lhs.m[0][1]*rhs; c.m[1][1]=lhs.m[1][1]*rhs; c.m[2][1]=lhs.m[2][1]*rhs; c.m[3][1]=lhs.m[3][1]*rhs;
    c.m[0][2]=lhs.m[0][2]*rhs; c.m[1][2]=lhs.m[1][2]*rhs; c.m[2][2]=lhs.m[2][2]*rhs; c.m[3][2]=lhs.m[3][2]*rhs;
    c.m[0][3]=lhs.m[0][3]*rhs; c.m[1][3]=lhs.m[1][3]*rhs; c.m[2][3]=lhs.m[2][3]*rhs; c.m[3][3]=lhs.m[3][3]*rhs;
    return(c);
    }

template <typename T, typename U>
mat4<T> operator* (const U &lhs, const mat4<T> &rhs)
    {
    return(rhs*lhs);
    }

template <typename T>
mat4<T> operator* (const mat4<T> &lhs, const mat4<T> &rhs)
    {
    mat4<T> c;
    int i,j;
    for (j=0;j<4;j++)
        {
        for (i=0;i<4;i++)
            {
            c.m[i][j]=rhs.m[0][j]*lhs.m[i][0]+rhs.m[1][j]*lhs.m[i][1]+rhs.m[2][j]*lhs.m[i][2]+rhs.m[3][j]*lhs.m[i][3];
            }
        }
    return(c);
    }

template <typename T, typename U>
mat4<T> operator/ (const mat4<T> &lhs, const U &rhs)
    {
    mat4<T> c;
    c.m[0][0]=lhs.m[0][0]/rhs; c.m[1][0]=lhs.m[1][0]/rhs; c.m[2][0]=lhs.m[2][0]/rhs; c.m[3][0]=lhs.m[3][0]/rhs;
    c.m[0][1]=lhs.m[0][1]/rhs; c.m[1][1]=lhs.m[1][1]/rhs; c.m[2][1]=lhs.m[2][1]/rhs; c.m[3][1]=lhs.m[3][1]/rhs;
    c.m[0][2]=lhs.m[0][2]/rhs; c.m[1][2]=lhs.m[1][2]/rhs; c.m[2][2]=lhs.m[2][2]/rhs; c.m[3][2]=lhs.m[3][2]/rhs;
    c.m[0][3]=lhs.m[0][3]/rhs; c.m[1][3]=lhs.m[1][3]/rhs; c.m[2][3]=lhs.m[2][3]/rhs; c.m[3][3]=lhs.m[3][3]/rhs;
    return(c);
    }

template <typename T>
mat4<T> operator/ (const mat4<T> &lhs, const mat4<T> &rhs)
    {
    return(lhs*rhs.inverse());
    }

template <typename T>
mat4<T>& operator+= (mat4<T> &lhs, const mat4<T> &rhs)
    {
    lhs=lhs+rhs;
    return(lhs);
    }

template <typename T>
mat4<T>& operator-= (mat4<T> &lhs, const mat4<T> &rhs)
    {
    lhs=lhs-rhs;
    return(lhs);
    }

template <typename T>
mat4<T>& operator*= (mat4<T> &lhs, const mat4<T> &rhs)
    {
    lhs=lhs*rhs;
    return(lhs);
    }

template <typename T, typename U>
mat4<T>& operator*= (mat4<T> &lhs, const U &rhs)
    {
    lhs=lhs*rhs;
    return(lhs);
    }

template <typename T>
mat4<T>& operator/= (mat4<T> &lhs, const mat4<T> &rhs)
    {
    lhs=lhs/rhs;
    return(lhs);
    }

template <typename T, typename U>
mat4<T>& operator/= (mat4<T> &lhs, const U &rhs)
    {
    lhs=lhs/rhs;
    return(lhs);
    }

template<typename T>
sf_inline bool 
operator == 
(const mat4<T> &lhs, const mat4<T> &rhs) {
    return((lhs.M[0]  == rhs.M[0])  &&
           (lhs.M[1]  == rhs.M[1])  &&
           (lhs.M[2]  == rhs.M[2])  &&
           (lhs.M[3]  == rhs.M[3])  &&
           (lhs.M[4]  == rhs.M[4])  &&
           (lhs.M[5]  == rhs.M[5])  &&
           (lhs.M[6]  == rhs.M[6])  &&
           (lhs.M[7]  == rhs.M[7])  &&
           (lhs.M[8]  == rhs.M[8])  &&
           (lhs.M[9]  == rhs.M[9])  &&
           (lhs.M[10] == rhs.M[10]) &&
           (lhs.M[11] == rhs.M[11]) &&
           (lhs.M[12] == rhs.M[12]) &&
           (lhs.M[13] == rhs.M[13]) &&
           (lhs.M[14] == rhs.M[14]) &&
           (lhs.M[15] == rhs.M[15]));
    }

template<typename T>
sf_inline bool 
operator != 
(const mat4<T> &lhs, const mat4<T> &rhs) {
    return((lhs.M[0]  != rhs.M[0])  ||
           (lhs.M[1]  != rhs.M[1])  ||
           (lhs.M[2]  != rhs.M[2])  ||
           (lhs.M[3]  != rhs.M[3])  ||
           (lhs.M[4]  != rhs.M[4])  ||
           (lhs.M[5]  != rhs.M[5])  ||
           (lhs.M[6]  != rhs.M[6])  ||
           (lhs.M[7]  != rhs.M[7])  ||
           (lhs.M[8]  != rhs.M[8])  ||
           (lhs.M[9]  != rhs.M[9])  ||
           (lhs.M[10] != rhs.M[10]) ||
           (lhs.M[11] != rhs.M[11]) ||
           (lhs.M[12] != rhs.M[12]) ||
           (lhs.M[13] != rhs.M[13]) ||
           (lhs.M[14] != rhs.M[14]) ||
           (lhs.M[15] != rhs.M[15]));
}

template<typename T>
sf_inline std::ostream& 
operator << 
(std::ostream &os, const mat4<T> &rhs) {
    std::ios_base::fmtflags f = os.flags();
    os << std::fixed;
    os << std::endl;
    os << "| " << std::setprecision(5) << std::setw(10) << rhs.x0 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.x1 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.x2 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.x3 
       << " |" << std::endl;
    os << "| " << std::setprecision(5) << std::setw(10) << rhs.y0 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.y1 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.y2 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.y3 
       << " |" << std::endl;
    os << "| " << std::setprecision(5) << std::setw(10) << rhs.z0 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.z1 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.z2 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.z3 
       << " |" << std::endl;
    os << "| " << std::setprecision(5) << std::setw(10) << rhs.w0 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.w1 
       << " "  << std::setprecision(5) << std::setw(10) << rhs.w2
       << " "  << std::setprecision(5) << std::setw(10) << rhs.w3 
       << " |" << std::endl;
    os.flags(f);
    return os;
}

} // namespace sf
