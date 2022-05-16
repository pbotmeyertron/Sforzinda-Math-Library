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
#ifndef EPSILON
#define EPSILON         FLT_EPSILON  
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
// Generic Mathematical Utilities                                                 
//==============================================================================

/* Performs equality check using machine-epsilon. */
template<typename T>
sf_inline T
sf_math_utils_equals(T a, T b) {
    return std::abs((a - b) < EPSILON);
}

/* Performs non-equality check using machine-epsilon. */
template<typename T>
sf_inline T
sf_math_utils_not_equals(T a, T b) {
    return std::abs((a - b) >= EPSILON);
}

/* Mutliplies a value by itself. */
template<typename T>
sf_inline T
sf_math_utils_square(T a) {
    return a * a;
}

/* Mutliplies a value by itself thrice. */
template<typename T>
sf_inline T
sf_math_utils_cube(T a) {
    return a * a * a;
}

/* Calculates the size of an array in bytes. */
#define sf_math_utils_array_size(x) (sizeof(x) / sizeof((x)[0]))

/* Calculates the size of a structure member */
#define sf_math_utils_field_sizeof(t, f) (sizeof(((t*)0)->f))

/*---------------------------------*/
/* Type Reinterpretation Functions */
/*---------------------------------*/

/* Reinterprets a 32-bit f32 as a 32-bit unsigned integer. Avoids the
 * potential undefined behavior of reinterpret_cast<>. */
sf_inline u32 
sf_math_utils_reinterpret_f32_as_u32(f32 f) {
    u32 ret;
    std::memcpy(&ret, &f, sizeof(f));
    return ret;
}

/* Reinterprets a 32-bit unsigned integer as a 32-bit f32. Avoids the
 * potential undefined behavior of reinterpret_cast<>. */
sf_inline f32 
sf_math_utils_reinterpret_u32_as_f32(u32 u) {
    f32 ret;
    std::memcpy(&ret, &u, sizeof(u));
    return ret;
}

/* Reinterprets a 64-bit f32 as a 64-bit unsigned integer. Avoids the
 * potential undefined behavior of reinterpret_cast<>. */
sf_inline u64 
sf_math_utils_reinterpret_f64_as_u64(f64 d) {
    u64 ret;
    std::memcpy(&ret, &d, sizeof(d));
    return ret;
}

/* Reinterprets a 64-bit unsigned integer as a 64-bit f32. Avoids the
 * potential undefined behavior of reinterpret_cast<>. */
sf_inline f64 
sf_math_utils_reinterpret_u64_as_f64(u64 u) {
    f64 ret;
    std::memcpy(&ret, &u, sizeof(u));
    return ret;
}

/*---------------------*/
/* Type Sign Functions */
/*---------------------*/

/* Returns the sign of a 32-bit integer as +1, -1, or 0. */
sf_inline i32  
sf_math_utils_sign(i32 val) {
    return ((val >> 31) - (-val >> 31));
}

/* Returns the sign of a 64-bit integer as +1, -1, or 0. */
sf_inline i64  
sf_math_utils_sign(i64 val) {
    return ((val >> 63) - (-val >> 63));
}

/* Returns the sign of a 32-bit float as +1, -1, or 0. */
sf_inline 
f32 sf_math_utils_sign(f32 val) {
    return (f32)((val > 0.0f) - (val < 0.0f));
}

/* Returns the sign of a 64-bit float as +1, -1, or 0. */
sf_inline f64 
sf_math_utils_sign(f64 val) {
    return (f64)((val > 0.0f) - (val < 0.0f));
}

/*--------------------*/
/* Graphics Utilities */
/*--------------------*/

/* Converts degrees to radians. */
template<typename T>
sf_inline T 
sf_math_utils_degrees_to_radians(T deg) {
    return deg * PI / 180.0f;
}

/* Converts radians to degrees. */
template<typename T>
sf_inline T 
sf_math_utils_radians_to_degrees(T rad) {
    return rad * 180.0f / PI;
}

/* Clamp a number between min and max. */
template<typename T>
sf_inline T 
sf_math_utils_clamp(T val, T min, T max) {
    return std::min(std::max(val, min), max);
}

/* Clamp a number between zero and one. */
template<typename T>
sf_inline T 
sf_math_utils_clamp_zero_to_one(T val) {
    return sf_math_utils_clamp(val, 0.0f, 1.0f);
}

/* Linear interpolation between two numbers. */
template<typename T>
sf_inline T 
sf_math_utils_lerp(T from, T to, T t) {
    return from + t * (to - from);
}

/* Clamped linear interpolation. */
template<typename T>
sf_inline T 
sf_math_utils_clamped_lerp(T from, T to, T t) {
    return sf_math_utils_lerp(from, to, sf_math_utils_clamp_zero_to_one(t));
}

/* Step function. Returns 0.0 if x < edge, else 1.0. */
template<typename T>
sf_inline T 
sf_math_utils_step(T edge, T x) {
    return (x < edge) ? 0.0f : 1.0f;
}

/* Hermite interpolation. */
template<typename T>
sf_inline T 
sf_math_utils_hermite_interpolation(T t) {
    return t * t * (3.0f - 2.0f * t);
}

/* Threshold function with smooth transition. */
template<typename T>
sf_inline T 
sf_math_utils_smoothstep(T edge0, T edge1, T x) {
    T t;
    t = sf_math_utils_clamp_zero_to_one((x - edge0) / (edge1 - edge0));
    return sf_math_utils_hermite_interpolation(t);
}

/* Smoothstep function with Hermite interpolation. */
template<typename T>
sf_inline T 
sf_math_utils_smooth_hermite(T from, T to, T t) {
    return from + sf_math_utils_hermite_interpolation(t) * (to - from);
}

/* Clamped smoothstep with Hermite interpolation. */
template<typename T>
sf_inline T 
sf_math_utils_smooth_hermite_clamped(T from, T to, T t) {
    return sf_math_utils_smooth_hermite(from, to, sf_math_utils_clamp_zero_to_one(t));
}

/* Percentage of current value between start and end value. */
template<typename T>
sf_inline T 
sf_math_utils_percent(T from, T to, T current) {
    T t;
    if ((t = to - from) == 0.0f)
        return 1.0f;
    return (current - from) / t;
}

/* Clamped percentage of current value between start and end value. */
template<typename T>
sf_inline T 
sf_math_utils_percent_clamped(T from, T to, T current) {
    return sf_math_utils_clamp_zero_to_one(sf_math_utils_percent(from, to, current));
}

//==============================================================================
// Mathematical Types                                                  
//==============================================================================

/*
 * The following types are defined as template structs with overloads and
 * common functions used in graphics routines:
 *
 * vector2 - 2D Vector
 * vector3 - 3D Vector
 * vector4 - 4D Vector
 * matrix2 - 2x2 Matrix
 * matrix3 - 3x3 Matrix
 * matrix4 - 4x4 Matrix
 * quat - Quaternion
 */

/*-----------*/
/* 2D Vector */
/*-----------*/

template<typename T>
struct vector2 {
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

    vector2<T>() { 
        x = 0;
        y = 0; 
    } 

    vector2<T>(T cx, T cy) { 
        x = cx; 
        y = cy; 
    }

    vector2<T>(T cx) { 
        x = cx;
        y = cx; 
    }

    vector2<T>(const vector2<T>& v) { 
        x = v.x; 
        y = v.y; 
    }

    vector2<T>(T v[2]) { 
        x = v[0]; 
        y = v[1]; 
    }

    /* Index or subscript operand. */
    sf_inline T 
    operator [] 
    (u32 i) const {
        return v[i];
    }

    /* Index or subscript operand. */
    sf_inline T& 
    operator[] 
    (u32 i) {
        return v[i];
    }
}; // vector2

/*---------------------*/
/* 2D Vector Overloads */
/*---------------------*/

/* Add two vector2s. */
template<typename T>
sf_inline vector2<T> 
operator + 
(const vector2<T>& lhs, const vector2<T>& rhs) {
    vector2<T> c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    return c;
}

/* Add vector2 and scalar. */
template<typename T, typename U>
sf_inline vector2<T> 
operator + 
(const vector2<T>& lhs, const U& rhs) {
    vector2<T> c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    return c;
}

/* Add scalar and vector2. */
template<typename T, typename U>
sf_inline vector2<T> 
operator + 
(const U& lhs, const vector2<T>& rhs) {
    vector2<T> c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    return c;
}

/* Plus-equals operand with two vector2s. */
template<typename T>
sf_inline vector2<T>& 
operator += 
(vector2<T>& lhs, const vector2<T>& rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y;
    return lhs;
}

/* Plus-equals operand with a vector2 and scalar. */
template <typename T, typename U>
sf_inline vector2<T>& 
operator += 
(vector2<T>& lhs, const U& rhs) {
    lhs.x += rhs; 
    lhs.y += rhs;
    return lhs;
}

/* Unary minus operand. Makes vector2 negative. */
template<typename T>
sf_inline vector2<T> 
operator - 
(const vector2<T>& rhs) {
    vector2<T> c; 
    c.x =- rhs.x; 
    c.y =- rhs.y; 
    return c;
}

/* Subtracts a vector2 from a vector2. */
template <typename T>
sf_inline vector2<T> 
operator - 
(const vector2<T>& lhs, const vector2<T>& rhs) {
    vector2<T> c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    return c;
}

/* Subtracts a scalar from a vector2. */
template<typename T, typename U>
sf_inline vector2<T> 
operator - 
(const vector2<T>& lhs, const U& rhs) {
    vector2<T> c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    return c;
}

/* Subtracts a vector2 from a scalar. */
template<typename T, typename U>
sf_inline vector2<T> 
operator - 
(const U& lhs, const vector2<T>& rhs) {
    vector2<T> c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    return c;
}

/* Minus-equals operand for two vector2s. */
template<typename T>
sf_inline vector2<T>& 
operator -= 
(vector2<T>& lhs, const vector2<T>& rhs) {
    lhs.x-=rhs.x; lhs.y-=rhs.y;
    return lhs;
}

/* Minus-equals operand for vector2 and scalar. */
template <typename T, typename U>
sf_inline vector2<T>& 
operator -= 
(vector2<T>& lhs, const U& rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs;
    return lhs;
}

/* Multiplies two vector2s. */
template<typename T>
sf_inline vector2<T> 
operator * 
(const vector2<T>& lhs, const vector2<T>& rhs) {
    vector2<T> c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y;
    return c;
}

/* Multiplies a vector2 and scalar. */
template<typename T, typename U>
sf_inline vector2<T> 
operator * 
(const U& lhs, const vector2<T>& rhs) {
    vector2<T> c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs;
    return c;
}

/* Multiplies a scalar and vector2. */
template <typename T, typename U>
sf_inline vector2<T> 
operator * 
(const vector2<T>& lhs, const U& rhs) {
    vector2<T> c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    return c;
}

/* Multiply-equals operand for vector2. */
template <typename T>
sf_inline vector2<T>& 
operator *= 
(vector2<T>& lhs, const vector2<T>& rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y;
    return lhs;
}

/* Multiply-equals operand for vector2 and scalar. */
template<typename T, typename U>
sf_inline vector2<T>& 
operator *= 
(vector2<T>& lhs, const U& rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs;
    return lhs;
}

/* Divides two vector2. */
template<typename T>
sf_inline vector2<T> 
operator / 
(const vector2<T>& lhs, const vector2<T>& rhs) {
    vector2<T> c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y;
    return c;
}

/* Divides a vector2 by a scalar. */
template <typename T, typename U>
sf_inline vector2<T> 
operator / 
(const vector2<T>& lhs, const U& rhs) {
    vector2<T> c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs;
    return c;
}

/* Divide-equals operand for two vector2s. */
template <typename T>
sf_inline vector2<T>& 
operator /= 
(vector2<T>& lhs, const vector2<T>& rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y;
    return lhs;
}

/* Divide-equals operand for vector2 and scalar. */
template<typename T, typename U>
sf_inline vector2<T>& 
operator /= 
(vector2<T>& lhs, const U& rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs;
    return lhs;
}

/* Add one to each element in vector2. */
template<typename T>
sf_inline vector2<T>& 
operator ++ 
(vector2<T>& rhs) {
    ++rhs.x; 
    ++rhs.y;
    return rhs;
}

/* Add one to each element in vector2. */
template<typename T>
sf_inline vector2<T> 
operator ++ 
(vector2<T>& lhs, i32) {
    vector2<T> c = lhs;
    lhs.x++; 
    lhs.y++;
    return(c);
}

/* Subtract one from each element in vector2. */
template<typename T>
sf_inline vector2<T>& 
operator -- 
(vector2<T>& rhs) {
    --rhs.x; 
    --rhs.y;
    return rhs;
}

/* Subtract one from each element in vector2. */
template<typename T>
sf_inline vector2<T> 
operator -- 
(vector2<T>& lhs, i32) {
    vector2<T> c = lhs;
    lhs.x--; 
    lhs.y--;
    return c;
}

/* Tests two vector2s for equality. */
template<typename T>
sf_inline bool 
operator == 
(const vector2<T>& lhs, const vector2<T>& rhs) {
    return((lhs.x == rhs.x) && (lhs.y == rhs.y));
}

/* Tests two vector2s for non-equality. */
template<typename T>
sf_inline bool 
operator != 
(const vector2<T>& lhs, const vector2<T>& rhs) {
    return((lhs.x != rhs.x) || (lhs.y != rhs.y));
}

/* Allows for printing elements of vector2 to stdout. Thanks to Ray Tracing in One
 * Weekend for this. :) */
template<typename T>
sf_inline std::ostream& 
operator << 
(std::ostream& os, const vector2<T>& rhs) {
    os << "(" << rhs.x << "," << rhs.y << ")";
    return os;
}

/*---------------------*/
/* 2D Vector Functions */
/*---------------------*/

/* Returns the length (magnitude) of a 2D vector. */
template<typename T>
sf_inline T 
length(const vector2<T>& a) {
    return std::sqrt(sf_math_utils_square(a.x) + 
                     sf_math_utils_square(a.y));
}

/* Normalizes a 2D vector. */
template<typename T>
sf_inline vector2<T> 
normalize(const vector2<T>& a) {
    T mag = length(a);
    if (mag != 0.0f) {
        return(a /= mag);
    }
    return(vector2<T>(0.0f, 0.0f));
}

/* Returns the dot product of a 2D vector. */
template<typename T>
sf_inline T 
dot_product(const vector2<T>& a, const vector2<T>& b) {
    return(a.x * b.x + 
           a.y * b.y);
}

/* Returns the cross product of a 2D vector. */
template<typename T>
sf_inline vector2<T> 
cross_product(const vector2<T>& a, const vector2<T> b) {
    vector2<T> c;
    c.x = (a.x * b.y) - (a.y * b.x);
    c.y = (a.y * b.x) - (a.x * b.y);
    return c;
}

/* Rotate vector2 around origin counter-clockwise. */
template<typename T>
sf_inline vector2<T> 
rotate(const vector2<T>& a, T angle) {
    vector2<T> dest;
    T cos_angle, sin_angle, x1, y1;
    cos_angle = std::cos(angle);
    sin_angle = std::sin(angle);
    x1 = a.x;
    y1 = a.y;
    dest.x = (cos_angle * x1) - (sin_angle * y1);
    dest.y = (sin_angle * x1) + (cos_angle * y1);
    return dest;
}

/* Clamp a vector2 between min and max. */
template<typename T>
sf_inline vector2<T> 
clamp(const vector2<T>& a, T min, T max) {
    a.x = sf_math_utils_clamp(a.x, min, max);
    a.y = sf_math_utils_clamp(a.y, min, max);
    return a;
}

/* Returns the angle between two 2D vectors. */
template<typename T>
sf_inline T 
angle_between(const vector2<T>& a, const vector2<T>& b) {
    return dot_product(a, b) / (length(a) * length(b));
}

/* Returns the normal axis between two 2D vectors. */
template<typename T>
sf_inline vector2<T> 
normal_axis_between(const vector2<T>& a, const vector2<T>& b) {
    return normalize(cross_product(a, b));
}

/* Returns the distance between two 2D vectors. */
template<typename T>
sf_inline T 
distance(const vector2<T>& a, const vector2<T>& b) {
    return std::sqrt(sf_math_utils_square(b.x - a.x) + 
                     sf_math_utils_square(b.y - a.y));
}

/* Prints out the coordinates of a vector2. */
template<typename T>
sf_inline void 
print_vec(const vector2<T>& a) {
    printf("x:12f, y:12f\r\n", a.x, a.y);
}

/*-----------*/
/* 3D Vector */
/*-----------*/

template<typename T>
struct vector3 {
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

    vector3<T>() { 
        x = 0;
        y = 0;
        z = 0; 
    }

    vector3<T>(T cx, T cy, T cz) { 
        x = cx; 
        y = cy; 
        z = cz; 
    }

    vector3<T>(T cx) { 
        x = cx;
        y = cx;
        z = cx; 
    }

    /* Initialize a vector3 with a vector2 and a scalar. */
    vector3<T>(vector2<T> v, T cz) { 
        x = v.x; 
        y = v.y; 
        z = cz; 
    }

    vector3<T>(const vector3<T> &v) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
    }

    vector3<T>(T v[3]) { 
        x = v[0]; 
        y = v[1]; 
        z = v[2]; 
    }

    /* Index or subscript operand. */
    sf_inline T 
    operator [] 
    (u32 i) const {
        return v[i];
    }

    /* Index or subscript operand. */
    sf_inline T& 
    operator [] 
    (u32 i) {
        return v[i];
    }
}; // vector3

/*---------------------*/
/* 3D Vector Overloads */
/*---------------------*/

/* Add two vector3s. */
template<typename T>
sf_inline vector3<T> 
operator + 
(const vector3<T>& lhs, const vector3<T>& rhs) {
    vector3<T> c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    c.z = lhs.z + rhs.z; 
    return c;
}

/* Add vector3 and scalar. */
template<typename T, typename U>
sf_inline vector3<T> 
operator + 
(const vector3<T>& lhs, const U& rhs) {
    vector3<T> c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    c.z = lhs.z + rhs; 
    return c;
}

/* Add scalar and vector3. */
template<typename T, typename U>
sf_inline vector3<T> 
operator + 
(const U& lhs, const vector3<T>& rhs) {
    vector3<T> c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    c.z = lhs + rhs.z; 
    return c;
}

/* Plus-equals operand with two vector3s. */
template<typename T>
sf_inline vector3<T>& 
operator += 
(vector3<T>& lhs, const vector3<T>& rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y; 
    lhs.z += rhs.z;
    return lhs;
}

/* Plus-equals operand with a vector3 and scalar. */
template<typename T, typename U>
sf_inline vector3<T>& 
operator += 
(vector3<T>& lhs, const U& rhs) {
    lhs.x += rhs; 
    lhs.y += rhs; 
    lhs.z += rhs;
    return lhs;
}

/* Unary minus operand. Makes vector3 negative. */
template<typename T>
sf_inline vector3<T> 
operator - 
(const vector3<T>& rhs) {
    vector3<T> c; 
    c.x =- rhs.x; 
    c.y =- rhs.y; 
    c.z =- rhs.z; 
    return c;
}

/* Subtracts a vector3 from a vector3. */
template<typename T>
sf_inline vector3<T> 
operator - 
(const vector3<T>& lhs, const vector3<T>& rhs) {
    vector3<T> c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    c.z = lhs.z - rhs.z; 
    return c;
}

/* Subtracts a scalar from a vector3. */
template<typename T, typename U>
sf_inline vector3<T> 
operator - 
(const vector3<T>& lhs, const U& rhs) {
    vector3<T> c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    c.z = lhs.z - rhs; 
    return c;
}

/* Subtracts a vector3 from a scalar. */
template<typename T, typename U>
sf_inline vector3<T> 
operator - 
(const U& lhs, const vector3<T>& rhs) {
    vector3<T> c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    c.z = lhs - rhs.z; 
    return c;
}

/* Minus-equals operand for two vector3s. */
template<typename T>
sf_inline vector3<T>& 
operator -= 
(vector3<T>& lhs, const vector3<T>& rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y; 
    lhs.z -= rhs.z;
    return lhs;
}

/* Minus-equals operand for vector3 and scalar. */
template<typename T, typename U>
sf_inline vector3<T>& 
operator -= 
(vector3<T>& lhs, const U& rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs; 
    lhs.z -= rhs;
    return lhs;
}

/* Multiplies two vector3s. */
template<typename T>
sf_inline vector3<T> 
operator * 
(const vector3<T>& lhs, const vector3<T>& rhs) {
    vector3<T> c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y; 
    c.z = rhs.z * lhs.z;
    return c;
}

/* Multiplies a vector3 and scalar. */
template<typename T, typename U>
sf_inline vector3<T> 
operator * 
(const U &lhs, const vector3<T> &rhs) {
    vector3<T> c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs; 
    c.z = rhs.z * lhs;
    return(c);
}

/* Multiplies a scalar and vector3. */
template<typename T, typename U>
sf_inline vector3<T> 
operator * 
(const vector3<T>& lhs, const U& rhs) {
    vector3<T> c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    c.z = rhs * lhs.z;
    return c;
}

/* Multiply-equals operand for vector3. */
template<typename T>
sf_inline vector3<T>& 
operator *= 
(vector3<T>& lhs, const vector3<T>& rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y; 
    lhs.z *= rhs.z;
    return lhs;
}

/* Multiply-equals operand for vector3 and scalar. */
template<typename T, typename U>
sf_inline vector3<T>& 
operator *= 
(vector3<T>& lhs, const U& rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs; 
    lhs.z *= rhs;
    return lhs;
}

/* Divides two vector3s. */
template<typename T>
sf_inline vector3<T> 
operator / 
(const vector3<T>& lhs, const vector3<T>& rhs) {
    vector3<T> c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y; 
    c.z = lhs.z / rhs.z;
    return c;
}

/* Divides a vector3 by a scalar. */
template<typename T, typename U>
sf_inline vector3<T> 
operator / 
(const vector3<T>& lhs, const U& rhs) {
    vector3<T> c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs; 
    c.z = lhs.z / rhs;
    return c;
}

/* Divide-equals operand for two vector3s. */
template<typename T>
sf_inline vector3<T>& 
operator /= 
(vector3<T>& lhs, const vector3<T>& rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y; 
    lhs.z /= rhs.z;
    return(lhs);
}

/* Divide-equals operand for vector3 and scalar. */
template<typename T, typename U>
sf_inline vector3<T>& 
operator /= 
(vector3<T>& lhs, const U& rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs; 
    lhs.z /= rhs;
    return lhs;
}

/* Add one to each element in vector3. */
template<typename T>
sf_inline vector3<T>& 
operator ++ 
(vector3<T>& rhs) {
    ++rhs.x; 
    ++rhs.y; 
    ++rhs.z;
    return rhs;
}

/* Add one to each element in vector3. */
template<typename T>
sf_inline vector3<T> 
operator ++ 
(vector3<T>& lhs, i32) {
    vector3<T> c = lhs;
    lhs.x++; 
    lhs.y++; 
    lhs.z++;
    return c;
}

/* Subtract one from each element in vector3. */
template<typename T>
sf_inline vector3<T>& 
operator -- 
(vector3<T>& rhs) {
    --rhs.x; 
    --rhs.y; 
    --rhs.z;
    return rhs;
}

/* Subtract one from each element in vector3. */
template<typename T>
sf_inline vector3<T> 
operator -- 
(vector3<T>& lhs, i32) {
    vector3<T> c = lhs;
    lhs.x--; 
    lhs.y--; 
    lhs.z--;
    return c;
}

/* Tests two vector3s for equality. */
template<typename T>
sf_inline bool 
operator == 
(const vector3<T>& lhs, const vector3<T>& rhs) {
    return((lhs.x == rhs.x) && 
           (lhs.y == rhs.y) && 
           (lhs.z == rhs.z));
}

/* Tests two vector3s for non-equality. */
template<typename T>
sf_inline bool 
operator != 
(const vector3<T>& lhs, const vector3<T>& rhs) {
    return((lhs.x != rhs.x) || 
           (lhs.y != rhs.y) || 
           (lhs.z != rhs.z));
}

/* Allows for printing elements of vector3 to stdout. Thanks to Ray Tracing in One
 * Weekend for this. :) */
template<typename T>
std::ostream& 
operator << 
(std::ostream& os, const vector3<T>& rhs) {
    os << "(" << rhs.x << "," << rhs.y << "," << rhs.z << ")";
    return os;
}

/*---------------------*/
/* 3D Vector Functions */
/*---------------------*/

/* Returns the length (magnitude) of a 3D vector. */
template<typename T>
sf_inline T 
length(const vector3<T>& a) {
    return std::sqrt(sf_math_utils_square(a.x) + 
                     sf_math_utils_square(a.y) + 
                     sf_math_utils_square(a.z));
}

/* Normalizes a 3D vector. */
template<typename T>
sf_inline vector3<T> 
normalize(const vector3<T>& a) {
    T mag = length(a);
    if (mag != 0.0f) {
        return a /= mag;
    }
    return(vector3<T>(0.0f, 0.0f, 0.0f));
}

/* Returns the dot product of a 3D vector. */
template<typename T>
sf_inline T 
dot_product(const vector3<T>& a, const vector3<T>& b) {
    return(a.x * b.x + 
           a.y * b.y + 
           a.z * b.z);
}

/* Returns the cross product of a 3D vector. */
template<typename T>
sf_inline vector3<T> 
cross_product(const vector3<T>& a, const vector3<T>& b) {
    vector3<T> c;
    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);
    return c;
}

/* Returns the angle between two 3D vectors. */
template<typename T>
sf_inline T 
angle_between(const vector3<T>& a, const vector3<T>& b) {
    T c;
    c = dot_product(a, b) / (length(a) * length(b));
    return 2.0f * std::acos(c);
}

/* Returns the normal axis between two 3D vectors. */
template<typename T>
sf_inline vector3<T> 
normal_axis_between(const vector3<T>& a, const vector3<T>& b) {
    return normalize(cross_product(a, b));
}

/* Returns the distance between two 3D vectors. */
template<typename T>
sf_inline T 
distance(const vector3<T>& a, const vector3<T>& b) {
    return std::sqrt(sf_math_utils_square(b.x - a.x) + 
                     sf_math_utils_square(b.y - a.y) +
                     sf_math_utils_square(b.z - a.z));
}

/* Prints out the coordinates of a vector3. */
template<typename T>
sf_inline void 
print_vec(const vector3<T>& a) {
    printf("x:12f, y:12f, z:12f\r\n", a.x, a.y, a.z);
}

/*-----------*/
/* 4D Vector */
/*-----------*/

template<typename T>
struct vector4 {
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

    vector4<T>() { 
        x = 0;
        y = 0;
        z = 0;
        w = 0; 
    }

    vector4<T>(T cx, T cy, T cz, T cw) { 
        x = cx; 
        y = cy; 
        z = cz; 
        w = cw; 
    }

    vector4<T>(T cx) { 
        x = cx;
        y = cx;
        z = cx;
        w = cx; 
    }

    /* Initialize a vector4 with a vector3 and a scalar. */
    vector4<T>(vector3<T> v, T cw) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
        w = cw; 
    }

    /* Initialize a vector4 with two vector2s. */
    vector4<T>(vector2<T> v, vector2<T> u) { 
        x = v.x; 
        y = v.y; 
        z = u.x; 
        w = u.y; 
    }   

    vector4<T>(const vector4<T> &v) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
        w = v.w; 
    }

    vector4<T>(T v[4]) { 
        x = v[0]; 
        y = v[1]; 
        z = v[2]; 
        w = v[3]; 
    }

    /* Index or subscript operand. */
    sf_inline T 
    operator [] 
    (u32 i) const {
        return v[i];
    }

    /* Index or subscript operand. */
    sf_inline T& 
    operator [] 
    (u32 i) {
        return v[i];
    }
};

/*---------------------*/
/* 4D Vector Overloads */
/*---------------------*/

/* Add two vector4s. */
template<typename T>
sf_inline vector4<T> 
operator + 
(const vector4<T>& lhs, const vector4<T>& rhs) {
    vector4<T> c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    c.z = lhs.z + rhs.z; 
    c.w = lhs.w + rhs.w; 
    return c;
}

/* Add vector4 and scalar. */
template<typename T, typename U>
sf_inline vector4<T> 
operator + 
(const vector4<T>& lhs, const U& rhs) {
    vector4<T> c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    c.z = lhs.z + rhs; 
    c.w = lhs.w + rhs; 
    return c;
}

/* Add scalar and vector4. */
template<typename T, typename U>
sf_inline vector4<T> 
operator + 
(const U& lhs, const vector4<T>& rhs) {
    vector4<T> c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    c.z = lhs + rhs.z; 
    c.w = lhs + rhs.w; 
    return c;
}

/* Plus-equals operand with two vector4s. */
template<typename T>
sf_inline vector4<T>& 
operator += 
(vector4<T>& lhs, const vector4<T>& rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y; 
    lhs.z += rhs.z; 
    lhs.w += rhs.w;
    return lhs;
}

/* Plus-equals operand with a vector4 and scalar. */
template<typename T, typename U>
sf_inline vector4<T>& 
operator += 
(vector4<T>& lhs, const U& rhs) {
    lhs.x += rhs; 
    lhs.y += rhs; 
    lhs.z += rhs; 
    lhs.w += rhs;
    return lhs;
}

/* Unary minus operand. Makes vector4 negative. */
template<typename T>
sf_inline vector4<T> 
operator - 
(const vector4<T>& rhs) {
    vector4<T> c; 
    c.x =- rhs.x; 
    c.y =- rhs.y; 
    c.z =- rhs.z; 
    c.w =- rhs.w; 
    return c;
}

/* Subtracts a vector4 from a vector4. */
template<typename T>
sf_inline vector4<T> 
operator - 
(const vector4<T>& lhs, const vector4<T>& rhs) {
    vector4<T> c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    c.z = lhs.z - rhs.z; 
    c.w = lhs.w - rhs.w; 
    return c;
}

/* Subtracts a scalar from a vector4. */
template<typename T, typename U>
sf_inline vector4<T> 
operator - 
(const vector4<T>& lhs, const U& rhs) {
    vector4<T> c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    c.z = lhs.z - rhs; 
    c.w = lhs.w - rhs; 
    return c;
}

/* Subtracts a vector4 from a scalar. */
template<typename T, typename U>
sf_inline vector4<T> 
operator - 
(const U& lhs, const vector4<T>& rhs) {
    vector4<T> c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    c.z = lhs - rhs.z; 
    c.w = lhs - rhs.w; 
    return c;
}

/* Minus-equals operand for two vector4s. */
template<typename T>
sf_inline vector4<T>& 
operator -= 
(vector4<T>& lhs, const vector4<T>& rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y; 
    lhs.z -= rhs.z; 
    lhs.w -= rhs.w;
    return lhs;
}

/* Minus-equals operand for vector4 and scalar. */
template<typename T, typename U>
sf_inline vector4<T>& 
operator -= 
(vector4<T>& lhs, const U& rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs; 
    lhs.z -= rhs; 
    lhs.w -= rhs;
    return lhs;
}

/* Multiplies two vector4s. */
template<typename T>
sf_inline vector4<T> 
operator * 
(const vector4<T>& lhs, const vector4<T>& rhs) {
    vector4<T> c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y; 
    c.z = rhs.z * lhs.z; 
    c.w = rhs.w * lhs.w;
    return c;
}

/* Multiplies a vector4 and scalar. */
template<typename T, typename U>
sf_inline vector4<T> 
operator * 
(const U& lhs, const vector4<T>& rhs) {
    vector4<T> c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs; 
    c.z = rhs.z * lhs; 
    c.w = rhs.w * lhs;
    return c;
}

/* Multiplies a scalar and vector4. */
template<typename T, typename U>
sf_inline vector4<T> 
operator * 
(const vector4<T>& lhs, const U& rhs) {
    vector4<T> c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    c.z = rhs * lhs.z;
    c.w = rhs * lhs.w;
    return c;
}

/* Multiply-equals operand for vector4. */
template<typename T>
sf_inline vector4<T>& 
operator *= 
(vector4<T>& lhs, const vector4<T>& rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y; 
    lhs.z *= rhs.z; 
    lhs.w *= rhs.w;
    return lhs;
}

/* Multiply-equals operand for vector4 and scalar. */
template<typename T, typename U>
sf_inline vector4<T>& 
operator *= 
(vector4<T>& lhs, const U& rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs; 
    lhs.z *= rhs; 
    lhs.w *= rhs;
    return lhs;
}

/* Divides two vector4s. */
template<typename T>
sf_inline vector4<T> 
operator / 
(const vector4<T>& lhs, const vector4<T>& rhs) {
    vector4<T> c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y; 
    c.z = lhs.z / rhs.z; 
    c.w = lhs.w / rhs.w;
    return c;
}

/* Divides a vector4 by a scalar. */
template<typename T, typename U>
sf_inline vector4<T> 
operator / 
(const vector4<T>& lhs, const U& rhs) {
    vector4<T> c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs; 
    c.z = lhs.z / rhs; 
    c.w = lhs.w / rhs;
    return c;
}

/* Divide-equals operand for two vector4s. */
template<typename T>
sf_inline vector4<T>& 
operator /= 
(vector4<T>& lhs, const vector4<T>& rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y; 
    lhs.z /= rhs.z; 
    lhs.w /= rhs.w;
    return lhs;
}

/* Divide-equals operand for vector4 and scalar. */
template<typename T, typename U>
sf_inline vector4<T>& 
operator /= 
(vector4<T>& lhs, const U& rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs; 
    lhs.z /= rhs; 
    lhs.w /= rhs;
    return lhs;
}

/* Add one to each element in vector4. */
template<typename T>
sf_inline vector4<T>& 
operator ++ 
(vector4<T>& rhs) {
    ++rhs.x; 
    ++rhs.y; 
    ++rhs.z; 
    ++rhs.w;
    return rhs;
}

/* Add one to each element in vector4. */
template<typename T>
sf_inline vector4<T> 
operator ++ 
(vector4<T>& lhs, i32) {
    vector4<T> c = lhs;
    lhs.x++; 
    lhs.y++; 
    lhs.z++; 
    lhs.w++;
    return c;
}

/* Subtract one from each element in vector4. */
template<typename T>
sf_inline vector4<T>& 
operator -- 
(vector4<T>& rhs) {
    --rhs.x; 
    --rhs.y; 
    --rhs.z; 
    --rhs.w;
    return rhs;
}

/* Subtract one from each element in vector4. */
template<typename T>
sf_inline vector4<T> 
operator -- 
(vector4<T>& lhs, i32) {
    vector4<T> c = lhs;
    lhs.x--; 
    lhs.y--; 
    lhs.z--; 
    lhs.w--;
    return c;
}

/* Tests two vector4s for equality. */
template<typename T>
sf_inline bool 
operator == 
(const vector4<T>& lhs, const vector4<T>& rhs) {
    return((lhs.x == rhs.x) && 
           (lhs.y == rhs.y) && 
           (lhs.z == rhs.z) && 
           (lhs.w == rhs.w));
}

/* Tests two vector4s for non-equality. */
template<typename T>
sf_inline bool 
operator != 
(const vector4<T>& lhs, const vector4<T>& rhs) {
    return((lhs.x != rhs.x) || 
           (lhs.y != rhs.y) || 
           (lhs.z != rhs.z) || 
           (lhs.w != rhs.w));
}

/* Allows for printing elements of vector4 to stdout. Thanks to Ray Tracing in One
 * Weekend for this. :) */
template<typename T>
sf_inline std::ostream& 
operator << 
(std::ostream& os, const vector4<T>& rhs) {
    os    << "(" << 
    rhs.x << "," << 
    rhs.y << "," << 
    rhs.z << "," << 
    rhs.w << ")";
    return os;
}

/*---------------------*/
/* 4D Vector Functions */
/*---------------------*/

/* Returns the length (magnitude) of a 4D vector. */
template<typename T>
sf_inline T 
length(const vector4<T>& a) {
    return std::sqrt(sf_math_utils_square(a.x) + 
                     sf_math_utils_square(a.y) + 
                     sf_math_utils_square(a.z) +
                     sf_math_utils_square(a.w));
}

/* Normalizes a 4D vector. */
template<typename T>
sf_inline vector4<T> 
normalize(const vector4<T>& a) {
    T mag = length(a);
    if (mag != 0.0f) {
        return(a /= mag);
    }
    return(vector4<T>(0.0f, 0.0f, 0.0f, 0.0f));
}

/* Returns the dot product of a 4D vector. */
template<typename T>
sf_inline T 
dot_product(const vector4<T>& a, const vector4<T>& b) {
    return(a.x * b.x + 
           a.y * b.y + 
           a.z * b.z +
           a.w * b.w);
}

/* Returns the cross product of a 4D vector. */
template<typename T>
sf_inline vector4<T> 
cross_product(const vector4<T>& a, const vector4<T>& b) {
    vector4<T> c;
    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);
    c.w = (a.w * b.w) - (a.w * b.w); // evaluates to zero
    return c;
}

/* Returns the distance between two 4D vectors. */
template<typename T>
sf_inline T 
distance(const vector4<T>& a, const vector4<T>& b) {
    return std::sqrt(sf_math_utils_square(b.x - a.x) + 
                     sf_math_utils_square(b.y - a.y) + 
                     sf_math_utils_square(b.z - a.z) + 
                     sf_math_utils_square(b.w - a.w));
}

/* Prints out the coordinates of a vector4. */
template<typename T>
sf_inline void 
print_vec(const vector4<T>& a) {
    printf("x:12f, y:12f, z:12f, w:12f\r\n", a.x, a.y, a.z, a.w);
}

//==============================================================================
// matrix2                                                 
//==============================================================================
// TODO: Come back and define matrix2
/* 2x2 Matrix */

/*-----------*/
/* 3D Matrix */
/*-----------*/

template<typename T>
struct matrix3 {
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

    matrix3<T>() { 
        x0 = 0; y0 = 0; z0 = 0;
        x1 = 0; y1 = 0; z1 = 0;
        x2 = 0; y2 = 0; z2 = 0;
    }

    matrix3<T>(vector3<T> v1, vector3<T> v2, vector3<T> v3) { 
        x0 = v1.x; y0 = v1.y; z0 = v1.z; 
        x1 = v2.x; y1 = v2.y; z1 = v2.z; 
        x2 = v3.x; y2 = v3.y; z2 = v3.z; 
    }

    matrix3<T>(const matrix3<T>& v) { 
        x0 = v.x0; y0 = v.y0; z0 = v.z0; 
        x1 = v.x1; y1 = v.y1; z1 = v.z1; 
        x2 = v.x2; y2 = v.y2; z2 = v.z2; 
    }
};

/*---------------------*/
/* 3D Matrix Overloads */
/*---------------------*/

/* Add two matrix3s. */
template<typename T>
sf_inline matrix3<T> 
operator + 
(const matrix3<T>& lhs, const matrix3<T>& rhs) {
    matrix3<T> c;
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

/* matrix3 plus-equals operand. */
template<typename T>
sf_inline matrix3<T>& 
operator += 
(matrix3<T>& lhs, const matrix3<T>& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

/* Unary minus operand. Makes matrix3 negative. */
template<typename T>
sf_inline matrix3<T> 
operator - 
(const matrix3<T>& rhs) {
    matrix3<T> c;
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

/* Subtract a matrix3 from a matrix3. */
template<typename T>
sf_inline matrix3<T> 
operator - 
(const matrix3<T>& lhs, const matrix3<T>& rhs) {
    matrix3<T> c;
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

/* matrix3 minus-equals operand. */
template<typename T>
sf_inline matrix3<T>& 
operator -= 
(matrix3<T>& lhs, const matrix3<T>& rhs) {
    lhs = lhs - rhs;
    return lhs;
}

/* Multiply a matrix3 with a vector3. */
template<typename T>
sf_inline vector3<T> 
operator * 
(const matrix3<T>& lhs, const vector3<T>& rhs) {
    vector3<T> c;
    c.x = rhs.x * lhs.x0 + rhs.y * lhs.x1 + rhs.z * lhs.x2;
    c.y = rhs.x * lhs.y0 + rhs.y * lhs.y1 + rhs.z * lhs.y2;
    c.z = rhs.x * lhs.z0 + rhs.y * lhs.z1 + rhs.z * lhs.z2;
    return c;
}

/* Multiply a vector3 with a matrix3. */
template<typename T>
sf_inline vector3<T> 
operator * 
(const vector3<T>& lhs, const matrix3<T>& rhs) {
    vector3<T> c;
    c.x = lhs.x * rhs.x0 + lhs.y * rhs.y0 + lhs.z * rhs.z0;
    c.y = lhs.x * rhs.x1 + lhs.y * rhs.y1 + lhs.z * rhs.z1;
    c.z = lhs.x * rhs.x2 + lhs.y * rhs.y2 + lhs.z * rhs.z2;
    return c;
}

/* Multiply a matrix3 with a scalar. */
template<typename T, typename U>
sf_inline matrix3<T> 
operator * 
(const matrix3<T>& lhs, const U& rhs) {
    matrix3<T> c;
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

/* Multiply a scalar with a matrix3. */
template<typename T, typename U>
sf_inline matrix3<T> 
operator * 
(const U& lhs, const matrix3<T>& rhs) {
    return(rhs * lhs);
}

/* Multiply two matrix3s. */
template<typename T>
sf_inline matrix3<T> 
operator * 
(const matrix3<T>& lhs, const matrix3<T>& rhs) {
    matrix3<T> c;
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

/* Multiply-equals operand with two matrix3s. */
template<typename T>
sf_inline matrix3<T>& 
operator *= 
(matrix3<T>& lhs, const matrix3<T>& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Multiply-equals operand with matrix3 and scalar. */
template<typename T, typename U>
sf_inline matrix3<T>& 
operator *= 
(matrix3<T>& lhs, const U& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Tests for equality between two matrix3s. */
template<typename T>
sf_inline bool 
operator == 
(const matrix3<T>& lhs, const matrix3<T>& rhs) {
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

/* Tests for non-equality between two matrix3s. */
template<typename T>
sf_inline bool 
operator != 
(const matrix3<T>& lhs, const matrix3<T>& rhs) {
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

/* Allows for printing elements of matrix3 to stdout. */
template<typename T>
sf_inline std::ostream& 
operator << 
(std::ostream& os, const matrix3<T>& rhs) {
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
    return os;
}

/*---------------------*/
/* 3D Matrix Functions */
/*---------------------*/

/* Copy the values from one mat3 to another. */
template<typename T>
sf_inline matrix3<T> 
copy(const matrix3<T>& src, const matrix3<T>& dest) {
    dest.m[0][0] = src.m[0][0];
    dest.m[0][1] = src.m[0][1];
    dest.m[0][2] = src.m[0][2];
    dest.m[1][0] = src.m[1][0];
    dest.m[1][1] = src.m[1][1];
    dest.m[1][2] = src.m[1][2];
    dest.m[2][0] = src.m[2][0];
    dest.m[2][1] = src.m[2][1];
    dest.m[2][2] = src.m[2][2];
    return dest;
}

/* Matrix transposition. */
template<typename T>
sf_inline matrix3<T>  
transpose(const matrix3<T>& transpose) {
    matrix3<T> mat;
    /* row 1 */
    transpose.m[0][0] = mat.m[0][0]; 
    transpose.m[1][0] = mat.m[0][1]; 
    transpose.m[2][0] = mat.m[0][2];
    /* row 2 */
    transpose.m[0][1] = mat.m[1][0]; 
    transpose.m[1][1] = mat.m[1][1]; 
    transpose.m[2][1] = mat.m[1][2];
    /* row 3 */
    transpose.m[0][2] = mat.m[2][0]; 
    transpose.m[1][2] = mat.m[2][1]; 
    transpose.m[2][2] = mat.m[2][2];
    return transpose;
}

/* Mat3 determinant. */
template<typename T>
sf_inline T 
determinant(const matrix3<T>& det) {
    return(det.m[0][0] * (det.m[1][1] * det.m[2][2] - det.m[2][1] * det.m[1][2]) - 
           det.m[1][0] * (det.m[0][1] * det.m[2][2] - det.m[2][1] * det.m[0][2]) + 
           det.m[2][0] * (det.m[0][1] * det.m[1][2] - det.m[1][1] * det.m[0][2]));
}

/* Mat3 inverse. */
template<typename T>
sf_inline matrix3<T>
inverse(const matrix3<T>& a) {
    matrix3<T> dest;
    T determinant;
    dest.m[0][0] =   a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1];
    dest.m[0][1] = -(a.m[0][1] * a.m[2][2] - a.m[2][1] * a.m[0][2]);
    dest.m[0][2] =   a.m[0][1] * a.m[1][2] - a.m[1][1] * a.m[0][2];
    dest.m[1][0] = -(a.m[1][0] * a.m[2][2] - a.m[2][0] * a.m[1][2]);
    dest.m[1][1] =   a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0];
    dest.m[1][2] = -(a.m[0][0] * a.m[1][2] - a.m[1][0] * a.m[0][2]);
    dest.m[2][0] =   a.m[1][0] * a.m[2][1] - a.m[2][0] * a.m[1][1];
    dest.m[2][1] = -(a.m[0][0] * a.m[2][1] - a.m[2][0] * a.m[0][1]);
    dest.m[2][2] =   a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0];
    determinant = 1.0f / (a.m[0][0] * dest.m[0][0] + 
                          a.m[0][1] * dest.m[1][0] + 
                          a.m[0][2] * dest.m[2][0]);
    return dest * determinant;
}

/* Print out the formatted values for a mat3. */
template<typename T>
sf_inline void 
print_matrix(i32 new_line) {
    matrix3<T> mat;
    printf("| %10.5f %10.5f %10.5f |\n", mat.m[0][0], mat.m[0][1], mat.m[0][2]);
    printf("| %10.5f %10.5f %10.5f |\n", mat.m[1][0], mat.m[1][1], mat.m[1][2]);
    printf("| %10.5f %10.5f %10.5f |\n", mat.m[2][0], mat.m[2][1], mat.m[2][2]);
    if (new_line) { printf("\n"); }
}

/*-----------*/
/* 4D Matrix */
/*-----------*/

template<typename T>
struct matrix4 {
    union {
        struct alignas(8 * sizeof(T)) { 
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

    matrix4<T>() { 
        x0 = 0; y0 = 0; z0 = 0; w0 = 0;
        x1 = 0; y1 = 0; z1 = 0; w1 = 0;
        x2 = 0; y2 = 0; z2 = 0; w2 = 0;
        x3 = 0; y3 = 0; z3 = 0; w3 = 0; 
    }

    matrix4<T>(vector4<T> v1, vector4<T> v2, vector4<T> v3, vector4<T> v4) { 
        x0 = v1.x; y0 = v1.y; z0 = v1.z; w0 = v1.w; 
        x1 = v2.x; y1 = v2.y; z1 = v2.z; w1 = v2.w; 
        x2 = v3.x; y2 = v3.y; z2 = v3.z; w2 = v3.w; 
        x3 = v4.x; y3 = v4.y; z3 = v4.z; w3 = v4.w; 
    }

    matrix4<T>(const matrix4<T> &v) { 
        x0 = v.x0; y0 = v.y0; z0 = v.z0; w0 = v.w0; 
        x1 = v.x1; y1 = v.y1; z1 = v.z1; w1 = v.w1; 
        x2 = v.x2; y2 = v.y2; z2 = v.z2; w2 = v.w2; 
        x3 = v.x3; y3 = v.y3; z3 = v.z3; w3 = v.w3; 
    }

    /* Beginning of matrix4 inverse() function */
    
    sf_inline void
    matrix4_scale(matrix4<T> mat, T s) {
        mat.m[0][0] *= s; mat.m[0][1] *= s; mat.m[0][2] *= s; mat.m[0][3] *= s;
        mat.m[1][0] *= s; mat.m[1][1] *= s; mat.m[1][2] *= s; mat.m[1][3] *= s;
        mat.m[2][0] *= s; mat.m[2][1] *= s; mat.m[2][2] *= s; mat.m[2][3] *= s;
        mat.m[3][0] *= s; mat.m[3][1] *= s; mat.m[3][2] *= s; mat.m[3][3] *= s;
    }
    
    sf_inline void
    matrix4_inverse(matrix4<T> mat, matrix4<T> dest) {
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

        matrix4_scale(dest, det);
    }  

    sf_inline matrix4<T> const inverse_0() const {
        matrix4<T> mat;
        matrix4<T> r;
        matrix4_inv(mat.m, r.m);
        return r;
    }

/* This algorithm is almost directly borrowed from GLM. */

    sf_inline matrix4<T> const inverse_1() const {
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
        vector4<T> f0(c00, c00, c02, c03);
        vector4<T> f1(c04, c04, c06, c07);
        vector4<T> f2(c08, c08, c10, c11);
        vector4<T> f3(c12, c12, c14, c15);
        vector4<T> f4(c16, c16, c18, c19);
        vector4<T> f5(c20, c20, c22, c23);
        vector4<T> v0(m[0][1], m[0][0], m[0][0], m[0][0]);
        vector4<T> v1(m[1][1], m[1][0], m[1][0], m[1][0]);
        vector4<T> v2(m[2][1], m[2][0], m[2][0], m[2][0]);
        vector4<T> v3(m[3][1], m[3][0], m[3][0], m[3][0]);
        vector4<T> i0(v1 * f0 - v2 * f1 + v3 * f2);
        vector4<T> i1(v0 * f0 - v2 * f3 + v3 * f4);
        vector4<T> i2(v0 * f1 - v1 * f3 + v3 * f5);
        vector4<T> i3(v0 * f2 - v1 * f4 + v2 * f5);
        vector4<T> sA( 1,-1, 1,-1);
        vector4<T> sB(-1, 1,-1, 1);
        matrix4<T> inv(i0 * sA, 
                    i1 * sB, 
                    i2 * sA, 
                    i3 * sB);
        vector4<T> r0(inv.m[0][0],
                   inv.m[0][1],
                   inv.m[0][2],
                   inv.m[0][3]);
        vector4<T> d0(m[0][0] * r0.x,
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

/*---------------------*/
/* 4D Matrix Overloads */
/*---------------------*/

/* Add two matrix4s. */
template<typename T>
sf_inline matrix4<T> 
operator + 
(const matrix4<T>& lhs, const matrix4<T>& rhs) {
    matrix4<T> c;
    /* row 1 */
    c.m[0][0] = lhs.m[0][0] + rhs.m[0][0]; 
    c.m[1][0] = lhs.m[1][0] + rhs.m[1][0]; 
    c.m[2][0] = lhs.m[2][0] + rhs.m[2][0]; 
    c.m[3][0] = lhs.m[3][0] + rhs.m[3][0];
    /* row 2 */
    c.m[0][1] = lhs.m[0][1] + rhs.m[0][1]; 
    c.m[1][1] = lhs.m[1][1] + rhs.m[1][1]; 
    c.m[2][1] = lhs.m[2][1] + rhs.m[2][1]; 
    c.m[3][1] = lhs.m[3][1] + rhs.m[3][1];
    /* row 3 */
    c.m[0][2] = lhs.m[0][2] + rhs.m[0][2]; 
    c.m[1][2] = lhs.m[1][2] + rhs.m[1][2]; 
    c.m[2][2] = lhs.m[2][2] + rhs.m[2][2]; 
    c.m[3][2] = lhs.m[3][2] + rhs.m[3][2];
    /* row 4 */
    c.m[0][3] = lhs.m[0][3] + rhs.m[0][3]; 
    c.m[1][3] = lhs.m[1][3] + rhs.m[1][3]; 
    c.m[2][3] = lhs.m[2][3] + rhs.m[2][3]; 
    c.m[3][3] = lhs.m[3][3] + rhs.m[3][3];
    return c;
}

/* matrix4 plus-equals operand. */
template<typename T>
sf_inline matrix4<T>& 
operator += 
(matrix4<T>& lhs, const matrix4<T>& rhs) {
    lhs = lhs + rhs;
    return lhs;
}


/* Unary minus operand. Makes matrix4 negative. */
template<typename T>
sf_inline matrix4<T> 
operator - 
(const matrix4<T>& rhs) {
    matrix4<T> c;
    /* row 1 */
    c.x0 =- rhs.x0; 
    c.y0 =- rhs.y0; 
    c.z0 =- rhs.z0; 
    c.w0 =- rhs.w0;
    /* row 2 */
    c.x1 =- rhs.x1; 
    c.y1 =- rhs.y1; 
    c.z1 =- rhs.z1; 
    c.w1 =- rhs.w1;
    /* row 3 */
    c.x2 =- rhs.x2; 
    c.y2 =- rhs.y2; 
    c.z2 =- rhs.z2; 
    c.w2 =- rhs.w2;
    /* row 4 */
    c.x3 =- rhs.x3; 
    c.y3 =- rhs.y3; 
    c.z3 =- rhs.z3; 
    c.w3 =- rhs.w3;
    return c;
}

/* Subtract a matrix4 from a matrix3. */
template<typename T>
sf_inline matrix4<T> 
operator - 
(const matrix4<T>& lhs, const matrix4<T>& rhs) {
    matrix4<T> c;
    /* row 1 */
    c.m[0][0] = lhs.m[0][0] - rhs.m[0][0]; 
    c.m[1][0] = lhs.m[1][0] - rhs.m[1][0]; 
    c.m[2][0] = lhs.m[2][0] - rhs.m[2][0]; 
    c.m[3][0] = lhs.m[3][0] - rhs.m[3][0];
    /* row 2 */
    c.m[0][1] = lhs.m[0][1] - rhs.m[0][1]; 
    c.m[1][1] = lhs.m[1][1] - rhs.m[1][1]; 
    c.m[2][1] = lhs.m[2][1] - rhs.m[2][1]; 
    c.m[3][1] = lhs.m[3][1] - rhs.m[3][1];
    /* row 3 */
    c.m[0][2] = lhs.m[0][2] - rhs.m[0][2]; 
    c.m[1][2] = lhs.m[1][2] - rhs.m[1][2]; 
    c.m[2][2] = lhs.m[2][2] - rhs.m[2][2]; 
    c.m[3][2] = lhs.m[3][2] - rhs.m[3][2];
    /* row 4 */
    c.m[0][3] = lhs.m[0][3] - rhs.m[0][3]; 
    c.m[1][3] = lhs.m[1][3] - rhs.m[1][3]; 
    c.m[2][3] = lhs.m[2][3] - rhs.m[2][3]; 
    c.m[3][3] = lhs.m[3][3] - rhs.m[3][3];
    return c;
}

/* matrix4 minus-equals operand. */
template<typename T>
sf_inline matrix4<T>& 
operator -= 
(matrix4<T>& lhs, const matrix4<T>& rhs) {
    lhs = lhs - rhs;
    return lhs;
}


/* Multiply a matrix4 with a vector4. */
template<typename T>
sf_inline vector4<T> 
operator * 
(const matrix4<T>& lhs, const vector4<T>& rhs) {
    vector4<T> c;
    c.x = rhs.x * lhs.x0 + rhs.y * lhs.x1 + rhs.z * lhs.x2 + rhs.w * lhs.x3;
    c.y = rhs.x * lhs.y0 + rhs.y * lhs.y1 + rhs.z * lhs.y2 + rhs.w * lhs.y3;
    c.z = rhs.x * lhs.z0 + rhs.y * lhs.z1 + rhs.z * lhs.z2 + rhs.w * lhs.z3;
    c.w = rhs.x * lhs.w0 + rhs.y * lhs.w1 + rhs.z * lhs.w2 + rhs.w * lhs.w3;
    return c;
}

/* Multiply a vector4 with a matrix4. */
template<typename T>
sf_inline vector4<T> 
operator * 
(const vector4<T>& lhs, const matrix4<T>& rhs) {
    vector4<T> c;
    c.x = lhs.x * rhs.x0 + lhs.y * rhs.y0 + lhs.z * rhs.z0 + lhs.w * rhs.w0;
    c.y = lhs.x * rhs.x1 + lhs.y * rhs.y1 + lhs.z * rhs.z1 + lhs.w * rhs.w1;
    c.z = lhs.x * rhs.x2 + lhs.y * rhs.y2 + lhs.z * rhs.z2 + lhs.w * rhs.w2;
    c.w = lhs.x * rhs.x3 + lhs.y * rhs.y3 + lhs.z * rhs.z3 + lhs.w * rhs.w3;
    return c;
}

/* Multiply a matrix4 with a scalar. */
template<typename T, typename U>
sf_inline matrix4<T> 
operator * 
(const matrix4<T>& lhs, const U& rhs) {
    matrix4<T> c;
    /* row 1 */
    c.m[0][0] = lhs.m[0][0] * rhs; 
    c.m[1][0] = lhs.m[1][0] * rhs; 
    c.m[2][0] = lhs.m[2][0] * rhs; 
    c.m[3][0] = lhs.m[3][0] * rhs;
    /* row 2 */
    c.m[0][1] = lhs.m[0][1] * rhs; 
    c.m[1][1] = lhs.m[1][1] * rhs; 
    c.m[2][1] = lhs.m[2][1] * rhs; 
    c.m[3][1] = lhs.m[3][1] * rhs;
    /* row 3 */
    c.m[0][2] = lhs.m[0][2] * rhs; 
    c.m[1][2] = lhs.m[1][2] * rhs; 
    c.m[2][2] = lhs.m[2][2] * rhs; 
    c.m[3][2] = lhs.m[3][2] * rhs;
    /* row 4 */
    c.m[0][3] = lhs.m[0][3] * rhs; 
    c.m[1][3] = lhs.m[1][3] * rhs; 
    c.m[2][3] = lhs.m[2][3] * rhs; 
    c.m[3][3] = lhs.m[3][3] * rhs;
    return c;
}

/* Multiply a scalar with a matrix4. */
template<typename T, typename U>
sf_inline matrix4<T> 
operator * 
(const U& lhs, const matrix4<T>& rhs) {
    return(rhs * lhs);
}

/* Multiply two matrix4s. */
template<typename T>
sf_inline matrix4<T> 
operator * 
(const matrix4<T>& lhs, const matrix4<T>& rhs) {
    matrix4<T> c;
    for (u32 j = 0; j < 4; ++j) {
        for (u32 i = 0; i < 4; ++i) {
            c.m[i][j] = rhs.m[0][j] * lhs.m[i][0] + 
                        rhs.m[1][j] * lhs.m[i][1] + 
                        rhs.m[2][j] * lhs.m[i][2] + 
                        rhs.m[3][j] * lhs.m[i][3];
            }
        } 
    return c;

    /* The following code represents the 'unrolled loop' version of matrix4
     * multiplication. In my testing on modern compilers, the version above is
     * faster or the same, although this was not always the case. I will keep
     * this here in case that somehow changes in the future. */

    // matrix4<T> c;
    // c.m[0][0] = lhs.m[0][0] * rhs.m[0][0] + 
    //             lhs.m[1][0] * rhs.m[0][1] + 
    //             lhs.m[2][0] * rhs.m[0][2] + 
    //             lhs.m[3][0] * rhs.m[0][3];
    // c.m[0][1] = lhs.m[0][1] * rhs.m[0][0] + 
    //             lhs.m[1][1] * rhs.m[0][1] + 
    //             lhs.m[2][1] * rhs.m[0][2] + 
    //             lhs.m[3][1] * rhs.m[0][3];
    // c.m[0][2] = lhs.m[0][2] * rhs.m[0][0] + 
    //             lhs.m[1][2] * rhs.m[0][1] + 
    //             lhs.m[2][2] * rhs.m[0][2] + 
    //             lhs.m[3][2] * rhs.m[0][3];
    // c.m[0][3] = lhs.m[0][3] * rhs.m[0][0] + 
    //             lhs.m[1][3] * rhs.m[0][1] + 
    //             lhs.m[2][3] * rhs.m[0][2] + 
    //             lhs.m[3][3] * rhs.m[0][3];
    // c.m[1][0] = lhs.m[0][0] * rhs.m[1][0] + 
    //             lhs.m[1][0] * rhs.m[1][1] + 
    //             lhs.m[2][0] * rhs.m[1][2] + 
    //             lhs.m[3][0] * rhs.m[1][3];
    // c.m[1][1] = lhs.m[0][1] * rhs.m[1][0] + 
    //             lhs.m[1][1] * rhs.m[1][1] + 
    //             lhs.m[2][1] * rhs.m[1][2] + 
    //             lhs.m[3][1] * rhs.m[1][3];
    // c.m[1][2] = lhs.m[0][2] * rhs.m[1][0] + 
    //             lhs.m[1][2] * rhs.m[1][1] + 
    //             lhs.m[2][2] * rhs.m[1][2] + 
    //             lhs.m[3][2] * rhs.m[1][3];
    // c.m[1][3] = lhs.m[0][3] * rhs.m[1][0] + 
    //             lhs.m[1][3] * rhs.m[1][1] + 
    //             lhs.m[2][3] * rhs.m[1][2] + 
    //             lhs.m[3][3] * rhs.m[1][3];
    // c.m[2][0] = lhs.m[0][0] * rhs.m[2][0] + 
    //             lhs.m[1][0] * rhs.m[2][1] + 
    //             lhs.m[2][0] * rhs.m[2][2] + 
    //             lhs.m[3][0] * rhs.m[2][3];
    // c.m[2][1] = lhs.m[0][1] * rhs.m[2][0] + 
    //             lhs.m[1][1] * rhs.m[2][1] + 
    //             lhs.m[2][1] * rhs.m[2][2] + 
    //             lhs.m[3][1] * rhs.m[2][3];
    // c.m[2][2] = lhs.m[0][2] * rhs.m[2][0] + 
    //             lhs.m[1][2] * rhs.m[2][1] + 
    //             lhs.m[2][2] * rhs.m[2][2] + 
    //             lhs.m[3][2] * rhs.m[2][3];
    // c.m[2][3] = lhs.m[0][3] * rhs.m[2][0] + 
    //             lhs.m[1][3] * rhs.m[2][1] + 
    //             lhs.m[2][3] * rhs.m[2][2] + 
    //             lhs.m[3][3] * rhs.m[2][3];
    // c.m[3][0] = lhs.m[0][0] * rhs.m[3][0] + 
    //             lhs.m[1][0] * rhs.m[3][1] + 
    //             lhs.m[2][0] * rhs.m[3][2] + 
    //             lhs.m[3][0] * rhs.m[3][3];
    // c.m[3][1] = lhs.m[0][1] * rhs.m[3][0] + 
    //             lhs.m[1][1] * rhs.m[3][1] + 
    //             lhs.m[2][1] * rhs.m[3][2] + 
    //             lhs.m[3][1] * rhs.m[3][3];
    // c.m[3][2] = lhs.m[0][2] * rhs.m[3][0] + 
    //             lhs.m[1][2] * rhs.m[3][1] + 
    //             lhs.m[2][2] * rhs.m[3][2] + 
    //             lhs.m[3][2] * rhs.m[3][3];
    // c.m[3][3] = lhs.m[0][3] * rhs.m[3][0] + 
    //             lhs.m[1][3] * rhs.m[3][1] + 
    //             lhs.m[2][3] * rhs.m[3][2] + 
    //             lhs.m[3][3] * rhs.m[3][3];
    // return c;
}

/* Multiply-equals operand with two matrix4s. */
template<typename T>
sf_inline matrix4<T>& 
operator *= 
(matrix4<T>& lhs, const matrix4<T>& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Multiply-equals operand with matrix4 and scalar. */
template<typename T, typename U>
sf_inline matrix4<T>& 
operator *= 
(matrix4<T>& lhs, const U& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Tests for equality between two matrix4s. */
template<typename T>
sf_inline bool 
operator == 
(const matrix4<T>& lhs, const matrix4<T>& rhs) {
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

/* Tests for non-equality between two matrix4s. */
template<typename T>
sf_inline bool 
operator != 
(const matrix4<T>& lhs, const matrix4<T>& rhs) {
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
(std::ostream& os, const matrix4<T>& rhs) {
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

/*---------------------*/
/* 4D Matrix Functions */
/*---------------------*/

template<typename T>
sf_inline matrix4<T> 
identity(matrix4<T> mat) {
	mat.m[0][0] = 1.0f, mat.m[1][0] = 0.0f, mat.m[2][0] = 0.0f, mat.m[3][0] = 0.0f;
	mat.m[0][1] = 0.0f, mat.m[1][1] = 1.0f, mat.m[2][1] = 0.0f, mat.m[3][1] = 0.0f;
	mat.m[0][2] = 0.0f, mat.m[1][2] = 0.0f, mat.m[2][2] = 1.0f, mat.m[3][2] = 0.0f;
	mat.m[0][3] = 0.0f, mat.m[1][3] = 0.0f, mat.m[2][3] = 0.0f, mat.m[3][3] = 1.0f;
	return mat;
}

template<typename T>
sf_inline matrix4<T> 
transpose(const matrix4<T>& transpose) {
    matrix4<T> mat;
    /* row 1 */
    transpose.m[0][0] = mat.m[0][0]; 
    transpose.m[1][0] = mat.m[0][1]; 
    transpose.m[2][0] = mat.m[0][2];
    transpose.m[3][0] = mat.m[0][3];
    /* row 2 */
    transpose.m[0][1] = mat.m[1][0]; 
    transpose.m[1][1] = mat.m[1][1]; 
    transpose.m[2][1] = mat.m[1][2];
    transpose.m[3][1] = mat.m[1][3];
    /* row 3 */
    transpose.m[0][2] = mat.m[2][0]; 
    transpose.m[1][2] = mat.m[2][1]; 
    transpose.m[2][2] = mat.m[2][2];
    transpose.m[3][2] = mat.m[2][3];
    /* row 4 */
    transpose.m[0][3] = mat.m[3][0]; 
    transpose.m[1][3] = mat.m[3][1]; 
    transpose.m[2][3] = mat.m[3][2];
    transpose.m[3][3] = mat.m[3][3];
    return transpose;
}

template<typename T>
sf_inline matrix4<T> 
determinant(const matrix4<T> det) {
    T t[6];

    t[0] = det.m[2][2] * det.m[3][3] - det.m[3][2] * det.m[2][3];
    t[1] = det.m[2][1] * det.m[3][3] - det.m[3][1] * det.m[2][3];
    t[2] = det.m[2][1] * det.m[3][2] - det.m[3][1] * det.m[2][2];
    t[3] = det.m[2][0] * det.m[3][3] - det.m[3][0] * det.m[2][3];
    t[4] = det.m[2][0] * det.m[3][2] - det.m[3][0] * det.m[2][2];
    t[5] = det.m[2][0] * det.m[3][1] - det.m[3][0] * det.m[2][1];

    return(det.m[0][0] * (det.m[1][1] * t[0] - det.m[1][2] * t[1] + det.m[1][3] * t[2]) - 
           det.m[0][1] * (det.m[1][0] * t[0] - det.m[1][2] * t[3] + det.m[1][3] * t[4]) + 
           det.m[0][2] * (det.m[1][0] * t[1] - det.m[1][1] * t[3] + det.m[1][3] * t[5]) - 
           det.m[0][3] * (det.m[1][0] * t[2] - det.m[1][1] * t[4] + det.m[1][2] * t[5]));
}

template<typename T>
sf_inline matrix4<T>
inverse(const matrix4<T> mat) {
    T t[6];
    T determinant;
    matrix4<T> dest;

    t[0] = mat.m[2][2] * mat.m[3][3] - mat.m[3][2] * mat.m[2][3]; 
    t[1] = mat.m[2][1] * mat.m[3][3] - mat.m[3][1] * mat.m[2][3]; 
    t[2] = mat.m[2][1] * mat.m[3][2] - mat.m[3][1] * mat.m[2][2];
    t[3] = mat.m[2][0] * mat.m[3][3] - mat.m[3][0] * mat.m[2][3]; 
    t[4] = mat.m[2][0] * mat.m[3][2] - mat.m[3][0] * mat.m[2][2]; 
    t[5] = mat.m[2][0] * mat.m[3][1] - mat.m[3][0] * mat.m[2][1];

    dest.m[0][0] =   mat.m[1][1] * t[0] - mat.m[1][2] * t[1] + mat.m[1][3] * t[2];
    dest.m[1][0] = -(mat.m[1][0] * t[0] - mat.m[1][2] * t[3] + mat.m[1][3] * t[4]);
    dest.m[2][0] =   mat.m[1][0] * t[1] - mat.m[1][1] * t[3] + mat.m[1][3] * t[5];
    dest.m[3][0] = -(mat.m[1][0] * t[2] - mat.m[1][1] * t[4] + mat.m[1][2] * t[5]);
    
    dest.m[0][1] = -(mat.m[0][1] * t[0] - mat.m[0][2] * t[1] + mat.m[0][3] * t[2]);
    dest.m[1][1] =   mat.m[0][0] * t[0] - mat.m[0][2] * t[3] + mat.m[0][3] * t[4];
    dest.m[2][1] = -(mat.m[0][0] * t[1] - mat.m[0][1] * t[3] + mat.m[0][3] * t[5]);
    dest.m[3][1] =   mat.m[0][0] * t[2] - mat.m[0][1] * t[4] + mat.m[0][2] * t[5];
    
    t[0] = mat.m[1][2] * mat.m[3][3] - mat.m[3][2] * mat.m[1][3]; 
    t[1] = mat.m[1][1] * mat.m[3][3] - mat.m[3][1] * mat.m[1][3]; 
    t[2] = mat.m[1][1] * mat.m[3][2] - mat.m[3][1] * mat.m[1][2];
    t[3] = mat.m[1][0] * mat.m[3][3] - mat.m[3][0] * mat.m[1][3]; 
    t[4] = mat.m[1][0] * mat.m[3][2] - mat.m[3][0] * mat.m[1][2]; 
    t[5] = mat.m[1][0] * mat.m[3][1] - mat.m[3][0] * mat.m[1][1];
    
    dest.m[0][2] =   mat.m[0][1] * t[0] - mat.m[0][2] * t[1] + mat.m[0][3] * t[2];
    dest.m[1][2] = -(mat.m[0][0] * t[0] - mat.m[0][2] * t[3] + mat.m[0][3] * t[4]);
    dest.m[2][2] =   mat.m[0][0] * t[1] - mat.m[0][1] * t[3] + mat.m[0][3] * t[5];
    dest.m[3][2] = -(mat.m[0][0] * t[2] - mat.m[0][1] * t[4] + mat.m[0][2] * t[5]);
        
    t[0] = mat.m[1][2] * mat.m[2][3] - mat.m[2][2] * mat.m[1][3]; 
    t[1] = mat.m[1][1] * mat.m[2][3] - mat.m[2][1] * mat.m[1][3]; 
    t[2] = mat.m[1][1] * mat.m[2][2] - mat.m[2][1] * mat.m[1][2];
    t[3] = mat.m[1][0] * mat.m[2][3] - mat.m[2][0] * mat.m[1][3]; 
    t[4] = mat.m[1][0] * mat.m[2][2] - mat.m[2][0] * mat.m[1][2]; 
    t[5] = mat.m[1][0] * mat.m[2][1] - mat.m[2][0] * mat.m[1][1];
        
    dest.m[0][3] = -(mat.m[0][1] * t[0] - mat.m[0][2] * t[1] + mat.m[0][3] * t[2]);
    dest.m[1][3] =   mat.m[0][0] * t[0] - mat.m[0][2] * t[3] + mat.m[0][3] * t[4];
    dest.m[2][3] = -(mat.m[0][0] * t[1] - mat.m[0][1] * t[3] + mat.m[0][3] * t[5]);
    dest.m[3][3] =   mat.m[0][0] * t[2] - mat.m[0][1] * t[4] + mat.m[0][2] * t[5];

    determinant = 1.0f / (mat.m[0][0] * dest.m[0][0] + 
                          mat.m[0][1] * dest.m[1][0] + 
                          mat.m[0][2] * dest.m[2][0] + 
                          mat.m[0][3] * dest.m[3][0]);

    return(dest * determinant);
}

template<typename T>
sf_inline matrix4<T> 
translate(const vector3<T>& t) {
    matrix4<T> r(1.0f);
    r.m[3][0] = t.x;
    r.m[3][1] = t.y;
    r.m[3][2] = t.z;
    return r;     
}    
    
template<typename T>
sf_inline matrix4<T> 
scale(const vector3<T>& s) {
    matrix4<T> r;
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    return r;
}

template<typename T>
sf_inline matrix4<T> 
rotate(const T ang, const i32 type) {
    matrix4<T> r;
    T c = std::cos(ang);
    T s = std::sin(ang);
    switch(type) {
        case 0:
    		r.m[1][1] =  c;
    		r.m[2][2] =  c;
    		r.m[1][2] =  s;
    		r.m[2][1] = -s;
    		break;
    	case 1:
    		r.m[0][0] =  c;
    		r.m[2][2] =  c;
    		r.m[2][0] =  s;
    		r.m[0][2] = -s;
    		break;
    	case 2:
    		r.m[0][0] =  c;
    		r.m[1][1] =  c;
    		r.m[0][1] =  s;
    		r.m[1][0] = -s;
    		break;
    	}	
    return r;
}

template<typename T>
sf_inline matrix4<T> 
lookat(const vector3<T>& eye, const vector3<T>& target, const vector3<T>& up) {
    matrix4<T> observer;
    vector3<T> n = target - eye;
    n = normalize(n);
    T b = dot_product(up, n);
    T ab = std::sqrt(1.0f - sf_math_utils_square(b));
    observer.m[0][2] = n.x;
    observer.m[1][2] = n.y;
    observer.m[2][2] = n.z;
    observer.m[0][1] = (up.x - b * n.x) / ab;
    observer.m[1][1] = (up.y - b * n.y) / ab;
    observer.m[2][1] = (up.z - b * n.z) / ab;
    observer.m[0][0] = observer.m[1][2] * observer.m[2][1] - observer.m[1][1] * observer.m[2][2];
    observer.m[1][0] = observer.m[2][2] * observer.m[0][1] - observer.m[2][1] * observer.m[0][2];
    observer.m[2][0] = observer.m[0][2] * observer.m[1][1] - observer.m[0][1] * observer.m[1][2];
    matrix4<T> r2 = translate(-eye);      
    observer = r2 * observer;
    return observer;
}

template<typename T>
sf_inline matrix4<T> 
perspective_projection(const T& angle_of_view, const T& z_near, const T& z_far) { 
    matrix4<T> mat;
    T scale = 1 / std::tan(angle_of_view * 0.5 * PI / 180); 
    mat.m[0][0] = scale;
    mat.m[1][1] = scale;
    mat.m[2][2] = -z_far / (z_far - z_near);
    mat.m[3][2] = -z_far * z_near / (z_far - z_near);
    mat.m[2][3] = -1; // set w = -z 
    mat.m[3][3] = 0; 
    return mat;
} 

/*---------------------------*/
/* Mathematical Type Aliases */
/*---------------------------*/

typedef vector2<f32> vec2f;
typedef vector2<f64> vec2;

typedef vector3<f32> vec3f;
typedef vector3<f64> vec3;

typedef vector4<f32> vec4f;
typedef vector4<f64> vec4;

typedef matrix3<f32> mat3f;
typedef matrix3<f64> mat3;

typedef matrix4<f32> mat4f;
typedef matrix4<f64> mat4;

} // namespace sf
