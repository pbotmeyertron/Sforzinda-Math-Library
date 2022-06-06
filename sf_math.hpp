#pragma once

#include <cstdint>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <complex>
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

#ifndef sf_inline
    #define sf_inline static inline
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
#define SQRT_2          1.414213562373095048801688724209698079  
#endif


/* sqrt(3) */
#ifndef SQRT_3
#define SQRT_3          1.732050807568877293527446341505872366  
#endif

/* sqrt(5) */
#ifndef SQRT_5
#define SQRT_5          2.236067977499789696409173668731276235  
#endif

/* sqrt(1/2) */
#ifndef SQRT_1_DIV_2
#define SQRT_1_DIV_2    0.707106781186547524400844362104849039  
#endif

/* sqrt(1/3) */
#ifndef SQRT_1_DIV_3
#define SQRT_1_DIV_3    0.577350269189625764509148780501957455  
#endif

/* pi */
#ifndef PI
#define PI              3.141592653589793238462643383279502884  
#endif

/* pi * 2 */
#ifndef TAU
#define TAU             6.283185307179586476925286766559005774
#endif

/* pi/2 */
#ifndef PI_DIV_2
#define PI_DIV_2        1.570796326794896619231321691639751442  
#endif

/* pi/4 */
#ifndef PI_DIV_4
#define PI_DIV_4        0.785398163397448309615660845819875721  
#endif

/* sqrt(pi) */
#ifndef SQRT_PI
#define SQRT_PI         1.772453850905516027298167483341145183  
#endif

/* e */
#ifndef E
#define E               2.718281828459045235360287471352662498        
#endif

/* ln(2) */
#ifndef LN_2
#define LN_2            0.693147180559945309417232121458176568  
#endif

/* ln(10) */
#ifndef LN_10
#define LN_10           2.302585092994045684017991454684364208  
#endif

/* ln(pi) */
#ifndef LN_PI
#define LN_PI           1.144729885849400174143427351353058712  
#endif

/* log_2(e) */
#ifndef LOG_BASE_2_E 
#define LOG_BASE_2_E    1.442695040888963407359924681001892137     
#endif  
  
/* log_10(e) */
#ifndef LOG_BASE_10_E
#define LOG_BASE_10_E   0.434294481903251827651128918916605082  
#endif

/* Euler-Mascheroni Constant */
#ifndef EULER
#define EULER           0.577215664901532860606512090082402431  
#endif

/* Golden rhsatio */
#ifndef PHI
#define PHI             1.618033988749894848204586834365638118  
#endif

/* Apery's Constant */
#ifndef APERY
#define APERY           1.202056903159594285399738161511449991  
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
[[nodiscard]] sf_inline constexpr T
sf_math_utils_equals(T a, T b) {
    return std::abs((a - b) < EPSILON);
}

/* Performs non-equality check using machine-epsilon. */
template<typename T>
[[nodiscard]] sf_inline constexpr T
sf_math_utils_not_equals(T a, T b) {
    return std::abs((a - b) >= EPSILON);
}

/* Mutliplies a value by itself. */
template<typename T>
[[nodiscard]] sf_inline constexpr T
sf_math_utils_square(T a) {
    return a * a;
}

/* Mutliplies a value by itself thrice. */
template<typename T>
[[nodiscard]] sf_inline constexpr T
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
    return (val >> 31) - (-val >> 31);
}

/* Returns the sign of a 64-bit integer as +1, -1, or 0. */
sf_inline i64  
sf_math_utils_sign(i64 val) {
    return (val >> 63) - (-val >> 63);
}

/* Returns the sign of a 32-bit float as +1, -1, or 0. */
sf_inline 
f32 sf_math_utils_sign(f32 val) {
    return static_cast<f32>((val > 0.0f) - (val < 0.0f));
}

/* Returns the sign of a 64-bit float as +1, -1, or 0. */
sf_inline f64 
sf_math_utils_sign(f64 val) {
    return static_cast<f64>((val > 0.0f) - (val < 0.0f));
}

/*--------------------*/
/* Graphics Utilities */
/*--------------------*/

/* Converts degrees to radians. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_degrees_to_radians(T deg) {
    return deg * PI / 180.0f;
}

/* Converts radians to degrees. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_radians_to_degrees(T rad) {
    return rad * 180.0f / PI;
}

/* Clamp a number between min and max. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_clamp(T val, T min, T max) {
    return std::min(std::max(val, min), max);
}

/* Clamp a number between zero and one. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_clamp_zero_to_one(T val) {
    return sf_math_utils_clamp(val, 0.0f, 1.0f);
}

/* Linear interpolation between two numbers. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_lerp(T from, T to, T t) {
    return from + t * (to - from);
}

/* Clamped linear interpolation. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_clamped_lerp(T from, T to, T t) {
    return sf_math_utils_lerp(from, to, sf_math_utils_clamp_zero_to_one(t));
}

/* Step function. Returns 0.0 if x < edge, else 1.0. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_step(T edge, T x) {
    return (x < edge) ? 0.0f : 1.0f;
}

/* Hermite interpolation. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_hermite_interpolation(T t) {
    return sf_math_utils_square(t) * (3.0f - (2.0f * t));
}

/* Threshold function with smooth transition. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_smoothstep(T edge0, T edge1, T x) {
    T t;
    t = sf_math_utils_clamp_zero_to_one((x - edge0) / (edge1 - edge0));
    return sf_math_utils_hermite_interpolation(t);
}

/* Smoothstep function with Hermite interpolation. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_smooth_hermite(T from, T to, T t) {
    return from + sf_math_utils_hermite_interpolation(t) * (to - from);
}

/* Clamped smoothstep with Hermite interpolation. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_smooth_hermite_clamped(T from, T to, T t) {
    return sf_math_utils_smooth_hermite(from, to, sf_math_utils_clamp_zero_to_one(t));
}

/* Percentage of current value between start and end value. */
template<typename T>
[[nodiscard]] sf_inline T 
sf_math_utils_percent(T from, T to, T current) {
    T t;
    if ((t = to - from) == 0.0f)
        return 1.0f;
    return (current - from) / t;
}

/* Clamped percentage of current value between start and end value. */
template<typename T>
[[nodiscard]] sf_inline T 
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
 * vec2    - 2D Vector
 * vec3    - 3D Vector
 * vec4    - 4D Vector
 * matrix2    - 2x2 Matrix
 * mat3    - 3x3 Matrix
 * mat4    - 4x4 Matrix
 * quat - quat
 */

/*-----------*/
/* 2D Vector */
/*-----------*/

struct vec2 {
    union {
        struct sf_align(8) {
            /* Coordinate notation. */
            f32 x, y; 
        };
        struct sf_align(8) {
            /* Array notation. */
            f32 v[2]; 
        };
    };

    vec2() { 
        x = 0;
        y = 0; 
    } 

    vec2(f32 cx, f32 cy) { 
        x = cx; 
        y = cy; 
    }

    vec2(f32 cx) { 
        x = cx;
        y = cx; 
    }

    vec2(const vec2& v) { 
        x = v.x; 
        y = v.y; 
    }

    vec2(f32 v[2]) { 
        x = v[0]; 
        y = v[1]; 
    }

    /* Index or subscript operand. */
    [[nodiscard]] constexpr inline f32 
    operator [] 
    (u32 i) const {
        return v[i];
    }

    /* Index or subscript operand. */
    [[nodiscard]] constexpr inline f32& 
    operator[] 
    (u32 i) {
        return v[i];
    }

}; // vec2

/*---------------------*/
/* 2D Vector Overloads */
/*---------------------*/

/* Add two vec2s. */
[[nodiscard]] sf_inline vec2 
operator + 
(const vec2& lhs, const vec2& rhs) {
    vec2 c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    return c;
}

/* Add vec2 and scalar. */
[[nodiscard]] sf_inline vec2 
operator + 
(const vec2& lhs, const f32& rhs) {
    vec2 c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    return c;
}

/* Add scalar and vec2. */
[[nodiscard]] sf_inline vec2 
operator + 
(const f32& lhs, const vec2& rhs) {
    vec2 c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    return c;
}

/* Plus-equals operand with two vec2s. */
[[nodiscard]] sf_inline vec2& 
operator += 
(vec2& lhs, const vec2& rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y;
    return lhs;
}

/* Plus-equals operand with a vec2 and scalar. */
[[nodiscard]] sf_inline vec2& 
operator += 
(vec2& lhs, const f32& rhs) {
    lhs.x += rhs; 
    lhs.y += rhs;
    return lhs;
}

/* Unary minus operand. Makes vec2 negative. */
[[nodiscard]] sf_inline vec2 
operator - 
(const vec2& rhs) {
    vec2 c; 
    c.x = -rhs.x; 
    c.y = -rhs.y; 
    return c;
}

/* Subtracts a vec2 from a vec2. */
[[nodiscard]] sf_inline vec2 
operator - 
(const vec2& lhs, const vec2& rhs) {
    vec2 c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    return c;
}

/* Subtracts a scalar from a vec2. */
[[nodiscard]] sf_inline vec2 
operator - 
(const vec2& lhs, const f32& rhs) {
    vec2 c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    return c;
}

/* Subtracts a vec2 from a scalar. */
[[nodiscard]] sf_inline vec2 
operator - 
(const f32& lhs, const vec2& rhs) {
    vec2 c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    return c;
}

/* Minus-equals operand for two vec2s. */
[[nodiscard]] sf_inline vec2& 
operator -= 
(vec2& lhs, const vec2& rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y;
    return lhs;
}

/* Minus-equals operand for vec2 and scalar. */
[[nodiscard]] sf_inline vec2& 
operator -= 
(vec2& lhs, const f32& rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs;
    return lhs;
}

/* Multiplies two vec2s. */
[[nodiscard]] sf_inline vec2 
operator * 
(const vec2& lhs, const vec2& rhs) {
    vec2 c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y;
    return c;
}

/* Multiplies a vec2 and scalar. */
[[nodiscard]] sf_inline vec2 
operator * 
(const f32& lhs, const vec2& rhs) {
    vec2 c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs;
    return c;
}

/* Multiplies a scalar and vec2. */
[[nodiscard]] sf_inline vec2 
operator * 
(const vec2& lhs, const f32& rhs) {
    vec2 c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    return c;
}

/* Multiply-equals operand for vec2. */
[[nodiscard]] sf_inline vec2& 
operator *= 
(vec2& lhs, const vec2& rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y;
    return lhs;
}

/* Multiply-equals operand for vec2 and scalar. */
[[nodiscard]] sf_inline vec2& 
operator *= 
(vec2& lhs, const f32& rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs;
    return lhs;
}

/* Divides two vec2. */
[[nodiscard]] sf_inline vec2 
operator / 
(const vec2& lhs, const vec2& rhs) {
    vec2 c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y;
    return c;
}

/* Divides a vec2 by a scalar. */
[[nodiscard]] sf_inline vec2 
operator / 
(const vec2& lhs, const f32& rhs) {
    vec2 c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs;
    return c;
}

/* Divide-equals operand for two vec2s. */
[[nodiscard]] sf_inline vec2& 
operator /= 
(vec2& lhs, const vec2& rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y;
    return lhs;
}

/* Divide-equals operand for vec2 and scalar. */
[[nodiscard]] sf_inline vec2& 
operator /= 
(vec2& lhs, const f32& rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs;
    return lhs;
}

/* Add one to each element in vec2. */
[[nodiscard]] sf_inline vec2& 
operator ++ 
(vec2& rhs) {
    ++rhs.x; 
    ++rhs.y;
    return rhs;
}

/* Add one to each element in vec2. */
[[nodiscard]] sf_inline vec2 
operator ++ 
(vec2& lhs, i32) {
    vec2 c = lhs;
    lhs.x++; 
    lhs.y++;
    return(c);
}

/* Subtract one from each element in vec2. */
[[nodiscard]] sf_inline vec2& 
operator -- 
(vec2& rhs) {
    --rhs.x; 
    --rhs.y;
    return rhs;
}

/* Subtract one from each element in vec2. */
[[nodiscard]] sf_inline vec2 
operator -- 
(vec2& lhs, i32) {
    vec2 c = lhs;
    lhs.x--; 
    lhs.y--;
    return c;
}

/* Tests two vec2s for equality. */
[[nodiscard]] sf_inline bool 
operator == 
(const vec2& lhs, const vec2& rhs) {
    return (lhs.x == rhs.x) && 
           (lhs.y == rhs.y);
}

/* Tests two vec2s for non-equality. */
[[nodiscard]] sf_inline bool 
operator != 
(const vec2& lhs, const vec2& rhs) {
    return (lhs.x != rhs.x) || (lhs.y != rhs.y);
}

/* Allows for printing elements of vec2 to stdout. Thanks to rhsay Tracing in One
 * Weekend for this. :) */
[[nodiscard]] sf_inline std::ostream& 
operator << 
(std::ostream& os, const vec2& rhs) {
    os << "(" << rhs.x << "," << rhs.y << ")";
    return os;
}

/*---------------------*/
/* 2D Vector Functions */
/*---------------------*/

/* Returns the length (magnitude) of a 2D vector. */
[[nodiscard]] sf_inline f32 
length(const vec2& a) {
    return std::sqrt(sf_math_utils_square(a.x) + 
                     sf_math_utils_square(a.y));
}

/* Normalizes a 2D vector. */
[[nodiscard]] sf_inline vec2 
normalize(vec2& a) {
    f32 mag = length(a);
    if (mag != 0.0f) {
        return(a /= mag);
    }
    return vec2(0.0f, 0.0f);
}

/* Returns the dot product of a 2D vector. */
[[nodiscard]] sf_inline f32 
dot_product(const vec2& a, const vec2& b) {
    return a.x * b.x + 
           a.y * b.y;
}

/* Returns the cross product of a 2D vector. */
[[nodiscard]] sf_inline vec2 
cross_product(const vec2& a, const vec2 b) {
    vec2 c;
    c.x = (a.x * b.y) - (a.y * b.x);
    c.y = (a.y * b.x) - (a.x * b.y);
    return c;
}

/* Rotate vec2 around origin counter-clockwise. */
[[nodiscard]] sf_inline vec2 
rotate(const vec2& a, f32 angle) {
    vec2 dest;
    f32 cos_angle, sin_angle, x1, y1;
    cos_angle = std::cos(angle);
    sin_angle = std::sin(angle);
    x1 = a.x;
    y1 = a.y;
    dest.x = (cos_angle * x1) - (sin_angle * y1);
    dest.y = (sin_angle * x1) + (cos_angle * y1);
    return dest;
}

/* Clamp a vec2 between min and max. */
[[nodiscard]] sf_inline vec2 
clamp(vec2& a, f32 min, f32 max) {
    a.x = sf_math_utils_clamp(a.x, min, max);
    a.y = sf_math_utils_clamp(a.y, min, max);
    return a;
}

/* Returns the angle between two 2D vectors. */
[[nodiscard]] sf_inline f32 
angle_between(const vec2& a, const vec2& b) {
    return dot_product(a, b) / (length(a) * length(b));
}

/* Returns the distance between two 2D vectors. */
[[nodiscard]] sf_inline f32 
distance(const vec2& a, const vec2& b) {
    return std::sqrt(sf_math_utils_square(b.x - a.x) + 
                     sf_math_utils_square(b.y - a.y));
}

/*-----------*/
/* 3D Vector */
/*-----------*/

struct vec3 {
    union {
        struct sf_align(16) { 
            /* Coordinate notation. */
            f32 x, y, z; 
        };
        struct sf_align(16 ){ 
            /* Array notation. */
            f32 v[3]; 
        };
    };

    vec3() { 
        x = 0;
        y = 0;
        z = 0; 
    }

    vec3(f32 cx, f32 cy, f32 cz) { 
        x = cx; 
        y = cy; 
        z = cz; 
    }

    vec3(f32 cx) { 
        x = cx;
        y = cx;
        z = cx; 
    }

    /* Initialize a vec3 with a vec2 and a scalar. */
    vec3(vec2 v, f32 cz) { 
        x = v.x; 
        y = v.y; 
        z = cz; 
    }

    vec3(const vec3& v) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
    }

    vec3(f32 v[3]) { 
        x = v[0]; 
        y = v[1]; 
        z = v[2]; 
    }

    /* Index or subscript operand. */
    [[nodiscard]] constexpr inline f32 
    operator [] 
    (u32 i) const {
        return v[i];
    }

    /* Index or subscript operand. */
    [[nodiscard]] constexpr inline f32& 
    operator [] 
    (u32 i) {
        return v[i];
    }

}; // vec3

/*---------------------*/
/* 3D Vector Overloads */
/*---------------------*/

/* Add two vec3s. */
[[nodiscard]] sf_inline vec3 
operator + 
(const vec3& lhs, const vec3& rhs) {
    vec3 c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    c.z = lhs.z + rhs.z; 
    return c;
}

/* Add vec3 and scalar. */
[[nodiscard]] sf_inline vec3 
operator + 
(const vec3& lhs, const f32& rhs) {
    vec3 c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    c.z = lhs.z + rhs; 
    return c;
}

/* Add scalar and vec3. */
[[nodiscard]] sf_inline vec3 
operator + 
(const f32& lhs, const vec3& rhs) {
    vec3 c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    c.z = lhs + rhs.z; 
    return c;
}

/* Plus-equals operand with two vec3s. */
[[nodiscard]] sf_inline vec3& 
operator += 
(vec3& lhs, const vec3& rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y; 
    lhs.z += rhs.z;
    return lhs;
}

/* Plus-equals operand with a vec3 and scalar. */
[[nodiscard]] sf_inline vec3& 
operator += 
(vec3& lhs, const f32& rhs) {
    lhs.x += rhs; 
    lhs.y += rhs; 
    lhs.z += rhs;
    return lhs;
}

/* Unary minus operand. Makes vec3 negative. */
[[nodiscard]] sf_inline vec3 
operator - 
(const vec3& rhs) {
    vec3 c; 
    c.x = -rhs.x; 
    c.y = -rhs.y; 
    c.z = -rhs.z; 
    return c;
}

/* Subtracts a vec3 from a vec3. */
[[nodiscard]] sf_inline vec3 
operator - 
(const vec3& lhs, const vec3& rhs) {
    vec3 c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    c.z = lhs.z - rhs.z; 
    return c;
}

/* Subtracts a scalar from a vec3. */
[[nodiscard]] sf_inline vec3 
operator - 
(const vec3& lhs, const f32& rhs) {
    vec3 c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    c.z = lhs.z - rhs; 
    return c;
}

/* Subtracts a vec3 from a scalar. */
[[nodiscard]] sf_inline vec3 
operator - 
(const f32& lhs, const vec3& rhs) {
    vec3 c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    c.z = lhs - rhs.z; 
    return c;
}

/* Minus-equals operand for two vec3s. */
[[nodiscard]] sf_inline vec3& 
operator -= 
(vec3& lhs, const vec3& rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y; 
    lhs.z -= rhs.z;
    return lhs;
}

/* Minus-equals operand for vec3 and scalar. */
[[nodiscard]] sf_inline vec3& 
operator -= 
(vec3& lhs, const f32& rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs; 
    lhs.z -= rhs;
    return lhs;
}

/* Multiplies two vec3s. */
[[nodiscard]] sf_inline vec3 
operator * 
(const vec3& lhs, const vec3& rhs) {
    vec3 c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y; 
    c.z = rhs.z * lhs.z;
    return c;
}

/* Multiplies a vec3 and scalar. */
[[nodiscard]] sf_inline vec3 
operator * 
(const f32 &lhs, const vec3 &rhs) {
    vec3 c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs; 
    c.z = rhs.z * lhs;
    return(c);
}

/* Multiplies a scalar and vec3. */
[[nodiscard]] sf_inline vec3 
operator * 
(const vec3& lhs, const f32& rhs) {
    vec3 c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    c.z = rhs * lhs.z;
    return c;
}

/* Multiply-equals operand for vec3. */
[[nodiscard]] sf_inline vec3& 
operator *= 
(vec3& lhs, const vec3& rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y; 
    lhs.z *= rhs.z;
    return lhs;
}

/* Multiply-equals operand for vec3 and scalar. */
[[nodiscard]] sf_inline vec3& 
operator *= 
(vec3& lhs, const f32& rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs; 
    lhs.z *= rhs;
    return lhs;
}

/* Divides two vec3s. */
[[nodiscard]] sf_inline vec3 
operator / 
(const vec3& lhs, const vec3& rhs) {
    vec3 c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y; 
    c.z = lhs.z / rhs.z;
    return c;
}

/* Divides a vec3 by a scalar. */
[[nodiscard]] sf_inline vec3 
operator / 
(const vec3& lhs, const f32& rhs) {
    vec3 c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs; 
    c.z = lhs.z / rhs;
    return c;
}

/* Divide-equals operand for two vec3s. */
[[nodiscard]] sf_inline vec3& 
operator /= 
(vec3& lhs, const vec3& rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y; 
    lhs.z /= rhs.z;
    return(lhs);
}

/* Divide-equals operand for vec3 and scalar. */
[[nodiscard]] sf_inline vec3& 
operator /= 
(vec3& lhs, const f32& rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs; 
    lhs.z /= rhs;
    return lhs;
}

/* Add one to each element in vec3. */
[[nodiscard]] sf_inline vec3& 
operator ++ 
(vec3& rhs) {
    ++rhs.x; 
    ++rhs.y; 
    ++rhs.z;
    return rhs;
}

/* Add one to each element in vec3. */
[[nodiscard]] sf_inline vec3 
operator ++ 
(vec3& lhs, i32) {
    vec3 c = lhs;
    lhs.x++; 
    lhs.y++; 
    lhs.z++;
    return c;
}

/* Subtract one from each element in vec3. */
[[nodiscard]] sf_inline vec3& 
operator -- 
(vec3& rhs) {
    --rhs.x; 
    --rhs.y; 
    --rhs.z;
    return rhs;
}

/* Subtract one from each element in vec3. */
[[nodiscard]] sf_inline vec3 
operator -- 
(vec3& lhs, i32) {
    vec3 c = lhs;
    lhs.x--; 
    lhs.y--; 
    lhs.z--;
    return c;
}

/* Tests two vec3s for equality. */
[[nodiscard]] sf_inline bool 
operator == 
(const vec3& lhs, const vec3& rhs) {
    return((lhs.x == rhs.x) && 
           (lhs.y == rhs.y) && 
           (lhs.z == rhs.z));
}

/* Tests two vec3s for non-equality. */
[[nodiscard]] sf_inline bool 
operator != 
(const vec3& lhs, const vec3& rhs) {
    return((lhs.x != rhs.x) || 
           (lhs.y != rhs.y) || 
           (lhs.z != rhs.z));
}

/* Allows for printing elements of vec3 to stdout. Thanks to rhsay Tracing in One
 * Weekend for this. :) */
[[nodiscard]] sf_inline std::ostream& 
operator << 
(std::ostream& os, const vec3& rhs) {
    os << "(" << rhs.x << "," << rhs.y << "," << rhs.z << ")";
    return os;
}

/*---------------------*/
/* 3D Vector Functions */
/*---------------------*/

/* Returns the length (magnitude) of a 3D vector. */
[[nodiscard]] sf_inline f32 
length(const vec3& a) {
    return std::sqrt(sf_math_utils_square(a.x) + 
                     sf_math_utils_square(a.y) + 
                     sf_math_utils_square(a.z));
}

/* Normalizes a 3D vector. */
[[nodiscard]] sf_inline vec3 
normalize(vec3& a) {
    f32 mag = length(a);
    if (mag != 0.0f) {
        return a /= mag;
    }
    return vec3(0.0f, 0.0f, 0.0f);
}

/* Returns the dot product of a 3D vector. */
[[nodiscard]] sf_inline f32 
dot_product(const vec3& a, const vec3& b) {
    return a.x * b.x + 
           a.y * b.y + 
           a.z * b.z;
}

/* Returns the cross product of a 3D vector. */
[[nodiscard]] sf_inline vec3 
cross_product(const vec3& a, const vec3& b) {
    vec3 c;
    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);
    return c;
}

/* Returns the angle between two 3D vectors. */
[[nodiscard]] sf_inline f32 
angle_between(const vec3& a, const vec3& b) {
    f32 c;
    c = dot_product(a, b) / (length(a) * length(b));
    return 2.0f * std::acos(c);
}

/* Returns the distance between two 3D vectors. */
[[nodiscard]] sf_inline f32 
distance(const vec3& a, const vec3& b) {
    return std::sqrt(sf_math_utils_square(b.x - a.x) + 
                     sf_math_utils_square(b.y - a.y) +
                     sf_math_utils_square(b.z - a.z));
}

/*-----------*/
/* 4D Vector */
/*-----------*/

struct vec4 {
    union {
        struct sf_align(16) { 
            /* Coordinate notation. */
            f32 x, y, z, w; 
        };
        struct sf_align(16) { 
            /* Array notation. */
            f32 v[4]; 
        };
    };

    vec4() { 
        x = 0;
        y = 0;
        z = 0;
        w = 0; 
    }

    vec4(f32 cx, f32 cy, f32 cz, f32 cw) { 
        x = cx; 
        y = cy; 
        z = cz; 
        w = cw; 
    }

    vec4(f32 cx) { 
        x = cx;
        y = cx;
        z = cx;
        w = cx; 
    }

    /* Initialize a vec4 with a vec3 and a scalar. */
    vec4(vec3 v, f32 cw) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
        w = cw; 
    }

    /* Initialize a vec4 with two vec2s. */
    vec4(vec2 v, vec2 u) { 
        x = v.x; 
        y = v.y; 
        z = u.x; 
        w = u.y; 
    }   

    vec4(const vec4& v) { 
        x = v.x; 
        y = v.y; 
        z = v.z; 
        w = v.w; 
    }

    vec4(f32 v[4]) { 
        x = v[0]; 
        y = v[1]; 
        z = v[2]; 
        w = v[3]; 
    }

    /* Index or subscript operand. */
    [[nodiscard]] constexpr inline f32 
    operator [] 
    (u32 i) const {
        return v[i];
    }

    /* Index or subscript operand. */
    [[nodiscard]] constexpr inline f32& 
    operator [] 
    (u32 i) {
        return v[i];
    }

}; // vec4

/*---------------------*/
/* 4D Vector Overloads */
/*---------------------*/

/* Add two vec4s. */
[[nodiscard]] sf_inline vec4 
operator + 
(const vec4& lhs, const vec4& rhs) {
    vec4 c; 
    c.x = lhs.x + rhs.x; 
    c.y = lhs.y + rhs.y; 
    c.z = lhs.z + rhs.z; 
    c.w = lhs.w + rhs.w; 
    return c;
}

/* Add vec4 and scalar. */
[[nodiscard]] sf_inline vec4 
operator + 
(const vec4& lhs, const f32& rhs) {
    vec4 c; 
    c.x = lhs.x + rhs; 
    c.y = lhs.y + rhs; 
    c.z = lhs.z + rhs; 
    c.w = lhs.w + rhs; 
    return c;
}

/* Add scalar and vec4. */
[[nodiscard]] sf_inline vec4 
operator + 
(const f32& lhs, const vec4& rhs) {
    vec4 c; 
    c.x = lhs + rhs.x; 
    c.y = lhs + rhs.y; 
    c.z = lhs + rhs.z; 
    c.w = lhs + rhs.w; 
    return c;
}

/* Plus-equals operand with two vec4s. */
[[nodiscard]] sf_inline vec4& 
operator += 
(vec4& lhs, const vec4& rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y; 
    lhs.z += rhs.z; 
    lhs.w += rhs.w;
    return lhs;
}

/* Plus-equals operand with a vec4 and scalar. */
[[nodiscard]] sf_inline vec4& 
operator += 
(vec4& lhs, const f32& rhs) {
    lhs.x += rhs; 
    lhs.y += rhs; 
    lhs.z += rhs; 
    lhs.w += rhs;
    return lhs;
}

/* Unary minus operand. Makes vec4 negative. */
[[nodiscard]] sf_inline vec4 
operator - 
(const vec4& rhs) {
    vec4 c; 
    c.x = -rhs.x; 
    c.y = -rhs.y; 
    c.z = -rhs.z; 
    c.w = -rhs.w; 
    return c;
}

/* Subtracts a vec4 from a vec4. */
[[nodiscard]] sf_inline vec4 
operator - 
(const vec4& lhs, const vec4& rhs) {
    vec4 c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    c.z = lhs.z - rhs.z; 
    c.w = lhs.w - rhs.w; 
    return c;
}

/* Subtracts a scalar from a vec4. */
[[nodiscard]] sf_inline vec4 
operator - 
(const vec4& lhs, const f32& rhs) {
    vec4 c; 
    c.x = lhs.x - rhs; 
    c.y = lhs.y - rhs; 
    c.z = lhs.z - rhs; 
    c.w = lhs.w - rhs; 
    return c;
}

/* Subtracts a vec4 from a scalar. */
[[nodiscard]] sf_inline vec4 
operator - 
(const f32& lhs, const vec4& rhs) {
    vec4 c; 
    c.x = lhs - rhs.x; 
    c.y = lhs - rhs.y; 
    c.z = lhs - rhs.z; 
    c.w = lhs - rhs.w; 
    return c;
}

/* Minus-equals operand for two vec4s. */
[[nodiscard]] sf_inline vec4& 
operator -= 
(vec4& lhs, const vec4& rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y; 
    lhs.z -= rhs.z; 
    lhs.w -= rhs.w;
    return lhs;
}

/* Minus-equals operand for vec4 and scalar. */
[[nodiscard]] sf_inline vec4& 
operator -= 
(vec4& lhs, const f32& rhs) {
    lhs.x -= rhs; 
    lhs.y -= rhs; 
    lhs.z -= rhs; 
    lhs.w -= rhs;
    return lhs;
}

/* Multiplies two vec4s. */
[[nodiscard]] sf_inline vec4 
operator * 
(const vec4& lhs, const vec4& rhs) {
    vec4 c;
    c.x = rhs.x * lhs.x; 
    c.y = rhs.y * lhs.y; 
    c.z = rhs.z * lhs.z; 
    c.w = rhs.w * lhs.w;
    return c;
}

/* Multiplies a vec4 and scalar. */
[[nodiscard]] sf_inline vec4 
operator * 
(const f32& lhs, const vec4& rhs) {
    vec4 c;
    c.x = rhs.x * lhs; 
    c.y = rhs.y * lhs; 
    c.z = rhs.z * lhs; 
    c.w = rhs.w * lhs;
    return c;
}

/* Multiplies a scalar and vec4. */
[[nodiscard]] sf_inline vec4 
operator * 
(const vec4& lhs, const f32& rhs) {
    vec4 c;
    c.x = rhs * lhs.x;
    c.y = rhs * lhs.y;
    c.z = rhs * lhs.z;
    c.w = rhs * lhs.w;
    return c;
}

/* Multiply-equals operand for vec4. */
[[nodiscard]] sf_inline vec4& 
operator *= 
(vec4& lhs, const vec4& rhs) {
    lhs.x *= rhs.x; 
    lhs.y *= rhs.y; 
    lhs.z *= rhs.z; 
    lhs.w *= rhs.w;
    return lhs;
}

/* Multiply-equals operand for vec4 and scalar. */
[[nodiscard]] sf_inline vec4& 
operator *= 
(vec4& lhs, const f32& rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs; 
    lhs.z *= rhs; 
    lhs.w *= rhs;
    return lhs;
}

/* Divides two vec4s. */
[[nodiscard]] sf_inline vec4 
operator / 
(const vec4& lhs, const vec4& rhs) {
    vec4 c;
    c.x = lhs.x / rhs.x; 
    c.y = lhs.y / rhs.y; 
    c.z = lhs.z / rhs.z; 
    c.w = lhs.w / rhs.w;
    return c;
}

/* Divides a vec4 by a scalar. */
[[nodiscard]] sf_inline vec4 
operator / 
(const vec4& lhs, const f32& rhs) {
    vec4 c;
    c.x = lhs.x / rhs; 
    c.y = lhs.y / rhs; 
    c.z = lhs.z / rhs; 
    c.w = lhs.w / rhs;
    return c;
}

/* Divide-equals operand for two vec4s. */
[[nodiscard]] sf_inline vec4& 
operator /= 
(vec4& lhs, const vec4& rhs) {
    lhs.x /= rhs.x; 
    lhs.y /= rhs.y; 
    lhs.z /= rhs.z; 
    lhs.w /= rhs.w;
    return lhs;
}

/* Divide-equals operand for vec4 and scalar. */
[[nodiscard]] sf_inline vec4& 
operator /= 
(vec4& lhs, const f32& rhs) {
    lhs.x /= rhs; 
    lhs.y /= rhs; 
    lhs.z /= rhs; 
    lhs.w /= rhs;
    return lhs;
}

/* Add one to each element in vec4. */
[[nodiscard]] sf_inline vec4& 
operator ++ 
(vec4& rhs) {
    ++rhs.x; 
    ++rhs.y; 
    ++rhs.z; 
    ++rhs.w;
    return rhs;
}

/* Add one to each element in vec4. */
[[nodiscard]] sf_inline vec4 
operator ++ 
(vec4& lhs, i32) {
    vec4 c = lhs;
    lhs.x++; 
    lhs.y++; 
    lhs.z++; 
    lhs.w++;
    return c;
}

/* Subtract one from each element in vec4. */
[[nodiscard]] sf_inline vec4& 
operator -- 
(vec4& rhs) {
    --rhs.x; 
    --rhs.y; 
    --rhs.z; 
    --rhs.w;
    return rhs;
}

/* Subtract one from each element in vec4. */
[[nodiscard]] sf_inline vec4 
operator -- 
(vec4& lhs, i32) {
    vec4 c = lhs;
    lhs.x--; 
    lhs.y--; 
    lhs.z--; 
    lhs.w--;
    return c;
}

/* Tests two vec4s for equality. */
[[nodiscard]] sf_inline bool 
operator == 
(const vec4& lhs, const vec4& rhs) {
    return((lhs.x == rhs.x) && 
           (lhs.y == rhs.y) && 
           (lhs.z == rhs.z) && 
           (lhs.w == rhs.w));
}

/* Tests two vec4s for non-equality. */
[[nodiscard]] sf_inline bool 
operator != 
(const vec4& lhs, const vec4& rhs) {
    return((lhs.x != rhs.x) || 
           (lhs.y != rhs.y) || 
           (lhs.z != rhs.z) || 
           (lhs.w != rhs.w));
}

/* Allows for printing elements of vec4 to stdout. Thanks to rhsay Tracing in One
 * Weekend for this. :) */
[[nodiscard]] sf_inline std::ostream& 
operator << 
(std::ostream& os, const vec4& rhs) {
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
[[nodiscard]] sf_inline f32 
length(const vec4& a) {
    return std::sqrt(sf_math_utils_square(a.x) + 
                     sf_math_utils_square(a.y) + 
                     sf_math_utils_square(a.z) +
                     sf_math_utils_square(a.w));
}

/* Normalizes a 4D vector. */
[[nodiscard]] sf_inline vec4 
normalize(vec4& a) {
    f32 mag = length(a);
    if (mag != 0.0f) {
        return(a /= mag);
    }
    return vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

/* Returns the dot product of a 4D vector. */
[[nodiscard]] sf_inline f32 
dot_product(const vec4& a, const vec4& b) {
    return a.x * b.x + 
           a.y * b.y + 
           a.z * b.z +
           a.w * b.w;
}

/* Returns the cross product of a 4D vector. */
[[nodiscard]] sf_inline vec4 
cross_product(const vec4& a, const vec4& b) {
    vec4 c;
    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);
    c.w = (a.w * b.w) - (a.w * b.w); // evaluates to zero
    return c;
}

/* Returns the distance between two 4D vectors. */
[[nodiscard]] sf_inline f32 
distance(const vec4& a, const vec4& b) {
    return std::sqrt(sf_math_utils_square(b.x - a.x) + 
                     sf_math_utils_square(b.y - a.y) + 
                     sf_math_utils_square(b.z - a.z) + 
                     sf_math_utils_square(b.w - a.w));
}

//==============================================================================
// matrix2                                                 
//==============================================================================
// TODO: Come back and define matrix2
/* 2x2 Matrix */

/*-----------*/
/* 3D Matrix */
/*-----------*/

struct mat3 {
    union {
        struct sf_align(32) { 
            /* reference matrix [row][column] */
            f32 m[3][3]; 
        };
        struct sf_align(32) { 
            f32 x0, x1, x2;
            f32 y0, y1, y2;
            f32 z0, z1, z2; 
        };
        struct sf_align(32) { 
            f32 M[9]; 
        };
    };

    mat3() { 
        x0 = 0; y0 = 0; z0 = 0;
        x1 = 0; y1 = 0; z1 = 0;
        x2 = 0; y2 = 0; z2 = 0;
    }

    mat3(vec3 v1, vec3 v2, vec3 v3) { 
        x0 = v1.x; y0 = v1.y; z0 = v1.z; 
        x1 = v2.x; y1 = v2.y; z1 = v2.z; 
        x2 = v3.x; y2 = v3.y; z2 = v3.z; 
    }

    mat3(const mat3& v) { 
        x0 = v.x0; y0 = v.y0; z0 = v.z0; 
        x1 = v.x1; y1 = v.y1; z1 = v.z1; 
        x2 = v.x2; y2 = v.y2; z2 = v.z2; 
    }

}; // mat3

/*---------------------*/
/* 3D Matrix Overloads */
/*---------------------*/

/* Add two mat3s. */
[[nodiscard]] sf_inline mat3 
operator + 
(const mat3& lhs, const mat3& rhs) {
    mat3 c;
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

/* mat3 plus-equals operand. */
[[nodiscard]] sf_inline mat3& 
operator += 
(mat3& lhs, const mat3& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

/* Unary minus operand. Makes mat3 negative. */
[[nodiscard]] sf_inline mat3 
operator - 
(const mat3& rhs) {
    mat3 c;
    /* row 1 */
    c.x0 = -rhs.x0; 
    c.y0 = -rhs.y0; 
    c.z0 = -rhs.z0;
    /* row 2 */
    c.x1 = -rhs.x1; 
    c.y1 = -rhs.y1; 
    c.z1 = -rhs.z1;
    /* row 3 */
    c.x2 = -rhs.x2; 
    c.y2 = -rhs.y2; 
    c.z2 = -rhs.z2;
    return c;
}

/* Subtract a mat3 from a mat3. */
[[nodiscard]] sf_inline mat3 
operator - 
(const mat3& lhs, const mat3& rhs) {
    mat3 c;
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

/* mat3 minus-equals operand. */
[[nodiscard]] sf_inline mat3& 
operator -= 
(mat3& lhs, const mat3& rhs) {
    lhs = lhs - rhs;
    return lhs;
}

/* Multiply a mat3 with a vec3. */
[[nodiscard]] sf_inline vec3 
operator * 
(const mat3& lhs, const vec3& rhs) {
    vec3 c;
    c.x = rhs.x * lhs.x0 + rhs.y * lhs.x1 + rhs.z * lhs.x2;
    c.y = rhs.x * lhs.y0 + rhs.y * lhs.y1 + rhs.z * lhs.y2;
    c.z = rhs.x * lhs.z0 + rhs.y * lhs.z1 + rhs.z * lhs.z2;
    return c;
}

/* Multiply a vec3 with a mat3. */
[[nodiscard]] sf_inline vec3 
operator * 
(const vec3& lhs, const mat3& rhs) {
    vec3 c;
    c.x = lhs.x * rhs.x0 + lhs.y * rhs.y0 + lhs.z * rhs.z0;
    c.y = lhs.x * rhs.x1 + lhs.y * rhs.y1 + lhs.z * rhs.z1;
    c.z = lhs.x * rhs.x2 + lhs.y * rhs.y2 + lhs.z * rhs.z2;
    return c;
}

/* Multiply a mat3 with a scalar. */
[[nodiscard]] sf_inline mat3 
operator * 
(const mat3& lhs, const f32& rhs) {
    mat3 c;
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
[[nodiscard]] sf_inline mat3 
operator * 
(const f32& lhs, const mat3& rhs) {
    return rhs * lhs;
}

/* Multiply two mat3s. */
[[nodiscard]] sf_inline mat3 
operator * 
(const mat3& lhs, const mat3& rhs) {
    mat3 c;
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
[[nodiscard]] sf_inline mat3& 
operator *= 
(mat3& lhs, const mat3& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Multiply-equals operand with mat3 and scalar. */
[[nodiscard]] sf_inline mat3& 
operator *= 
(mat3& lhs, const f32& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Tests for equality between two mat3s. */
[[nodiscard]] sf_inline bool 
operator == 
(const mat3& lhs, const mat3& rhs) {
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
[[nodiscard]] sf_inline bool 
operator != 
(const mat3& lhs, const mat3& rhs) {
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

/* Allows for printing elements of mat3 to stdout. */
[[nodiscard]] sf_inline std::ostream& 
operator << 
(std::ostream& os, const mat3& rhs) {
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
[[nodiscard]] sf_inline mat3 
copy(const mat3& src, mat3& dest) {
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
[[nodiscard]] sf_inline mat3  
transpose(mat3& transpose) {
    mat3 mat;
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
[[nodiscard]] sf_inline f32 
determinant(const mat3& det) {
    return det.m[0][0] * (det.m[1][1] * det.m[2][2] - det.m[2][1] * det.m[1][2]) - 
           det.m[1][0] * (det.m[0][1] * det.m[2][2] - det.m[2][1] * det.m[0][2]) + 
           det.m[2][0] * (det.m[0][1] * det.m[1][2] - det.m[1][1] * det.m[0][2]);
}

/* Mat3 inverse. */
[[nodiscard]] sf_inline mat3
inverse(const mat3& a) {
    mat3 dest;
    f32 determinant;
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

/*-----------*/
/* 4D Matrix */
/*-----------*/

struct mat4 {
    union {
        struct sf_align(32) { 
            f32 m[4][4]; 
        };
        struct sf_align(32) { 
            f32 M[16]; 
        };
        struct sf_align(32) { 
            f32 x0, x1, x2, x3;
            f32 y0, y1, y2, y3; 
            f32 z0, z1, z2, z3; 
            f32 w0, w1, w2, w3; 
        };
    };

    mat4() { 
        x0 = 0; y0 = 0; z0 = 0; w0 = 0;
        x1 = 0; y1 = 0; z1 = 0; w1 = 0;
        x2 = 0; y2 = 0; z2 = 0; w2 = 0;
        x3 = 0; y3 = 0; z3 = 0; w3 = 0; 
    }

    mat4(vec4 v1, vec4 v2, vec4 v3, vec4 v4) { 
        x0 = v1.x; y0 = v1.y; z0 = v1.z; w0 = v1.w; 
        x1 = v2.x; y1 = v2.y; z1 = v2.z; w1 = v2.w; 
        x2 = v3.x; y2 = v3.y; z2 = v3.z; w2 = v3.w; 
        x3 = v4.x; y3 = v4.y; z3 = v4.z; w3 = v4.w; 
    }

    mat4(const mat4& v) { 
        x0 = v.x0; y0 = v.y0; z0 = v.z0; w0 = v.w0; 
        x1 = v.x1; y1 = v.y1; z1 = v.z1; w1 = v.w1; 
        x2 = v.x2; y2 = v.y2; z2 = v.z2; w2 = v.w2; 
        x3 = v.x3; y3 = v.y3; z3 = v.z3; w3 = v.w3; 
    }

    [[nodiscard]] inline mat4 const transpose() const noexcept {
        mat4 transpose;
        transpose.m[0][0] = m[0][0]; 
        transpose.m[1][0] = m[0][1]; 
        transpose.m[2][0] = m[0][2];
        transpose.m[3][0] = m[0][3];
        transpose.m[0][1] = m[1][0]; 
        transpose.m[1][1] = m[1][1]; 
        transpose.m[2][1] = m[1][2];
        transpose.m[3][1] = m[1][3];
        transpose.m[0][2] = m[2][0]; 
        transpose.m[1][2] = m[2][1]; 
        transpose.m[2][2] = m[2][2];
        transpose.m[3][2] = m[2][3];
        transpose.m[0][3] = m[3][0]; 
        transpose.m[1][3] = m[3][1]; 
        transpose.m[2][3] = m[3][2];
        transpose.m[3][3] = m[3][3];
        return transpose;
    }

    [[nodiscard]] inline f32 const determinant() const noexcept {
        f32 f0 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        f32 f1 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        f32 f2 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        f32 f3 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
        f32 f4 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
        f32 f5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
        vec4 dc = vec4((m[1][1] * f0 - m[2][1] * f1 + m[3][1] * f2), 
                                  -(m[0][1] * f0 - m[2][1] * f3 + m[3][1] * f4), 
                                   (m[0][1] * f1 - m[1][1] * f3 + m[3][1] * f5),
                                  -(m[0][1] * f2 - m[1][1] * f4 + m[2][1] * f5));
        return m[0][0] * dc.x + 
               m[1][0] * dc.y + 
               m[2][0] * dc.z + 
               m[3][0] * dc.w;
    }

    [[nodiscard]] inline mat4 const inverse() const noexcept {
        f32 c00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        f32 c02 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
        f32 c03 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
        f32 c04 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        f32 c06 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
        f32 c07 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
        f32 c08 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        f32 c10 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
        f32 c11 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        f32 c12 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
        f32 c14 = m[0][1] * m[3][3] - m[0][3] * m[3][1];
        f32 c15 = m[0][1] * m[3][2] - m[0][2] * m[3][1];
        f32 c16 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
        f32 c18 = m[0][1] * m[2][3] - m[0][3] * m[2][1];
        f32 c19 = m[0][1] * m[2][2] - m[0][2] * m[2][1];
        f32 c20 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
        f32 c22 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
        f32 c23 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        vec4 f0(c00, c00, c02, c03);
        vec4 f1(c04, c04, c06, c07);
        vec4 f2(c08, c08, c10, c11);
        vec4 f3(c12, c12, c14, c15);
        vec4 f4(c16, c16, c18, c19);
        vec4 f5(c20, c20, c22, c23);
        vec4 v0(m[0][1], m[0][0], m[0][0], m[0][0]);
        vec4 v1(m[1][1], m[1][0], m[1][0], m[1][0]);
        vec4 v2(m[2][1], m[2][0], m[2][0], m[2][0]);
        vec4 v3(m[3][1], m[3][0], m[3][0], m[3][0]);
        vec4 i0(v1 * f0 - v2 * f1 + v3 * f2);
        vec4 i1(v0 * f0 - v2 * f3 + v3 * f4);
        vec4 i2(v0 * f1 - v1 * f3 + v3 * f5);
        vec4 i3(v0 * f2 - v1 * f4 + v2 * f5);
        vec4 sA( 1, -1,  1, -1);
        vec4 sB(-1,  1, -1,  1);
        mat4 inverse(i0 * sA, i1 * sB, i2 * sA, i3 * sB);
        vec4 r0(inverse.m[0][0],
                inverse.m[0][1],
                inverse.m[0][2],
                inverse.m[0][3]);
        vec4 d0(m[0][0] * r0.x, 
                m[1][0] * r0.y, 
                m[2][0] * r0.z, 
                m[3][0] * r0.w);
        f32 d1 = (d0.x + d0.y) + (d0.z + d0.w);
        f32 inverse_determinant = 1.0f / d1;
        mat4 dest;  
        dest.m[0][0] = inverse.m[0][0] * inverse_determinant; 
        dest.m[1][0] = inverse.m[1][0] * inverse_determinant; 
        dest.m[2][0] = inverse.m[2][0] * inverse_determinant; 
        dest.m[3][0] = inverse.m[3][0] * inverse_determinant;
        dest.m[0][1] = inverse.m[0][1] * inverse_determinant; 
        dest.m[1][1] = inverse.m[1][1] * inverse_determinant; 
        dest.m[2][1] = inverse.m[2][1] * inverse_determinant; 
        dest.m[3][1] = inverse.m[3][1] * inverse_determinant;
        dest.m[0][2] = inverse.m[0][2] * inverse_determinant; 
        dest.m[1][2] = inverse.m[1][2] * inverse_determinant; 
        dest.m[2][2] = inverse.m[2][2] * inverse_determinant; 
        dest.m[3][2] = inverse.m[3][2] * inverse_determinant;
        dest.m[0][3] = inverse.m[0][3] * inverse_determinant; 
        dest.m[1][3] = inverse.m[1][3] * inverse_determinant; 
        dest.m[2][3] = inverse.m[2][3] * inverse_determinant; 
        dest.m[3][3] = inverse.m[3][3] * inverse_determinant;
        return dest;
    }

}; // mat4

/*---------------------*/
/* 4D Matrix Overloads */
/*---------------------*/

/* Add two mat4s. */
[[nodiscard]] sf_inline mat4 
operator + 
(const mat4& lhs, const mat4& rhs) {
    mat4 c;
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

/* mat4 plus-equals operand. */
[[nodiscard]] sf_inline mat4& 
operator += 
(mat4& lhs, const mat4& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

/* Unary minus operand. Makes mat4 negative. */
[[nodiscard]] sf_inline mat4 
operator - 
(const mat4& rhs) {
    mat4 c;
    /* row 1 */
    c.x0 = -rhs.x0; 
    c.y0 = -rhs.y0; 
    c.z0 = -rhs.z0;
    c.w0 = -rhs.w0;
    /* row 2 */
    c.x1 = -rhs.x1; 
    c.y1 = -rhs.y1; 
    c.z1 = -rhs.z1; 
    c.w1 = -rhs.w1;
    /* row 3 */
    c.x2 = -rhs.x2; 
    c.y2 = -rhs.y2; 
    c.z2 = -rhs.z2; 
    c.w2 = -rhs.w2;
    /* row 4 */
    c.x3 = -rhs.x3; 
    c.y3 = -rhs.y3; 
    c.z3 = -rhs.z3; 
    c.w3 = -rhs.w3;
    return c;
}   
 
/* Subtract a mat4 from a mat3. */
[[nodiscard]] sf_inline mat4 
operator - 
(const mat4& lhs, const mat4& rhs) {
    mat4 c;
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

/* mat4 minus-equals operand. */
[[nodiscard]] sf_inline mat4& 
operator -= 
(mat4& lhs, const mat4& rhs) {
    lhs = lhs - rhs;
    return lhs;
}


/* Multiply a mat4 with a vec4. */
[[nodiscard]] sf_inline vec4 
operator * 
(const mat4& lhs, const vec4& rhs) {
    vec4 c;
    c.x = rhs.x * lhs.x0 + rhs.y * lhs.x1 + rhs.z * lhs.x2 + rhs.w * lhs.x3;
    c.y = rhs.x * lhs.y0 + rhs.y * lhs.y1 + rhs.z * lhs.y2 + rhs.w * lhs.y3;
    c.z = rhs.x * lhs.z0 + rhs.y * lhs.z1 + rhs.z * lhs.z2 + rhs.w * lhs.z3;
    c.w = rhs.x * lhs.w0 + rhs.y * lhs.w1 + rhs.z * lhs.w2 + rhs.w * lhs.w3;
    return c;
}

/* Multiply a vec4 with a mat4. */
[[nodiscard]] sf_inline vec4 
operator * 
(const vec4& lhs, const mat4& rhs) {
    vec4 c;
    c.x = lhs.x * rhs.x0 + lhs.y * rhs.y0 + lhs.z * rhs.z0 + lhs.w * rhs.w0;
    c.y = lhs.x * rhs.x1 + lhs.y * rhs.y1 + lhs.z * rhs.z1 + lhs.w * rhs.w1;
    c.z = lhs.x * rhs.x2 + lhs.y * rhs.y2 + lhs.z * rhs.z2 + lhs.w * rhs.w2;
    c.w = lhs.x * rhs.x3 + lhs.y * rhs.y3 + lhs.z * rhs.z3 + lhs.w * rhs.w3;
    return c;
}

/* Multiply a mat4 with a scalar. */
[[nodiscard]] sf_inline mat4 
operator * 
(const mat4& lhs, const f32& rhs) {
    mat4 c;
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

/* Multiply a scalar with a mat4. */
[[nodiscard]] sf_inline mat4 
operator * 
(const f32& lhs, const mat4& rhs) {
    return(rhs * lhs);
}

/* Multiply two mat4s. */
[[nodiscard]] sf_inline mat4 
operator * 
(const mat4& lhs, const mat4& rhs) {
    mat4 c;
    for (u32 j = 0; j < 4; ++j) {
        for (u32 i = 0; i < 4; ++i) {
            c.m[i][j] = rhs.m[0][j] * lhs.m[i][0] + 
                        rhs.m[1][j] * lhs.m[i][1] + 
                        rhs.m[2][j] * lhs.m[i][2] + 
                        rhs.m[3][j] * lhs.m[i][3];
            }
        } 
    return c;

    /* The following code represents the 'unrolled loop' version of mat4
     * multiplication. In my testing on modern compilers, the version above is
     * faster or the same, although this was not always the case. I will keep
     * this here in case that somehow changes in the future. */

    // mat4 c;
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

/* Multiply-equals operand with two mat4s. */
[[nodiscard]] sf_inline mat4& 
operator *= 
(mat4& lhs, const mat4& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Multiply-equals operand with mat4 and scalar. */
[[nodiscard]] sf_inline mat4& 
operator *= 
(mat4& lhs, const f32& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

/* Tests for equality between two mat4s. */
[[nodiscard]] sf_inline bool 
operator == 
(const mat4& lhs, const mat4& rhs) {
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

/* Tests for non-equality between two mat4s. */
[[nodiscard]] sf_inline bool 
operator != 
(const mat4& lhs, const mat4& rhs) {
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

[[nodiscard]] sf_inline std::ostream& 
operator << 
(std::ostream& os, const mat4& rhs) {
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

[[nodiscard]] sf_inline mat4 
identity(mat4 mat) {
    mat.m[0][0] = 1.0f, mat.m[1][0] = 0.0f, mat.m[2][0] = 0.0f, mat.m[3][0] = 0.0f;
    mat.m[0][1] = 0.0f, mat.m[1][1] = 1.0f, mat.m[2][1] = 0.0f, mat.m[3][1] = 0.0f;
    mat.m[0][2] = 0.0f, mat.m[1][2] = 0.0f, mat.m[2][2] = 1.0f, mat.m[3][2] = 0.0f;
    mat.m[0][3] = 0.0f, mat.m[1][3] = 0.0f, mat.m[2][3] = 0.0f, mat.m[3][3] = 1.0f;
    return mat;
}

[[nodiscard]] sf_inline mat4 
transpose(mat4& transpose) {
    mat4 mat;
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

[[nodiscard]] sf_inline f32
determinant(const mat4 det) {
    f32 t[6];
    t[0] = det.m[2][2] * det.m[3][3] - det.m[3][2] * det.m[2][3];
    t[1] = det.m[2][1] * det.m[3][3] - det.m[3][1] * det.m[2][3];
    t[2] = det.m[2][1] * det.m[3][2] - det.m[3][1] * det.m[2][2];
    t[3] = det.m[2][0] * det.m[3][3] - det.m[3][0] * det.m[2][3];
    t[4] = det.m[2][0] * det.m[3][2] - det.m[3][0] * det.m[2][2];
    t[5] = det.m[2][0] * det.m[3][1] - det.m[3][0] * det.m[2][1];
    return det.m[0][0] * (det.m[1][1] * t[0] - det.m[1][2] * t[1] + det.m[1][3] * t[2]) - 
           det.m[0][1] * (det.m[1][0] * t[0] - det.m[1][2] * t[3] + det.m[1][3] * t[4]) + 
           det.m[0][2] * (det.m[1][0] * t[1] - det.m[1][1] * t[3] + det.m[1][3] * t[5]) - 
           det.m[0][3] * (det.m[1][0] * t[2] - det.m[1][1] * t[4] + det.m[1][2] * t[5]);
}

[[nodiscard]] sf_inline mat4
inverse(const mat4 mat) {
    f32 t[6];
    f32 determinant;
    mat4 dest;

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

    /* This version is slightly faster than the member function version because
     * it can take advantage of the inlined operator overloading for multiplying mat4
     * and f32. */
    return dest * determinant;
}

[[nodiscard]] sf_inline mat4 
translate(const vec3& t) {
    mat4 r;
    r.m[3][0] = t.x;
    r.m[3][1] = t.y;
    r.m[3][2] = t.z;
    return r;     
}    

[[nodiscard]] sf_inline mat4 
scale(const vec3& s) {
    mat4 r;
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    return r;
}

[[nodiscard]] sf_inline mat4 
rotate(const f32 ang, const i32 type) {
    mat4 r;
    f32 c = std::cos(ang);
    f32 s = std::sin(ang);
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

[[nodiscard]] sf_inline mat4 
camera_view(const vec3& eye, const vec3& target, const vec3& up) {
    mat4 observer;
    vec3 n = target - eye;
    n = normalize(n);
    f32 b = dot_product(up, n);
    f32 ab = std::sqrt(1.0f - sf_math_utils_square(b));
    observer.m[0][2] = n.x;
    observer.m[1][2] = n.y;
    observer.m[2][2] = n.z;
    observer.m[0][1] = (up.x - b * n.x) / ab;
    observer.m[1][1] = (up.y - b * n.y) / ab;
    observer.m[2][1] = (up.z - b * n.z) / ab;
    observer.m[0][0] = observer.m[1][2] * observer.m[2][1] - observer.m[1][1] * observer.m[2][2];
    observer.m[1][0] = observer.m[2][2] * observer.m[0][1] - observer.m[2][1] * observer.m[0][2];
    observer.m[2][0] = observer.m[0][2] * observer.m[1][1] - observer.m[0][1] * observer.m[1][2];
    mat4 r2 = translate(-eye);      
    observer = r2 * observer;
    return observer;
}

[[nodiscard]] sf_inline mat4 
perspective_projection(const f32& angle_of_view, const f32& z_near, const f32& z_far) { 
    mat4 mat;
    f32 scale = 1 / std::tan(angle_of_view * 0.5 * PI / 180); 
    mat.m[0][0] = scale;
    mat.m[1][1] = scale;
    mat.m[2][2] = -z_far / (z_far - z_near);
    mat.m[3][2] = -z_far * z_near / (z_far - z_near);
    mat.m[2][3] = -1; // set w = -z 
    mat.m[3][3] = 0; 
    return mat;
} 

/*------------*/
/* Quaternion */
/*------------*/

struct quat {
    union {
        struct sf_align(16) { 
            f32 w,x,y,z; 
        };
        struct sf_align(16) { 
            f32 X,Y,Z,W; 
        };
        struct sf_align(16) { 
            f32 V[4]; 
        };
    };

        quat() { 
            x = 0;
            y = 0;
            z = 0;
            w = 0; 
        }

        quat(f32 cw, f32 cx, f32 cy, f32 cz) { 
            x = cx; 
            y = cy; 
            z = cz; 
            w = cw; 
        }

        /* Euler angle initialization. y = yaw, z = pitch, x = roll. */
        quat(const vec3& euler_angles) {
            vec3 cos_angles, sin_angles;
            cos_angles.x = std::cos(euler_angles.x * 0.5f);
            cos_angles.y = std::cos(euler_angles.y * 0.5f);
            cos_angles.z = std::cos(euler_angles.z * 0.5f);
            sin_angles.x = std::sin(euler_angles.x * 0.5f);
            sin_angles.y = std::sin(euler_angles.y * 0.5f);
            sin_angles.z = std::sin(euler_angles.z * 0.5f);
            w = cos_angles.x * cos_angles.y * cos_angles.z - 
                sin_angles.x * sin_angles.y * sin_angles.z;
            x = cos_angles.x * cos_angles.y * sin_angles.z + 
                sin_angles.x * sin_angles.y * cos_angles.z;
            y = sin_angles.x * cos_angles.y * cos_angles.z + 
                cos_angles.x * sin_angles.y * sin_angles.z;
            z = cos_angles.x * sin_angles.y * cos_angles.z - 
                sin_angles.x * cos_angles.y * sin_angles.z;
        }
    
        quat(const quat& s) { 
            x = s.x; 
            y = s.y; 
            z = s.z; 
            w = s.w; 
        }

        quat(f32 v[4]) { 
            w = v[0]; 
            x = v[1]; 
            y = v[2]; 
            z = v[3]; 
        }

        quat(std::complex<f32>& c) { 
            w = c.real(); 
            x = c.imag(); 
            y = 0;
            z = 0; 
        }

        quat(const f32& s, const vec3& v) { 
            w = s; 
            x = v.x; 
            y = v.y; 
            z = v.z; 
        }

        quat(const f32& s) { 
            w = s; 
            x = 0.0f;
            y = 0.0f;
            z = 0.0f; 
        }

        /*-----------------------------*/
        /* Quaternion Member Functions */
        /*-----------------------------*/

        [[nodiscard]] inline quat square() const noexcept {
            quat r;
            r.w = sf_math_utils_square(w) - (sf_math_utils_square(x) + 
                                             sf_math_utils_square(y) + 
                                             sf_math_utils_square(z));
            r.x = 2.0f * w * x; 
            r.y = 2.0f * w * y; 
            r.z = 2.0f * w * z;
            return r;
        }

        [[nodiscard]] inline f32 dot() const noexcept {
            f32 r = sf_math_utils_square(x) + 
                    sf_math_utils_square(y) + 
                    sf_math_utils_square(z) + 
                    sf_math_utils_square(w);
            return r;
        }

        [[nodiscard]] inline f32 length() const noexcept {
            f32 r;
            r = std::sqrt(sf_math_utils_square(x) + 
                          sf_math_utils_square(y) + 
                          sf_math_utils_square(z) + 
                          sf_math_utils_square(w));
            return r;
        }

        [[nodiscard]] inline f32 normalize() {
            f32 mag = std::sqrt(sf_math_utils_square(x) +
                                sf_math_utils_square(y) +
                                sf_math_utils_square(z) +
                                sf_math_utils_square(w));
            if (mag != 0.0f) { 
                x /= mag; 
                y /= mag; 
                z /= mag; 
                w /= mag; 
            } else { 
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                w = 0.0f; 
            }
            return mag;
        }

        [[nodiscard]] inline quat conjugate() const noexcept {
            return quat(w, -x, -y, -z);
        }

        [[nodiscard]] inline quat inverse() const noexcept {
            return quat(w / this->dot(), 
                       -x / this->dot(),
                       -y / this->dot(),
                       -z / this->dot());
        }

        /* Quaternion must be normalized before invoking roll(), pitch(), and
         * yaw(). */

        [[nodiscard]] inline f32 roll() noexcept {
            f32 x_axis = 1.0f - (2.0f * (sf_math_utils_square(x) + sf_math_utils_square(z)));
            f32 y_axis = 2.0f * (w * x - y * z);
            if ((x_axis == 0.0f) && 
                (y_axis == 0.0f)) { 
                return 0.0f; 
            }
            return std::atan2(x_axis, x_axis);
        }

        [[nodiscard]] inline f32 pitch() noexcept {
            f32 v = 2.0f * (x * y + z * w);
            return std::asin((v < -1.0f) ? -1.0f : 
                             (v >  1.0f) ?  1.0f : v);
        }

        [[nodiscard]] inline f32 yaw() noexcept {
            f32 x_axis = 1.0f - (2.0f * sf_math_utils_square(y) + sf_math_utils_square(z));
            f32 y_axis = 2.0f * (w * y - x * z);
            if ((x_axis == 0.0f) && (y_axis == 0.0f)) { 
                return((2.0f * std::atan2(x,w))); 
            }
            return std::atan2(y_axis, x_axis);
        }

        [[nodiscard]] constexpr inline f32 
        operator[] 
        (i32 i) const {
            return V[i];
        }

        [[nodiscard]] constexpr inline f32& 
        operator[] 
        (i32 i) {
            return V[i];
        }
    };

/*----------------------*/
/* Quaternion Overloads */
/*----------------------*/

[[nodiscard]] sf_inline quat 
operator + 
(const quat& lhs, const quat& rhs) {
    return quat(lhs.w + rhs.w, 
                lhs.x + rhs.x, 
                lhs.y + rhs.y, 
                lhs.z + rhs.z);
}

[[nodiscard]] sf_inline quat& 
operator += 
(quat& lhs, const quat& rhs) {
    lhs.x += rhs.x; 
    lhs.y += rhs.y; 
    lhs.z += rhs.z; 
    lhs.w += rhs.w;
    return lhs;
}

[[nodiscard]] sf_inline quat 
operator - 
(const quat& rhs) {
    quat c; 
    c.x = -rhs.x; 
    c.y = -rhs.y; 
    c.z = -rhs.z; 
    c.w = -rhs.w; 
    return c;
}

[[nodiscard]] sf_inline quat 
operator - 
(const quat& lhs, const quat& rhs) {
    quat c; 
    c.x = lhs.x - rhs.x; 
    c.y = lhs.y - rhs.y; 
    c.z = lhs.z - rhs.z; 
    c.w = lhs.w - rhs.w;
    return c;
}

[[nodiscard]] sf_inline quat& 
operator -= 
(quat& lhs, const quat& rhs) {
    lhs.x -= rhs.x; 
    lhs.y -= rhs.y; 
    lhs.z -= rhs.z; 
    lhs.w -= rhs.w;
    return lhs;
}

[[nodiscard]] sf_inline quat 
operator * 
(const quat& lhs, const quat& rhs) {
    quat c;
    c.w = lhs.w * rhs.w - lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    c.x = lhs.w * rhs.x + rhs.w * lhs.x + lhs.y * rhs.z - lhs.z * rhs.y;
    c.y = lhs.w * rhs.y + rhs.w * lhs.y + lhs.z * rhs.x - lhs.x * rhs.z;
    c.z = lhs.w * rhs.z + rhs.w * lhs.z + lhs.x * rhs.y - lhs.y * rhs.x;
    return c; 
}

[[nodiscard]] sf_inline quat 
operator * 
(const f32& lhs, const quat& rhs) {
    quat c;
    c.x=rhs.x*lhs; 
    c.y=rhs.y*lhs; 
    c.z=rhs.z*lhs; 
    c.w=rhs.w*lhs;
    return c;
}

[[nodiscard]] sf_inline quat 
operator * 
(const quat& lhs, const f32& rhs) {
    return rhs * lhs;
}

[[nodiscard]] sf_inline vec3 
operator * 
(const quat& lhs, const vec3& rhs) {
    vec3 qv(lhs.x, lhs.y, lhs.z);
    vec3 uv  = cross_product(qv, rhs);
    vec3 uuv = cross_product(qv, uv);
    return(rhs + ((uv * lhs.w) + uuv) * 2.0f);
}

[[nodiscard]] sf_inline quat& 
operator *= 
(quat& lhs, const quat& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

[[nodiscard]] sf_inline quat& 
operator *= 
(quat& lhs, const f32& rhs) {
    lhs.x *= rhs; 
    lhs.y *= rhs; 
    lhs.z *= rhs; 
    lhs.w *= rhs;
    return lhs;
}

[[nodiscard]] sf_inline bool 
operator == 
(const quat& lhs, const quat& rhs) {
    return((lhs.x == rhs.x) &&
           (lhs.y == rhs.y) &&
           (lhs.z == rhs.z) &&
           (lhs.w == rhs.w));
}

[[nodiscard]] sf_inline bool 
operator != 
(const quat& lhs, const quat& rhs) {
    return((lhs.x != rhs.x) ||
           (lhs.y != rhs.y) ||
           (lhs.z != rhs.z) ||
           (lhs.w != rhs.w));
}


[[nodiscard]] sf_inline std::ostream& 
operator << 
(std::ostream& os, const quat& rhs) {
    os << "(" << rhs.w << "+" 
              << rhs.x << "i+" 
              << rhs.y << "j+" 
              << rhs.z << "k)";
    return os;
}

/*----------------------*/
/* Quaternion Functions */
/*----------------------*/

[[nodiscard]] sf_inline vec3 
cross_product(const vec3& v, const quat& q) {
    return q.inverse() * v;
}

[[nodiscard]] sf_inline vec3 
cross_product(const quat& q, const vec3& v) {
    return q * v;
}

[[nodiscard]] sf_inline quat 
cross_product(const quat& q1, const quat& q2) {
    quat r;
    r.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    r.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    r.y = q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z;
    r.z = q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x;
    return r;
}

[[nodiscard]] sf_inline quat 
rotate(const quat& q, const f32& a, const vec3& v) {
    f32  s = std::sin(a * 0.5f);
    quat r = q * quat(std::cos(a * 0.5f), v.x * s, v.y * s, v.z * s);
    return r;
}

[[nodiscard]] sf_inline vec3 
rotate_point(const vec3& p, const f32& a, const vec3& v) {
    f32 s = std::sin(a * 0.5f);
    quat q = quat(std::cos(a * 0.5f), v.x * s, v.y * s, v.z * s);
    quat point(0.0, p.x, p.y, p.z);
    quat qn = q * point * q.conjugate();
    vec3 r(qn.x, qn.y, qn.z);
    return r;
}

sf_inline void 
axis_angle(const quat& q, vec3& axis, f32& theta) {
    f32 mag = std::sqrt(sf_math_utils_square(q.x) +
                        sf_math_utils_square(q.y) + 
                        sf_math_utils_square(q.z));
    theta = 2.0f * std::atan2(mag, q.w);
    axis  = vec3(q.x, q.y, q.z) / mag;
}

[[nodiscard]] sf_inline quat 
lerp(const quat& q1, const quat& q2, const f32& t) {
    return 1.0f - t * (q1 + t * q2);
}

} // namespace sf
