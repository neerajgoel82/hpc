// math_utils.h
// Header file example: declarations only
// This is the interface that users see

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// Header guard prevents multiple inclusion
// Alternative: #pragma once (modern, but not standard)

// ===== Function Declarations =====
int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
float divide(float a, float b);

// ===== Inline Functions (defined in header) =====
inline int square(int x) {
    return x * x;
}

inline int cube(int x) {
    return x * x * x;
}

// ===== Struct Definition (in header) =====
struct Vector2D {
    float x;
    float y;
};

// ===== Function declarations for struct operations =====
float length(const Vector2D& v);
Vector2D add(const Vector2D& a, const Vector2D& b);
Vector2D subtract(const Vector2D& a, const Vector2D& b);
float dot(const Vector2D& a, const Vector2D& b);
void print(const Vector2D& v);

// ===== Constants (in header with inline or constexpr) =====
constexpr float PI = 3.14159265359f;
constexpr float E = 2.71828182846f;

// ===== Inline helper functions =====
inline float radToDeg(float rad) {
    return rad * 180.0f / PI;
}

inline float degToRad(float deg) {
    return deg * PI / 180.0f;
}

#endif // MATH_UTILS_H

/*
HEADER FILE BEST PRACTICES:

1. ALWAYS use header guards or #pragma once
2. Declarations only (usually)
3. Inline functions OK in headers
4. constexpr functions must be in headers
5. Template functions must be in headers
6. Don't put "using namespace" in headers
7. Include only what you need
8. Use forward declarations when possible

WHAT GOES IN HEADERS:
✓ Function declarations
✓ Class/struct definitions
✓ Template definitions
✓ Inline functions
✓ constexpr functions
✓ Constants (const/constexpr)
✓ Type aliases (using, typedef)
✓ Enums

WHAT GOES IN .cpp FILES:
✓ Function definitions (non-inline)
✓ Global variable definitions
✓ Static variables
✓ Implementation details

GPU/CUDA:
- .cuh files for CUDA headers
- __device__ functions often in headers (inline)
- Kernel declarations in headers
- Similar rules apply
*/
