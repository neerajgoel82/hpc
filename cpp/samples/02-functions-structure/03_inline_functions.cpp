// 03_inline_functions.cpp
// Inline functions and when to use them
// Compile: g++ -std=c++17 -O2 -o inline_functions 03_inline_functions.cpp

#include <iostream>
#include <chrono>

// ===== Inline Function =====
// Hint to compiler to replace function call with function body
inline int square(int x) {
    return x * x;
}

inline float fastMult(float a, float b) {
    return a * b;
}

// ===== Regular Function (for comparison) =====
int squareNormal(int x) {
    return x * x;
}

// ===== Larger Function (poor inline candidate) =====
inline int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);  // Recursive, won't inline well
}

// ===== Class with Inline Members =====
class Vector2D {
private:
    float x, y;
public:
    // Constructor defined in class = implicitly inline
    Vector2D(float x_, float y_) : x(x_), y(y_) {}

    // Member functions defined in class are implicitly inline
    float getX() const { return x; }
    float getY() const { return y; }

    // Inline member function (explicit)
    inline float lengthSquared() const {
        return x * x + y * y;
    }

    // Larger function, not ideal for inline
    float length() const;  // Defined outside
};

// Define outside class (not inline unless specified)
float Vector2D::length() const {
    return std::sqrt(x * x + y * y);
}

// ===== constexpr implies inline =====
constexpr int cube(int x) {
    return x * x * x;
}

// ===== Inline vs Macro =====
#define SQUARE_MACRO(x) ((x) * (x))  // Old C style, avoid!

inline int squareInline(int x) {
    return x * x;  // Type-safe, debuggable
}

// ===== GPU-style inline functions =====
inline float dot2D(float x1, float y1, float x2, float y2) {
    return x1 * x2 + y1 * y2;
}

inline float dot3D(float x1, float y1, float z1, float x2, float y2, float z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
}

// ===== Performance Test =====
void performanceTest() {
    const int iterations = 10000000;

    // Test inline function
    auto start = std::chrono::high_resolution_clock::now();
    volatile int sum1 = 0;  // volatile prevents optimization
    for (int i = 0; i < iterations; i++) {
        sum1 += square(i % 100);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test normal function
    start = std::chrono::high_resolution_clock::now();
    volatile int sum2 = 0;
    for (int i = 0; i < iterations; i++) {
        sum2 += squareNormal(i % 100);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Inline function time: " << duration1.count() << " μs" << std::endl;
    std::cout << "Normal function time: " << duration2.count() << " μs" << std::endl;
    std::cout << "Note: With -O2 optimization, compiler may inline both!" << std::endl;
}

int main() {
    // ===== Basic Usage =====
    int x = 5;
    std::cout << "square(" << x << ") = " << square(x) << std::endl;
    std::cout << "cube(" << x << ") = " << cube(x) << std::endl;
    std::cout << std::endl;

    // ===== Class Usage =====
    Vector2D v(3.0f, 4.0f);
    std::cout << "Vector: (" << v.getX() << ", " << v.getY() << ")" << std::endl;
    std::cout << "Length squared: " << v.lengthSquared() << std::endl;
    std::cout << "Length: " << v.length() << std::endl;
    std::cout << std::endl;

    // ===== Macro vs Inline =====
    int a = 5;
    std::cout << "Macro square: " << SQUARE_MACRO(a) << std::endl;
    std::cout << "Inline square: " << squareInline(a) << std::endl;

    // Macro problem:
    int result = SQUARE_MACRO(a++);  // Expands to ((a++) * (a++)) - WRONG!
    std::cout << "Macro with a++: " << result << ", a = " << a << std::endl;

    int b = 5;
    result = squareInline(b++);  // Correct: b++ evaluated once
    std::cout << "Inline with b++: " << result << ", b = " << b << std::endl;
    std::cout << std::endl;

    // ===== Performance Test =====
    std::cout << "Performance comparison:" << std::endl;
    performanceTest();

    return 0;
}

/*
LEARNING NOTES:

INLINE KEYWORD:
- Hint to compiler to replace call with function body
- Reduces function call overhead
- Compiler may ignore hint (it's just a suggestion)
- With -O2 optimization, compiler inlines automatically

WHEN TO INLINE:
✓ Small, frequently called functions
✓ Simple getters/setters
✓ Math operations (add, multiply, etc.)
✓ Functions in tight loops
✗ Large functions
✗ Recursive functions
✗ Complex functions with loops

IMPLICIT INLINE:
- constexpr functions
- Functions defined inside class definition
- Template functions (in headers)

INLINE vs MACRO:
Macros (#define):
- Text substitution (preprocessor)
- No type checking
- Can cause unexpected behavior
- Hard to debug

Inline functions:
✓ Type-safe
✓ Debuggable
✓ Scoped
✓ Follow C++ rules

HEADER FILES:
- Inline functions can (and should) be in headers
- Regular functions in headers cause multiple definition errors
- Inline = "multiple definitions OK, use any one"

GPU RELEVANCE:
- GPU device functions often inline
- CUDA __forceinline__ for aggressive inlining
- Small math functions should be inline
- Critical for GPU performance (no function call overhead)

CUDA EXAMPLE:
__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

COMPILER OPTIMIZATION:
- Modern compilers inline automatically with -O2/-O3
- Link-time optimization (LTO) can inline across files
- inline keyword is mostly documentation now
- But still useful in headers

TRY THIS:
1. Create inline functions for min() and max()
2. Compare performance of inline vs non-inline (compile with -O0)
3. Create a class with inline getters/setters
4. Why would a recursive function be a poor inline candidate?
5. Create inline vector math functions (add, subtract, multiply)
*/
