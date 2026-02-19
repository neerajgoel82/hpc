/*
 * MODULE 7: Templates
 * File: 01_function_templates.cpp
 *
 * TOPIC: Function Templates - Generic Programming
 *
 * CONCEPTS:
 * - Template syntax and instantiation
 * - Type deduction
 * - Template specialization
 * - Multiple template parameters
 *
 * GPU RELEVANCE:
 * - Generic vector/matrix operations for different types (float, double, int)
 * - Thrust library uses templates extensively
 * - Generic GPU kernels (template kernels)
 * - Type-safe GPU code without runtime overhead
 *
 * COMPILE: g++ -std=c++17 -o function_templates 01_function_templates.cpp
 */

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

// Example 1: Basic function template
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Example 2: Template with multiple parameters
template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// Example 3: Template for swapping
template <typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// Example 4: Generic vector operations
template <typename T>
struct Vec3T {
    T x, y, z;

    Vec3T(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) {}

    Vec3T operator+(const Vec3T& v) const {
        return Vec3T(x + v.x, y + v.y, z + v.z);
    }

    Vec3T operator-(const Vec3T& v) const {
        return Vec3T(x - v.x, y - v.y, z - v.z);
    }

    Vec3T operator*(T scalar) const {
        return Vec3T(x * scalar, y * scalar, z * scalar);
    }

    T dot(const Vec3T& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    T length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    void print() const {
        std::cout << "(" << x << ", " << y << ", " << z << ")";
    }
};

// Type aliases for common uses
using Vec3f = Vec3T<float>;
using Vec3d = Vec3T<double>;
using Vec3i = Vec3T<int>;

// Example 5: Function template with non-type parameter
template <typename T, int N>
T sumArray(T (&arr)[N]) {
    T sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return sum;
}

// Example 6: Template specialization
template <typename T>
void printType(T value) {
    std::cout << "Generic type: " << value << "\n";
}

// Specialized version for const char*
template <>
void printType<const char*>(const char* value) {
    std::cout << "C-string: \"" << value << "\"\n";
}

// Example 7: Template for finding minimum in array
template <typename T>
T findMin(const T* arr, int size) {
    T minVal = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] < minVal) {
            minVal = arr[i];
        }
    }
    return minVal;
}

// Example 8: Generic lerp (linear interpolation)
// Note: T must support +, -, and * operations
template <typename T, typename U>
auto lerp(const T& a, const T& b, U t) -> T {
    T diff = b - a;
    T scaled = diff * t;
    return a + scaled;
}

// Example 9: Template with multiple type parameters
template <typename T, typename Func>
void forEach(T* arr, int size, Func operation) {
    for (int i = 0; i < size; ++i) {
        operation(arr[i]);
    }
}

void demonstrateBasicTemplates() {
    std::cout << "\n=== Basic Function Templates ===\n";

    std::cout << "max(5, 10) = " << max(5, 10) << "\n";
    std::cout << "max(3.14, 2.71) = " << max(3.14, 2.71) << "\n";
    std::cout << "max('a', 'z') = " << max('a', 'z') << "\n";

    // Mixed types
    std::cout << "\nadd(5, 3.14) = " << add(5, 3.14) << "\n";
    std::cout << "add(2.5f, 7) = " << add(2.5f, 7) << "\n";

    // Swap
    int x = 10, y = 20;
    std::cout << "\nBefore swap: x=" << x << ", y=" << y << "\n";
    swap(x, y);
    std::cout << "After swap: x=" << x << ", y=" << y << "\n";
}

void demonstrateGenericVectors() {
    std::cout << "\n=== Generic Vector Operations ===\n";

    Vec3f vf(1.0f, 2.0f, 3.0f);
    Vec3d vd(1.0, 2.0, 3.0);
    Vec3i vi(1, 2, 3);

    std::cout << "Float vector: ";
    vf.print();
    std::cout << ", length = " << vf.length() << "\n";

    std::cout << "Double vector: ";
    vd.print();
    std::cout << ", length = " << vd.length() << "\n";

    std::cout << "Int vector: ";
    vi.print();
    std::cout << "\n";

    Vec3f vf2(4.0f, 5.0f, 6.0f);
    Vec3f sum = vf + vf2;
    std::cout << "\nSum: ";
    sum.print();
    std::cout << "\nDot product: " << vf.dot(vf2) << "\n";
}

void demonstrateArrayTemplates() {
    std::cout << "\n=== Array Templates ===\n";

    int intArr[] = {1, 2, 3, 4, 5};
    float floatArr[] = {1.1f, 2.2f, 3.3f};

    std::cout << "Sum of int array: " << sumArray(intArr) << "\n";
    std::cout << "Sum of float array: " << sumArray(floatArr) << "\n";

    std::cout << "\nMin of int array: " << findMin(intArr, 5) << "\n";
    std::cout << "Min of float array: " << findMin(floatArr, 3) << "\n";
}

void demonstrateSpecialization() {
    std::cout << "\n=== Template Specialization ===\n";

    printType(42);
    printType(3.14);
    printType("Hello");  // Uses specialized version
}

void demonstrateLerp() {
    std::cout << "\n=== Generic Lerp (Linear Interpolation) ===\n";

    float a = 0.0f, b = 10.0f;
    std::cout << "lerp(0, 10, 0.5) = " << lerp(a, b, 0.5f) << "\n";
    std::cout << "lerp(0, 10, 0.25) = " << lerp(a, b, 0.25f) << "\n";

    Vec3f v1(0, 0, 0);
    Vec3f v2(10, 20, 30);
    Vec3f mid = lerp(v1, v2, 0.5f);
    std::cout << "\nVector lerp: ";
    mid.print();
    std::cout << "\n";
}

void demonstrateForEach() {
    std::cout << "\n=== Template with Function Parameter ===\n";

    int arr[] = {1, 2, 3, 4, 5};

    std::cout << "Printing each element:\n";
    forEach(arr, 5, [](int x) {
        std::cout << "  " << x << "\n";
    });

    std::cout << "\nDoubling each element:\n";
    forEach(arr, 5, [](int& x) {
        x *= 2;
    });

    forEach(arr, 5, [](int x) {
        std::cout << "  " << x << "\n";
    });
}

// GPU-relevant example: Generic kernel-style function
template <typename T>
void parallelMultiply(T* arr, int size, T multiplier) {
    // In GPU code, this would be: __global__ void kernel(T* arr, ...)
    // Each element would be processed by a different thread
    for (int i = 0; i < size; ++i) {
        arr[i] *= multiplier;
    }
}

void demonstrateGPUStyle() {
    std::cout << "\n=== GPU-Style Generic Operations ===\n";

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    std::cout << "Before: ";
    for (float x : data) std::cout << x << " ";
    std::cout << "\n";

    parallelMultiply(data, 5, 2.0f);

    std::cout << "After multiplying by 2: ";
    for (float x : data) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "\nGPU CONNECTION: CUDA kernels can be templated!\n";
    std::cout << "Example: template<typename T> __global__ void kernel(T* data)\n";
}

int main() {
    std::cout << "=== MODULE 7: Function Templates Demo ===\n";

    demonstrateBasicTemplates();
    demonstrateGenericVectors();
    demonstrateArrayTemplates();
    demonstrateSpecialization();
    demonstrateLerp();
    demonstrateForEach();
    demonstrateGPUStyle();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS:
 *
 * 1. TEMPLATE SYNTAX:
 *    template <typename T>
 *    T functionName(T param) { ... }
 *
 * 2. TYPE DEDUCTION:
 *    Compiler automatically deduces T from arguments
 *    max(5, 10) -> T = int
 *    max(3.14, 2.71) -> T = double
 *
 * 3. ADVANTAGES:
 *    - Code reuse without runtime overhead
 *    - Type safety at compile time
 *    - Zero-cost abstraction
 *
 * 4. MULTIPLE PARAMETERS:
 *    template <typename T, typename U>
 *    template <typename T, int N>  // Non-type parameter
 *
 * TRY THIS:
 * 1. Create template for 4x4 matrix with any numeric type
 * 2. Implement generic clamp(value, min, max)
 * 3. Add template for computing array average
 * 4. Create generic normalize function for vectors
 * 5. Implement template for dot product
 * 6. Add bounds-checking template wrapper
 *
 * GPU CONNECTION:
 * - CUDA kernels can be templated
 * - Thrust library is fully templated
 * - Type-generic GPU algorithms
 * - No runtime overhead
 * - Compile-time polymorphism (vs virtual functions)
 */
