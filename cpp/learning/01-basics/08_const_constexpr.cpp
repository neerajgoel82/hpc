// 08_const_constexpr.cpp
// Understanding const and constexpr
// Compile: g++ -std=c++17 -o const_constexpr 08_const_constexpr.cpp

#include <iostream>

// ===== Compile-time Constants with constexpr =====
constexpr int BLOCK_SIZE = 256;          // GPU thread block size
constexpr float PI = 3.14159265359f;
constexpr int WARP_SIZE = 32;            // CUDA warp size (constant)

// constexpr function (evaluated at compile time if possible)
constexpr int square(int x) {
    return x * x;
}

constexpr int grid_size(int total, int block) {
    return (total + block - 1) / block;  // Ceiling division
}

int main() {
    // ===== const Variables =====
    const int max_threads = 1024;
    // max_threads = 2048;  // ERROR: cannot modify const

    const float gravity = 9.81f;
    std::cout << "Gravity: " << gravity << " m/s^2" << std::endl;

    // ===== const with Pointers (Tricky!) =====
    int value = 42;
    int other = 100;

    // Pointer to const int (cannot change the value through pointer)
    const int* ptr1 = &value;
    // *ptr1 = 50;  // ERROR: cannot modify
    ptr1 = &other;  // OK: can change where pointer points

    // Const pointer to int (cannot change where pointer points)
    int* const ptr2 = &value;
    *ptr2 = 50;     // OK: can modify the value
    // ptr2 = &other;  // ERROR: cannot change pointer

    // Const pointer to const int (cannot change either)
    const int* const ptr3 = &value;
    // *ptr3 = 50;     // ERROR
    // ptr3 = &other;  // ERROR

    std::cout << "Value after modification: " << value << std::endl;

    // ===== constexpr (Compile-Time Evaluation) =====
    constexpr int array_size = 100;
    int array[array_size];  // Array size must be known at compile time

    // constexpr functions evaluated at compile time
    constexpr int sq = square(10);  // Computed at compile time!
    std::cout << "Square of 10: " << sq << std::endl;

    // Can also call at runtime (acts like regular function)
    int x = 5;
    int sq_runtime = square(x);  // Computed at runtime
    std::cout << "Square of " << x << ": " << sq_runtime << std::endl;

    // ===== GPU-Related Example =====
    constexpr int TOTAL_ELEMENTS = 10000;
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int NUM_BLOCKS = grid_size(TOTAL_ELEMENTS, THREADS_PER_BLOCK);

    std::cout << "\nGPU Configuration:" << std::endl;
    std::cout << "Total elements: " << TOTAL_ELEMENTS << std::endl;
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Number of blocks: " << NUM_BLOCKS << std::endl;

    // ===== const References =====
    int num = 42;
    const int& ref = num;  // Reference to const
    // ref = 50;  // ERROR: cannot modify through const reference
    num = 50;  // OK: can modify original variable
    std::cout << "Num: " << num << ", Ref: " << ref << std::endl;

    // ===== Using const with Arrays =====
    const int primes[] = {2, 3, 5, 7, 11};
    std::cout << "First prime: " << primes[0] << std::endl;
    // primes[0] = 1;  // ERROR: array is const

    // ===== Practical Example: Circle Area =====
    constexpr float radius = 5.0f;
    constexpr float area = PI * square(radius);  // Computed at compile time!
    std::cout << "\nCircle with radius " << radius << " has area " << area << std::endl;

    return 0;
}

/*
LEARNING NOTES:

const:
- Value cannot be modified after initialization
- Runtime constant (value can be determined at runtime)
- Use for variables that shouldn't change

constexpr:
- Compile-time constant (value must be known at compile time)
- Can be used in places requiring compile-time values (array sizes, template parameters)
- constexpr functions can be evaluated at compile time
- If arguments are not constant, acts like regular function

POINTER SYNTAX (read right-to-left):
- const int* ptr      → pointer to const int
- int* const ptr      → const pointer to int
- const int* const ptr → const pointer to const int

WHEN TO USE:
- const: Values that don't change (configuration, constants)
- constexpr: Compile-time computations, array sizes, template parameters

BENEFITS:
- Prevents accidental modification
- Enables compiler optimizations
- Documents intent
- Allows compile-time computation

GPU RELEVANCE:
- GPU thread block sizes often constexpr (BLOCK_SIZE = 256)
- Warp size is compile-time constant (32 on NVIDIA GPUs)
- Grid dimensions can be computed at compile time
- CUDA constant memory stores const data
- Many GPU parameters are constexpr for performance

TRY THIS:
1. Create constexpr functions for: cube(x), factorial(n), power(base, exp)
2. What happens if you try to use a non-constexpr value as array size?
3. Declare a const pointer to a const float and try to modify both
4. Create a constexpr function to compute number of warps in a block
5. Why would you use constexpr instead of #define?

CONSTEXPR vs #define:
#define PI 3.14          // Preprocessor macro (text replacement, no type safety)
constexpr float PI = 3.14f;  // Type-safe, scoped constant (preferred)

CUDA EXAMPLE YOU'LL SEE:
constexpr int THREADS = 256;
kernel<<<NUM_BLOCKS, THREADS>>>(data);
*/
