/*
 * Homework: 08_const_constexpr.cpp
 *
 * Complete the exercises below based on the concepts from 08_const_constexpr.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 08_const_constexpr.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * 1. Create constexpr functions for: cube(x), factorial(n), power(base, exp)
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

int main() {
    std::cout << "Homework: 08_const_constexpr\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
