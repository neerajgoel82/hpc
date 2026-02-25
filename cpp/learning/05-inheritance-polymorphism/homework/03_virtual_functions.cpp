/*
 * Homework: 03_virtual_functions.cpp
 *
 * Complete the exercises below based on the concepts from 03_virtual_functions.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 03_virtual_functions.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Add another level of inheritance and trace virtual calls
 * 2. Implement a visitor pattern using virtual functions
 * 3. Create a command pattern with virtual execute() method
 * 4. Test performance with deeper inheritance hierarchies
 * 5. Implement a factory pattern returning polymorphic objects
 * 6. Create an abstract Shader class with concrete implementations
 *
 * COMMON MISTAKES:
 * - Calling virtual functions in constructors (doesn't work as expected)
 * - Forgetting 'override' keyword (typo becomes hidden bug)
 * - Not understanding early vs late binding
 * - Excessive use of virtual functions where templates would be better
 *
 * GPU CONNECTION:
 * - CPU-side scene management uses virtual functions
 * - GPU kernels DON'T use virtual functions (no vtable support)
 * - GPU alternatives:
 *   * Templates for compile-time polymorphism
 *   * Switch statements on type enum
 *   * Separate arrays per type
 * - Virtual function overhead negligible for CPU prep work
 *
 * PERFORMANCE NOTES:
 * - Virtual call overhead: ~1-3 nanoseconds
 * - Usually negligible compared to actual work
 * - Can prevent some compiler optimizations (inlining)
 * - 'final' keyword can enable devirtualization
 * - Profile before optimizing!
 *
 * WHEN TO USE VIRTUAL:
 * - Polymorphic behavior needed at runtime
 * - Type not known at compile-time
 * - Clean interface for heterogeneous collections
 *
 * WHEN NOT TO USE VIRTUAL:
 * - Performance-critical tight loops (use templates)
 * - GPU device code (not supported)
 * - Type known at compile-time (use templates)
 */

int main() {
    std::cout << "Homework: 03_virtual_functions\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
