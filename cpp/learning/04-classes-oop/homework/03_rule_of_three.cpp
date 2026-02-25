/*
 * Homework: 03_rule_of_three.cpp
 *
 * Complete the exercises below based on the concepts from 03_rule_of_three.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 03_rule_of_three.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Uncomment the BadVector example and see the crash
 * 2. Add move constructor and move assignment (Rule of Five)
 * 3. Implement a Matrix class with proper Rule of Three
 * 4. Add a swap() method and implement copy-and-swap idiom
 * 5. Create a reference-counted smart pointer class
 * 6. Profile the cost of copying large GPU buffers
 *
 * COMMON MISTAKES:
 * - Forgetting to check for self-assignment
 * - Shallow copy when deep copy is needed
 * - Not returning *this from assignment operator
 * - Memory leak in assignment (forgetting to free old memory)
 *
 * GPU CONNECTION:
 * - GPU memory can't be implicitly copied
 * - cudaMemcpy is explicit, expensive operation
 * - Modern CUDA code often uses move semantics instead (Module 9)
 * - Understanding copy costs is crucial for GPU performance
 * - Many GPU classes disable copying entirely (delete copy operations)
 *
 * MODERN C++ NOTE:
 * - C++11 adds Rule of Five (add move constructor & move assignment)
 * - Modern code often uses unique_ptr/shared_ptr instead of raw pointers
 * - Move semantics (Module 9) often preferred over copying for GPU data
 */

int main() {
    std::cout << "Homework: 03_rule_of_three\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
