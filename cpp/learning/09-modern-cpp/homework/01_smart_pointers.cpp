/*
 * Homework: 01_smart_pointers.cpp
 *
 * Complete the exercises below based on the concepts from 01_smart_pointers.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 01_smart_pointers.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Create texture cache using shared_ptr
 * 2. Implement scene graph with weak_ptr for parent references
 * 3. Add custom deleter for CUDA memory
 * 4. Build resource manager with smart pointers
 * 5. Implement observer pattern with weak_ptr
 *
 * GPU CONNECTION:
 * - Texture management: shared_ptr for cache
 * - Mesh data: unique_ptr for exclusive ownership
 * - Scene graph: weak_ptr for parent pointers
 * - GPU buffer RAII: unique_ptr with custom deleter
 * - Reference counting prevents memory leaks
 *
 * BEST PRACTICES:
 * - Default to unique_ptr
 * - Use shared_ptr when sharing is needed
 * - Use weak_ptr to break cycles
 * - Never manually delete smart pointer contents
 * - Prefer make_unique/make_shared
 */

int main() {
    std::cout << "Homework: 01_smart_pointers\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
