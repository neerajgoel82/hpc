/*
 * Homework: 02_raii.cpp
 *
 * Complete the exercises below based on the concepts from 02_raii.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 02_raii.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Create a LockGuard class that acquires/releases a mutex
 * 2. Add error checking to GPUMemory (handle allocation failures)
 * 3. Create a BufferPool class that reuses GPU memory buffers
 * 4. Implement a TextureRAII class for OpenGL textures
 * 5. Add logging to track all resource acquisitions and releases
 * 6. Uncomment the throw statement to see exception safety in action
 *
 * COMMON MISTAKES:
 * - Forgetting to delete copy constructor/assignment (can cause double-free)
 * - Not checking if resource acquisition succeeded
 * - Mixing RAII with manual resource management
 * - Not making destructor virtual in base classes
 *
 * GPU CONNECTION:
 * - RAII is the GOLD STANDARD for GPU memory management
 * - Prevents memory leaks that crash GPU applications
 * - Modern libraries (Thrust, cuDNN) use RAII extensively
 * - Example: cudaMalloc in constructor, cudaFree in destructor
 *
 * REAL CUDA EXAMPLE PATTERN:
 *
 * class CudaMemory {
 *     void* d_ptr;
 *     size_t bytes;
 * public:
 *     CudaMemory(size_t n) : bytes(n) {
 *         cudaMalloc(&d_ptr, bytes);
 *     }
 *     ~CudaMemory() {
 *         cudaFree(d_ptr);
 *     }
 *     // ... methods ...
 * };
 */

int main() {
    std::cout << "Homework: 02_raii\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
