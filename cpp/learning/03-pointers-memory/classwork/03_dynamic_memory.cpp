// 03_dynamic_memory.cpp
// Dynamic memory allocation with new/delete
// Compile: g++ -std=c++17 -o dynamic_memory 03_dynamic_memory.cpp

#include <iostream>

int main() {
    // ===== Stack vs Heap =====
    int stackVar = 10;  // Stack: automatic, limited size, fast
    std::cout << "Stack variable: " << stackVar << std::endl;

    int* heapVar = new int(20);  // Heap: manual, large size, slower
    std::cout << "Heap variable: " << *heapVar << std::endl;
    delete heapVar;  // MUST free heap memory
    heapVar = nullptr;  // Good practice
    std::cout << std::endl;

    // ===== Single Object Allocation =====
    int* ptr = new int;      // Allocate uninitialized
    *ptr = 42;
    std::cout << "*ptr = " << *ptr << std::endl;
    delete ptr;

    int* ptr2 = new int(100);  // Allocate with value
    std::cout << "*ptr2 = " << *ptr2 << std::endl;
    delete ptr2;

    int* ptr3 = new int{200};  // C++11 uniform initialization
    std::cout << "*ptr3 = " << *ptr3 << std::endl;
    delete ptr3;
    std::cout << std::endl;

    // ===== Array Allocation =====
    int size = 5;
    int* arr = new int[size];  // Allocate array

    // Initialize array
    for (int i = 0; i < size; i++) {
        arr[i] = i * 10;
    }

    std::cout << "Dynamic array: ";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    delete[] arr;  // MUST use delete[] for arrays, not delete!
    arr = nullptr;
    std::cout << std::endl;

    // ===== Large Allocation =====
    int n = 1000000;  // 1 million ints
    int* largeArray = new int[n];

    std::cout << "Allocated " << n << " ints ("
              << (n * sizeof(int)) / (1024 * 1024) << " MB)" << std::endl;

    // Use array
    largeArray[0] = 1;
    largeArray[n - 1] = 999999;

    std::cout << "First element: " << largeArray[0] << std::endl;
    std::cout << "Last element: " << largeArray[n - 1] << std::endl;

    delete[] largeArray;
    std::cout << std::endl;

    // ===== Struct Allocation =====
    struct Particle {
        float x, y, z;
        float vx, vy, vz;
        int id;
    };

    Particle* particle = new Particle;
    particle->x = 1.0f;
    particle->y = 2.0f;
    particle->z = 3.0f;
    particle->id = 42;

    std::cout << "Particle ID: " << particle->id << std::endl;
    std::cout << "Position: (" << particle->x << ", "
              << particle->y << ", " << particle->z << ")" << std::endl;

    delete particle;
    std::cout << std::endl;

    // ===== Array of Structs =====
    int numParticles = 3;
    Particle* particles = new Particle[numParticles];

    for (int i = 0; i < numParticles; i++) {
        particles[i].x = static_cast<float>(i);
        particles[i].y = static_cast<float>(i * 2);
        particles[i].z = 0.0f;
        particles[i].id = i;
    }

    std::cout << "Particles:" << std::endl;
    for (int i = 0; i < numParticles; i++) {
        std::cout << "  Particle " << particles[i].id
                  << ": (" << particles[i].x << ", "
                  << particles[i].y << ", "
                  << particles[i].z << ")" << std::endl;
    }

    delete[] particles;
    std::cout << std::endl;

    // ===== Memory Leak Example (DON'T DO THIS!) =====
    void* leak = new int[100];  // Allocated but never deleted
    // Memory leak! This memory is lost until program ends

    // ===== Double Delete (DON'T DO THIS!) =====
    // int* ptr = new int(5);
    // delete ptr;
    // delete ptr;  // CRASH! Double delete is undefined behavior

    // ===== Null Pointer Safety =====
    int* safePtr = nullptr;
    delete safePtr;  // Safe: deleting nullptr does nothing
    std::cout << "Deleting nullptr is safe" << std::endl;

    return 0;
}

/*
LEARNING NOTES:

STACK MEMORY:
- Automatic allocation/deallocation
- Limited size (~1-8 MB)
- Fast access
- Destroyed when out of scope
- Use for small, short-lived objects

HEAP MEMORY:
- Manual allocation/deallocation
- Large size (GB)
- Slower access
- Persists until freed
- Use for large or long-lived objects

NEW/DELETE OPERATORS:
new Type           - Allocate single object
new Type[size]     - Allocate array
delete ptr         - Free single object
delete[] arr       - Free array

CRITICAL RULES:
1. Every 'new' needs a 'delete'
2. Every 'new[]' needs a 'delete[]'
3. Never 'delete' stack variables
4. Never 'delete' same pointer twice
5. Set pointers to nullptr after delete

COMMON ERRORS:
Memory Leak:
  ptr = new int;
  // Forgot to delete!

Wrong Delete:
  arr = new int[10];
  delete arr;  // WRONG! Should be delete[]

Dangling Pointer:
  delete ptr;
  *ptr = 5;  // CRASH! ptr is dangling

Double Delete:
  delete ptr;
  delete ptr;  // CRASH!

MEMORY LEAK DETECTION:
- Valgrind (Linux/Mac)
- AddressSanitizer
- Visual Studio memory profiler
- Always test for leaks!

GPU RELEVANCE - CRITICAL!:
GPU memory is heap memory!

cudaMalloc(&d_ptr, size);     // Like 'new' for GPU
cudaMemcpy(d_ptr, h_ptr, ...); // Copy to GPU
kernel<<<...>>>(d_ptr);        // Use in kernel
cudaFree(d_ptr);               // Like 'delete' for GPU

GPU memory management:
- Explicit allocation (cudaMalloc)
- Explicit deallocation (cudaFree)
- Manual copy (cudaMemcpy)
- Memory leaks are easy!
- Understanding new/delete essential

MODERN C++ ALTERNATIVE:
Use smart pointers (Module 9) to avoid manual memory management:
std::unique_ptr<int> ptr = std::make_unique<int>(42);
// Automatically deleted, no leaks!

TRY THIS:
1. Allocate array, double all values, print, free
2. Create a memory leak intentionally and run Valgrind
3. What happens if you delete[] a variable allocated with new?
4. Allocate a 2D array dynamically
5. Create a function that returns a dynamically allocated array
*/
