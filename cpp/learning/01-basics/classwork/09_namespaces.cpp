// 09_namespaces.cpp
// Organizing code with namespaces
// Compile: g++ -std=c++17 -o namespaces 09_namespaces.cpp

#include <iostream>

// ===== Defining a Namespace =====
namespace math {
    const float PI = 3.14159f;

    float square(float x) {
        return x * x;
    }

    float circle_area(float radius) {
        return PI * square(radius);
    }
}

// ===== Another Namespace (can have same function names!) =====
namespace physics {
    const float G = 9.81f;  // Gravity

    float kinetic_energy(float mass, float velocity) {
        return 0.5f * mass * velocity * velocity;
    }
}

// ===== Nested Namespaces =====
namespace gpu {
    namespace cuda {
        const int WARP_SIZE = 32;

        int calculate_blocks(int total, int threads_per_block) {
            return (total + threads_per_block - 1) / threads_per_block;
        }
    }

    // C++17 nested namespace syntax (cleaner)
    namespace opencl {
        const int WAVEFRONT_SIZE = 64;  // AMD terminology
    }
}

// ===== Anonymous/Unnamed Namespace (file scope) =====
namespace {
    // Only visible in this file (like 'static' in C)
    int internal_counter = 0;

    void increment_counter() {
        internal_counter++;
    }
}

int main() {
    // ===== Using Namespaces =====

    // Option 1: Full qualification
    std::cout << "PI = " << math::PI << std::endl;
    std::cout << "Square of 5 = " << math::square(5) << std::endl;

    // Option 2: Using declaration (for specific items)
    using math::circle_area;
    std::cout << "Circle area (r=3) = " << circle_area(3.0f) << std::endl;

    // Option 3: Using directive (brings entire namespace)
    using namespace physics;
    std::cout << "Kinetic energy = " << kinetic_energy(10.0f, 5.0f) << " J" << std::endl;

    // ===== Accessing Nested Namespaces =====
    std::cout << "\nGPU Info:" << std::endl;
    std::cout << "CUDA warp size: " << gpu::cuda::WARP_SIZE << std::endl;
    std::cout << "OpenCL wavefront size: " << gpu::opencl::WAVEFRONT_SIZE << std::endl;

    int blocks = gpu::cuda::calculate_blocks(1000, 256);
    std::cout << "Blocks needed: " << blocks << std::endl;

    // ===== Using Nested Namespace =====
    using namespace gpu::cuda;
    std::cout << "Warp size: " << WARP_SIZE << std::endl;

    // ===== Anonymous Namespace =====
    increment_counter();
    increment_counter();
    std::cout << "\nInternal counter: " << internal_counter << std::endl;

    // ===== std Namespace =====
    // std::cout is actually std namespace's cout object
    // std::string is std namespace's string class

    // You could do this (but NOT recommended for std):
    // using namespace std;
    // cout << "Now I don't need std::" << endl;  // AVOID IN REAL CODE!

    // Better: use specific declarations
    using std::cout;
    using std::endl;
    cout << "This is cleaner than 'using namespace std'" << endl;

    return 0;
}

/*
LEARNING NOTES:

NAMESPACE SYNTAX:
- namespace name { ... }  → Define namespace
- name::item             → Access namespace item
- using name::item;      → Bring specific item into scope
- using namespace name;  → Bring entire namespace into scope (use sparingly!)

WHY USE NAMESPACES:
- Avoid name conflicts (multiple libraries can have same names)
- Organize code logically
- Create library boundaries

BEST PRACTICES:
- Use full qualification (std::cout) in headers
- Use 'using' declarations in .cpp files if needed
- NEVER use "using namespace" in header files
- Avoid "using namespace std" in general (pollutes namespace)

NESTED NAMESPACES:
C++98/11/14: namespace A { namespace B { ... } }
C++17+:      namespace A::B { ... }  (cleaner)

ANONYMOUS NAMESPACE:
- namespace { ... }
- Items visible only in current file
- Alternative to 'static' for file-local scope

COMMON NAMESPACES YOU'LL SEE:
- std::     → Standard library (string, vector, cout, etc.)
- cuda::    → CUDA toolkit namespace (in modern CUDA)
- thrust::  → CUDA thrust library
- cv::      → OpenCV (computer vision)

GPU RELEVANCE:
- CUDA uses namespaces (cuda::std::, thrust::, cooperative_groups::)
- Organizing GPU utilities in namespaces
- Separating device and host code logically
- Third-party GPU libraries use namespaces

TRY THIS:
1. Create a namespace 'geometry' with functions for rectangle area and perimeter
2. Create nested namespaces: graphics::d2 and graphics::d3 with different shapes
3. What happens if two namespaces have functions with same name?
4. Why is "using namespace std;" considered bad practice in large projects?
5. Create a namespace for GPU memory operations (allocate, free, copy)

EXAMPLE STRUCTURE:
namespace myapp {
    namespace gpu {
        void init_device();
        void allocate_memory();
    }
    namespace cpu {
        void process_data();
    }
}

REAL CUDA EXAMPLE:
#include <cuda/std/complex>
using cuda::std::complex;  // Complex numbers on GPU
*/
