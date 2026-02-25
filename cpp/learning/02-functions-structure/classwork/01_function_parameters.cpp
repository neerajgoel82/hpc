// 01_function_parameters.cpp
// Understanding different ways to pass parameters
// Compile: g++ -std=c++17 -o function_parameters 01_function_parameters.cpp

#include <iostream>
#include <string>

// ===== Pass by Value =====
// Creates a copy of the argument
void passByValue(int x) {
    x = 100;  // Only modifies the copy
    std::cout << "Inside passByValue: x = " << x << std::endl;
}

// ===== Pass by Reference =====
// Works with the original variable (no copy)
void passByReference(int& x) {
    x = 100;  // Modifies the original
    std::cout << "Inside passByReference: x = " << x << std::endl;
}

// ===== Pass by Const Reference =====
// Read-only access, no copy (efficient for large objects)
void passByConstReference(const std::string& str) {
    // str = "new value";  // ERROR: cannot modify
    std::cout << "String: " << str << std::endl;
}

// ===== Pass by Pointer =====
// Can modify through pointer, can be nullptr
void passByPointer(int* ptr) {
    if (ptr != nullptr) {
        *ptr = 200;  // Modify through pointer
    }
}

// ===== Multiple parameters with different types =====
void compute(int value, int& result, const int& multiplier) {
    result = value * multiplier;
}

// ===== Large struct example =====
struct ParticleData {
    float position[3];
    float velocity[3];
    float mass;
    int id;
};

// Inefficient: copies entire struct (32 bytes)
void processParticleCopy(ParticleData particle) {
    particle.mass *= 2.0f;  // Only modifies copy
}

// Efficient: uses reference (no copy)
void processParticleRef(ParticleData& particle) {
    particle.mass *= 2.0f;  // Modifies original
}

// Read-only access, no copy
void printParticle(const ParticleData& particle) {
    std::cout << "Particle ID: " << particle.id
              << ", Mass: " << particle.mass << std::endl;
}

int main() {
    // ===== Pass by Value Demo =====
    int a = 10;
    std::cout << "Before passByValue: a = " << a << std::endl;
    passByValue(a);
    std::cout << "After passByValue: a = " << a << std::endl;
    std::cout << std::endl;

    // ===== Pass by Reference Demo =====
    int b = 10;
    std::cout << "Before passByReference: b = " << b << std::endl;
    passByReference(b);
    std::cout << "After passByReference: b = " << b << std::endl;
    std::cout << std::endl;

    // ===== Const Reference Demo =====
    std::string message = "Hello GPU Programming";
    passByConstReference(message);
    std::cout << std::endl;

    // ===== Pointer Demo =====
    int c = 10;
    std::cout << "Before passByPointer: c = " << c << std::endl;
    passByPointer(&c);  // Pass address of c
    std::cout << "After passByPointer: c = " << c << std::endl;

    // Can pass nullptr
    passByPointer(nullptr);  // Safe because we check inside
    std::cout << std::endl;

    // ===== Multiple parameters =====
    int value = 5;
    int result;
    int multiplier = 7;
    compute(value, result, multiplier);
    std::cout << value << " * " << multiplier << " = " << result << std::endl;
    std::cout << std::endl;

    // ===== Struct passing comparison =====
    ParticleData particle = {{1.0f, 2.0f, 3.0f}, {0.1f, 0.2f, 0.3f}, 5.0f, 42};

    std::cout << "Original mass: " << particle.mass << std::endl;
    processParticleCopy(particle);  // Copy, no change
    std::cout << "After copy: " << particle.mass << std::endl;

    processParticleRef(particle);  // Reference, modifies
    std::cout << "After reference: " << particle.mass << std::endl;

    printParticle(particle);

    return 0;
}

/*
LEARNING NOTES:

PASS BY VALUE:
- Creates a copy
- Original unchanged
- Use for small types (int, float, char)
- Overhead for large objects

PASS BY REFERENCE (&):
- No copy created
- Can modify original
- Must pass valid object (can't be null)
- Use for output parameters

PASS BY CONST REFERENCE (const &):
- No copy created
- Cannot modify
- Use for large input objects (strings, vectors, structs)
- Best for read-only access to large data

PASS BY POINTER (*):
- Can be nullptr
- Can modify through pointer
- Legacy style, references preferred in modern C++
- Use when nullptr is valid

WHEN TO USE WHAT:
- Small types (int, float): by value
- Large types, read-only: const reference
- Large types, modify: reference
- Optional parameter: pointer (or std::optional)
- Arrays: pointer or reference to array

GPU RELEVANCE:
- Understanding pass-by-reference critical for GPU memory
- GPU kernels receive pointers to device memory
- Large data structures (particle arrays) passed by pointer/reference
- Avoiding copies is critical for performance

CUDA EXAMPLE:
__global__ void kernel(float* data, int size) {
    // data is pointer to GPU memory
}

TRY THIS:
1. Create a function that swaps two integers using references
2. Create a function that takes a const reference to a vector
3. What happens if you pass nullptr to passByReference?
4. Why is const reference preferred over pass-by-value for strings?
5. Create a function with mix of value, reference, and pointer parameters
*/
