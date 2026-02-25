/*
 * MODULE 4: Classes and Object-Oriented Programming
 * File: 03_rule_of_three.cpp
 *
 * TOPIC: Rule of Three (Copy Constructor, Copy Assignment, Destructor)
 *
 * CONCEPTS:
 * - If a class needs a custom destructor, it probably needs:
 *   1. Destructor
 *   2. Copy constructor
 *   3. Copy assignment operator
 * - Deep copy vs shallow copy
 * - Resource ownership and copying
 *
 * GPU RELEVANCE:
 * - GPU memory can't be naively copied (requires explicit copies)
 * - Understanding deep/shallow copy prevents GPU memory corruption
 * - Particle systems, mesh data often need proper copying
 * - Texture and buffer objects need careful copy semantics
 *
 * COMPILE: g++ -std=c++17 -o rule_of_three 03_rule_of_three.cpp
 */

#include <iostream>
#include <cstring>
#include <algorithm>

// BAD EXAMPLE: Missing copy operations (dangerous!)
class BadVector {
private:
    float* data;
    size_t size;

public:
    BadVector(size_t n) : size(n) {
        data = new float[size];
        std::cout << "BadVector allocated " << size << " floats at " << data << "\n";
    }

    ~BadVector() {
        std::cout << "BadVector deallocating at " << data << "\n";
        delete[] data;
    }

    // MISSING: Copy constructor and copy assignment!
    // Compiler generates shallow copies, leading to double-delete!

    void set(size_t index, float value) {
        if (index < size) data[index] = value;
    }

    float get(size_t index) const {
        return (index < size) ? data[index] : 0.0f;
    }
};

// GOOD EXAMPLE: Complete Rule of Three implementation
class Vector3D {
private:
    float* components;  // [x, y, z]
    static const size_t SIZE = 3;

public:
    // Constructor
    Vector3D(float x = 0.0f, float y = 0.0f, float z = 0.0f) {
        components = new float[SIZE];
        components[0] = x;
        components[1] = y;
        components[2] = z;
        std::cout << "Vector3D constructed: " << this << "\n";
    }

    // Destructor
    ~Vector3D() {
        std::cout << "Vector3D destroyed: " << this << " (data at " << components << ")\n";
        delete[] components;
    }

    // Copy Constructor (deep copy)
    Vector3D(const Vector3D& other) {
        std::cout << "Vector3D COPY CONSTRUCTOR: " << &other << " -> " << this << "\n";
        components = new float[SIZE];
        std::copy(other.components, other.components + SIZE, components);
    }

    // Copy Assignment Operator (deep copy)
    Vector3D& operator=(const Vector3D& other) {
        std::cout << "Vector3D COPY ASSIGNMENT: " << &other << " -> " << this << "\n";

        // Check for self-assignment
        if (this == &other) {
            std::cout << "  Self-assignment detected, skipping\n";
            return *this;
        }

        // Copy data
        std::copy(other.components, other.components + SIZE, components);

        return *this;
    }

    // Accessors
    float x() const { return components[0]; }
    float y() const { return components[1]; }
    float z() const { return components[2]; }

    void setX(float val) { components[0] = val; }
    void setY(float val) { components[1] = val; }
    void setZ(float val) { components[2] = val; }

    void print() const {
        std::cout << "Vector3D(" << x() << ", " << y() << ", " << z() << ")\n";
    }
};

// ADVANCED EXAMPLE: Rule of Three for GPU-like buffer
class GPUBuffer {
private:
    float* hostData;    // CPU memory
    size_t numElements;

public:
    // Constructor
    GPUBuffer(size_t n) : numElements(n) {
        hostData = new float[numElements];
        std::fill(hostData, hostData + numElements, 0.0f);
        std::cout << "GPUBuffer constructed: " << numElements << " elements\n";
    }

    // Destructor
    ~GPUBuffer() {
        std::cout << "GPUBuffer destroyed: " << numElements << " elements\n";
        delete[] hostData;
    }

    // Copy Constructor (deep copy - expensive operation!)
    GPUBuffer(const GPUBuffer& other) : numElements(other.numElements) {
        std::cout << "GPUBuffer COPY CONSTRUCTOR (expensive!): copying "
                  << numElements << " elements\n";

        hostData = new float[numElements];
        std::copy(other.hostData, other.hostData + numElements, hostData);

        // In real GPU code, this would involve:
        // 1. Allocate new GPU memory
        // 2. Copy from source GPU buffer to new GPU buffer
        // 3. Very expensive operation!
    }

    // Copy Assignment Operator
    GPUBuffer& operator=(const GPUBuffer& other) {
        std::cout << "GPUBuffer COPY ASSIGNMENT (expensive!): ";

        if (this == &other) {
            std::cout << "self-assignment\n";
            return *this;
        }

        // If sizes differ, reallocate
        if (numElements != other.numElements) {
            std::cout << "resizing from " << numElements
                      << " to " << other.numElements << " elements\n";
            delete[] hostData;
            numElements = other.numElements;
            hostData = new float[numElements];
        } else {
            std::cout << "same size (" << numElements << " elements)\n";
        }

        std::copy(other.hostData, other.hostData + numElements, hostData);
        return *this;
    }

    // Utility methods
    void set(size_t index, float value) {
        if (index < numElements) {
            hostData[index] = value;
        }
    }

    float get(size_t index) const {
        return (index < numElements) ? hostData[index] : 0.0f;
    }

    size_t size() const { return numElements; }

    void print() const {
        std::cout << "GPUBuffer[" << numElements << "]: [";
        for (size_t i = 0; i < std::min(numElements, size_t(10)); ++i) {
            std::cout << hostData[i];
            if (i < numElements - 1) std::cout << ", ";
        }
        if (numElements > 10) std::cout << "...";
        std::cout << "]\n";
    }
};

// Demonstrate problems without Rule of Three
void demonstrateBadCopy() {
    std::cout << "\n=== BAD: Missing Copy Operations ===\n";
    std::cout << "WARNING: This example shows what NOT to do!\n\n";

    // Uncommenting this will cause a double-delete crash:
    /*
    BadVector v1(3);
    v1.set(0, 1.0f);

    BadVector v2 = v1;  // Shallow copy! Both point to same memory
    // When v2 and v1 go out of scope, both try to delete the same memory!
    */

    std::cout << "Example commented out to prevent crash.\n";
    std::cout << "Problem: Shallow copy means both objects share the same pointer.\n";
    std::cout << "When both destructors run, memory is deleted twice = CRASH!\n";
}

// Demonstrate proper Rule of Three
void demonstrateGoodCopy() {
    std::cout << "\n=== GOOD: Proper Rule of Three ===\n";

    Vector3D v1(1.0f, 2.0f, 3.0f);
    std::cout << "v1: ";
    v1.print();

    // Copy constructor
    std::cout << "\nCalling copy constructor:\n";
    Vector3D v2 = v1;  // or Vector3D v2(v1);
    std::cout << "v2: ";
    v2.print();

    // Modify v2, v1 should be unchanged
    v2.setX(100.0f);
    std::cout << "\nAfter modifying v2:\n";
    std::cout << "v1: ";
    v1.print();
    std::cout << "v2: ";
    v2.print();

    // Copy assignment
    std::cout << "\nCalling copy assignment:\n";
    Vector3D v3;
    v3 = v1;
    std::cout << "v3: ";
    v3.print();

    // Self-assignment test
    std::cout << "\nTesting self-assignment:\n";
    v3 = v3;

    std::cout << "\nAll vectors are independent (deep copies)!\n";
}

// Demonstrate with GPU-like buffer
void demonstrateGPUBuffer() {
    std::cout << "\n=== GPU Buffer with Rule of Three ===\n";

    GPUBuffer buf1(5);
    buf1.set(0, 10.0f);
    buf1.set(1, 20.0f);
    buf1.set(2, 30.0f);

    std::cout << "\nOriginal buffer:\n";
    buf1.print();

    // Copy constructor
    std::cout << "\nCopying buffer (like cudaMemcpy):\n";
    GPUBuffer buf2 = buf1;
    buf2.print();

    // Modify copy
    buf2.set(0, 999.0f);

    std::cout << "\nAfter modifying copy:\n";
    std::cout << "buf1: ";
    buf1.print();
    std::cout << "buf2: ";
    buf2.print();

    // Copy assignment with different size
    std::cout << "\nCopy assignment with different size:\n";
    GPUBuffer buf3(8);
    buf3 = buf1;
    buf3.print();
}

// Pass by value (triggers copy constructor)
void processByValue(Vector3D vec) {
    std::cout << "Inside processByValue: ";
    vec.print();
    vec.setX(999.0f);
    // vec is destroyed here (destructor called)
}

// Pass by reference (no copy)
void processByReference(const Vector3D& vec) {
    std::cout << "Inside processByReference: ";
    vec.print();
    // No copy, no destructor call
}

void demonstratePassingSemantics() {
    std::cout << "\n=== Passing by Value vs Reference ===\n";

    Vector3D v(5.0f, 6.0f, 7.0f);

    std::cout << "\nPassing by value (triggers copy):\n";
    processByValue(v);
    std::cout << "After call, original: ";
    v.print();

    std::cout << "\nPassing by reference (no copy):\n";
    processByReference(v);
    std::cout << "After call, original: ";
    v.print();
}

int main() {
    std::cout << "=== MODULE 4: Rule of Three Demo ===\n";

    demonstrateBadCopy();
    demonstrateGoodCopy();
    demonstrateGPUBuffer();
    demonstratePassingSemantics();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. RULE OF THREE:
 *    - If you need a destructor, you need copy constructor & copy assignment
 *    - Required when managing resources (memory, files, GPU memory)
 *
 * 2. SHALLOW vs DEEP COPY:
 *    - Shallow: Copy pointer (both objects share same memory) - DANGEROUS!
 *    - Deep: Allocate new memory and copy data - SAFE but expensive
 *
 * 3. IMPLEMENTATION CHECKLIST:
 *    - Destructor: Free resources
 *    - Copy constructor: Create independent copy
 *    - Copy assignment: Check self-assignment, copy data, return *this
 *
 * 4. COPY ASSIGNMENT PATTERN:
 *    a) Check for self-assignment (if (this == &other) return *this;)
 *    b) Free old resources if needed (or reuse if same size)
 *    c) Allocate new resources
 *    d) Copy data
 *    e) Return *this
 *
 * TRY THIS:
 * 1. Uncomment the BadVector example and see the crash
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
