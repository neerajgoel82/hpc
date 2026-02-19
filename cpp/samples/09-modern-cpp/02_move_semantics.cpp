/*
 * MODULE 9: Modern C++
 * File: 02_move_semantics.cpp
 *
 * TOPIC: Move Semantics and Rvalue References
 *
 * COMPILE: g++ -std=c++17 -o move_semantics 02_move_semantics.cpp
 */

#include <iostream>
#include <utility>
#include <vector>
#include <cstring>

class GPUBuffer {
private:
    float* data;
    size_t size;

public:
    // Constructor
    GPUBuffer(size_t n) : size(n) {
        data = new float[size];
        std::cout << "  Constructor: allocated " << size << " floats\n";
    }

    // Destructor
    ~GPUBuffer() {
        std::cout << "  Destructor: freeing " << size << " floats\n";
        delete[] data;
    }

    // Copy constructor (expensive!)
    GPUBuffer(const GPUBuffer& other) : size(other.size) {
        std::cout << "  COPY Constructor: copying " << size << " floats (EXPENSIVE!)\n";
        data = new float[size];
        std::memcpy(data, other.data, size * sizeof(float));
    }

    // Copy assignment (expensive!)
    GPUBuffer& operator=(const GPUBuffer& other) {
        std::cout << "  COPY Assignment: copying " << size << " floats (EXPENSIVE!)\n";
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new float[size];
            std::memcpy(data, other.data, size * sizeof(float));
        }
        return *this;
    }

    // Move constructor (cheap!)
    GPUBuffer(GPUBuffer&& other) noexcept : data(other.data), size(other.size) {
        std::cout << "  MOVE Constructor: stealing resources (CHEAP!)\n";
        other.data = nullptr;
        other.size = 0;
    }

    // Move assignment (cheap!)
    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        std::cout << "  MOVE Assignment: stealing resources (CHEAP!)\n";
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    size_t getSize() const { return size; }
};

// Factory function returning by value (move semantics)
GPUBuffer createBuffer(size_t size) {
    std::cout << "Creating buffer in factory...\n";
    GPUBuffer buf(size);
    return buf;  // Move, not copy!
}

int main() {
    std::cout << "=== Move Semantics Demo ===\n\n";

    // Move constructor
    std::cout << "=== Move Constructor ===\n";
    GPUBuffer buf1(1000);
    GPUBuffer buf2 = std::move(buf1);  // Move, not copy
    std::cout << "buf1 size after move: " << buf1.getSize() << "\n\n";

    // Move assignment
    std::cout << "=== Move Assignment ===\n";
    GPUBuffer buf3(500);
    GPUBuffer buf4(200);
    buf4 = std::move(buf3);  // Move, not copy
    std::cout << "\n";

    // Return value optimization + move
    std::cout << "=== Factory Function (Move) ===\n";
    GPUBuffer buf5 = createBuffer(2000);
    std::cout << "\n";

    // Vector with move semantics
    std::cout << "=== Vector with Moves ===\n";
    std::vector<GPUBuffer> buffers;
    buffers.push_back(GPUBuffer(100));  // Moved into vector
    buffers.push_back(GPUBuffer(200));
    std::cout << "\nVector size: " << buffers.size() << "\n";

    std::cout << "\nGPU CONNECTION:\n";
    std::cout << "  - Moving GPU resources instead of copying\n";
    std::cout << "  - Huge performance win for large buffers\n";
    std::cout << "  - Essential for modern C++ GPU code\n";

    return 0;
}

/*
 * TRY THIS:
 * 1. Add move semantics to your Vec3/Mat4 classes
 * 2. Implement texture class with move support
 * 3. Profile copy vs move performance
 * 4. Use std::move with unique_ptr
 * 5. Implement swap using moves
 */
