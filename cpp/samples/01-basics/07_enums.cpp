// 07_enums.cpp
// Enumerations: named constants for better code readability
// Compile: g++ -std=c++17 -o enums 07_enums.cpp

#include <iostream>

// ===== Traditional enum (C-style) =====
enum Color {
    RED,      // 0 by default
    GREEN,    // 1
    BLUE      // 2
};

// Enum with explicit values
enum Status {
    SUCCESS = 0,
    ERROR = -1,
    PENDING = 100,
    TIMEOUT = 101
};

// ===== enum class (C++11 - strongly typed, recommended) =====
enum class Direction {
    NORTH,
    SOUTH,
    EAST,
    WEST
};

enum class GPUMemoryType {
    DEVICE = 0,    // GPU global memory
    HOST = 1,      // CPU memory
    MANAGED = 2,   // Unified memory
    CONSTANT = 3   // GPU constant memory
};

// Enum class with explicit underlying type
enum class ErrorCode : int {
    SUCCESS = 0,
    INVALID_ARGUMENT = 1,
    OUT_OF_MEMORY = 2,
    DEVICE_ERROR = 3
};

int main() {
    // ===== Using Traditional enum =====
    Color favorite = RED;

    std::cout << "Favorite color value: " << favorite << std::endl;  // Prints 0

    if (favorite == RED) {
        std::cout << "You like red!" << std::endl;
    }

    // Traditional enum can implicitly convert to int
    int color_value = GREEN;
    std::cout << "Green as int: " << color_value << std::endl;

    // ===== Using enum with switch =====
    Status current_status = PENDING;

    switch (current_status) {
        case SUCCESS:
            std::cout << "Operation succeeded!" << std::endl;
            break;
        case ERROR:
            std::cout << "Operation failed!" << std::endl;
            break;
        case PENDING:
            std::cout << "Operation pending..." << std::endl;
            break;
        case TIMEOUT:
            std::cout << "Operation timed out!" << std::endl;
            break;
    }

    // ===== Using enum class (preferred in modern C++) =====
    Direction heading = Direction::NORTH;  // Must use scope resolution

    // Cannot implicitly convert to int (type safe!)
    // int dir_value = heading;  // ERROR: won't compile
    int dir_value = static_cast<int>(heading);  // Must explicitly cast
    std::cout << "Direction value: " << dir_value << std::endl;

    // enum class comparison
    if (heading == Direction::NORTH) {
        std::cout << "Heading north!" << std::endl;
    }

    // ===== GPU Memory Type Example =====
    GPUMemoryType mem_type = GPUMemoryType::DEVICE;

    switch (mem_type) {
        case GPUMemoryType::DEVICE:
            std::cout << "Allocating on GPU device memory" << std::endl;
            break;
        case GPUMemoryType::HOST:
            std::cout << "Allocating on CPU host memory" << std::endl;
            break;
        case GPUMemoryType::MANAGED:
            std::cout << "Allocating managed (unified) memory" << std::endl;
            break;
        case GPUMemoryType::CONSTANT:
            std::cout << "Allocating on GPU constant memory" << std::endl;
            break;
    }

    // ===== Error Code Example =====
    ErrorCode result = ErrorCode::SUCCESS;

    if (result == ErrorCode::SUCCESS) {
        std::cout << "GPU operation completed successfully!" << std::endl;
    } else {
        std::cout << "Error code: " << static_cast<int>(result) << std::endl;
    }

    // ===== Practical Example: Function that returns enum =====
    auto check_value = [](int val) -> Direction {
        if (val < 0) return Direction::SOUTH;
        if (val > 0) return Direction::NORTH;
        if (val % 2 == 0) return Direction::EAST;
        return Direction::WEST;
    };

    Direction dir = check_value(10);
    std::cout << "Computed direction: " << static_cast<int>(dir) << std::endl;

    return 0;
}

/*
LEARNING NOTES:

TRADITIONAL enum:
- Values automatically numbered starting from 0
- Implicitly converts to int
- Pollutes namespace (RED is accessible anywhere)
- Can have value collisions

enum class (STRONGLY TYPED):
- Must use scope: Direction::NORTH
- Does NOT implicitly convert to int (type safe)
- Better for large projects (no namespace pollution)
- Recommended in modern C++

WHEN TO USE:
- Use enum class for new code (C++11+)
- Use traditional enum only for C compatibility
- Perfect for state machines, error codes, options

SYNTAX:
- enum Name { VALUE1, VALUE2 };           // Traditional
- enum class Name { VALUE1, VALUE2 };     // Modern (preferred)
- enum class Name : type { ... };         // With explicit type

GPU RELEVANCE:
- CUDA uses enums for memory types (cudaMemcpyHostToDevice, etc.)
- Error codes in GPU APIs are often enums
- Kernel configuration options (cache preferences, etc.)
- Makes GPU code more readable and maintainable

TRY THIS:
1. Create an enum class for days of the week
2. Write a function that takes a day enum and returns if it's a weekend
3. Create an enum for GPU execution modes (synchronous, asynchronous, stream)
4. What happens if you try to assign an enum class to int without casting?
5. Create an enum with values that are powers of 2 (for bitwise flags)

EXAMPLE:
enum class MemoryOperation {
    ALLOCATE,
    COPY,
    FREE
};

CUDA EXAMPLE YOU'LL SEE LATER:
enum cudaMemcpyKind {
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice
};
*/
