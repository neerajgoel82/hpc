// 02_types_and_variables.cpp
// Understanding C++ types and how they map to hardware
// Compile: g++ -std=c++17 -o types 02_types_and_variables.cpp

#include <iostream>
#include <cstdint>  // For fixed-width integer types

int main() {
    // Integer types (important for GPU: exact sizes matter!)
    int a = 42;                    // Platform-dependent size (usually 32-bit)
    int32_t b = 100;               // Exactly 32 bits
    int64_t c = 1000000;           // Exactly 64 bits
    unsigned int d = 42;           // Unsigned: only positive values

    // Floating point (crucial for GPU computation)
    float f = 3.14f;               // 32-bit (GPU often uses this)
    double g = 3.14159265359;      // 64-bit (more precision, slower on many GPUs)

    // Boolean
    bool is_valid = true;

    // Character
    char letter = 'A';

    // Print sizes and values
    std::cout << "Size of int: " << sizeof(int) << " bytes" << std::endl;
    std::cout << "Size of int32_t: " << sizeof(int32_t) << " bytes" << std::endl;
    std::cout << "Size of int64_t: " << sizeof(int64_t) << " bytes" << std::endl;
    std::cout << "Size of float: " << sizeof(float) << " bytes" << std::endl;
    std::cout << "Size of double: " << sizeof(double) << " bytes" << std::endl;
    std::cout << "Size of bool: " << sizeof(bool) << " bytes" << std::endl;
    std::cout << "Size of char: " << sizeof(char) << " bytes" << std::endl;

    std::cout << "\nValues:" << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "c = " << c << std::endl;
    std::cout << "d = " << d << std::endl;
    std::cout << "f = " << f << std::endl;
    std::cout << "g = " << g << std::endl;
    std::cout << "is_valid = " << is_valid << std::endl;
    std::cout << "letter = " << letter << std::endl;

    return 0;
}

/*
LEARNING NOTES:
- sizeof() returns size in bytes
- GPUs care about exact sizes: float vs double affects performance
- Use fixed-width types (int32_t) when size matters
- 'f' suffix on float literals (3.14f) prevents implicit double conversion

TRY THIS:
1. Create variables of different types
2. Try arithmetic: a + b, f * 2.0f
3. What happens if you multiply float by double?
4. Calculate how many floats fit in 1GB: (1024*1024*1024) / sizeof(float)
*/
