// 04_header_example_main.cpp
// Using functions from separate header/implementation files
// Compile: g++ -std=c++17 -o header_example 04_header_example_main.cpp math_utils.cpp

#include <iostream>
#include "math_utils.h"  // Include our header

int main() {
    std::cout << "=== Basic Math Functions ===" << std::endl;
    std::cout << "add(5, 3) = " << add(5, 3) << std::endl;
    std::cout << "subtract(10, 4) = " << subtract(10, 4) << std::endl;
    std::cout << "multiply(7, 6) = " << multiply(7, 6) << std::endl;
    std::cout << "divide(15.0f, 3.0f) = " << divide(15.0f, 3.0f) << std::endl;
    std::cout << std::endl;

    std::cout << "=== Inline Functions ===" << std::endl;
    std::cout << "square(5) = " << square(5) << std::endl;
    std::cout << "cube(3) = " << cube(3) << std::endl;
    std::cout << std::endl;

    std::cout << "=== Vector Operations ===" << std::endl;
    Vector2D v1 = {3.0f, 4.0f};
    Vector2D v2 = {1.0f, 2.0f};

    std::cout << "v1 = ";
    print(v1);
    std::cout << std::endl;

    std::cout << "v2 = ";
    print(v2);
    std::cout << std::endl;

    std::cout << "length(v1) = " << length(v1) << std::endl;
    std::cout << "dot(v1, v2) = " << dot(v1, v2) << std::endl;

    Vector2D sum = add(v1, v2);
    std::cout << "v1 + v2 = ";
    print(sum);
    std::cout << std::endl;

    Vector2D diff = subtract(v1, v2);
    std::cout << "v1 - v2 = ";
    print(diff);
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "=== Constants and Helpers ===" << std::endl;
    std::cout << "PI = " << PI << std::endl;
    std::cout << "E = " << E << std::endl;
    std::cout << "90 degrees = " << degToRad(90.0f) << " radians" << std::endl;
    std::cout << "PI radians = " << radToDeg(PI) << " degrees" << std::endl;

    return 0;
}

/*
COMPILING MULTI-FILE PROJECTS:

METHOD 1: Compile separately
g++ -std=c++17 -c math_utils.cpp -o math_utils.o
g++ -std=c++17 -c 04_header_example_main.cpp -o main.o
g++ main.o math_utils.o -o program

METHOD 2: Compile together
g++ -std=c++17 -o program 04_header_example_main.cpp math_utils.cpp

METHOD 3: Using wildcard
g++ -std=c++17 -o program *.cpp

PROJECT STRUCTURE:
project/
├── include/
│   └── math_utils.h
├── src/
│   ├── math_utils.cpp
│   └── main.cpp
└── build/
    └── (compiled files)

INCLUDE PATHS:
g++ -I./include -o program src/*.cpp

ADVANTAGES OF SEPARATE FILES:
1. Organization - Logical grouping
2. Reusability - Use same code in multiple projects
3. Compilation speed - Only recompile changed files
4. Team work - Multiple people can work on different files
5. Libraries - Create reusable libraries

GPU EXAMPLE STRUCTURE:
cuda-project/
├── include/
│   ├── kernels.cuh       # CUDA kernel declarations
│   └── utils.h           # CPU utility functions
├── src/
│   ├── kernels.cu        # CUDA kernel implementations
│   ├── utils.cpp         # CPU implementations
│   └── main.cpp          # Main program
└── CMakeLists.txt        # Build configuration

TRY THIS:
1. Add more functions to math_utils.h/cpp
2. Create a separate geometry.h/cpp for shapes
3. What happens if you forget to include the header?
4. Remove the header guard - what error do you get?
5. Try to use a function without including its header
*/
