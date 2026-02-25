// math_utils.cpp
// Implementation file: function definitions

#include "math_utils.h"
#include <cmath>
#include <iostream>

// ===== Basic Math Functions =====
int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

float divide(float a, float b) {
    if (b == 0.0f) {
        std::cerr << "Error: Division by zero!" << std::endl;
        return 0.0f;
    }
    return a / b;
}

// ===== Vector2D Functions =====
float length(const Vector2D& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

Vector2D add(const Vector2D& a, const Vector2D& b) {
    return {a.x + b.x, a.y + b.y};
}

Vector2D subtract(const Vector2D& a, const Vector2D& b) {
    return {a.x - b.x, a.y - b.y};
}

float dot(const Vector2D& a, const Vector2D& b) {
    return a.x * b.x + a.y * b.y;
}

void print(const Vector2D& v) {
    std::cout << "(" << v.x << ", " << v.y << ")";
}

/*
IMPLEMENTATION FILE (.cpp):

- Includes its own header
- Contains function definitions
- Can include other headers as needed
- Implementation details hidden from users
- Can change implementation without recompiling users

COMPILATION:
g++ -c math_utils.cpp -o math_utils.o    # Compile to object file
g++ main.cpp math_utils.o -o program     # Link with main

Or compile together:
g++ main.cpp math_utils.cpp -o program

LINKING:
- Linker combines object files
- Resolves function calls to definitions
- Creates final executable

GPU/CUDA:
- .cu files compiled with nvcc
- Can link .o files from g++ and nvcc
- Separable compilation for CUDA code
- Device code linking
*/
