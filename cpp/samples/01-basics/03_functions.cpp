// 03_functions.cpp
// Functions: building blocks of programs (and GPU kernels!)
// Compile: g++ -std=c++17 -o functions 03_functions.cpp

#include <iostream>
#include <cmath>  // For sqrt()

// Function declaration (prototype)
int add(int a, int b);
float vectorLength(float x, float y);
void printSquare(int n);

// Function that returns nothing (void)
void greet(std::string name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}

// Function with return value
int add(int a, int b) {
    return a + b;
}

// Function with floating point math (common in GPU)
float vectorLength(float x, float y) {
    return std::sqrt(x*x + y*y);  // Pythagorean theorem
}

// Function that prints without returning
void printSquare(int n) {
    std::cout << n << " squared is " << (n * n) << std::endl;
}

// Function overloading: same name, different parameters
int multiply(int a, int b) {
    return a * b;
}

float multiply(float a, float b) {
    return a * b;
}

int main() {
    // Call functions
    greet("GPU Programmer");

    int sum = add(5, 7);
    std::cout << "5 + 7 = " << sum << std::endl;

    float len = vectorLength(3.0f, 4.0f);
    std::cout << "Vector length: " << len << std::endl;

    printSquare(8);

    // Function overloading in action
    std::cout << "Int multiply: " << multiply(5, 6) << std::endl;
    std::cout << "Float multiply: " << multiply(5.5f, 2.0f) << std::endl;

    return 0;
}

/*
LEARNING NOTES:
- Functions must be declared before use (prototype at top)
- void = returns nothing
- C++ allows function overloading (same name, different parameters)
- GPU kernels are similar to functions but execute in parallel

TRY THIS:
1. Write a function that calculates: (a + b) * c
2. Write a function to normalize a vector: divide by its length
3. What happens if you call multiply(5, 3.0f)? Why?
4. Add a function that takes 3 float parameters
*/
