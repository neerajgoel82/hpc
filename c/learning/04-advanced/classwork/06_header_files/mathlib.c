/*
 * File: mathlib.c
 * Topic: Implementation File
 *
 * This file implements the functions declared in mathlib.h.
 * Implementation files contain the actual function definitions.
 *
 * Key Concepts:
 * - Include your own header file
 * - Implement all declared functions
 * - Can have private helper functions
 */

#include "mathlib.h"  // Include our own header
#include <stdio.h>
#include <math.h>     // For sqrt() if needed

// ==================== BASIC ARITHMETIC ====================

double add(double a, double b) {
    return a + b;
}

double subtract(double a, double b) {
    return a - b;
}

double multiply(double a, double b) {
    return a * b;
}

double divide(double a, double b) {
    if (b == 0) {
        fprintf(stderr, "Error: Division by zero\n");
        return 0;
    }
    return a / b;
}

// ==================== ADVANCED MATH ====================

double power(double base, int exponent) {
    if (exponent == 0) return 1;

    double result = 1;
    int abs_exp = exponent < 0 ? -exponent : exponent;

    for (int i = 0; i < abs_exp; i++) {
        result *= base;
    }

    return exponent < 0 ? 1.0 / result : result;
}

double squareRoot(double n) {
    if (n < 0) {
        fprintf(stderr, "Error: Square root of negative number\n");
        return 0;
    }

    // Newton's method for square root
    if (n == 0) return 0;

    double guess = n / 2.0;
    double epsilon = 0.00001;

    while ((guess * guess - n) > epsilon || (n - guess * guess) > epsilon) {
        guess = (guess + n / guess) / 2.0;
    }

    return guess;
}

double absolute(double n) {
    return n < 0 ? -n : n;
}

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }

    return true;
}

long long factorial(int n) {
    if (n < 0) return -1;  // Error
    if (n == 0 || n == 1) return 1;

    long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }

    return result;
}

int gcd(int a, int b) {
    // Euclidean algorithm
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// ==================== GEOMETRY ====================

double distance(Point p1, Point p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return squareRoot(dx * dx + dy * dy);
}

double circleArea(double radius) {
    if (radius < 0) {
        fprintf(stderr, "Error: Negative radius\n");
        return 0;
    }
    return PI * radius * radius;
}

double rectangleArea(double length, double width) {
    if (length < 0 || width < 0) {
        fprintf(stderr, "Error: Negative dimensions\n");
        return 0;
    }
    return length * width;
}

// ==================== COMPLEX NUMBERS ====================

Complex createComplex(double real, double imag) {
    Complex c;
    c.real = real;
    c.imag = imag;
    return c;
}

Complex complexAdd(Complex c1, Complex c2) {
    Complex result;
    result.real = c1.real + c2.real;
    result.imag = c1.imag + c2.imag;
    return result;
}

Complex complexMultiply(Complex c1, Complex c2) {
    Complex result;
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    result.real = c1.real * c2.real - c1.imag * c2.imag;
    result.imag = c1.real * c2.imag + c1.imag * c2.real;
    return result;
}

void printComplex(Complex c) {
    if (c.imag >= 0) {
        printf("%.2f + %.2fi\n", c.real, c.imag);
    } else {
        printf("%.2f - %.2fi\n", c.real, -c.imag);
    }
}

// ==================== UTILITIES ====================

void printLibraryInfo(void) {
    printf("====================================\n");
    printf("Math Library v%s\n", MATHLIB_VERSION);
    printf("====================================\n");
    printf("A simple mathematical library\n");
    printf("demonstrating multi-file projects.\n");
    printf("====================================\n");
}

const char* getVersion(void) {
    return MATHLIB_VERSION;
}

/*
 * IMPLEMENTATION FILE GUIDELINES:
 *
 * 1. Include corresponding header first
 * 2. Include other headers as needed
 * 3. Implement all declared functions
 * 4. Can have static helper functions (private)
 * 5. Error checking and validation
 * 6. Documentation in header, implementation here
 *
 * STATIC FUNCTIONS:
 * - Use 'static' for functions only used in this file
 * - They won't be visible to other files
 * - Good for helper/utility functions
 *
 * Example:
 * static int helperFunction(int x) {
 *     return x * 2;
 * }
 */
