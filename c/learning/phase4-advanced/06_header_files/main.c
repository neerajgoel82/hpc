/*
 * File: main.c
 * Topic: Using Header Files - Main Program
 *
 * This file demonstrates how to use a custom library (mathlib).
 * It includes the header file and uses the library functions.
 *
 * Compilation:
 *   gcc main.c mathlib.c -o math_program
 *   ./math_program
 *
 * Or compile separately and link:
 *   gcc -c mathlib.c -o mathlib.o
 *   gcc -c main.c -o main.o
 *   gcc mathlib.o main.o -o math_program
 *   ./math_program
 */

#include <stdio.h>
#include "mathlib.h"  // Include our custom library

int main() {
    // Display library info
    printLibraryInfo();

    // ==================== BASIC ARITHMETIC ====================

    printf("\n=== Basic Arithmetic ===\n");
    printf("10 + 5 = %.2f\n", add(10, 5));
    printf("10 - 5 = %.2f\n", subtract(10, 5));
    printf("10 * 5 = %.2f\n", multiply(10, 5));
    printf("10 / 5 = %.2f\n", divide(10, 5));
    printf("10 / 0 = %.2f (error handled)\n", divide(10, 0));

    // ==================== ADVANCED MATH ====================

    printf("\n=== Advanced Math ===\n");
    printf("2^10 = %.0f\n", power(2, 10));
    printf("2^-3 = %.4f\n", power(2, -3));
    printf("sqrt(16) = %.2f\n", squareRoot(16));
    printf("sqrt(2) = %.4f\n", squareRoot(2));
    printf("abs(-42) = %.0f\n", absolute(-42));
    printf("abs(3.14) = %.2f\n", absolute(3.14));

    printf("\n=== Prime Numbers ===\n");
    int numbers[] = {2, 7, 10, 17, 20, 29, 30};
    for (int i = 0; i < 7; i++) {
        printf("%d is %s\n", numbers[i],
               isPrime(numbers[i]) ? "prime" : "not prime");
    }

    printf("\n=== Factorial ===\n");
    for (int i = 0; i <= 10; i++) {
        printf("%d! = %lld\n", i, factorial(i));
    }

    printf("\n=== GCD ===\n");
    printf("GCD(48, 18) = %d\n", gcd(48, 18));
    printf("GCD(100, 50) = %d\n", gcd(100, 50));
    printf("GCD(17, 19) = %d\n", gcd(17, 19));

    // ==================== GEOMETRY ====================

    printf("\n=== Geometry ===\n");

    Point p1 = {0, 0};
    Point p2 = {3, 4};
    printf("Distance from (%.0f,%.0f) to (%.0f,%.0f) = %.2f\n",
           p1.x, p1.y, p2.x, p2.y, distance(p1, p2));

    printf("Area of circle (radius 5) = %.2f\n", circleArea(5));
    printf("Area of circle (radius 10) = %.2f\n", circleArea(10));

    printf("Area of rectangle (5 x 10) = %.2f\n", rectangleArea(5, 10));

    // ==================== COMPLEX NUMBERS ====================

    printf("\n=== Complex Numbers ===\n");

    Complex c1 = createComplex(3, 4);
    Complex c2 = createComplex(1, 2);

    printf("c1 = ");
    printComplex(c1);

    printf("c2 = ");
    printComplex(c2);

    Complex sum = complexAdd(c1, c2);
    printf("c1 + c2 = ");
    printComplex(sum);

    Complex product = complexMultiply(c1, c2);
    printf("c1 * c2 = ");
    printComplex(product);

    // ==================== CONSTANTS ====================

    printf("\n=== Mathematical Constants ===\n");
    printf("PI = %.10f\n", PI);
    printf("E = %.10f\n", E);

    printf("\n=== Library Version ===\n");
    printf("Version: %s\n", getVersion());

    return 0;
}

/*
 * MULTI-FILE PROJECT STRUCTURE:
 *
 * mathlib.h       - Interface (declarations)
 * mathlib.c       - Implementation (definitions)
 * main.c          - Application (uses the library)
 *
 * COMPILATION STEPS:
 *
 * Method 1: Compile everything together
 *   gcc main.c mathlib.c -o program
 *
 * Method 2: Separate compilation (better for large projects)
 *   Step 1: Compile source files to object files
 *     gcc -c mathlib.c -o mathlib.o
 *     gcc -c main.c -o main.o
 *
 *   Step 2: Link object files
 *     gcc mathlib.o main.o -o program
 *
 * Method 3: Using make (see Makefile)
 *   make
 *
 * WHY SEPARATE FILES:
 * - Organization: Related code grouped together
 * - Reusability: Library can be used in multiple projects
 * - Maintainability: Changes in one place
 * - Compilation: Only recompile changed files
 * - Team work: Different people work on different modules
 * - Testing: Easier to test individual modules
 *
 * INCLUDE PATH:
 * - "mathlib.h" - Look in current directory first
 * - <stdio.h>   - Look in system directories
 *
 * HEADER FILE RULES:
 * 1. Declare in header, define in source
 * 2. Include guard in every header
 * 3. Include only what you need
 * 4. Don't include implementation details
 *
 * BEST PRACTICES:
 * - One header per module
 * - Match header and source file names
 * - Minimize dependencies
 * - Document in headers
 * - Keep interfaces stable
 * - Use const for parameters that won't change
 */
