/*
 * File: mathlib.h
 * Topic: Header File Example
 *
 * This is a header file that declares the interface for a simple math library.
 * Header files contain function declarations, type definitions, and macros
 * that can be shared across multiple source files.
 *
 * Key Concepts:
 * - Header guards (prevent multiple inclusion)
 * - Function declarations (prototypes)
 * - Separation of interface and implementation
 * - #include usage
 */

#ifndef MATHLIB_H    // Header guard start
#define MATHLIB_H    // Define this symbol

// Include standard headers if needed
#include <stdbool.h>

/*
 * HEADER GUARDS:
 * Prevent the header from being included multiple times.
 * Without guards, you'd get "redefinition" errors.
 *
 * Pattern:
 * #ifndef FILENAME_H
 * #define FILENAME_H
 * ... contents ...
 * #endif
 */

// ==================== CONSTANTS ====================

#define MATHLIB_VERSION "1.0.0"
#define PI 3.14159265359
#define E  2.71828182846

// ==================== TYPE DEFINITIONS ====================

// Point in 2D space
typedef struct {
    double x;
    double y;
} Point;

// Complex number
typedef struct {
    double real;
    double imag;
} Complex;

// ==================== FUNCTION DECLARATIONS ====================

/*
 * Basic arithmetic operations
 */

// Add two numbers
double add(double a, double b);

// Subtract b from a
double subtract(double a, double b);

// Multiply two numbers
double multiply(double a, double b);

// Divide a by b (returns 0 if b is 0)
double divide(double a, double b);

/*
 * Advanced math operations
 */

// Calculate power (a^b)
double power(double base, int exponent);

// Calculate square root (simple approximation)
double squareRoot(double n);

// Calculate absolute value
double absolute(double n);

// Check if number is prime
bool isPrime(int n);

// Calculate factorial
long long factorial(int n);

// Calculate greatest common divisor
int gcd(int a, int b);

/*
 * Geometry operations
 */

// Calculate distance between two points
double distance(Point p1, Point p2);

// Calculate area of circle
double circleArea(double radius);

// Calculate area of rectangle
double rectangleArea(double length, double width);

/*
 * Complex number operations
 */

// Create complex number
Complex createComplex(double real, double imag);

// Add two complex numbers
Complex complexAdd(Complex c1, Complex c2);

// Multiply two complex numbers
Complex complexMultiply(Complex c1, Complex c2);

// Print complex number
void printComplex(Complex c);

/*
 * Utility functions
 */

// Print library information
void printLibraryInfo(void);

// Get library version
const char* getVersion(void);

#endif  // MATHLIB_H (Header guard end)

/*
 * HEADER FILE BEST PRACTICES:
 *
 * 1. Always use header guards
 * 2. Only include necessary headers
 * 3. Declare, don't define (except inline functions)
 * 4. Use const where appropriate
 * 5. Document all functions
 * 6. Group related declarations
 * 7. Use consistent naming
 * 8. Avoid using "using namespace" (in C++)
 *
 * WHAT GOES IN HEADER FILES:
 * ✓ Function declarations
 * ✓ Type definitions (struct, typedef, enum)
 * ✓ Macro definitions
 * ✓ Extern variable declarations
 * ✓ Inline function definitions
 * ✓ Constants
 *
 * WHAT DOESN'T GO IN HEADER FILES:
 * ✗ Function implementations (except inline)
 * ✗ Global variable definitions
 * ✗ using directives
 * ✗ Executable code
 *
 * WHY USE HEADER FILES:
 * - Code organization
 * - Reusability
 * - Interface/implementation separation
 * - Easier maintenance
 * - Multiple file projects
 * - Library creation
 */
