/*
 * File: 02_data_types.c
 * Topic: Basic Data Types in C
 *
 * C has several built-in data types. Understanding them is crucial
 * for efficient memory usage and preventing bugs.
 *
 * Key Concepts:
 * - Integer types: int, short, long, long long
 * - Floating-point types: float, double
 * - Character type: char
 * - sizeof operator
 * - Format specifiers for printf()
 */

#include <stdio.h>

int main() {
    // Integer types
    int age = 25;                    // Typical integer
    short small_num = 100;           // Smaller range, less memory
    long big_num = 1000000L;         // Larger range
    long long huge_num = 9223372036854775807LL;  // Very large numbers

    // Floating-point types
    float pi = 3.14159f;             // Single precision (6-7 decimal digits)
    double precise_pi = 3.14159265359;  // Double precision (15-16 digits)

    // Character type
    char grade = 'A';                // Single character
    char newline = '\n';             // Special character (newline)

    // Display values with appropriate format specifiers
    printf("=== Integer Types ===\n");
    printf("int: %d (size: %zu bytes)\n", age, sizeof(int));
    printf("short: %hd (size: %zu bytes)\n", small_num, sizeof(short));
    printf("long: %ld (size: %zu bytes)\n", big_num, sizeof(long));
    printf("long long: %lld (size: %zu bytes)\n", huge_num, sizeof(long long));

    printf("\n=== Floating-Point Types ===\n");
    printf("float: %.5f (size: %zu bytes)\n", pi, sizeof(float));
    printf("double: %.11f (size: %zu bytes)\n", precise_pi, sizeof(double));

    printf("\n=== Character Type ===\n");
    printf("char: %c (ASCII: %d, size: %zu byte)\n", grade, grade, sizeof(char));

    // Constants
    const int MAX_SCORE = 100;       // Cannot be changed after initialization
    printf("\n=== Constants ===\n");
    printf("Maximum score: %d\n", MAX_SCORE);

    return 0;
}

/*
 * FORMAT SPECIFIERS QUICK REFERENCE:
 * %d   - int
 * %hd  - short
 * %ld  - long
 * %lld - long long
 * %f   - float or double
 * %c   - char
 * %s   - string
 * %zu  - size_t (used by sizeof)
 *
 * EXERCISES:
 * 1. Declare a variable for your height in cm (int) and print it
 * 2. Create a variable for your GPA (use double) and print with 2 decimal places
 * 3. Store your first initial in a char and print it
 * 4. Try to modify MAX_SCORE and observe the compiler warning
 * 5. Use sizeof() to find the size of 'char newline'
 */
