/*
 * File: 02_preprocessor.c
 * Topic: C Preprocessor Directives
 *
 * The preprocessor processes your code before compilation.
 * It handles #include, #define, conditional compilation, and more.
 *
 * Key Concepts:
 * - #include directive
 * - #define macros
 * - Conditional compilation
 * - Predefined macros
 * - Macro functions
 */

#include <stdio.h>

// Simple macro definitions
#define PI 3.14159
#define MAX_SIZE 100
#define GREETING "Hello from macro!"

// Macro functions
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define IS_EVEN(n) ((n) % 2 == 0)

// Multi-line macros
#define PRINT_ARRAY(arr, size) \
    do { \
        printf("["); \
        for (int i = 0; i < (size); i++) { \
            printf("%d", arr[i]); \
            if (i < (size) - 1) printf(", "); \
        } \
        printf("]\n"); \
    } while(0)

// Conditional compilation
#define DEBUG 1
#define VERSION 2

#if DEBUG
    #define DEBUG_PRINT(msg) printf("DEBUG: %s\n", msg)
#else
    #define DEBUG_PRINT(msg)
#endif

// Stringification
#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)

// Token pasting
#define CONCAT(a, b) a##b

int main() {
    // Using simple macros
    printf("=== Simple Macros ===\n");
    printf("PI = %f\n", PI);
    printf("MAX_SIZE = %d\n", MAX_SIZE);
    printf("GREETING: %s\n", GREETING);

    float radius = 5.0;
    float area = PI * radius * radius;
    printf("Area of circle (radius %.1f) = %.2f\n", radius, area);

    // Using macro functions
    printf("\n=== Macro Functions ===\n");
    int x = 5;
    printf("SQUARE(%d) = %d\n", x, SQUARE(x));
    printf("SQUARE(3 + 2) = %d\n", SQUARE(3 + 2));

    int a = 10, b = 20;
    printf("MAX(%d, %d) = %d\n", a, b, MAX(a, b));
    printf("MIN(%d, %d) = %d\n", a, b, MIN(a, b));

    int num = 7;
    if (IS_EVEN(num)) {
        printf("%d is even\n", num);
    } else {
        printf("%d is odd\n", num);
    }

    // Multi-line macro
    printf("\n=== Multi-line Macro ===\n");
    int arr[] = {1, 2, 3, 4, 5};
    printf("Array: ");
    PRINT_ARRAY(arr, 5);

    // Debug macros
    printf("\n=== Conditional Compilation ===\n");
    DEBUG_PRINT("This is a debug message");
    printf("DEBUG is %s\n", DEBUG ? "enabled" : "disabled");

    // Predefined macros
    printf("\n=== Predefined Macros ===\n");
    printf("File: %s\n", __FILE__);
    printf("Line: %d\n", __LINE__);
    printf("Date: %s\n", __DATE__);
    printf("Time: %s\n", __TIME__);

    #ifdef __STDC__
        printf("Standard C: Yes\n");
    #endif

    // Stringification
    printf("\n=== Stringification ===\n");
    int value = 42;
    printf("Variable name: %s, value: %d\n", STRINGIFY(value), value);
    printf("PI as string: %s\n", TO_STRING(PI));

    // Token pasting
    printf("\n=== Token Pasting ===\n");
    int var1 = 10;
    int var2 = 20;
    printf("var1 = %d\n", var1);
    printf("var2 = %d\n", var2);
    printf("CONCAT(var, 1) = %d\n", CONCAT(var, 1));

    // Conditional compilation demo
    printf("\n=== Version-based Compilation ===\n");
    #if VERSION >= 2
        printf("Using version 2 features\n");
    #elif VERSION == 1
        printf("Using version 1 features\n");
    #else
        printf("Using basic features\n");
    #endif

    // ifdef/ifndef
    printf("\n=== Checking if Defined ===\n");
    #ifdef PI
        printf("PI is defined as: %f\n", PI);
    #endif

    #ifndef UNDEFINED_MACRO
        printf("UNDEFINED_MACRO is not defined\n");
    #endif

    // Platform-specific code
    printf("\n=== Platform Detection ===\n");
    #ifdef _WIN32
        printf("Compiling for Windows\n");
    #elif defined(__APPLE__)
        printf("Compiling for macOS\n");
    #elif defined(__linux__)
        printf("Compiling for Linux\n");
    #else
        printf("Unknown platform\n");
    #endif

    return 0;
}

/*
 * PREPROCESSOR DIRECTIVES:
 *
 * #include <file>     - Include system header
 * #include "file"     - Include user header
 * #define NAME value  - Define constant
 * #define NAME(x)     - Define macro function
 * #undef NAME         - Undefine macro
 * #ifdef NAME         - If defined
 * #ifndef NAME        - If not defined
 * #if condition       - Conditional
 * #elif condition     - Else if
 * #else               - Else
 * #endif              - End conditional
 * #error message      - Generate error
 * #pragma             - Implementation-specific
 *
 * MACRO BEST PRACTICES:
 * 1. Use parentheses in macro definitions
 *    - Bad:  #define SQUARE(x) x * x
 *    - Good: #define SQUARE(x) ((x) * (x))
 * 2. Use uppercase for macro names
 * 3. Avoid side effects in macro arguments
 * 4. Use do-while(0) for multi-statement macros
 * 5. Consider inline functions instead of macros
 *
 * PREDEFINED MACROS:
 * __FILE__    - Current filename
 * __LINE__    - Current line number
 * __DATE__    - Compilation date
 * __TIME__    - Compilation time
 * __STDC__    - 1 if standard C
 * __func__    - Current function name (C99)
 *
 * COMMON USES:
 * - Constants that never change
 * - Platform-specific code
 * - Debug/release builds
 * - Feature flags
 * - Code generation
 * - Header guards
 *
 * MACRO vs CONST vs INLINE:
 * - Macro: Text replacement, no type checking
 * - const: Type-safe constant
 * - inline: Type-safe function, better than macro
 *
 * EXERCISES:
 * 1. Create macros for geometric formulas
 * 2. Write a macro to swap two values
 * 3. Create debug macros for logging
 * 4. Implement conditional compilation for different platforms
 * 5. Create a macro to check array bounds
 * 6. Write a macro to measure execution time
 * 7. Create assertion macros
 */
