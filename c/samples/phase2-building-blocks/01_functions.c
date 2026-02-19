/*
 * File: 01_functions.c
 * Topic: Functions in C
 *
 * Functions are reusable blocks of code that perform specific tasks.
 * They help organize code, improve readability, and reduce repetition.
 *
 * Key Concepts:
 * - Function declaration (prototype)
 * - Function definition
 * - Function call
 * - Parameters and arguments
 * - Return values
 * - void functions
 */

#include <stdio.h>

// Function declarations (prototypes)
void greet(void);
void greetPerson(char name[]);
int add(int a, int b);
int multiply(int a, int b);
float divide(float a, float b);
int isEven(int num);
void printLine(int length);

int main() {
    // Calling a simple function
    printf("=== Simple Function Call ===\n");
    greet();

    // Calling function with parameters
    printf("\n=== Function with Parameters ===\n");
    greetPerson("Alice");
    greetPerson("Bob");

    // Functions that return values
    printf("\n=== Functions with Return Values ===\n");
    int sum = add(5, 3);
    printf("5 + 3 = %d\n", sum);

    int product = multiply(4, 7);
    printf("4 * 7 = %d\n", product);

    float quotient = divide(10.0, 3.0);
    printf("10 / 3 = %.2f\n", quotient);

    // Using function return value in expressions
    int result = add(10, 20) + multiply(3, 4);
    printf("add(10, 20) + multiply(3, 4) = %d\n", result);

    // Function returning boolean-like value
    printf("\n=== Boolean-like Functions ===\n");
    int num = 8;
    if (isEven(num)) {
        printf("%d is even\n", num);
    } else {
        printf("%d is odd\n", num);
    }

    // Using helper functions
    printf("\n=== Using Helper Functions ===\n");
    printLine(40);
    printf("This is a section\n");
    printLine(40);

    return 0;
}

// Function definitions

// void function - no return value
void greet(void) {
    printf("Hello, World!\n");
}

// Function with parameter
void greetPerson(char name[]) {
    printf("Hello, %s!\n", name);
}

// Function with return value
int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

float divide(float a, float b) {
    if (b != 0) {
        return a / b;
    } else {
        printf("Error: Division by zero!\n");
        return 0;
    }
}

// Function returning 1 (true) or 0 (false)
int isEven(int num) {
    return (num % 2 == 0);
}

// Helper function for formatting
void printLine(int length) {
    for (int i = 0; i < length; i++) {
        printf("-");
    }
    printf("\n");
}

/*
 * FUNCTION SYNTAX:
 *
 * return_type function_name(parameter_type parameter_name) {
 *     // function body
 *     return value;  // if return_type is not void
 * }
 *
 * WHY USE FUNCTIONS?
 * 1. Code reusability - write once, use many times
 * 2. Better organization - break complex problems into smaller parts
 * 3. Easier debugging - isolate problems
 * 4. Easier testing - test individual functions
 * 5. Readability - main() shows high-level logic
 *
 * EXERCISES:
 * 1. Write a function to calculate the area of a circle
 * 2. Create a function to check if a number is prime
 * 3. Write a function to find the maximum of three numbers
 * 4. Create a function to calculate factorial
 * 5. Write a function to convert Celsius to Fahrenheit
 * 6. Create a function to print the first n Fibonacci numbers
 * 7. Write a function to check if a year is a leap year
 * 8. Create a function to reverse a number
 */
