/*
 * File: 03_operators.c
 * Topic: Operators in C
 *
 * Operators allow you to perform operations on variables and values.
 * C has arithmetic, relational, logical, and assignment operators.
 *
 * Key Concepts:
 * - Arithmetic operators: +, -, *, /, %
 * - Relational operators: ==, !=, <, >, <=, >=
 * - Logical operators: &&, ||, !
 * - Increment/Decrement: ++, --
 * - Assignment operators: =, +=, -=, *=, /=
 */

#include <stdio.h>

int main() {
    // Arithmetic Operators
    printf("=== Arithmetic Operators ===\n");
    int a = 10, b = 3;
    printf("a = %d, b = %d\n", a, b);
    printf("a + b = %d\n", a + b);       // Addition
    printf("a - b = %d\n", a - b);       // Subtraction
    printf("a * b = %d\n", a * b);       // Multiplication
    printf("a / b = %d\n", a / b);       // Division (integer)
    printf("a %% b = %d\n", a % b);      // Modulus (remainder)

    // Floating-point division
    printf("a / (float)b = %.2f\n", a / (float)b);

    // Increment and Decrement
    printf("\n=== Increment/Decrement ===\n");
    int x = 5;
    printf("x = %d\n", x);
    printf("x++ = %d (post-increment, returns then increments)\n", x++);
    printf("x is now = %d\n", x);
    printf("++x = %d (pre-increment, increments then returns)\n", ++x);
    printf("x is now = %d\n", x);

    // Relational Operators
    printf("\n=== Relational Operators ===\n");
    int num1 = 10, num2 = 20;
    printf("num1 = %d, num2 = %d\n", num1, num2);
    printf("num1 == num2: %d\n", num1 == num2);  // Equal to
    printf("num1 != num2: %d\n", num1 != num2);  // Not equal to
    printf("num1 < num2: %d\n", num1 < num2);    // Less than
    printf("num1 > num2: %d\n", num1 > num2);    // Greater than
    printf("num1 <= num2: %d\n", num1 <= num2);  // Less than or equal
    printf("num1 >= num2: %d\n", num1 >= num2);  // Greater than or equal
    printf("(In C, 1 = true, 0 = false)\n");

    // Logical Operators
    printf("\n=== Logical Operators ===\n");
    int p = 1, q = 0;  // 1 = true, 0 = false
    printf("p = %d (true), q = %d (false)\n", p, q);
    printf("p && q (AND): %d\n", p && q);        // Both must be true
    printf("p || q (OR): %d\n", p || q);         // At least one must be true
    printf("!p (NOT): %d\n", !p);                // Negation

    // Compound Assignment Operators
    printf("\n=== Compound Assignment ===\n");
    int value = 10;
    printf("Initial value = %d\n", value);
    value += 5;  // Same as: value = value + 5
    printf("After value += 5: %d\n", value);
    value -= 3;  // Same as: value = value - 3
    printf("After value -= 3: %d\n", value);
    value *= 2;  // Same as: value = value * 2
    printf("After value *= 2: %d\n", value);
    value /= 4;  // Same as: value = value / 4
    printf("After value /= 4: %d\n", value);

    return 0;
}

/*
 * EXERCISES:
 * 1. Write a program to calculate the area of a circle (area = π * r²)
 * 2. Calculate compound interest: A = P(1 + r)^t
 * 3. Swap two numbers using arithmetic operators (without a temp variable)
 * 4. Check if a number is even or odd using the modulus operator
 * 5. Experiment with operator precedence: what is the result of 2 + 3 * 4?
 */
