/*
 * File: 07_loops.c
 * Topic: Loops in C (while, do-while, for)
 *
 * Loops allow you to execute a block of code repeatedly.
 * C has three types of loops: while, do-while, and for.
 *
 * Key Concepts:
 * - while loop
 * - do-while loop
 * - for loop
 * - Nested loops
 * - break and continue statements
 * - Infinite loops
 */

#include <stdio.h>

int main() {
    // while loop - checks condition before executing
    printf("=== while Loop ===\n");
    int count = 1;
    while (count <= 5) {
        printf("Count: %d\n", count);
        count++;
    }

    // do-while loop - executes at least once, then checks condition
    printf("\n=== do-while Loop ===\n");
    int num = 1;
    do {
        printf("Number: %d\n", num);
        num++;
    } while (num <= 5);

    // Demonstrating do-while executes at least once
    printf("\nEven with false condition, do-while runs once:\n");
    int x = 10;
    do {
        printf("x = %d (condition is false, but this prints)\n", x);
    } while (x < 5);

    // for loop - most common loop for counting
    printf("\n=== for Loop ===\n");
    for (int i = 1; i <= 5; i++) {
        printf("i = %d\n", i);
    }

    // for loop - counting backwards
    printf("\nCounting backwards:\n");
    for (int i = 5; i >= 1; i--) {
        printf("%d ", i);
    }
    printf("\n");

    // for loop - different increments
    printf("\nSkipping by 2:\n");
    for (int i = 0; i <= 10; i += 2) {
        printf("%d ", i);
    }
    printf("\n");

    // Nested loops
    printf("\n=== Nested Loops ===\n");
    printf("Multiplication table (1-5):\n");
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= 5; j++) {
            printf("%4d", i * j);
        }
        printf("\n");
    }

    // break statement - exits the loop immediately
    printf("\n=== break Statement ===\n");
    printf("Finding first number divisible by 7:\n");
    for (int i = 1; i <= 50; i++) {
        if (i % 7 == 0) {
            printf("Found: %d\n", i);
            break;  // Exit the loop
        }
    }

    // continue statement - skips rest of current iteration
    printf("\n=== continue Statement ===\n");
    printf("Odd numbers from 1 to 10:\n");
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) {
            continue;  // Skip even numbers
        }
        printf("%d ", i);
    }
    printf("\n");

    // Practical example: Sum of numbers
    printf("\n=== Sum of First N Numbers ===\n");
    int n = 10;
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    printf("Sum of first %d numbers: %d\n", n, sum);

    // Practical example: Factorial
    printf("\n=== Factorial ===\n");
    int factorial_num = 5;
    long long factorial = 1;
    for (int i = 1; i <= factorial_num; i++) {
        factorial *= i;
    }
    printf("%d! = %lld\n", factorial_num, factorial);

    // Pattern printing with nested loops
    printf("\n=== Pattern Printing ===\n");
    printf("Right triangle pattern:\n");
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= i; j++) {
            printf("* ");
        }
        printf("\n");
    }

    printf("\nInverted triangle pattern:\n");
    for (int i = 5; i >= 1; i--) {
        for (int j = 1; j <= i; j++) {
            printf("* ");
        }
        printf("\n");
    }

    // Interactive example with while loop
    printf("\n=== Interactive Example ===\n");
    int input;
    printf("Enter numbers (0 to stop):\n");
    sum = 0;
    int count_nums = 0;

    while (1) {  // Infinite loop
        scanf("%d", &input);
        if (input == 0) {
            break;  // Exit when user enters 0
        }
        sum += input;
        count_nums++;
    }

    if (count_nums > 0) {
        printf("Sum: %d, Average: %.2f\n", sum, (float)sum / count_nums);
    }

    return 0;
}

/*
 * CHOOSING THE RIGHT LOOP:
 * - for loop: When you know how many iterations you need
 * - while loop: When you don't know iterations, check condition first
 * - do-while loop: When you want to execute at least once
 *
 * EXERCISES:
 * 1. Print all even numbers from 1 to 100
 * 2. Calculate the sum of digits of a number (e.g., 123 -> 1+2+3 = 6)
 * 3. Print the Fibonacci sequence up to n terms
 * 4. Check if a number is prime
 * 5. Create a number guessing game using a loop
 * 6. Print a pyramid pattern:
 *        *
 *       ***
 *      *****
 *     *******
 * 7. Find the GCD of two numbers using a loop
 * 8. Reverse a number (e.g., 123 becomes 321)
 */
