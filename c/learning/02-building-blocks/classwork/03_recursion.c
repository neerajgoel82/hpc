/*
 * File: 03_recursion.c
 * Topic: Recursion in C
 *
 * A recursive function is one that calls itself. It's useful for
 * problems that can be broken down into smaller similar problems.
 *
 * Key Concepts:
 * - Base case (stopping condition)
 * - Recursive case
 * - Call stack
 * - When to use recursion vs iteration
 */

#include <stdio.h>

// Function prototypes
int factorial(int n);
int fibonacci(int n);
int sum_of_digits(int n);
void countdown(int n);
int power(int base, int exp);
int gcd(int a, int b);

int main() {
    // Factorial example
    printf("=== Factorial (Recursive) ===\n");
    for (int i = 0; i <= 6; i++) {
        printf("%d! = %d\n", i, factorial(i));
    }

    // Fibonacci example
    printf("\n=== Fibonacci Sequence ===\n");
    printf("First 10 Fibonacci numbers:\n");
    for (int i = 0; i < 10; i++) {
        printf("F(%d) = %d\n", i, fibonacci(i));
    }

    // Sum of digits
    printf("\n=== Sum of Digits ===\n");
    int num = 12345;
    printf("Sum of digits of %d = %d\n", num, sum_of_digits(num));

    // Countdown
    printf("\n=== Countdown ===\n");
    countdown(5);

    // Power calculation
    printf("\n=== Power Calculation ===\n");
    int base = 2, exp = 5;
    printf("%d^%d = %d\n", base, exp, power(base, exp));

    // GCD (Greatest Common Divisor)
    printf("\n=== GCD (Euclidean Algorithm) ===\n");
    int a = 48, b = 18;
    printf("GCD of %d and %d = %d\n", a, b, gcd(a, b));

    return 0;
}

// Factorial: n! = n * (n-1)!
// Base case: 0! = 1
int factorial(int n) {
    // Base case
    if (n == 0 || n == 1) {
        return 1;
    }
    // Recursive case
    return n * factorial(n - 1);
}

// Fibonacci: F(n) = F(n-1) + F(n-2)
// Base cases: F(0) = 0, F(1) = 1
int fibonacci(int n) {
    // Base cases
    if (n == 0) return 0;
    if (n == 1) return 1;

    // Recursive case
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Sum of digits: sum(1234) = 4 + sum(123)
// Base case: n < 10 returns n
int sum_of_digits(int n) {
    // Base case
    if (n < 10) {
        return n;
    }
    // Recursive case: last digit + sum of remaining digits
    return (n % 10) + sum_of_digits(n / 10);
}

// Countdown from n to 1
void countdown(int n) {
    // Base case
    if (n <= 0) {
        printf("Blastoff!\n");
        return;
    }
    // Recursive case
    printf("%d...\n", n);
    countdown(n - 1);
}

// Power: base^exp = base * base^(exp-1)
// Base case: base^0 = 1
int power(int base, int exp) {
    // Base case
    if (exp == 0) {
        return 1;
    }
    // Recursive case
    return base * power(base, exp - 1);
}

// GCD using Euclidean algorithm
// gcd(a, b) = gcd(b, a % b)
// Base case: gcd(a, 0) = a
int gcd(int a, int b) {
    // Base case
    if (b == 0) {
        return a;
    }
    // Recursive case
    return gcd(b, a % b);
}

/*
 * RECURSION REQUIREMENTS:
 * 1. Base case - condition to stop recursion
 * 2. Recursive case - function calls itself with different parameter
 * 3. Progress toward base case - each call should get closer to base case
 *
 * RECURSION vs ITERATION:
 *
 * Use Recursion when:
 * - Problem naturally breaks into similar subproblems
 * - Tree or graph traversal
 * - Code is clearer and more elegant
 * Examples: Fibonacci, tree traversal, divide-and-conquer
 *
 * Use Iteration when:
 * - Simple counting or accumulation
 * - Performance is critical (recursion has overhead)
 * - Deep recursion might cause stack overflow
 * Examples: Simple loops, array processing
 *
 * COMMON PITFALLS:
 * - Forgetting base case -> infinite recursion -> stack overflow
 * - Not progressing toward base case
 * - Too many recursive calls -> poor performance (fibonacci!)
 *
 * EXERCISES:
 * 1. Write recursive function to reverse a number
 * 2. Implement binary search recursively
 * 3. Calculate sum of first n natural numbers recursively
 * 4. Print numbers from 1 to n using recursion
 * 5. Find maximum element in array using recursion
 * 6. Check if a string is palindrome using recursion
 * 7. Calculate length of string recursively
 * 8. Implement tower of Hanoi puzzle
 */
