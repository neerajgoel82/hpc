/*
 * File: 05_if_else.c
 * Topic: Conditional Statements (if, else, else if)
 *
 * Conditional statements allow your program to make decisions
 * and execute different code based on conditions.
 *
 * Key Concepts:
 * - if statement
 * - if-else statement
 * - else if ladder
 * - Nested if statements
 * - Conditional (ternary) operator
 */

#include <stdio.h>

int main() {
    // Simple if statement
    printf("=== Simple if Statement ===\n");
    int age = 20;
    if (age >= 18) {
        printf("You are an adult.\n");
    }

    // if-else statement
    printf("\n=== if-else Statement ===\n");
    int number = -5;
    if (number >= 0) {
        printf("%d is positive.\n", number);
    } else {
        printf("%d is negative.\n", number);
    }

    // else if ladder
    printf("\n=== else if Ladder ===\n");
    int score = 85;
    if (score >= 90) {
        printf("Grade: A\n");
    } else if (score >= 80) {
        printf("Grade: B\n");
    } else if (score >= 70) {
        printf("Grade: C\n");
    } else if (score >= 60) {
        printf("Grade: D\n");
    } else {
        printf("Grade: F\n");
    }

    // Nested if statements
    printf("\n=== Nested if Statements ===\n");
    int age_check = 25;
    int has_license = 1;  // 1 = true, 0 = false

    if (age_check >= 18) {
        printf("Age requirement met.\n");
        if (has_license) {
            printf("You can drive!\n");
        } else {
            printf("You need a license to drive.\n");
        }
    } else {
        printf("You are too young to drive.\n");
    }

    // Multiple conditions with logical operators
    printf("\n=== Multiple Conditions ===\n");
    int temperature = 25;
    int is_raining = 0;

    if (temperature > 20 && !is_raining) {
        printf("Perfect weather for a walk!\n");
    } else if (temperature > 20 && is_raining) {
        printf("Warm but rainy. Take an umbrella.\n");
    } else {
        printf("Maybe stay indoors.\n");
    }

    // Ternary operator (conditional operator)
    printf("\n=== Ternary Operator ===\n");
    int num = 10;
    char *result = (num % 2 == 0) ? "even" : "odd";
    printf("%d is %s.\n", num, result);

    // Another ternary example
    int a = 15, b = 20;
    int max = (a > b) ? a : b;
    printf("Maximum of %d and %d is %d\n", a, b, max);

    // Practical example: Login system
    printf("\n=== Practical Example: Simple Login ===\n");
    int user_id;
    int password;

    printf("Enter user ID: ");
    scanf("%d", &user_id);
    printf("Enter password: ");
    scanf("%d", &password);

    if (user_id == 1234 && password == 5678) {
        printf("Login successful! Welcome.\n");
    } else if (user_id == 1234) {
        printf("Incorrect password.\n");
    } else {
        printf("User ID not found.\n");
    }

    return 0;
}

/*
 * EXERCISES:
 * 1. Write a program to check if a year is a leap year
 *    (divisible by 4, but not 100, unless also divisible by 400)
 * 2. Create a simple calculator that takes two numbers and an operator
 * 3. Check if a number is positive, negative, or zero
 * 4. Write a program to find the largest of three numbers
 * 5. Create a BMI calculator that categorizes:
 *    - Underweight: BMI < 18.5
 *    - Normal: 18.5 <= BMI < 25
 *    - Overweight: 25 <= BMI < 30
 *    - Obese: BMI >= 30
 */
